#!/usr/bin/env python3

"""
SAC training with a staged curriculum.

    .venv-drl/bin/python -m robot_controller.sac_train --steps 2000000

WHY A CURRICULUM
----------------
Starting on dense mazes with fast obstacles gives SAC almost no
successful trajectories to learn from, so the terminal reward never
enters the critic's estimates and the policy settles on whatever is
least bad among the outcomes it HAS seen - usually creeping around
safely until the step limit. That failure was measured directly earlier
in this project: success collapsed 20.7% -> 4.5% while timeouts rose to
59%.

ADVANCE ON COMPETENCE, NOT ON STEP COUNT
----------------------------------------
Promotion is on a rolling success rate, and there is a DEMOTION rule as
well. Promotion at a noisy plateau will occasionally fire on a lucky
window; without demotion the ladder then sits at a stage the agent
cannot actually do and never recovers.

The 60% promotion threshold is measured, not guessed: pilot runs on this
task promoted reliably at 60-72% per stage and never sustained 80%, so
an 80% bar would have stalled the ladder at stage 2.

DIFFICULTY SORTING USES BFS - TRAINING ONLY
-------------------------------------------
`SacMazeEnv._bfs` measures how far the goal is along the route the robot
must actually drive, so "goal within 2 cells" means two steps and not
two cells of straight-line distance that happen to be separated by a
wall. Measured on these layouts, 35.7% of Manhattan-adjacent cells are
behind a wall with true paths averaging 2.88 steps and reaching 23, so a
straight-line curriculum never delivers the easy stage it promises.

That BFS is a difficulty sorter for the TRAINER. It is not in the
observation and not in the control loop. The policy has no access to it
and must not be given any - wiring it in would destroy the unknown-maze
property the whole design rests on.
"""

import argparse
import multiprocessing as mp
import os
import time
from collections import deque

import numpy as np

# Torch defaults to one thread per core. With N env subprocesses also
# running, the two oversubscribe the machine and both slow down: a
# measured smoke run managed 86 steps/s against a ~4300 ideal. The env
# workers are the parallelism here; the gradient step is small enough
# (2x256 MLP, batch 256) that extra threads buy little.
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))

from . import sac_model_path
from .sac_agent import SAC
from .sac_env import SacMazeEnv
from .sac_observation import OBS_DIM


# ============================================================
# CURRICULUM
# ============================================================

STAGES = [
    dict(
        name="1 open, static, near goals",
        wall_prob_range=(0.15, 0.15),
        n_obstacles=(0, 0),
        obstacle_speed=(0.0, 0.0),
        max_goal_cells=2,
    ),
    dict(
        name="2 denser walls, any goal",
        wall_prob_range=(0.15, 0.35),
        n_obstacles=(0, 0),
        obstacle_speed=(0.0, 0.0),
        max_goal_cells=None,
    ),
    dict(
        name="3 slow obstacles",
        wall_prob_range=(0.15, 0.35),
        n_obstacles=(1, 2),
        obstacle_speed=(0.0, 0.10),
        max_goal_cells=None,
    ),
    dict(
        name="4 full randomisation",
        wall_prob_range=(0.15, 0.45),
        n_obstacles=(0, 4),
        obstacle_speed=(0.0, 0.30),
        max_goal_cells=None,
    ),
]

PROMOTE_AT = 0.60
DEMOTE_AT = 0.40
WINDOW = 100


# ============================================================
# VECTORISED ENVS
# ============================================================

def _worker(remote, seed, kwargs):

    env = SacMazeEnv(seed=seed, **kwargs)

    try:
        while True:

            cmd, data = remote.recv()

            if cmd == "step":
                obs, rew, term, trunc, info = env.step(data)
                if term or trunc:
                    final_obs, final_info = obs, info
                    obs, _ = env.reset()
                    remote.send((obs, rew, term, trunc, final_info, final_obs))
                else:
                    remote.send((obs, rew, term, trunc, None, None))

            elif cmd == "reset":
                remote.send(env.reset()[0])

            elif cmd == "stage":
                env.set_stage(**data)
                remote.send(True)

            elif cmd == "close":
                remote.close()
                break

    except KeyboardInterrupt:
        pass


class VecEnv:
    """Minimal subprocess vector env. No gym dependency."""

    def __init__(self, n, seed=0, **kwargs):

        ctx = mp.get_context("fork")

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(n)]
        )

        self.procs = []

        for i, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=_worker, args=(wr, seed + i, kwargs),
                            daemon=True)
            p.start()
            self.procs.append(p)
            wr.close()

        self.n = n

    def reset(self):

        for r in self.remotes:
            r.send(("reset", None))

        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions):

        for r, a in zip(self.remotes, actions):
            r.send(("step", a))

        obs, rews, terms, truncs, finals, final_obs = [], [], [], [], [], []

        for r in self.remotes:
            o, rw, te, tr, fi, fo = r.recv()
            obs.append(o)
            rews.append(rw)
            terms.append(te)
            truncs.append(tr)
            finals.append(fi)
            final_obs.append(fo)

        return (np.stack(obs), np.array(rews, dtype=np.float32),
                np.array(terms), np.array(truncs), finals, final_obs)

    def set_stage(self, **kw):

        for r in self.remotes:
            r.send(("stage", kw))

        return [r.recv() for r in self.remotes]

    def close(self):

        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass

        for p in self.procs:
            p.join(timeout=2)


# ============================================================
# TRAIN
# ============================================================

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--buffer", type=int, default=400_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--utd", type=float, default=0.25,
                   help="update-to-data ratio: gradient steps per env "
                        "step. A SAC gradient step costs 13-22 ms here "
                        "(six forwards and three backwards through twin "
                        "critics plus actor), so UTD 1.0 puts 2M steps "
                        "at 7+ hours of gradient time alone; 0.25 puts "
                        "it near 1.8 h. Lower is faster per step but "
                        "less sample-efficient - judge by wall clock to "
                        "a given success rate, never by steps/s.")
    p.add_argument("--name", default="policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-stage", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--save-every", type=int, default=50_000)
    args = p.parse_args()

    out_dir = sac_model_path.ensure_dir()
    ckpt = sac_model_path.checkpoint_path(args.name)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(out_dir, "tb", args.name))
    except Exception:
        writer = None

    stage = args.start_stage

    envs = VecEnv(args.envs, seed=args.seed,
                  **{k: v for k, v in STAGES[stage].items() if k != "name"})

    agent = SAC(
        obs_dim=OBS_DIM,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer,
        batch_size=args.batch,
        learning_starts=args.learning_starts,
        seed=args.seed,
    )

    print(f"checkpoint -> {sac_model_path.describe(args.name)}")
    print(f"stage {stage}: {STAGES[stage]['name']}")

    obs = envs.reset()

    outcomes = deque(maxlen=WINDOW)
    returns = deque(maxlen=WINDOW)
    ep_return = np.zeros(args.envs)

    reward_terms = {}
    t0 = time.time()
    grad_debt = 0.0

    stage_entered_at = 0

    for step in range(0, args.steps, args.envs):

        if len(agent.buffer) < args.learning_starts:
            actions = np.random.uniform(-1, 1, size=(args.envs, 2)
                                        ).astype(np.float32)
        else:
            actions = agent.act(obs)

        next_obs, rews, terms, truncs, finals, final_obs = envs.step(actions)

        ep_return += rews

        for i in range(args.envs):

            # The stored next_obs must be the TRUE next state. When a
            # worker auto-resets, next_obs is already the new episode's
            # first observation, so use the saved final one instead -
            # otherwise the critic learns a transition that bridges two
            # unrelated episodes.
            stored_next = (
                final_obs[i] if final_obs[i] is not None else next_obs[i]
            )

            # Bootstrap through truncation: only true termination ends
            # the world.
            agent.buffer.add(obs[i], actions[i], rews[i], stored_next,
                             bool(terms[i]))

            if finals[i] is not None:

                info = finals[i]

                outcomes.append(
                    "reached" if info["reached"]
                    else "collided" if info["collided"]
                    else "timeout"
                )

                returns.append(ep_return[i])
                ep_return[i] = 0.0

                for k, v in info.items():
                    if k.startswith("r_"):
                        reward_terms.setdefault(k, []).append(v)

        obs = next_obs

        # ---- learn --------------------------------------
        grad_debt += args.utd * args.envs

        while grad_debt >= 1.0:
            metrics = agent.update()
            grad_debt -= 1.0
            if metrics is None:
                grad_debt = 0.0
                break

        # ---- stage management ---------------------------
        if len(outcomes) == WINDOW and step - stage_entered_at > 20_000:

            success = sum(o == "reached" for o in outcomes) / WINDOW

            if success >= PROMOTE_AT and stage < len(STAGES) - 1:
                stage += 1
                envs.set_stage(
                    **{k: v for k, v in STAGES[stage].items() if k != "name"}
                )
                outcomes.clear()
                stage_entered_at = step
                print(f"  [{step:>9,}] PROMOTE -> stage {stage}: "
                      f"{STAGES[stage]['name']}  (success {success:.0%})",
                      flush=True)

            elif success < DEMOTE_AT and stage > 0:
                stage -= 1
                envs.set_stage(
                    **{k: v for k, v in STAGES[stage].items() if k != "name"}
                )
                outcomes.clear()
                stage_entered_at = step
                print(f"  [{step:>9,}] DEMOTE  -> stage {stage}: "
                      f"{STAGES[stage]['name']}  (success {success:.0%})",
                      flush=True)

        # ---- logging ------------------------------------
        if step % args.log_every < args.envs and outcomes:

            n = len(outcomes)
            succ = sum(o == "reached" for o in outcomes) / n
            coll = sum(o == "collided" for o in outcomes) / n
            tout = sum(o == "timeout" for o in outcomes) / n
            sps = step / max(time.time() - t0, 1e-9)

            print(
                f"  {step:>9,}  stage {stage}  success {succ:5.1%}  "
                f"collision {coll:5.1%}  timeout {tout:5.1%}  "
                f"({sps:,.0f} steps/s)",
                flush=True,
            )

            if writer is not None:

                writer.add_scalar("outcome/success", succ, step)
                writer.add_scalar("outcome/collision", coll, step)
                writer.add_scalar("outcome/timeout", tout, step)
                writer.add_scalar("curriculum/stage", stage, step)

                if returns:
                    writer.add_scalar("episode/return", np.mean(returns), step)

                    # Degenerate-policy check: return must track SUCCESS,
                    # not episode length. If this correlation goes
                    # negative the agent is farming something.
                    if len(returns) == len(outcomes) and len(returns) > 10:
                        succ_vec = np.array(
                            [o == "reached" for o in outcomes], dtype=float
                        )
                        if succ_vec.std() > 0:
                            writer.add_scalar(
                                "episode/return_success_corr",
                                float(np.corrcoef(returns, succ_vec)[0, 1]),
                                step,
                            )

                for k, v in reward_terms.items():
                    if v:
                        writer.add_scalar(f"reward/{k}", float(np.mean(v)),
                                          step)

                reward_terms.clear()

        if step % args.save_every < args.envs and step > 0:
            agent.save(ckpt)

    agent.save(ckpt)
    envs.close()

    if writer is not None:
        writer.close()

    print(f"\nsaved -> {ckpt}.pt")
    print(f"reached stage {stage} ({STAGES[stage]['name']})")
    print(f"wall clock: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

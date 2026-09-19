#!/usr/bin/env python3

"""
Soft Actor-Critic. Standard implementation, nothing exotic.

    twin Q critics with target networks, polyak tau = 0.005
    automatic entropy tuning, target entropy = -dim(A)
    gamma 0.99, lr 3e-4, batch 256
    MLP 2x256 ReLU, tanh-squashed Gaussian policy

REPLAY BUFFER SIZE
------------------
The spec asked for 1e6. At 151-dim float32 observations that is

    1e6 * 151 * 4 bytes * 2 (obs and next_obs) = 1.21 GB

on a machine with roughly 3 GB free that must also hold the torch
runtime, the parallel environment processes and the networks. The
default here is 400k (0.48 GB); raise it with --buffer if you have the
memory. `ReplayBuffer` warns at construction if the request looks larger
than available RAM, rather than dying with an opaque MemoryError part
way through a long run.

CHECKPOINTS
-----------
`save()` writes the ObsSpec alongside the weights and `load()` verifies
it. A checkpoint can therefore never be silently paired with a
mismatched observation builder - the failure mode that produced a
"trained" policy behaving randomly earlier in this project.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sac_observation import OBS_DIM, ObsSpec


ACTION_DIM = 2

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

EPS = 1e-6


def mlp(in_dim, out_dim, hidden=(256, 256)):

    layers = []
    last = in_dim

    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h

    layers.append(nn.Linear(last, out_dim))

    return nn.Sequential(*layers)


# ============================================================
# NETWORKS
# ============================================================

class Actor(nn.Module):
    """Tanh-squashed Gaussian."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACTION_DIM, hidden=(256, 256)):

        super().__init__()

        self.net = mlp(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def forward(self, obs):

        mu, log_std = self.net(obs).chunk(2, dim=-1)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mu, log_std

    def sample(self, obs, deterministic=False, with_logprob=True):

        mu, log_std = self(obs)

        if deterministic:
            return torch.tanh(mu), None

        std = log_std.exp()

        normal = torch.distributions.Normal(mu, std)

        u = normal.rsample()

        action = torch.tanh(u)

        if not with_logprob:
            return action, None

        # Change-of-variables correction for the tanh squash:
        #   log p(a) = log p(u) - sum log(1 - tanh(u)^2)
        log_prob = normal.log_prob(u).sum(-1)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + EPS).sum(-1)

        return action, log_prob


class Critic(nn.Module):
    """Twin Q. Both heads in one module so they update together."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACTION_DIM, hidden=(256, 256)):

        super().__init__()

        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):

        x = torch.cat([obs, act], dim=-1)

        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ============================================================
# REPLAY
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity, obs_dim=OBS_DIM, act_dim=ACTION_DIM):

        need = capacity * obs_dim * 4 * 2 / 1e9

        try:
            import shutil  # noqa: F401
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) / 1e6
                        break
                else:
                    avail = None
        except Exception:
            avail = None

        if avail is not None and need > avail * 0.7:
            print(
                f"WARNING: replay buffer needs ~{need:.2f} GB but only "
                f"{avail:.2f} GB is available.\n"
                f"         Lower --buffer or the run will be killed "
                f"part way through."
            )

        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)

        # `done` is TERMINATION only, never truncation. Bootstrapping
        # must continue through a time-limit cutoff: the episode ended
        # because the clock ran out, not because the world did, and
        # treating it as terminal teaches the agent that the world stops
        # existing at step 800.
        self.done = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, act, rew, next_obs, done):

        i = self.ptr

        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def __len__(self):

        return self.capacity if self.full else self.ptr

    def sample(self, batch_size, device):

        idx = np.random.randint(0, len(self), size=batch_size)

        t = lambda a: torch.as_tensor(a[idx], device=device)  # noqa: E731

        return t(self.obs), t(self.act), t(self.rew), t(self.next_obs), \
            t(self.done)


# ============================================================
# AGENT
# ============================================================

class SAC:

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACTION_DIM, hidden=(256, 256),
                 lr=3e-4, gamma=0.99, tau=0.005, buffer_size=400_000,
                 batch_size=256, learning_starts=10_000, device="cpu",
                 seed=None):

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device(device)

        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning. log_alpha is optimised rather than
        # alpha so it stays positive without a constraint.
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        self.updates = 0

    @property
    def alpha(self):

        return self.log_alpha.exp().detach()

    # ----------------------------------------------------
    # ACTING
    # ----------------------------------------------------

    @torch.no_grad()
    def act(self, obs, deterministic=False):

        single = np.asarray(obs).ndim == 1

        o = torch.as_tensor(
            np.atleast_2d(np.asarray(obs, dtype=np.float32)),
            device=self.device,
        )

        a, _ = self.actor.sample(o, deterministic=deterministic,
                                 with_logprob=False)

        a = a.cpu().numpy()

        return a[0] if single else a

    # ----------------------------------------------------
    # LEARNING
    # ----------------------------------------------------

    def update(self):

        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None

        obs, act, rew, next_obs, done = self.buffer.sample(
            self.batch_size, self.device
        )

        # ---- critic ----------------------------------------
        with torch.no_grad():

            next_act, next_logp = self.actor.sample(next_obs)

            tq1, tq2 = self.critic_target(next_obs, next_act)

            target_q = torch.min(tq1, tq2) - self.alpha * next_logp

            backup = rew + self.gamma * (1.0 - done) * target_q

        q1, q2 = self.critic(obs, act)

        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- actor -----------------------------------------
        for p in self.critic.parameters():
            p.requires_grad_(False)

        new_act, logp = self.actor.sample(obs)

        q1_pi, q2_pi = self.critic(obs, new_act)

        actor_loss = (self.alpha * logp - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

        # ---- temperature -----------------------------------
        alpha_loss = -(
            self.log_alpha * (logp.detach() + self.target_entropy)
        ).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---- polyak ----------------------------------------
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)

        self.updates += 1

        return {
            "critic_loss": critic_loss.detach().item(),
            "actor_loss": actor_loss.detach().item(),
            "alpha": self.alpha.item(),
            "entropy": (-logp).detach().mean().item(),
        }

    # ----------------------------------------------------
    # PERSISTENCE
    # ----------------------------------------------------

    def save(self, path):
        """Weights + ObsSpec. The spec is not optional metadata."""

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "obs_spec": ObsSpec().as_dict(),
                "arch": {
                    "obs_dim": self.obs_dim,
                    "act_dim": self.act_dim,
                    "hidden": list(self.hidden),
                },
                "updates": self.updates,
            },
            path + ".pt",
        )

    @classmethod
    def load(cls, path, device="cpu", **kw):

        blob = torch.load(path + ".pt", map_location=device,
                          weights_only=False)

        # Refuse a checkpoint built against a different observation.
        ObsSpec.check(blob["obs_spec"])

        arch = blob["arch"]

        agent = cls(
            obs_dim=arch["obs_dim"],
            act_dim=arch["act_dim"],
            hidden=tuple(arch["hidden"]),
            device=device,
            **kw,
        )

        agent.actor.load_state_dict(blob["actor"])
        agent.critic.load_state_dict(blob["critic"])
        agent.critic_target.load_state_dict(blob["critic_target"])

        with torch.no_grad():
            agent.log_alpha.copy_(blob["log_alpha"].to(device))

        agent.updates = blob.get("updates", 0)

        return agent

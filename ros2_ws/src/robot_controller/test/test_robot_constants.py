"""robot_constants.py must agree with robot.urdf.xacro.

The xacro is the source of truth - Gazebo reads it, so the simulated
robot's behaviour follows from it. If the Python constants drift, the
policy is trained against a robot that does not exist and the gap shows
up only as unexplained sim-to-sim error.

Runs under `colcon test`, so drift fails in CI.
"""

import os
import re

import pytest

from robot_controller import robot_constants as C


XACRO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "robot_description", "urdf", "robot.urdf.xacro",
)


def xacro_properties():
    """Parse <xacro:property name=".." value=".."/> into a dict."""

    if not os.path.isfile(XACRO):
        pytest.skip(f"xacro not found at {XACRO}")

    text = open(XACRO).read()

    out = {}
    for m in re.finditer(
        r'<xacro:property\s+name="([^"]+)"\s+value="([^"]+)"', text
    ):
        name, value = m.group(1), m.group(2)
        try:
            out[name] = float(value)
        except ValueError:
            pass

    return out


@pytest.mark.parametrize("xacro_name,python_value", [
    ("wheelbase", C.WHEELBASE),
    ("track_width", C.TRACK_WIDTH),
    ("wheel_radius", C.WHEEL_RADIUS),
    ("max_steer", C.MAX_STEER),
    ("chassis_length", C.CHASSIS_LENGTH),
    ("chassis_width", C.CHASSIS_PLATE_WIDTH),
])
def test_matches_xacro(xacro_name, python_value):

    props = xacro_properties()

    assert xacro_name in props, f"{xacro_name} missing from the xacro"

    assert props[xacro_name] == pytest.approx(python_value, abs=1e-9), (
        f"{xacro_name}: xacro={props[xacro_name]} "
        f"robot_constants={python_value}"
    )


def test_lidar_matches_xacro():
    """Beam count and range come from the sensor block, not a property."""

    text = open(XACRO).read()

    samples = int(re.search(r"<samples>(\d+)</samples>", text).group(1))
    rmin = float(re.search(r"<min>([\d.]+)</min>", text).group(1))
    rmax = float(re.search(r"<max>([\d.]+)</max>", text).group(1))

    assert samples == C.LIDAR_BEAMS
    assert rmin == pytest.approx(C.LIDAR_RANGE_MIN)
    assert rmax == pytest.approx(C.LIDAR_RANGE_MAX)


def test_derived_geometry_selfconsistent():
    """Guards against someone 'simplifying' the derived values."""

    import math

    assert C.MIN_TURN_RADIUS == pytest.approx(
        C.WHEELBASE / math.tan(C.MAX_STEER)
    )

    # Outer swept radius must account for the nose overhang; using
    # R + track/2 alone understates it by ~57 mm.
    assert C.SWEPT_RADIUS_OUTER > C.MIN_TURN_RADIUS + C.BODY_WIDTH / 2.0

    assert C.NOSE_OFFSET == pytest.approx(0.2415, abs=1e-6)

#!/usr/bin/env python3

import argparse
import math


def _vector(values, size, name):
    values = tuple(float(value) for value in values)

    if len(values) != size:
        raise ValueError("%s must contain exactly %d values" % (name, size))

    if not all(math.isfinite(value) for value in values):
        raise ValueError("%s contains NaN or Inf" % name)

    return values


def _finite_scalar(value, name):
    value = float(value)

    if not math.isfinite(value):
        raise ValueError("%s contains NaN or Inf" % name)

    return value


def _normalize_quaternion_wxyz(quaternion):
    w, x, y, z = _vector(quaternion, 4, "quaternion")

    norm = math.sqrt(w * w + x * x + y * y + z * z)

    if norm < 1.0e-12:
        raise ValueError("quaternion norm is too small")

    return w / norm, x / norm, y / norm, z / norm


def wrap_angle_rad(angle):
    """Wrap an angle to [-pi, pi)."""

    angle = _finite_scalar(angle, "angle")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def px4_ned_to_flu(vector):
    """PX4 world NED -> FlightLxx world FLU."""

    x, y, z = _vector(vector, 3, "ned_vector")
    return x, -y, -z


def px4_frd_to_flu(vector):
    """PX4 body FRD -> FlightLxx body FLU."""

    x, y, z = _vector(vector, 3, "frd_vector")
    return x, -y, -z


def px4_quaternion_to_flu(quaternion_wxyz):
    """PX4 quaternion -> FlightLxx quaternion."""

    w, x, y, z = _normalize_quaternion_wxyz(quaternion_wxyz)
    return w, x, -y, -z


def quaternion_to_ypr_rad(quaternion_wxyz):
    """Return yaw, pitch, roll in radians."""

    w, x, y, z = _normalize_quaternion_wxyz(quaternion_wxyz)

    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )

    pitch_value = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, pitch_value)))

    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    return yaw, pitch, roll


def vehicle_odometry_to_observation(
    position_ned,
    quaternion_ned_frd_wxyz,
    velocity_ned,
    angular_velocity_frd,
    physical_hover=(0.0, 0.0, 5.0),
    nn_goal=(0.0, 0.0, 5.0),
    physical_yaw_reference=0.0,
    nn_yaw_goal=0.0,
):
    """Build FlightLxx's 12D observation from PX4 odometry."""

    position_flu = px4_ned_to_flu(position_ned)

    quaternion_flu = px4_quaternion_to_flu(
        quaternion_ned_frd_wxyz
    )

    velocity_flu = px4_ned_to_flu(velocity_ned)

    angular_velocity_flu = px4_frd_to_flu(
        angular_velocity_frd
    )

    hover = _vector(
        physical_hover,
        3,
        "physical_hover",
    )

    goal = _vector(
        nn_goal,
        3,
        "nn_goal",
    )

    physical_yaw_reference = _finite_scalar(
        physical_yaw_reference,
        "physical_yaw_reference",
    )

    nn_yaw_goal = _finite_scalar(
        nn_yaw_goal,
        "nn_yaw_goal",
    )

    position_for_network = tuple(
        position_flu[index] - hover[index] + goal[index]
        for index in range(3)
    )

    raw_yaw, pitch, roll = quaternion_to_ypr_rad(
        quaternion_flu
    )

    yaw_for_network = wrap_angle_rad(
        raw_yaw
        - physical_yaw_reference
        + nn_yaw_goal
    )

    return (
        position_for_network[0],
        position_for_network[1],
        position_for_network[2],
        yaw_for_network,
        pitch,
        roll,
        velocity_flu[0],
        velocity_flu[1],
        velocity_flu[2],
        angular_velocity_flu[0],
        angular_velocity_flu[1],
        angular_velocity_flu[2],
    )


def main():
    parser = argparse.ArgumentParser(
        description="PX4 -> FlightLxx frame conversion checks"
    )
    parser.parse_args()

    observation = vehicle_odometry_to_observation(
        position_ned=(1.0, 2.0, -3.0),
        quaternion_ned_frd_wxyz=(1.0, 0.0, 0.0, 0.0),
        velocity_ned=(4.0, 5.0, -6.0),
        angular_velocity_frd=(0.1, 0.2, 0.3),
    )

    print("identity attitude observation:")
    print(tuple(round(value, 6) for value in observation))

    half_pi = math.pi / 4.0
    q_ned_yaw_90 = (
        math.cos(half_pi),
        0.0,
        0.0,
        math.sin(half_pi),
    )

    q_flu = px4_quaternion_to_flu(q_ned_yaw_90)
    raw_yaw, pitch, roll = quaternion_to_ypr_rad(q_flu)

    print("NED +90 deg yaw -> raw FLU yaw [rad]:")
    print(round(raw_yaw, 6))

    aligned_observation = vehicle_odometry_to_observation(
        position_ned=(0.0, 0.0, 0.0),
        quaternion_ned_frd_wxyz=q_ned_yaw_90,
        velocity_ned=(0.0, 0.0, 0.0),
        angular_velocity_frd=(0.0, 0.0, 0.0),
        physical_hover=(0.0, 0.0, 0.0),
        nn_goal=(0.0, 0.0, 0.0),
        physical_yaw_reference=raw_yaw,
        nn_yaw_goal=0.0,
    )

    print("aligned yaw with physical reference [rad]:")
    print(round(aligned_observation[3], 6))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
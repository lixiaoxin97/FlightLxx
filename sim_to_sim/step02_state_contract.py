#!/usr/bin/env python3

import argparse
import math


STATE_CONTRACT_VERSION = "world_yaw_rotation_eigen_zyx_v2"


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
    """Return signed yaw/roll and principal pitch, for heading calibration.

    Policy observations use quaternion_to_training_ypr_rad instead: the
    training environment's Eigen Euler-angle branch is different.
    """

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


def rotate_world_vector_yaw(vector, angle):
    """Apply the fixed world-axis rotation Rz(angle) to a world vector."""
    x, y, z = _vector(vector, 3, "world_vector")
    angle = _finite_scalar(angle, "world_yaw_rotation")
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y, z


def rotate_attitude_world_yaw(quaternion_wxyz, angle):
    """Left-multiply a body-to-world attitude by the world rotation Rz."""
    w, x, y, z = _normalize_quaternion_wxyz(quaternion_wxyz)
    angle = _finite_scalar(angle, "world_yaw_rotation")
    c, s = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return c*w - s*z, c*x - s*y, c*y + s*x, c*z + s*w


def quaternion_to_training_ypr_rad(quaternion_wxyz):
    """Match Eigen Matrix3::eulerAngles(2, 1, 0) used by getObs().

    These are ZYX Euler angles in radians, with yaw in [0, pi]. A negative
    principal yaw selects the equivalent alternate pitch/roll branch. Do not
    wrap yaw alone afterwards: all three angles describe the same rotation.
    Keep quaternion_to_ypr_rad for signed physical heading calibration.
    """
    w, x, y, z = _normalize_quaternion_wxyz(quaternion_wxyz)
    r00, r01, r02 = 1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)
    r10, r11, r12 = 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)
    r20, r21, r22 = 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
    yaw = math.atan2(r10, r00)
    cos_pitch_abs = math.hypot(r22, r21)
    if yaw < 0.0:
        yaw += math.pi
        pitch = math.atan2(-r20, -cos_pitch_abs)
    else:
        pitch = math.atan2(-r20, cos_pitch_abs)
    s, c = math.sin(yaw), math.cos(yaw)
    roll = math.atan2(s*r02 - c*r12, c*r11 - s*r01)
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
    """Build a consistent policy world frame from PX4 NED/FRD odometry.

    After NED/FRD -> world/body FLU, alpha = nn_yaw_goal - physical_yaw_reference:
      p_nn = Rz(alpha) * (p_flu - physical_hover) + nn_goal
      v_nn = Rz(alpha) * v_flu
      R_nn = Rz(alpha) * R_flu
    physical_hover is in the original world FLU axes; nn_goal is in network
    world axes. Body rates only change FRD -> FLU, never with world yaw.
    The reference is fixed for a run, so no rotating-frame velocity term is
    needed. Attitude is encoded using the training environment's Eigen ZYX.
    """

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

    world_yaw_rotation = wrap_angle_rad(nn_yaw_goal - physical_yaw_reference)
    position_error = tuple(
        position_flu[index] - hover[index]
        for index in range(3)
    )
    rotated_error = rotate_world_vector_yaw(position_error, world_yaw_rotation)
    position_for_network = tuple(
        rotated_error[index] + goal[index] for index in range(3)
    )
    velocity_for_network = rotate_world_vector_yaw(
        velocity_flu, world_yaw_rotation
    )
    quaternion_for_network = rotate_attitude_world_yaw(
        quaternion_flu, world_yaw_rotation
    )
    yaw_for_network, pitch, roll = quaternion_to_training_ypr_rad(
        quaternion_for_network
    )

    return (
        position_for_network[0],
        position_for_network[1],
        position_for_network[2],
        yaw_for_network,
        pitch,
        roll,
        velocity_for_network[0],
        velocity_for_network[1],
        velocity_for_network[2],
        angular_velocity_flu[0],
        angular_velocity_flu[1],
        angular_velocity_flu[2],
    )


def main():
    parser = argparse.ArgumentParser(
        description="PX4 -> FlightLxx frame conversion checks"
    )
    parser.add_argument("--self-test", action="store_true",
                        help="run offline frame regressions; no ROS commands")
    args = parser.parse_args()
    if args.self_test:
        import unittest
        import test_state_contract
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_state_contract)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

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

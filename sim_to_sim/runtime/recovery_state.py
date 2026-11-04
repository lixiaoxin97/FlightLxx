#!/usr/bin/env python3
"""Convert a FlightGym initial state into PX4/Gazebo spawn coordinates."""

import argparse
import json
import math

from state_contract import vehicle_odometry_to_observation


GRAVITY = 9.81
PROTOTYPE_MODEL_TO_BASELINK_Z_M = 0.0


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw-ax*bx-ay*by-az*bz,
            aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw)


def quaternion_conjugate(q):
    return q[0], -q[1], -q[2], -q[3]


def rotate_vector(q, vector):
    rotated = quaternion_multiply(
        quaternion_multiply(q, (0.0, *vector)), quaternion_conjugate(q)
    )
    return rotated[1:]


def quaternion_from_zyx(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (cy*cp*cr+sy*sp*sr, cy*cp*sr-sy*sp*cr,
            cy*sp*cr+sy*cp*sr, sy*cp*cr-cy*sp*sr)


def quaternion_to_zyx(q):
    w, x, y, z = q
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch_term = max(-1.0, min(1.0, 2*(w*y-z*x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw


def validate_training_state(state):
    state = tuple(float(value) for value in state)
    if len(state) != 12 or not all(math.isfinite(value) for value in state):
        raise ValueError("initial state must contain 12 finite numbers")
    if not (-1 <= state[0] <= 1 and -1 <= state[1] <= 1 and 4 <= state[2] <= 6):
        raise ValueError("position is outside random_initial_state() bounds")
    if not (-math.pi <= state[3] <= math.pi and -math.pi <= state[4] <= math.pi
            and 0 <= state[5] <= 2*math.pi):
        raise ValueError("Euler state is outside random_initial_state() bounds")
    if any(abs(value) > 1 for value in state[6:]):
        raise ValueError("velocity/rate is outside random_initial_state() bounds")
    return state


def recovery_spawn(state, attitude_encoding="eigen_zyx",
                   model_to_baselink_z=PROTOTYPE_MODEL_TO_BASELINK_Z_M):
    """Return the PX4 GZ spawn pose and equivalent expected policy input.

    FlightGym state coordinates are NWU/FLU. Gazebo uses ENU/FLU. PX4's
    PX4_GZ_MODEL_POSE controls the model frame. Recovery base_link has zero
    offset; the normal Prototype's +0.24 m offset can be supplied explicitly.
    Heading is kept absolute relative to the training frame; it is never
    recalibrated against the randomized initial yaw.
    """
    state = validate_training_state(state)
    roll, pitch, yaw = state[3:6]
    q_policy = quaternion_from_zyx(roll, pitch, yaw)
    q_nwu_to_enu = (math.sqrt(.5), 0.0, 0.0, math.sqrt(.5))
    q_enu_flu = quaternion_multiply(q_nwu_to_enu, q_policy)

    # NWU -> Gazebo ENU, then compensate the model-to-base_link offset.
    position_link_enu = (-state[1], state[0], state[2])
    offset_world = rotate_vector(q_enu_flu, (0.0, 0.0, model_to_baselink_z))
    position_model_enu = tuple(p-o for p, o in zip(position_link_enu, offset_world))
    pose = position_model_enu + quaternion_to_zyx(q_enu_flu)

    # GZBridge maps the canonical link pose/velocity to PX4 NED/FRD. Evaluate
    # that exact incoming odometry contract offline as the expected PX4 policy
    # observation. GZ OdometryPublisher reports linear velocity in body FLU.
    velocity_enu = (-state[7], state[6], state[8])
    velocity_body_flu = rotate_vector(quaternion_conjugate(q_enu_flu), velocity_enu)
    q_enu_to_ned = (0.0, math.sqrt(.5), math.sqrt(.5), 0.0)
    q_flu_to_frd = (0.0, 1.0, 0.0, 0.0)
    q_px4_frd_to_ned = quaternion_multiply(
        quaternion_multiply(q_enu_to_ned, q_enu_flu),
        quaternion_conjugate(q_flu_to_frd),
    )
    velocity_body_frd = (velocity_body_flu[0], -velocity_body_flu[1],
                         -velocity_body_flu[2])
    angular_velocity_frd = (state[9], -state[10], -state[11])
    expected = vehicle_odometry_to_observation(
        position_ned=(position_link_enu[1], position_link_enu[0],
                      -position_link_enu[2]),
        quaternion_ned_frd_wxyz=q_px4_frd_to_ned,
        velocity_ned=velocity_body_frd,
        angular_velocity_frd=angular_velocity_frd,
        physical_hover=(0.0, 0.0, 5.0),
        nn_goal=(0.0, 0.0, 5.0),
        physical_yaw_reference=0.0,
        nn_yaw_goal=0.0,
        attitude_encoding=attitude_encoding,
        pose_frame=1,
        velocity_frame=3,
    )
    return {
        "training_state": list(state),
        "px4_gz_model_pose": list(pose),
        "base_link_pose_enu": list(position_link_enu) + list(q_enu_flu),
        "gazebo_linear_velocity_world_enu": list(velocity_enu),
        "gazebo_linear_velocity_body_flu": list(velocity_body_flu),
        "gazebo_angular_velocity_body_flu": list(state[9:12]),
        "expected_policy_observation": list(expected),
        "attitude_encoding": attitude_encoding,
        "heading_reference_rad": 0.0,
        "model_to_baselink_offset_m": model_to_baselink_z,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=float, nargs=12, required=True,
                        help="[x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz]")
    parser.add_argument("--attitude-encoding", choices=("eigen_zyx", "principal_zyx"),
                        default="eigen_zyx")
    parser.add_argument("--model-to-baselink-z", type=float,
                        default=PROTOTYPE_MODEL_TO_BASELINK_Z_M)
    parser.add_argument("--shell", action="store_true",
                        help="print shell-quoted PX4_GZ_MODEL_POSE and vectors")
    args = parser.parse_args()
    result = recovery_spawn(args.state, args.attitude_encoding,
                            args.model_to_baselink_z)
    if args.shell:
        print("PX4_GZ_MODEL_POSE=" + ",".join("%.12g" % x for x in result["px4_gz_model_pose"]))
        print("S2S_EXPECTED_POLICY_OBSERVATION=" + ",".join(
            "%.12g" % x for x in result["expected_policy_observation"]))
        print("S2S_INITIAL_STATE=" + ",".join("%.12g" % x for x in result["training_state"]))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

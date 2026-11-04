#!/usr/bin/env python3
"""Inject and verify a FlightLxx 12D state in the Gazebo Prototype model.

The input vector follows train.disturbances.random_initial_state():
[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]. Position, linear
velocity, and angular velocity use the policy's FlightLxx NWU/FLU frames.
This utility only talks to Gazebo for state changes; it never arms or changes
PX4 flight mode. anchor enters a zero-twist pose hold; set releases that hold.
The initialization launcher uses this only on the disarmed Recovery model.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import uuid

from state_contract import quaternion_to_training_ypr_rad


DEFAULT_COMMAND_TOPIC = "/world/default/model/prototype_recovery_0/flightlxx/set_state"
DEFAULT_ACK_TOPIC = "/world/default/model/prototype_recovery_0/flightlxx/state_ack"


def finite(values, name, length):
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(x) for x in result):
        raise ValueError("%s must contain %d finite values" % (name, length))
    return result


def send(cli, command_topic, ack_topic, operation, values=()):
    request_id = uuid.uuid4().hex[:12]
    payload = ",".join((operation, request_id) +
                       tuple("%.17g" % value for value in values))
    result = subprocess.run(
        [str(cli), command_topic, ack_topic, request_id, payload],
        check=False, capture_output=True, text=True, timeout=8,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError("Gazebo state request failed: " + output)
    return request_id, output


def record(path, event):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def anchor(args):
    physical_anchor = finite(args.physical_anchor, "physical_anchor", 3)
    physical_hover = finite(args.physical_hover, "physical_hover", 3)
    nn_goal = finite(args.nn_goal, "nn_goal", 3)
    yaw = finite((args.physical_yaw_reference, args.nn_yaw_goal), "yaw", 2)
    values = physical_anchor + physical_hover + yaw[:1] + nn_goal + yaw[1:]
    request_id, output = send(args.cli, args.command_topic, args.ack_topic,
                              "anchor", values)
    print(output)
    record(args.log, {
        "kind": "anchor", "request_id": request_id,
        "wall_time_ns": time.time_ns(), "physical_anchor": physical_anchor,
        "physical_hover": physical_hover,
        "physical_yaw_reference": yaw[0], "nn_goal": nn_goal,
        "nn_yaw_goal": yaw[1], "ack": output,
    })


def set_state(args):
    state = finite(args.state, "state", 12)
    request_id, output = send(args.cli, args.command_topic, args.ack_topic,
                              "set", state)
    print(output)
    record(args.log, {
        "kind": "state", "request_id": request_id,
        "wall_time_ns": time.time_ns(), "state_training_order": state,
        "expected_policy_observation": expected_observation(state),
        "ack": output,
    })


def read_state(args):
    request_id, output = send(args.cli, args.command_topic, args.ack_topic,
                              "read")
    print(output)
    record(args.log, {
        "kind": "read", "request_id": request_id,
        "wall_time_ns": time.time_ns(), "ack": output,
    })


def prepare_state(args):
    state = finite(args.state, "state", 12)
    request_id, output = send(args.cli, args.command_topic, args.ack_topic, "prepare", state)
    print(output)
    record(args.log, {"kind": "prepare", "request_id": request_id,
        "wall_time_ns": time.time_ns(), "state_training_order": state, "ack": output})


def release_state(args):
    request_id, output = send(args.cli, args.command_topic, args.ack_topic, "release")
    print(output)
    record(args.log, {"kind": "release", "request_id": request_id,
        "wall_time_ns": time.time_ns(), "ack": output})


def expected_observation(state):
    roll, pitch, yaw = state[3:6]
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q_wxyz = (
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        cy*sp*cr + sy*cp*sr,
        sy*cp*cr - cy*sp*sr,
    )
    yaw_obs, pitch_obs, roll_obs = quaternion_to_training_ypr_rad(q_wxyz)
    return state[:3] + (yaw_obs, pitch_obs, roll_obs) + state[6:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=Path(__file__).resolve().parents[1] /
                        "build/state_injector/flightlxx_state_injector_cli")
    parser.add_argument("--log", type=Path, default=None,
                        help="append request/ack events to this JSONL file")
    parser.add_argument("--command-topic", default=DEFAULT_COMMAND_TOPIC)
    parser.add_argument("--ack-topic", default=DEFAULT_ACK_TOPIC)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    anchor_parser = subparsers.add_parser("anchor")
    anchor_parser.add_argument("--physical-anchor", type=float, nargs=3,
                               default=(0, 0, 5))
    anchor_parser.add_argument("--physical-hover", type=float, nargs=3,
                               default=(0, 0, 5))
    anchor_parser.add_argument("--physical-yaw-reference", type=float,
                               required=True)
    anchor_parser.add_argument("--nn-goal", type=float, nargs=3,
                               default=(0, 0, 5))
    anchor_parser.add_argument("--nn-yaw-goal", type=float, default=0.0)
    anchor_parser.set_defaults(function=anchor)

    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("--state", type=float, nargs=12, required=True)
    set_parser.set_defaults(function=set_state)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--state", type=float, nargs=12, required=True)
    prepare_parser.set_defaults(function=prepare_state)
    release_parser = subparsers.add_parser("release")
    release_parser.set_defaults(function=release_state)

    read_parser = subparsers.add_parser("read")
    read_parser.set_defaults(function=read_state)

    args = parser.parse_args()
    if not args.cli.is_file():
        parser.error("state injector CLI not found; run bash sim_to_sim/build_state_injector.sh")
    try:
        args.function(args)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

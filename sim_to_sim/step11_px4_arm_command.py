#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass


VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
PX4_SYSTEM_ID = 1
PX4_COMPONENT_ID = 1


@dataclass(frozen=True)
class ArmCommand:
    param1: float
    command: int = VEHICLE_CMD_COMPONENT_ARM_DISARM
    target_system: int = PX4_SYSTEM_ID
    target_component: int = PX4_COMPONENT_ID
    source_system: int = PX4_SYSTEM_ID
    source_component: int = PX4_COMPONENT_ID
    from_external: bool = True


def build_arm_command(arm: bool) -> ArmCommand:
    return ArmCommand(param1=1.0 if arm else 0.0)


def run_self_test() -> None:
    arm_cmd = build_arm_command(True)
    disarm_cmd = build_arm_command(False)

    assert arm_cmd.command == 400
    assert disarm_cmd.command == 400
    assert arm_cmd.param1 == 1.0
    assert disarm_cmd.param1 == 0.0
    assert arm_cmd.from_external is True
    assert disarm_cmd.from_external is True

    print("step11 self-test: PASS")


def import_ros():
    try:
        import rclpy
        from px4_msgs.msg import VehicleCommand
    except ImportError as exc:
        raise SystemExit(
            "ERROR: rclpy or px4_msgs is unavailable. "
            "Source ROS 2 Humble and px4_msgs first."
        ) from exc

    return rclpy, VehicleCommand


def publish_command(arm: bool, repeats: int, period: float) -> None:
    rclpy, VehicleCommand = import_ros()

    rclpy.init()
    node = rclpy.create_node("flightlxx_px4_arm_command")
    publisher = node.create_publisher(
        VehicleCommand,
        "/fmu/in/vehicle_command",
        10,
    )

    try:
        command = build_arm_command(arm)

        for index in range(repeats):
            message = VehicleCommand()
            message.timestamp = int(
                node.get_clock().now().nanoseconds / 1000
            )
            message.param1 = command.param1
            message.command = command.command
            message.target_system = command.target_system
            message.target_component = command.target_component
            message.source_system = command.source_system
            message.source_component = command.source_component
            message.from_external = command.from_external

            publisher.publish(message)

            action = "arm" if arm else "disarm"
            remaining = repeats - index - 1
            node.get_logger().info(
                f"published {action} command; remaining={remaining}"
            )

            time.sleep(period)

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--disarm",
        action="store_true",
        help="send disarm instead of arm",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="actually publish the command",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--period",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    if args.period <= 0:
        parser.error("--period must be > 0")

    return args


def main() -> int:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return 0

    arm = not args.disarm
    action = "arm" if arm else "disarm"

    if not args.send:
        command = build_arm_command(arm)
        print(
            f"dry-run: action={action}, "
            f"command={command.command}, "
            f"param1={command.param1}"
        )
        print("add --send to publish")
        return 0

    publish_command(
        arm=arm,
        repeats=args.repeats,
        period=args.period,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
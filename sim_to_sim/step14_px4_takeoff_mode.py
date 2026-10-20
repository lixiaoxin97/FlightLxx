#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass


VEHICLE_CMD_DO_SET_MODE = 176
PX4_CUSTOM_MAIN_MODE_AUTO = 4
PX4_CUSTOM_SUB_MODE_AUTO_TAKEOFF = 2
PX4_SYSTEM_ID = 1
PX4_COMPONENT_ID = 1


@dataclass(frozen=True)
class TakeoffModeCommand:
    command: int = VEHICLE_CMD_DO_SET_MODE
    param1: float = 1.0
    param2: float = float(PX4_CUSTOM_MAIN_MODE_AUTO)
    param3: float = float(PX4_CUSTOM_SUB_MODE_AUTO_TAKEOFF)
    target_system: int = PX4_SYSTEM_ID
    target_component: int = PX4_COMPONENT_ID
    source_system: int = PX4_SYSTEM_ID
    source_component: int = PX4_COMPONENT_ID
    from_external: bool = True


def build_takeoff_mode_command():
    return TakeoffModeCommand()


def run_self_test():
    command = build_takeoff_mode_command()

    assert command.command == 176
    assert command.param1 == 1.0
    assert command.param2 == 4.0
    assert command.param3 == 2.0
    assert command.target_system == 1
    assert command.target_component == 1
    assert command.source_system == 1
    assert command.source_component == 1
    assert command.from_external is True

    print("step14 self-test: PASS")


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Request PX4 multicopter AUTO_TAKEOFF mode"
    )

    parser.add_argument(
        "--send",
        action="store_true",
        help="actually publish the mode command",
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
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    if args.period <= 0.0:
        parser.error("--period must be > 0")

    return args


def publish_takeoff_mode(repeats, period):
    rclpy, VehicleCommand = import_ros()

    rclpy.init()
    node = rclpy.create_node("flightlxx_px4_takeoff_mode")

    publisher = node.create_publisher(
        VehicleCommand,
        "/fmu/in/vehicle_command",
        10,
    )

    try:
        command = build_takeoff_mode_command()

        for index in range(repeats):
            message = VehicleCommand()

            message.timestamp = int(
                node.get_clock().now().nanoseconds / 1000
            )

            message.param1 = command.param1
            message.param2 = command.param2
            message.param3 = command.param3
            message.command = command.command
            message.target_system = command.target_system
            message.target_component = command.target_component
            message.source_system = command.source_system
            message.source_component = command.source_component
            message.from_external = command.from_external

            publisher.publish(message)

            remaining = repeats - index - 1

            node.get_logger().info(
                "published AUTO_TAKEOFF mode command; "
                "remaining=%d" % remaining
            )

            time.sleep(period)

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


def main():
    args = parse_args()

    if args.self_test:
        run_self_test()
        return 0

    command = build_takeoff_mode_command()

    if not args.send:
        print(
            "dry-run: command=%d, param1=%.1f, "
            "main_mode=%.1f, sub_mode=%.1f"
            % (
                command.command,
                command.param1,
                command.param2,
                command.param3,
            )
        )
        print("add --send to publish")
        return 0

    publish_takeoff_mode(
        repeats=args.repeats,
        period=args.period,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
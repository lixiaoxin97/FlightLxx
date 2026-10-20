#!/usr/bin/env python3
"""Step 10: request PX4 Offboard mode without arming the vehicle."""

import argparse
import sys
import time


try:
    import rclpy
    from rclpy.node import Node
    from px4_msgs.msg import VehicleCommand
except ImportError:
    rclpy = None
    Node = object
    VehicleCommand = None


VEHICLE_CMD_DO_SET_MODE = 176


def fill_offboard_command(message, timestamp_us):
    """Fill PX4 VehicleCommand for custom mode 6: Offboard."""
    message.timestamp = int(timestamp_us)

    message.param1 = 1.0
    message.param2 = 6.0
    message.command = VEHICLE_CMD_DO_SET_MODE

    message.target_system = 1
    message.target_component = 1
    message.source_system = 1
    message.source_component = 1
    message.from_external = True


class FakeMessage:
    pass


def run_self_test():
    message = FakeMessage()

    fill_offboard_command(
        message,
        123456,
    )

    assert message.timestamp == 123456
    assert message.param1 == 1.0
    assert message.param2 == 6.0
    assert message.command == 176

    assert message.target_system == 1
    assert message.target_component == 1
    assert message.source_system == 1
    assert message.source_component == 1
    assert message.from_external is True

    print("step10 self-test: PASS")


if rclpy is not None:

    class OffboardCommandNode(Node):
        def __init__(self):
            super().__init__(
                "flightlxx_px4_offboard_command"
            )

            self.publisher = self.create_publisher(
                VehicleCommand,
                "/fmu/in/vehicle_command",
                10,
            )

        def publish_command(self, remaining):
            timestamp_us = (
                self.get_clock().now().nanoseconds
                // 1000
            )

            message = VehicleCommand()

            fill_offboard_command(
                message,
                timestamp_us,
            )

            self.publisher.publish(message)

            self.get_logger().info(
                "published Offboard mode command; "
                "remaining=%d"
                % remaining
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Request PX4 Offboard mode without arming"
        )
    )

    parser.add_argument(
        "--send",
        action="store_true",
        help="actually publish VehicleCommand",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="number of repeated commands",
    )

    parser.add_argument(
        "--period",
        type=float,
        default=0.1,
        help="seconds between repeated commands",
    )

    args = parser.parse_args()

    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    if args.period <= 0.0:
        parser.error("--period must be positive")

    if not args.send:
        print(
            "DRY-RUN: no VehicleCommand was published."
        )
        print(
            "Use --send only while step09 is already "
            "running with --publish."
        )
        return 0

    if rclpy is None:
        print(
            "ERROR: rclpy or px4_msgs is unavailable."
        )
        return 2

    rclpy.init()
    node = OffboardCommandNode()

    try:
        for index in range(args.repeat):
            remaining = args.repeat - index - 1

            node.publish_command(remaining)
            rclpy.spin_once(
                node,
                timeout_sec=0.0,
            )

            if remaining > 0:
                time.sleep(args.period)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        run_self_test()
        raise SystemExit(0)

    raise SystemExit(main())
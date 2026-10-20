#!/usr/bin/env python3

"""Step 7: bridge PX4 VehicleOdometry to the FlightLxx UDP policy server."""

import argparse
import time

from step02_state_contract import vehicle_odometry_to_observation
from step06_policy_client import PolicyClient


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile,
        ReliabilityPolicy,
        DurabilityPolicy,
    )
    from px4_msgs.msg import VehicleOdometry
except ImportError:
    rclpy = None
    Node = object
    QoSProfile = None
    ReliabilityPolicy = None
    DurabilityPolicy = None
    VehicleOdometry = None


def message_to_observation(
    message,
    physical_hover,
    nn_goal,
    physical_yaw_reference=0.0,
    nn_yaw_goal=0.0,
):
    """Convert one PX4 VehicleOdometry message to policy input."""

    return vehicle_odometry_to_observation(
        position_ned=message.position,
        quaternion_ned_frd_wxyz=message.q,
        velocity_ned=message.velocity,
        angular_velocity_frd=message.angular_velocity,
        physical_hover=physical_hover,
        nn_goal=nn_goal,
        physical_yaw_reference=physical_yaw_reference,
        nn_yaw_goal=nn_yaw_goal,
    )


if rclpy is not None:

    class Ros2PolicyBridge(Node):
        def __init__(
            self,
            topic,
            physical_hover,
            nn_goal,
            physical_yaw_reference,
            nn_yaw_goal,
            policy_client,
            max_rate,
            print_period,
        ):
            super().__init__("flightlxx_ros2_policy_bridge")

            self.physical_hover = tuple(physical_hover)
            self.nn_goal = tuple(nn_goal)
            self.physical_yaw_reference = float(
                physical_yaw_reference
            )
            self.nn_yaw_goal = float(nn_yaw_goal)

            self.policy_client = policy_client
            self.request_period = 1.0 / float(max_rate)
            self.print_period = float(print_period)

            self.next_request_time = 0.0
            self.last_print_time = 0.0
            self.last_error_time = 0.0

            self.message_count = 0
            self.request_count = 0
            self.sequence = 0

            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )

            self.subscription = self.create_subscription(
                VehicleOdometry,
                topic,
                self.odometry_callback,
                qos,
            )

            self.get_logger().info(
                "subscribed to %s; policy udp=%s:%d"
                % (
                    topic,
                    policy_client.server_address[0],
                    policy_client.server_address[1],
                )
            )

            self.get_logger().info(
                "physical_hover=%s, nn_goal=%s, "
                "physical_yaw_reference=%.6f, "
                "nn_yaw_goal=%.6f, max_rate=%.1f Hz"
                % (
                    self.physical_hover,
                    self.nn_goal,
                    self.physical_yaw_reference,
                    self.nn_yaw_goal,
                    max_rate,
                )
            )

        def _log_error_throttled(self, message):
            now = time.monotonic()

            if now - self.last_error_time >= 1.0:
                self.last_error_time = now
                self.get_logger().error(message)

        def odometry_callback(self, message):
            self.message_count += 1

            now = time.monotonic()

            if now < self.next_request_time:
                return

            self.next_request_time = now + self.request_period

            try:
                observation = message_to_observation(
                    message,
                    physical_hover=self.physical_hover,
                    nn_goal=self.nn_goal,
                    physical_yaw_reference=(
                        self.physical_yaw_reference
                    ),
                    nn_yaw_goal=self.nn_yaw_goal,
                )
            except (TypeError, ValueError) as exc:
                self._log_error_throttled(
                    "ignored invalid VehicleOdometry: %s" % exc
                )
                return

            self.sequence += 1

            try:
                action = self.policy_client.infer(
                    observation,
                    sequence=self.sequence,
                )
            except (
                OSError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                self._log_error_throttled(
                    "policy request failed: %s" % exc
                )
                return

            self.request_count += 1

            now = time.monotonic()

            if now - self.last_print_time < self.print_period:
                return

            self.last_print_time = now

            obs_text = ", ".join(
                "%.4f" % value for value in observation
            )

            action_text = ", ".join(
                "%.6f" % value for value in action
            )

            self.get_logger().info(
                "request[%d], PX4 messages=%d, "
                "obs=[%s], action=[%s]"
                % (
                    self.request_count,
                    self.message_count,
                    obs_text,
                    action_text,
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description="Bridge PX4 odometry to the FlightLxx policy server"
    )

    parser.add_argument(
        "--topic",
        default="/fmu/out/vehicle_odometry",
    )

    parser.add_argument(
        "--policy-host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--policy-port",
        type=int,
        default=5800,
    )

    parser.add_argument(
        "--physical-hover",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 5.0),
    )

    parser.add_argument(
        "--nn-goal",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 5.0),
    )

    parser.add_argument(
        "--physical-yaw-reference",
        type=float,
        default=0.0,
        help="PX4 raw FLU yaw mapped to network yaw 0",
    )

    parser.add_argument(
        "--nn-yaw-goal",
        type=float,
        default=0.0,
        help="network yaw goal in radians",
    )

    parser.add_argument(
        "--max-rate",
        type=float,
        default=30.0,
        help="maximum policy requests per second",
    )

    parser.add_argument(
        "--print-period",
        type=float,
        default=0.2,
        help="minimum seconds between printed samples",
    )

    args = parser.parse_args()

    if args.max_rate <= 0.0:
        parser.error("--max-rate must be positive")

    if rclpy is None:
        print(
            "ERROR: rclpy or px4_msgs is unavailable. "
            "Source ROS 2 Humble and build px4_msgs first."
        )
        return 2

    policy_client = PolicyClient(
        args.policy_host,
        args.policy_port,
        timeout=1.0,
    )

    try:
        print(
            "policy server ping:",
            policy_client.ping(),
        )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        policy_client.close()
        print(
            "ERROR: policy server is unavailable: %s"
            % exc
        )
        return 3

    rclpy.init()

    node = Ros2PolicyBridge(
        topic=args.topic,
        physical_hover=args.physical_hover,
        nn_goal=args.nn_goal,
        physical_yaw_reference=args.physical_yaw_reference,
        nn_yaw_goal=args.nn_yaw_goal,
        policy_client=policy_client,
        max_rate=args.max_rate,
        print_period=args.print_period,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        policy_client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
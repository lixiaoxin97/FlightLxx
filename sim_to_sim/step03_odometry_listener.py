#!/usr/bin/env python3

import argparse
import time

from step02_state_contract import vehicle_odometry_to_observation


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


if rclpy is not None:

    class OdometryListener(Node):
        def __init__(self, topic, physical_hover, nn_goal,
                     physical_yaw_reference=0.0, nn_yaw_goal=0.0):
            super().__init__("flightlxx_odometry_listener")

            self.physical_hover = tuple(physical_hover)
            self.nn_goal = tuple(nn_goal)
            self.physical_yaw_reference = float(physical_yaw_reference)
            self.nn_yaw_goal = float(nn_yaw_goal)
            self.last_print_time = 0.0
            self.message_count = 0

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

            self.get_logger().info("subscribed to %s" % topic)
            self.get_logger().info(
                "physical_hover=%s, nn_goal=%s"
                % (self.physical_hover, self.nn_goal)
            )

        def odometry_callback(self, message):
            self.message_count += 1

            try:
                observation = vehicle_odometry_to_observation(
                    position_ned=message.position,
                    quaternion_ned_frd_wxyz=message.q,
                    velocity_ned=message.velocity,
                    angular_velocity_frd=message.angular_velocity,
                    physical_hover=self.physical_hover,
                    nn_goal=self.nn_goal,
                    physical_yaw_reference=self.physical_yaw_reference,
                    nn_yaw_goal=self.nn_yaw_goal,
                )
            except (TypeError, ValueError) as exc:
                self.get_logger().warning(
                    "ignored invalid VehicleOdometry: %s" % exc
                )
                return

            now = time.monotonic()

            if now - self.last_print_time < 0.2:
                return

            self.last_print_time = now

            formatted = ", ".join(
                "%.4f" % value for value in observation
            )

            self.get_logger().info(
                "obs[%d] = [%s]"
                % (self.message_count, formatted)
            )


def main():
    parser = argparse.ArgumentParser(
        description="Print PX4 VehicleOdometry as a FlightLxx observation"
    )

    parser.add_argument(
        "--topic",
        default="/fmu/out/vehicle_odometry",
        help="PX4 VehicleOdometry topic",
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

    parser.add_argument("--physical-yaw-reference", type=float, default=0.0)
    parser.add_argument("--nn-yaw-goal", type=float, default=0.0)
    args = parser.parse_args()

    if rclpy is None:
        print(
            "ERROR: rclpy or px4_msgs is unavailable. "
            "Source ROS 2 Humble and build px4_msgs first."
        )
        return 2

    rclpy.init()

    node = OdometryListener(
        topic=args.topic,
        physical_hover=args.physical_hover,
        nn_goal=args.nn_goal,
        physical_yaw_reference=args.physical_yaw_reference,
        nn_yaw_goal=args.nn_yaw_goal,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        # Ctrl+C 时 rclpy 可能已经自动关闭 context，
        # 避免重复 shutdown 导致 RCLError。
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

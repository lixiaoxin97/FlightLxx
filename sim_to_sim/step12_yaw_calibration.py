#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

from step02_state_contract import (
    px4_quaternion_to_flu,
    quaternion_to_ypr_rad,
)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def raw_flu_yaw_from_px4_q(quaternion_wxyz) -> float:
    quaternion_flu = px4_quaternion_to_flu(quaternion_wxyz)
    yaw, _, _ = quaternion_to_ypr_rad(quaternion_flu)
    return yaw


def relative_yaw(raw_yaw: float, physical_yaw_reference: float) -> float:
    return wrap_angle(raw_yaw - physical_yaw_reference)


def run_self_test() -> None:
    identity = raw_flu_yaw_from_px4_q((1.0, 0.0, 0.0, 0.0))
    assert abs(identity) < 1.0e-12

    half_pi = math.pi / 4.0
    q_ned_yaw_90 = (
        math.cos(half_pi),
        0.0,
        0.0,
        math.sin(half_pi),
    )

    raw_yaw = raw_flu_yaw_from_px4_q(q_ned_yaw_90)

    assert abs(raw_yaw + math.pi / 2.0) < 1.0e-12
    assert abs(relative_yaw(raw_yaw, raw_yaw)) < 1.0e-12
    assert abs(wrap_angle(0.0)) < 1.0e-12
    assert abs(wrap_angle(3.0 * math.pi) + math.pi) < 1.0e-12

    print("step12 self-test: PASS")


def import_ros():
    try:
        import rclpy
        from px4_msgs.msg import VehicleOdometry
        from rclpy.qos import qos_profile_sensor_data
    except ImportError as exc:
        raise SystemExit(
            "ERROR: rclpy or px4_msgs is unavailable. "
            "Source ROS 2 Humble and px4_msgs first."
        ) from exc

    return rclpy, VehicleOdometry, qos_profile_sensor_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture PX4 heading for FlightLxx yaw alignment"
    )

    parser.add_argument(
        "--topic",
        default="/fmu/out/vehicle_odometry",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
    )
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be >= 1")

    if args.timeout <= 0.0:
        parser.error("--timeout must be > 0")

    return args


def run_ros(topic: str, sample_count: int, timeout: float) -> None:
    rclpy, VehicleOdometry, qos_profile_sensor_data = import_ros()

    rclpy.init()
    node = rclpy.create_node("flightlxx_yaw_calibration")

    sample_number = 0
    physical_yaw_reference = None

    def callback(message):
        nonlocal sample_number, physical_yaw_reference

        raw_yaw = raw_flu_yaw_from_px4_q(message.q)

        if physical_yaw_reference is None:
            physical_yaw_reference = raw_yaw

            node.get_logger().info(
                "captured physical_yaw_reference=%.6f rad (%.2f deg)"
                % (
                    physical_yaw_reference,
                    math.degrees(physical_yaw_reference),
                )
            )

        sample_number += 1

        aligned_yaw = relative_yaw(
            raw_yaw,
            physical_yaw_reference,
        )

        node.get_logger().info(
            "sample[%d]: raw_flu_yaw=%.6f, aligned_yaw=%.6f"
            % (
                sample_number,
                raw_yaw,
                aligned_yaw,
            )
        )

    node.create_subscription(
        VehicleOdometry,
        topic,
        callback,
        qos_profile_sensor_data,
    )

    node.get_logger().info("subscribed to %s" % topic)

    deadline = time.monotonic() + timeout

    try:
        while (
            rclpy.ok()
            and sample_number < sample_count
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    if sample_number == 0:
        raise SystemExit(
            "ERROR: no VehicleOdometry message received"
        )

    print(
        "physical_yaw_reference_rad: %.9f"
        % physical_yaw_reference
    )
    print(
        "physical_yaw_reference_deg: %.6f"
        % math.degrees(physical_yaw_reference)
    )
    print("network_yaw_goal_rad: 0.0")


def main() -> int:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return 0

    run_ros(
        topic=args.topic,
        sample_count=args.samples,
        timeout=args.timeout,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
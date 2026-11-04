#!/usr/bin/env python3
"""Verify a held recovery pose or capture released PX4 observations.

Read-only: this node never arms, publishes Offboard control, or changes PX4
parameters. The recovery model stays static until this gate succeeds. Output
position, quaternion, velocity, and rates are compared with the requested
FlightGym state; PX4 local-position validity, preflight health, reset counter,
and failsafe status are checked at the same time.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, VehicleLocalPosition, VehicleStatus

from recovery_state import (quaternion_from_zyx, recovery_spawn,
                            validate_training_state)
from observation import message_to_observation


def observation_quaternion(observation):
    yaw, pitch, roll = observation[3:6]
    return quaternion_from_zyx(roll, pitch, yaw)


def attitude_distance(actual, expected):
    dot = abs(sum(a*b for a, b in zip(actual, expected)))
    return 2.0*math.acos(max(-1.0, min(1.0, dot)))


class InitProbe(Node):
    def __init__(self, args, report):
        super().__init__("flightlxx_recovery_init_probe")
        self.args = args
        self.report = report
        self.latest = {}
        self.good_since = None
        self.good_samples = 0
        self.passed = False
        self.last_print = 0.0
        self.previous_reset = None
        self.history = deque(maxlen=min(50000, max(300, math.ceil(args.timeout*250))))
        qos = QoSProfile(depth=20,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(VehicleOdometry, args.odometry_topic,
                                 self.on_odometry, qos)
        self.create_subscription(VehicleLocalPosition, args.local_position_topic,
                                 self.on_local_position, qos)
        self.create_subscription(VehicleStatus, args.status_topic,
                                 self.on_status, qos)
        if args.ready_file:
            args.ready_file.parent.mkdir(parents=True, exist_ok=True)
            args.ready_file.write_text("subscriptions_created\n", encoding="utf-8")

    def on_odometry(self, msg):
        self.latest["odometry"] = (msg, time.monotonic())
        self.evaluate()

    def on_local_position(self, msg):
        self.latest["local_position"] = (msg, time.monotonic())

    def on_status(self, msg):
        self.latest["status"] = (msg, time.monotonic())

    def evaluate(self):
        if any(name not in self.latest for name in
               ("odometry", "local_position", "status")):
            return
        now = time.monotonic()
        odom, odom_at = self.latest["odometry"]
        local, local_at = self.latest["local_position"]
        status, status_at = self.latest["status"]
        # VehicleStatus is periodically published around 2 Hz when unchanged;
        # odometry/local position use a tighter freshness budget.
        if (max(now-odom_at, now-local_at) > self.args.max_age or
                now-status_at > self.args.status_max_age):
            self.good_since = None
            self.good_samples = 0
            self.report["stale_sample_count"] += 1
            return
        try:
            observation = message_to_observation(
                odom, physical_hover=(0, 0, 5), nn_goal=(0, 0, 5),
                physical_yaw_reference=0.0, nn_yaw_goal=0.0,
                attitude_encoding=self.args.attitude_encoding,
            )
        except (TypeError, ValueError) as exc:
            self.report["last_error"] = str(exc)
            self.good_since = None
            self.good_samples = 0
            return

        expected = self.args.expected
        pos_error = math.dist(observation[:3], expected[:3])
        quat_error = attitude_distance(observation_quaternion(observation),
                                       observation_quaternion(expected))
        vel_error = math.dist(observation[6:9], expected[6:9])
        rate_error = math.dist(observation[9:12], expected[9:12])
        local_valid = bool(local.xy_valid and local.z_valid and
                           local.v_xy_valid and local.v_z_valid)
        reasons = []
        if odom.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            reasons.append("unsupported_pose_frame")
        if pos_error > self.args.position_tolerance:
            reasons.append("position")
        if quat_error > self.args.attitude_tolerance:
            reasons.append("attitude")
        if vel_error > self.args.velocity_tolerance:
            reasons.append("velocity")
        if rate_error > self.args.rate_tolerance:
            reasons.append("body_rate")
        if not local_valid:
            reasons.append("local_position_invalid")
        if self.args.require_preflight and not status.pre_flight_checks_pass:
            reasons.append("px4_preflight_not_ready")
        if status.failsafe:
            reasons.append("px4_failsafe")
        if status.arming_state != VehicleStatus.ARMING_STATE_DISARMED:
            reasons.append("vehicle_not_disarmed")
        if self.previous_reset is not None and odom.reset_counter != self.previous_reset:
            reasons.append("estimator_reset")
        self.previous_reset = int(odom.reset_counter)
        if not all(math.isfinite(x) for x in observation):
            reasons.append("nonfinite_observation")

        sample = {
            "receive_monotonic_s": now,
            "receive_wall_time_ns": time.time_ns(),
            "timestamp": int(odom.timestamp),
            "timestamp_sample": int(odom.timestamp_sample),
            "pose_frame": int(odom.pose_frame),
            "velocity_frame": int(odom.velocity_frame),
            "reset_counter": int(odom.reset_counter),
            "observation": list(observation),
            "errors": {"position_m": pos_error, "attitude_rad": quat_error,
                       "velocity_m_s": vel_error, "body_rate_rad_s": rate_error},
            "local_position_valid": local_valid,
            "px4_preflight_pass": bool(status.pre_flight_checks_pass),
            "failsafe": bool(status.failsafe),
            "arming_state": int(status.arming_state),
            "nav_state": int(status.nav_state),
            "reasons": reasons,
        }
        self.history.append(sample)
        self.report["last_sample"] = sample
        self.report["sample_count"] += 1
        good = not reasons
        if good:
            if self.good_since is None:
                self.good_since = now
            self.good_samples += 1
        else:
            self.good_since = None
            self.good_samples = 0

        if now-self.last_print >= 1.0:
            self.last_print = now
            self.get_logger().info(
                "position=%.3f m attitude=%.3f rad velocity=%.3f m/s "
                "rate=%.3f rad/s valid=%s preflight=%s nav=%d arm=%d %s" %
                (pos_error, quat_error, vel_error, rate_error, local_valid,
                 status.pre_flight_checks_pass, status.nav_state,
                 status.arming_state, ",".join(reasons) or "MATCH")
            )

        if (self.args.phase == "hold" and good and self.good_since is not None and
                now-self.good_since >= self.args.stable_duration and
                self.good_samples >= self.args.minimum_samples):
            self.report["passed"] = True
            self.report["stable_duration_s"] = now-self.good_since
            self.report["stable_sample_count"] = self.good_samples
            self.report["initial_reset_counter"] = int(odom.reset_counter)
            self.report["history"] = list(self.history)
            self.passed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=float, nargs=12, required=True,
                        help="requested FlightGym state in training order")
    parser.add_argument("--attitude-encoding", choices=("eigen_zyx", "principal_zyx"),
                        default="eigen_zyx")
    parser.add_argument("--odometry-topic", default="/fmu/out/vehicle_odometry")
    parser.add_argument("--local-position-topic", default=None)
    parser.add_argument("--status-topic", default=None)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-age", type=float, default=0.25)
    parser.add_argument("--status-max-age", type=float, default=1.5)
    parser.add_argument("--stable-duration", type=float, default=0.75)
    parser.add_argument("--minimum-samples", type=int, default=40)
    parser.add_argument("--position-tolerance", type=float, default=0.06)
    parser.add_argument("--attitude-tolerance", type=float, default=0.06)
    parser.add_argument("--velocity-tolerance", type=float, default=0.06)
    parser.add_argument("--rate-tolerance", type=float, default=0.06)
    parser.add_argument("--require-preflight", action="store_true",
                        help="also require PX4 pre_flight_checks_pass")
    parser.add_argument("--phase", choices=("hold", "observe"), default="hold",
                        help="observe captures data without claiming an initial-state match")
    parser.add_argument("--ready-file", type=Path,
                        help="write a marker after creating the ROS subscriptions")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for name in ("timeout", "max_age", "status_max_age", "stable_duration",
                 "position_tolerance", "attitude_tolerance",
                 "velocity_tolerance", "rate_tolerance"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(name + " must be finite and positive")
    if args.minimum_samples < 1:
        parser.error("minimum-samples must be positive")
    state = validate_training_state(args.state)
    requested = recovery_spawn(state, args.attitude_encoding)["expected_policy_observation"]
    # The dedicated Recovery model remains static during estimator alignment;
    # its measured velocities/rates are zero. The requested trial velocities
    # are applied by the injector only after this pose/health gate passes.
    args.expected = list(requested)
    if args.phase == "hold":
        args.expected[6:12] = [0.0]*6
    if args.local_position_topic is None:
        from offboard_authority import px4_feedback_topic
        from px4_msgs.msg import VehicleLocalPosition
        args.local_position_topic = px4_feedback_topic(
            "/fmu/out/vehicle_local_position", VehicleLocalPosition)
    if args.status_topic is None:
        from offboard_authority import px4_feedback_topic
        args.status_topic = px4_feedback_topic("/fmu/out/vehicle_status", VehicleStatus)

    report = {
        "schema_version": 2,
        "phase": args.phase,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_state": list(state),
        "requested_policy_observation": list(requested),
        "comparison_policy_observation": list(args.expected),
        "attitude_encoding": args.attitude_encoding,
        "thresholds": {"position_m": args.position_tolerance,
                       "attitude_rad": args.attitude_tolerance,
                       "velocity_m_s": args.velocity_tolerance,
                       "body_rate_rad_s": args.rate_tolerance,
                       "stable_duration_s": args.stable_duration,
                       "minimum_samples": args.minimum_samples},
        "freshness_limits_s": {"odometry_and_local_position": args.max_age,
                              "vehicle_status": args.status_max_age},
        "policy_handover_validated": False,
        "topics": {"odometry": args.odometry_topic,
                   "local_position": args.local_position_topic,
                   "status": args.status_topic},
        "passed": False if args.phase == "hold" else None,
        "sample_count": 0,
        "stale_sample_count": 0,
    }
    rclpy.init()
    node = InitProbe(args, report)
    deadline = time.monotonic()+args.timeout
    while time.monotonic() < deadline and not getattr(node, "passed", False):
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["history"] = list(node.history)
    if args.phase == "observe":
        report["capture_complete"] = report["sample_count"] >= args.minimum_samples
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"],
                      "sample_count": report["sample_count"],
                      "last_sample": report.get("last_sample"),
                      "report": str(args.report)}, indent=2, allow_nan=False))
    return 0 if report.get("capture_complete") or report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

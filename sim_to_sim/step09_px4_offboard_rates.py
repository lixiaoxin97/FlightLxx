#!/usr/bin/env python3
"""Publish CTBR on a 50 Hz timer; run blocking UDP inference in one worker.

Odometry callbacks only cache the latest state. Each control tick submits at
most one inference job, then publishes the latest valid result (zero-order
hold). Slow inference never queues jobs or blocks the ROS executor. Linux /
Python / ROS timers provide a target period, not a hard real-time guarantee.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from step06_policy_client import PolicyClient
from step02_state_contract import STATE_CONTRACT_VERSION, wrap_angle_rad
from step07_ros2_policy_bridge import message_to_observation
from step08_px4_rates_contract import (
    DEFAULT_THRUST_MODEL,
    ThrustMapping,
    add_thrust_model_arguments,
    convert_raw_action_to_px4,
    map_collective_acceleration,
    thrust_model_from_args,
)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from px4_msgs.msg import (
        OffboardControlMode,
        VehicleRatesSetpoint,
        VehicleOdometry,
    )
except ImportError:
    rclpy = None
    Node = object


@dataclass(frozen=True)
class PolicyResult:
    sequence: int
    raw_action: tuple
    rates_frd: tuple
    thrust_body_frd: tuple
    collective_thrust: float
    observation_time: float
    started_at: float
    completed_at: float
    observation: tuple = ()
    source_odometry: dict = None
    thrust_mapping: ThrustMapping = None


def result_is_fresh(result, now, max_action_age, max_odometry_age):
    """Check both reply age and the age of the state used for that reply."""
    return (
        result is not None
        and 0.0 <= now - result.completed_at <= max_action_age
        and 0.0 <= now - result.observation_time <= max_odometry_age
    )


class PolicyWorker:
    """Own the UDP client on one thread; reject submissions while busy."""

    def __init__(self, client, client_factory=None, thrust_model=None):
        self._client = client
        self._client_factory = client_factory
        self._thrust_model = dict(thrust_model or DEFAULT_THRUST_MODEL)
        self._timeout = client.timeout
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._busy = False
        self._job = None
        self._sequence = 0
        self._result = None
        self._error = None
        self._success_count = 0
        self._thread = threading.Thread(
            target=self._run, name="flightlxx_policy_worker", daemon=True
        )
        self._thread.start()

    def submit(self, observation, observation_time, source_odometry=None):
        with self._lock:
            if self._stop.is_set() or self._busy:
                return False
            self._busy = True
            self._job = (
                tuple(observation), float(observation_time),
                dict(source_odometry or {}),
            )
            self._wake.set()
            return True

    def snapshot(self):
        with self._lock:
            return self._result, self._error, self._success_count

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=max(2.0, self._timeout + 0.5))
        return not self._thread.is_alive()

    def _run(self):
        try:
            while not self._stop.is_set():
                self._wake.wait()
                self._wake.clear()
                if self._stop.is_set():
                    break
                with self._lock:
                    job = self._job
                    self._job = None
                    self._sequence += 1
                    sequence = self._sequence
                if job is None:
                    continue
                observation, observation_time, source_odometry = job
                started_at = time.monotonic()
                try:
                    action = tuple(
                        self._client.infer(observation, sequence=sequence)
                    )
                    rates, thrust, collective, thrust_mapping = (
                        convert_raw_action_to_px4(
                            action, **self._thrust_model
                        )
                    )
                    result = PolicyResult(
                        sequence=sequence,
                        raw_action=action,
                        rates_frd=tuple(rates),
                        thrust_body_frd=tuple(thrust),
                        collective_thrust=float(collective),
                        observation_time=observation_time,
                        started_at=started_at,
                        completed_at=time.monotonic(),
                        observation=observation,
                        source_odometry=source_odometry,
                        thrust_mapping=thrust_mapping,
                    )
                    with self._lock:
                        self._result = result
                        self._success_count += 1
                        self._error = None
                except (OSError, TypeError, ValueError, RuntimeError) as exc:
                    with self._lock:
                        self._error = "policy/action conversion failed: %s" % exc
                    # A new socket discards late responses from a timed-out
                    # request instead of shifting every subsequent sequence.
                    if self._client_factory is not None:
                        self._client.close()
                        try:
                            self._client = self._client_factory()
                        except OSError as reopen_error:
                            with self._lock:
                                self._error = "policy socket reopen failed: %s" % (
                                    reopen_error,
                                )
                finally:
                    with self._lock:
                        self._busy = False
        finally:
            self._client.close()


def fill_px4_messages(
    offboard_message,
    rates_message,
    timestamp_us,
    rates_frd,
    thrust_body_frd,
):
    """Preserve the original CTBR message contract."""
    offboard_message.timestamp = int(timestamp_us)
    offboard_message.position = False
    offboard_message.velocity = False
    offboard_message.acceleration = False
    offboard_message.attitude = False
    offboard_message.body_rate = True
    offboard_message.thrust_and_torque = False
    offboard_message.direct_actuator = False
    rates_message.timestamp = int(timestamp_us)
    rates_message.roll = float(rates_frd[0])
    rates_message.pitch = float(rates_frd[1])
    rates_message.yaw = float(rates_frd[2])
    rates_message.thrust_body = [float(value) for value in thrust_body_frd]
    rates_message.reset_integral = False


if rclpy is not None:

    class Px4OffboardRatesNode(Node):
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
            publish_enabled,
            max_action_age=0.2,
            max_odometry_age=0.2,
            trace_enabled=True,
            thrust_model=None,
        ):
            super().__init__("flightlxx_px4_offboard_rates")
            self.physical_hover = tuple(physical_hover)
            self.nn_goal = tuple(nn_goal)
            self.physical_yaw_reference = float(physical_yaw_reference)
            self.nn_yaw_goal = float(nn_yaw_goal)
            self.control_rate = float(max_rate)
            self.control_period = 1.0 / self.control_rate
            self.print_period = float(print_period)
            self.publish_enabled = bool(publish_enabled)
            self.max_action_age = float(max_action_age)
            self.max_odometry_age = float(max_odometry_age)
            self.thrust_model = dict(thrust_model or DEFAULT_THRUST_MODEL)
            self.thrust_model_max_acceleration = map_collective_acceleration(
                0.0, **self.thrust_model
            ).maximum_acceleration_m_s2
            self.latest_odometry = None
            self.latest_odometry_time = 0.0
            self.last_print_time = 0.0
            self.last_error_time = 0.0
            self.message_count = 0
            self.publish_count = 0
            self.timer_count = 0
            self.last_stats_time = time.monotonic()
            self.last_stats_publish_count = 0
            self.last_stats_inference_count = 0
            self.publish_hz = 0.0
            self.infer_hz = 0.0
            self.trace_enabled = bool(trace_enabled)
            self.controller_run_id = uuid.uuid4().hex
            self.last_trace_sequence = None
            self.last_trace_config_time = -math.inf
            self.last_trace_error_time = -math.inf
            self.trace_error_count = 0
            self.trace_config = {
                "control_rate_hz": self.control_rate,
                "physical_hover": self.physical_hover,
                "nn_goal": self.nn_goal,
                "physical_yaw_reference": self.physical_yaw_reference,
                "nn_yaw_goal": self.nn_yaw_goal,
                "state_contract_version": STATE_CONTRACT_VERSION,
                "state_contract_sha256": hashlib.sha256(
                    Path(__file__).with_name("step02_state_contract.py").read_bytes()
                ).hexdigest(),
                "world_yaw_rotation_rad": wrap_angle_rad(
                    self.nn_yaw_goal - self.physical_yaw_reference
                ),
                "world_transform": "p=Rz(alpha)*(p_flu-hover)+goal; "
                                   "v=Rz(alpha)*v_flu; R=Rz(alpha)*R_flu",
                "attitude_encoding": "Eigen eulerAngles(2,1,0); yaw in [0,pi]",
                "body_rate_transform": "FRD->FLU only; independent of world yaw",
                "max_action_age_s": self.max_action_age,
                "max_odometry_age_s": self.max_odometry_age,
                "thrust_model": self.thrust_model,
                "thrust_model_max_acceleration_m_s2": (
                    self.thrust_model_max_acceleration
                ),
                "publish_enabled": self.publish_enabled,
                "policy_address": list(policy_client.server_address),
                "policy_timeout_s": policy_client.timeout,
                "model_path_hint": os.environ.get("S2S_MODEL"),
                "controller_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "observation_order": [
                    "x", "y", "z", "yaw", "pitch", "roll",
                    "vx", "vy", "vz", "wx", "wy", "wz",
                ],
                "rates_frame": "FRD", "rate_units": "rad/s",
                "source_age_basis": "odometry callback monotonic receive time",
            }

            # Latest-state sampling: do not process an old ten-message backlog.
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.subscription = self.create_subscription(
                VehicleOdometry, topic, self.odometry_callback, qos
            )
            self.offboard_publisher = self.create_publisher(
                OffboardControlMode, "/fmu/in/offboard_control_mode", 10
            )
            self.rates_publisher = self.create_publisher(
                VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", 10
            )
            self.trace_publisher = None
            if self.trace_enabled:
                self.trace_publisher = self.create_publisher(
                    String, "/flightlxx/sim_to_sim/policy_trace",
                    QoSProfile(
                        depth=10,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE,
                    ),
                )
            address = policy_client.server_address
            timeout = policy_client.timeout
            self.worker = PolicyWorker(
                policy_client,
                client_factory=lambda: PolicyClient(
                    address[0], address[1], timeout=timeout
                ),
                thrust_model=self.thrust_model,
            )
            # ROS callbacks stay short: this timer never performs UDP I/O.
            self.control_timer = self.create_timer(
                self.control_period, self.control_callback
            )
            self.get_logger().info(
                "subscribed to %s; policy udp=%s:%d"
                % (topic, address[0], address[1])
            )
            self.get_logger().info(
                "state_contract=%s; world_yaw_rotation=%.6f rad; "
                "policy attitude uses training Eigen ZYX branch"
                % (STATE_CONTRACT_VERSION, self.trace_config["world_yaw_rotation_rad"])
            )
            self.get_logger().info(
                "control timer=%.3f Hz (%.3f ms), publish=%s; "
                "max_action_age=%.3f s, max_odometry_age=%.3f s"
                % (
                    self.control_rate,
                    1000.0 * self.control_period,
                    "ENABLED" if self.publish_enabled else "DRY-RUN",
                    self.max_action_age,
                    self.max_odometry_age,
                )
            )

        def close(self):
            self.control_timer.cancel()
            if not self.worker.close():
                self.get_logger().warning("policy worker did not stop in time")

        def _log_error_throttled(self, message):
            now = time.monotonic()
            if now - self.last_error_time >= 1.0:
                self.last_error_time = now
                self.get_logger().error(message)

        def odometry_callback(self, message):
            self.message_count += 1
            self.latest_odometry = message
            self.latest_odometry_time = time.monotonic()

        def _publish_setpoint(self, rates_frd, thrust_body_frd):
            timestamp_us = self.get_clock().now().nanoseconds // 1000
            offboard_message = OffboardControlMode()
            rates_message = VehicleRatesSetpoint()
            fill_px4_messages(
                offboard_message,
                rates_message,
                timestamp_us,
                rates_frd,
                thrust_body_frd,
            )
            self.offboard_publisher.publish(offboard_message)
            self.rates_publisher.publish(rates_message)
            self.publish_count += 1
            return timestamp_us

        def _emit_trace(
            self, now, result, status, inference_count,
            submitted=False, error=None, output_timestamp_us=None,
            submitted_observation=None, submitted_source_odometry=None,
        ):
            """Best-effort telemetry only; no file I/O or inference here."""
            if self.trace_publisher is None:
                return
            try:
                data = {
                    "schema_version": 1, "event": "policy_tick",
                    "controller_run_id": self.controller_run_id,
                    "tick": self.timer_count, "status": status,
                    "control_monotonic_s": now,
                    "published": output_timestamp_us is not None,
                    "setpoint_timestamp_us": output_timestamp_us,
                    "inference_submitted": bool(submitted),
                    "submitted_observation": (
                        list(submitted_observation) if submitted else None
                    ),
                    "submitted_source_odometry": (
                        submitted_source_odometry if submitted else None
                    ),
                    "inference_count": inference_count,
                    "target_hz": self.control_rate,
                    "publish_hz": self.publish_hz,
                    "infer_hz": self.infer_hz,
                    "error": error, "trace_errors": self.trace_error_count,
                    "latest_odometry_age_ms": (
                        1000.0 * (now - self.latest_odometry_time)
                        if self.latest_odometry is not None else None
                    ),
                    "new_action": bool(
                        result is not None
                        and result.sequence != self.last_trace_sequence
                    ),
                    "result": None,
                }
                if result is not None:
                    data["result"] = {
                        "sequence": result.sequence,
                        "observation": list(result.observation),
                        "source_odometry": result.source_odometry or {},
                        "raw_action": list(result.raw_action),
                        "rates_frd": list(result.rates_frd),
                        "thrust_body_frd": list(result.thrust_body_frd),
                        "collective_log_m_s2": result.collective_thrust,
                        "thrust_mapping": {
                            "px4_fraction": (
                                result.thrust_mapping.px4_thrust_fraction
                            ),
                            "motor_output_fraction": (
                                result.thrust_mapping.motor_output_fraction
                            ),
                            "motor_speed_rad_s": (
                                result.thrust_mapping.motor_speed_rad_s
                            ),
                            "minimum_acceleration_m_s2": (
                                result.thrust_mapping.minimum_acceleration_m_s2
                            ),
                            "maximum_acceleration_m_s2": (
                                result.thrust_mapping.maximum_acceleration_m_s2
                            ),
                            "saturated_low": result.thrust_mapping.saturated_low,
                            "saturated_high": result.thrust_mapping.saturated_high,
                        },
                        "observation_monotonic_s": result.observation_time,
                        "started_monotonic_s": result.started_at,
                        "completed_monotonic_s": result.completed_at,
                        "action_age_ms": 1000.0 * (now - result.completed_at),
                        "state_age_ms": 1000.0 * (now - result.observation_time),
                        "inference_ms": 1000.0 * (
                            result.completed_at - result.started_at
                        ),
                    }
                if now - self.last_trace_config_time >= 1.0:
                    data["config"] = self.trace_config
                message = String()
                message.data = json.dumps(
                    data, separators=(",", ":"), allow_nan=False
                )
                self.trace_publisher.publish(message)
                if "config" in data:
                    self.last_trace_config_time = now
                if result is not None:
                    self.last_trace_sequence = result.sequence
            except Exception as exc:
                # Diagnostic failures must not interrupt the flight callback.
                self.trace_error_count += 1
                if now - self.last_trace_error_time >= 1.0:
                    self.last_trace_error_time = now
                    self.get_logger().warning("policy telemetry failed: %s" % exc)

        def _update_statistics(self, now, inference_count):
            elapsed = now - self.last_stats_time
            if elapsed >= 1.0:
                self.publish_hz = (
                    self.publish_count - self.last_stats_publish_count
                ) / elapsed
                self.infer_hz = (
                    inference_count - self.last_stats_inference_count
                ) / elapsed
                self.last_stats_time = now
                self.last_stats_publish_count = self.publish_count
                self.last_stats_inference_count = inference_count
                self.get_logger().info(
                    "timing: target_hz=%.2f publish_hz=%.2f infer_hz=%.2f "
                    "published=%d, PX4 messages=%d"
                    % (
                        self.control_rate,
                        self.publish_hz,
                        self.infer_hz,
                        self.publish_count,
                        self.message_count,
                    )
                )

        def control_callback(self):
            self.timer_count += 1
            result, worker_error, inference_count = self.worker.snapshot()
            now = time.monotonic()
            self._update_statistics(now, inference_count)
            if self.latest_odometry is None:
                self._emit_trace(now, result, "waiting_odometry", inference_count)
                self._log_error_throttled("waiting for VehicleOdometry")
                return
            if now - self.latest_odometry_time > self.max_odometry_age:
                self._emit_trace(now, result, "stale_odometry", inference_count)
                self._log_error_throttled(
                    "odometry is stale; CTBR and Offboard heartbeat paused"
                )
                return
            try:
                observation = message_to_observation(
                    self.latest_odometry,
                    physical_hover=self.physical_hover,
                    nn_goal=self.nn_goal,
                    physical_yaw_reference=self.physical_yaw_reference,
                    nn_yaw_goal=self.nn_yaw_goal,
                )
            except (TypeError, ValueError) as exc:
                self._emit_trace(
                    now, result, "invalid_odometry", inference_count,
                    error=str(exc),
                )
                self._log_error_throttled(
                    "ignored invalid VehicleOdometry: %s" % exc
                )
                return

            # No waiting: if the worker is still busy, skip this inference job.
            source_odometry = {
                name: int(getattr(self.latest_odometry, name, 0))
                for name in (
                    "timestamp", "timestamp_sample", "reset_counter",
                    "pose_frame", "velocity_frame",
                )
            }
            submitted = self.worker.submit(
                observation, self.latest_odometry_time, source_odometry
            )
            if worker_error is not None:
                self._log_error_throttled(worker_error)
            if not result_is_fresh(
                result, now, self.max_action_age, self.max_odometry_age
            ):
                self._emit_trace(
                    now, result, "no_fresh_action", inference_count,
                    submitted=submitted, error=worker_error,
                    submitted_observation=observation,
                    submitted_source_odometry=source_odometry,
                )
                self._log_error_throttled(
                    "no fresh policy action; CTBR and Offboard heartbeat paused"
                )
                return
            output_timestamp_us = None
            if self.publish_enabled:
                output_timestamp_us = self._publish_setpoint(
                    result.rates_frd, result.thrust_body_frd
                )
            self._emit_trace(
                now, result, "published" if self.publish_enabled else "dry_run",
                inference_count, submitted=submitted, error=worker_error,
                output_timestamp_us=output_timestamp_us,
                submitted_observation=observation,
                submitted_source_odometry=source_odometry,
            )
            if now - self.last_print_time < self.print_period:
                return
            self.last_print_time = now
            # Keep request[...] and raw= for the launcher's ready detection.
            self.get_logger().info(
                "request[%d], raw=%s, rates_frd=%s, thrust_body=%s, "
                "collective=%.4f m/s^2, rotor_speed=%.1f rad/s, "
                "thrust_sat=(low:%s high:%s), action_age_ms=%.2f, "
                "state_age_ms=%.2f, inference_ms=%.2f"
                % (
                    result.sequence,
                    tuple(round(value, 5) for value in result.raw_action),
                    tuple(round(value, 5) for value in result.rates_frd),
                    tuple(round(value, 5) for value in result.thrust_body_frd),
                    result.collective_thrust,
                    result.thrust_mapping.motor_speed_rad_s,
                    result.thrust_mapping.saturated_low,
                    result.thrust_mapping.saturated_high,
                    1000.0 * (now - result.completed_at),
                    1000.0 * (now - result.observation_time),
                    1000.0 * (result.completed_at - result.started_at),
                )
            )


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish FlightLxx CTBR on a fixed-period ROS 2 timer"
    )
    parser.add_argument("--topic", default="/fmu/out/vehicle_odometry")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5800)
    parser.add_argument(
        "--physical-hover", type=float, nargs=3, default=(0.0, 0.0, 5.0)
    )
    parser.add_argument(
        "--nn-goal", type=float, nargs=3, default=(0.0, 0.0, 5.0)
    )
    parser.add_argument(
        "--physical-yaw-reference", type=float, default=0.0,
        help="fixed physical FLU heading; rotates world attitude, position error, and velocity together",
    )
    parser.add_argument(
        "--nn-yaw-goal", type=float, default=0.0,
        help="network heading used in the same fixed world-frame rotation",
    )
    parser.add_argument(
        "--control-rate", "--max-rate", dest="max_rate", type=positive_float,
        default=50.0,
        help="fixed control timer frequency in Hz (default: 50); "
             "--max-rate remains accepted for existing launch commands",
    )
    parser.add_argument("--print-period", type=positive_float, default=0.2)
    parser.add_argument("--policy-timeout", type=positive_float, default=1.0)
    parser.add_argument("--max-action-age", type=positive_float, default=0.2)
    parser.add_argument("--max-odometry-age", type=positive_float, default=0.2)
    add_thrust_model_arguments(parser)
    parser.add_argument(
        "--no-policy-trace", action="store_true",
        help="disable diagnostic policy telemetry (enabled by default)",
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="actually publish OffboardControlMode and VehicleRatesSetpoint",
    )
    args = parser.parse_args()
    try:
        args.thrust_model = thrust_model_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    if rclpy is None:
        print("ERROR: rclpy, px4_msgs, or required PX4 messages are unavailable.")
        return 2
    policy_client = PolicyClient(
        args.policy_host, args.policy_port, timeout=args.policy_timeout
    )
    try:
        print("policy server ping:", policy_client.ping())
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        policy_client.close()
        print("ERROR: policy server is unavailable: %s" % exc)
        return 3
    rclpy.init()
    node = None
    try:
        node = Px4OffboardRatesNode(
            topic=args.topic,
            physical_hover=args.physical_hover,
            nn_goal=args.nn_goal,
            physical_yaw_reference=args.physical_yaw_reference,
            nn_yaw_goal=args.nn_yaw_goal,
            policy_client=policy_client,
            max_rate=args.max_rate,
            print_period=args.print_period,
            publish_enabled=args.publish,
            max_action_age=args.max_action_age,
            max_odometry_age=args.max_odometry_age,
            trace_enabled=not args.no_policy_trace,
            thrust_model=args.thrust_model,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        else:
            policy_client.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

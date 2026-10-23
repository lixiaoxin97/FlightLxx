#!/usr/bin/env python3
"""Measure PX4 body-rate response with small symmetric rate pulses.

The node publishes only an Offboard heartbeat until PX4 confirms armed
Offboard body-rate control. It then applies small roll and pitch rate pulses
around a constant hover-thrust setpoint. It never runs policy inference.
"""

import argparse
import json
import math
import time
import uuid

from offboard_authority import OffboardAuthority, px4_feedback_topic
from step08_px4_rates_contract import (
    DEFAULT_VEHICLE_PROFILE,
    add_thrust_model_arguments,
    map_collective_acceleration,
    thrust_model_from_args,
)


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from px4_msgs.msg import (
        OffboardControlMode,
        VehicleControlMode,
        VehicleLocalPosition,
        VehicleCommand,
        VehicleRatesSetpoint,
        VehicleStatus,
    )
    from std_msgs.msg import String
except ImportError:
    rclpy = None
    Node = object
    QoSProfile = None
    ReliabilityPolicy = None
    DurabilityPolicy = None
    OffboardControlMode = None
    VehicleControlMode = None
    VehicleLocalPosition = None
    VehicleCommand = None
    VehicleRatesSetpoint = None
    VehicleStatus = None
    String = None


def build_step_sequence(amplitude=0.12, pulse_duration=0.4,
                        recovery_duration=1.2, cycles=3,
                        initial_hold=3.0, final_hold=5.0):
    """Return (phase, duration_s, rates_frd) steps with zero net angle pulses."""
    steps = [("initial_hold", initial_hold, (0.0, 0.0, 0.0))]
    for axis_index, axis_name in ((0, "roll"), (1, "pitch")):
        for cycle in range(1, cycles + 1):
            positive = [0.0, 0.0, 0.0]
            negative = [0.0, 0.0, 0.0]
            positive[axis_index] = float(amplitude)
            negative[axis_index] = -float(amplitude)
            steps.extend((
                ("%s_%d_positive" % (axis_name, cycle), pulse_duration,
                 tuple(positive)),
                ("%s_%d_negative" % (axis_name, cycle), pulse_duration,
                 tuple(negative)),
                ("%s_%d_recovery" % (axis_name, cycle), recovery_duration,
                 (0.0, 0.0, 0.0)),
            ))
    steps.append(("final_hold", final_hold, (0.0, 0.0, 0.0)))
    return tuple(steps)


def step_at_elapsed(sequence, elapsed_s):
    """Return phase and rate setpoint for elapsed test time."""
    elapsed = max(0.0, float(elapsed_s))
    for phase, duration, rates in sequence:
        if elapsed < duration:
            return phase, rates
        elapsed -= duration
    return "complete_hold", (0.0, 0.0, 0.0)


def fill_heartbeat(message, timestamp_us):
    message.timestamp = int(timestamp_us)
    message.position = False
    message.velocity = False
    message.acceleration = False
    message.attitude = False
    message.body_rate = True
    message.thrust_and_torque = False
    message.direct_actuator = False


def fill_rate_setpoint(message, timestamp_us, rates_frd, thrust_body_frd):
    message.timestamp = int(timestamp_us)
    message.roll = float(rates_frd[0])
    message.pitch = float(rates_frd[1])
    message.yaw = float(rates_frd[2])
    message.thrust_body = [float(value) for value in thrust_body_frd]
    message.reset_integral = False


if rclpy is not None:

    class RateStepTestNode(Node):
        def __init__(self, args):
            super().__init__("flightlxx_px4_rate_step_test")
            self.args = args
            self.run_id = uuid.uuid4().hex
            self.authority = OffboardAuthority(args.max_feedback_age)
            self.status_received_at = None
            self.control_mode_received_at = None
            self.latest_position = None
            self.started_at = None
            self.last_phase = None
            self.aborted = False
            self.done_logged = False
            self.last_land_request = -math.inf
            self.sequence = build_step_sequence(
                amplitude=args.amplitude,
                pulse_duration=args.pulse_duration,
                recovery_duration=args.recovery_duration,
                cycles=args.cycles,
                initial_hold=args.initial_hold,
                final_hold=args.final_hold,
            )
            if args.hover_thrust_fraction is None:
                self.thrust = map_collective_acceleration(
                    args.hover_acceleration_m_s2, **args.thrust_model
                ).px4_thrust_fraction
            else:
                self.thrust = float(args.hover_thrust_fraction)
            self.thrust_body = (0.0, 0.0, -self.thrust)
            self.control_rate = float(args.control_rate)
            self.period = 1.0 / self.control_rate
            self.tick = 0

            best_effort = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            command_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(
                VehicleStatus,
                px4_feedback_topic("/fmu/out/vehicle_status", VehicleStatus),
                self.status_callback,
                best_effort,
            )
            self.create_subscription(
                VehicleControlMode,
                px4_feedback_topic(
                    "/fmu/out/vehicle_control_mode", VehicleControlMode
                ),
                self.control_mode_callback,
                best_effort,
            )
            self.create_subscription(
                VehicleLocalPosition,
                px4_feedback_topic(
                    "/fmu/out/vehicle_local_position", VehicleLocalPosition
                ),
                self.position_callback,
                best_effort,
            )
            self.heartbeat_publisher = self.create_publisher(
                OffboardControlMode,
                "/fmu/in/offboard_control_mode",
                command_qos,
            )
            self.rates_publisher = self.create_publisher(
                VehicleRatesSetpoint,
                "/fmu/in/vehicle_rates_setpoint",
                command_qos,
            )
            self.vehicle_command_publisher = self.create_publisher(
                VehicleCommand,
                "/fmu/in/vehicle_command",
                command_qos,
            )
            self.trace_publisher = self.create_publisher(
                String,
                "/flightlxx/sim_to_sim/policy_trace",
                QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                ),
            )
            self.timer = self.create_timer(self.period, self.control_tick)
            self._trace({
                "event": "rate_step_config",
                "controller_run_id": self.run_id,
                "config": {
                    "test": "px4_body_rate_symmetric_steps_v1",
                    "trial_label": args.trial_label,
                    "control_rate_hz": self.control_rate,
                    "amplitude_rad_s": args.amplitude,
                    "pulse_duration_s": args.pulse_duration,
                    "recovery_duration_s": args.recovery_duration,
                    "cycles_per_axis": args.cycles,
                    "hover_acceleration_m_s2": args.hover_acceleration_m_s2,
                    "hover_thrust_fraction_override": args.hover_thrust_fraction,
                    "hover_thrust_fraction": self.thrust,
                    "auto_land_at_completion": True,
                    "thrust_model": args.thrust_model,
                    "sequence": [
                        {"phase": n, "duration_s": d, "rates_frd_rad_s": r}
                        for n, d, r in self.sequence
                    ],
                },
            })
            self.get_logger().info(
                "rate-step test ready: profile=%s thrust=%.6f, amplitude=%.3f rad/s; "
                "will publish no rates until armed Offboard body-rate mode is confirmed"
                % (args.vehicle_profile, self.thrust, args.amplitude)
            )

        def _trace(self, data):
            data.setdefault("controller_run_id", self.run_id)
            data["monotonic_s"] = time.monotonic()
            message = String()
            message.data = json.dumps(data, separators=(",", ":"), allow_nan=False)
            self.trace_publisher.publish(message)

        def status_callback(self, message):
            now = time.monotonic()
            self.authority.update_status(message, now)
            self.status_received_at = now

        def control_mode_callback(self, message):
            now = time.monotonic()
            self.authority.update_control_mode(message, now)
            self.control_mode_received_at = now

        def position_callback(self, message):
            self.latest_position = message

        def _request_land(self, now):
            message = VehicleCommand()
            message.timestamp = self.get_clock().now().nanoseconds // 1000
            message.command = 21  # VEHICLE_CMD_NAV_LAND
            message.target_system = 1
            message.source_system = 1
            message.source_component = 1
            message.from_external = True
            self.vehicle_command_publisher.publish(message)
            self.last_land_request = now
            self._trace({"event": "rate_step_land_requested", "command": 21})
            self.get_logger().warning(
                "rate-step sequence complete; requested PX4 AUTO_LAND"
            )

        def control_tick(self):
            self.tick += 1
            now = time.monotonic()
            decision = self.authority.decision(now)
            if not self.args.publish:
                if self.tick % int(max(1, round(self.control_rate))) == 0:
                    self.get_logger().info(
                        "dry-run: authority=%s; no heartbeat or rates setpoint published"
                        % decision.status
                    )
                return

            if not decision.heartbeat:
                if self.started_at is not None and not self.aborted:
                    self.aborted = True
                    self._trace({
                        "event": "rate_step_aborted",
                        "reason": "PX4 authority lost: " + decision.status,
                    })
                    self.get_logger().error(
                        "PX4 authority lost (%s); rate-step sequence aborted"
                        % decision.status
                    )
                return

            timestamp_us = self.get_clock().now().nanoseconds // 1000
            heartbeat = OffboardControlMode()
            fill_heartbeat(heartbeat, timestamp_us)
            self.heartbeat_publisher.publish(heartbeat)

            if not decision.rates:
                return
            if self.started_at is None:
                self.started_at = now
                self.get_logger().info(
                    "PX4 armed Offboard confirmed; starting symmetric rate pulses"
                )
                self._trace({"event": "rate_step_started"})
            if self.aborted:
                return

            elapsed = now - self.started_at
            phase, rates = step_at_elapsed(self.sequence, elapsed)
            if phase != self.last_phase:
                self.last_phase = phase
                self.get_logger().info(
                    "rate-step phase=%s elapsed=%.3f setpoint_frd=%s"
                    % (phase, elapsed, tuple(round(x, 4) for x in rates))
                )
                self._trace({
                    "event": "rate_step_phase",
                    "phase": phase,
                    "elapsed_s": elapsed,
                    "rates_frd_rad_s": rates,
                })
            if phase == "complete_hold" and not self.done_logged:
                self.done_logged = True
                self.get_logger().info(
                    "rate-step sequence complete; holding zero body rates and hover thrust"
                )
                self._trace({"event": "rate_step_complete", "elapsed_s": elapsed})
            if phase == "complete_hold":
                status = self.authority.vehicle_status
                if (status is not None
                        and status.nav_state == status.NAVIGATION_STATE_OFFBOARD
                        and now - self.last_land_request >= 1.0):
                    self._request_land(now)

            setpoint = VehicleRatesSetpoint()
            fill_rate_setpoint(setpoint, timestamp_us, rates, self.thrust_body)
            self.rates_publisher.publish(setpoint)


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-rate", type=positive_float, default=50.0)
    parser.add_argument("--amplitude", type=positive_float, default=0.12,
                        help="symmetric roll/pitch rate pulse [rad/s]")
    parser.add_argument("--pulse-duration", type=positive_float, default=0.4)
    parser.add_argument("--recovery-duration", type=positive_float, default=1.2)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--initial-hold", type=positive_float, default=3.0)
    parser.add_argument("--final-hold", type=positive_float, default=5.0)
    parser.add_argument("--hover-acceleration-m-s2", type=positive_float, default=9.81)
    parser.add_argument("--hover-thrust-fraction", type=float, default=None,
                        help="use a measured PX4 hover fraction instead of the physical model")
    parser.add_argument("--max-feedback-age", type=positive_float, default=1.5)
    parser.add_argument("--trial-label", default="baseline",
                        help="label recorded in the experiment trace")
    parser.add_argument("--publish", action="store_true",
                        help="publish heartbeat/rates when PX4 confirms permission")
    add_thrust_model_arguments(parser)
    args = parser.parse_args(argv)
    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    try:
        args.thrust_model = thrust_model_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.hover_thrust_fraction is not None
            and (not math.isfinite(args.hover_thrust_fraction)
                 or not 0.0 <= args.hover_thrust_fraction <= 1.0)):
        parser.error("--hover-thrust-fraction must be finite and in [0, 1]")
    return args


def run_self_test():
    steps = build_step_sequence(amplitude=0.1, pulse_duration=0.4,
                                recovery_duration=1.0, cycles=2,
                                initial_hold=2.0, final_hold=3.0)
    assert steps[0] == ("initial_hold", 2.0, (0.0, 0.0, 0.0))
    assert steps[1][0] == "roll_1_positive"
    assert steps[1][2] == (0.1, 0.0, 0.0)
    assert steps[2][0] == "roll_1_negative"
    assert steps[2][2] == (-0.1, 0.0, 0.0)
    assert steps[7][0] == "pitch_1_positive"
    assert steps[7][2] == (0.0, 0.1, 0.0)
    assert step_at_elapsed(steps, 0.0) == ("initial_hold", (0.0, 0.0, 0.0))
    assert step_at_elapsed(steps, 2.0)[0] == "roll_1_positive"
    assert step_at_elapsed(steps, 2.41)[0] == "roll_1_negative"
    assert step_at_elapsed(steps, 1000.0) == ("complete_hold", (0.0, 0.0, 0.0))
    print("step16 self-test: PASS")


def main():
    args = parse_args()
    if rclpy is None:
        print("ERROR: rclpy, px4_msgs, or required PX4 messages are unavailable.")
        return 2
    if args.amplitude > 0.5:
        print("ERROR: refusing rate-step amplitude above 0.5 rad/s")
        return 2
    rclpy.init()
    node = None
    try:
        node = RateStepTestNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--self-test"]:
        run_self_test()
        raise SystemExit(0)
    raise SystemExit(main())

#!/usr/bin/env python3
"""Initialize the isolated Prototype Recovery, synchronize EKF, then fly NN.

Used by the fixed-PID launcher in a private Gazebo partition. All policy
inputs come from PX4 VehicleOdometry, never Gazebo ground truth. Gazebo state
is an independent initialization and handover check.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import hashlib
from pathlib import Path
import threading
import time
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.recovery_metrics import RecoveryTracker, METRICS_VERSION

from gz.transport13 import Node as GzNode
from gz.msgs10.stringmsg_pb2 import StringMsg
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from px4_msgs.msg import (
    VehicleOdometry,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleControlMode,
    VehicleCommand,
    VehicleCommandAck,
    OffboardControlMode,
    VehicleRatesSetpoint,
)

from offboard_authority import OffboardAuthority, px4_feedback_topic
from recovery_state import recovery_spawn, validate_training_state
from observation import message_to_observation
from policy_worker import (
    PolicyWorker,
    fill_offboard_heartbeat,
    fill_rates_setpoint,
    result_is_fresh,
)
from policy_client import PolicyClient
from thrust_mapping import PROTOTYPE_THRUST_MODEL
from initialization_gate import attitude_distance, observation_quaternion

BASE = "/world/default/model/prototype_recovery_0/flightlxx/"


def state_errors(observation, expected):
    return {
        "position_m": math.dist(observation[:3], expected[:3]),
        "attitude_rad": attitude_distance(
            observation_quaternion(observation), observation_quaternion(expected)
        ),
        "velocity_m_s": math.dist(observation[6:9], expected[6:9]),
        "rate_rad_s": math.dist(observation[9:], expected[9:]),
    }


def ready_for_release(
    observation, expected, local_valid, authority, result, reset_counter, now
):
    """Both current EKF state and the action's source must match this trial."""
    if not local_valid or not authority.rates:
        return False
    if not result_is_fresh(result, now, 0.1, 0.1):
        return False
    if (result.source_odometry or {}).get("reset_counter") != reset_counter:
        return False
    return all(
        value < 0.06 for value in state_errors(observation, expected).values()
    ) and all(
        value < 0.06 for value in state_errors(result.observation, expected).values()
    )


class RecoveryTrial(Node):
    def __init__(self, args):
        super().__init__("flightlxx_recovery_trial")
        self.args = args
        self.report = {
            "schema_version": 1,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "requested_state": list(args.state),
            "expected_observation": recovery_spawn(args.state, args.attitude_encoding)[
                "expected_policy_observation"
            ],
            "passed": False,
            "events": [],
            "control_source": "PX4 VehicleOdometry -> PPO -> CTBR -> PX4 rate loop",
            "initialization_passed": False,
            "policy_takeover_passed": False,
            "recovery_passed": False,
            "initializer_imu_conditioning": "held_only",
            "model": str(args.model),
            "attitude_encoding": args.attitude_encoding,
            "thresholds": {
                "position_m": 0.06,
                "attitude_rad": 0.06,
                "velocity_m_s": 0.06,
                "rate_rad_s": 0.06,
            },
        }
        self.report["model_sha256"] = hashlib.sha256(
            args.model.read_bytes()
        ).hexdigest()
        self.report["source_sha256"] = {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "recovery_controller.py",
                "recovery_vision_bridge.py",
                "state_contract.py",
                "thrust_mapping.py",
            )
        }
        self.expected = self.report["expected_observation"]
        self.authority = OffboardAuthority(1.5)
        self.odom = None
        self.odom_at = 0.0
        self.local = None
        self.local_at = 0.0
        self.telemetry = None
        self.telemetry_at = 0.0
        self.lock = threading.Lock()
        self.stage = "heartbeat"
        self.stage_at = time.monotonic()
        self.started = self.stage_at
        self.last_command = -math.inf
        self.good_since = None
        self.good_count = 0
        self.done = False
        self.last_print = 0.0
        self.control_count = 0
        self.release_time = None
        self.release_requested = None
        self.first_action = None
        self.success_since = None
        self.latest_result = None
        self.last_reset = None
        self.first_free = None
        self.prepared_once = False
        self.recovery_tracker = RecoveryTracker()
        self.release_sim_time = None
        self.report["metrics_version"] = METRICS_VERSION
        self.report["requested_duration_sim_s"] = args.duration
        self.report["trial_id"] = os.environ.get("S2S_TRIAL_ID")
        self.report["failure_category"] = None
        self.records = (args.output / "trial_trace.jsonl").open("w", buffering=1)
        self.gz = GzNode()
        self.gzpub = self.gz.advertise(BASE + "set_state", StringMsg)
        self.gz.subscribe(StringMsg, BASE + "state_ack/telemetry", self.on_truth)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self.on_odom, qos
        )
        self.create_subscription(
            VehicleLocalPosition,
            px4_feedback_topic("/fmu/out/vehicle_local_position", VehicleLocalPosition),
            self.on_local,
            qos,
        )
        self.create_subscription(
            VehicleStatus,
            px4_feedback_topic("/fmu/out/vehicle_status", VehicleStatus),
            lambda m: self.authority.update_status(m, time.monotonic()),
            qos,
        )
        self.create_subscription(
            VehicleControlMode,
            px4_feedback_topic("/fmu/out/vehicle_control_mode", VehicleControlMode),
            lambda m: self.authority.update_control_mode(m, time.monotonic()),
            qos,
        )
        self.create_subscription(
            VehicleCommandAck,
            px4_feedback_topic("/fmu/out/vehicle_command_ack", VehicleCommandAck),
            self.on_ack,
            qos,
        )
        self.hb = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", reliable
        )
        self.rates = self.create_publisher(
            VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", reliable
        )
        self.cmd = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", reliable
        )
        self.worker = self.build_worker()
        self.create_timer(0.02, self.tick)
        self.event("started", phase="heartbeat")

    def build_worker(self):
        client = PolicyClient("127.0.0.1", self.args.port, 0.2)
        client.ping()
        return PolicyWorker(
            client,
            lambda: PolicyClient("127.0.0.1", self.args.port, 0.2),
            PROTOTYPE_THRUST_MODEL,
        )

    def control_metadata(self):
        return {}

    def prepare_operation(self):
        return 'prepare',self.args.state

    def preparation_ready(self,good,obs,truth,result,trace,now):
        return self.good_count >= 3 and now-self.good_since >= .04 if self.good_since is not None else False

    def preparation_complete(self,obs,result,now):
        self.transition('arm')

    def release_matched_state(self,obs,result,now):
        self.report['release_observation']=list(obs)
        self.report['release_errors']=self.errors(obs,self.expected)
        self.report['release_policy_observation']=list(result.observation)
        self.report['release_action']=list(result.raw_action)
        self.report['initialization_passed']=True
        self.report['armed_before_release']=True
        self.release_requested=now
        self.report['release_request_monotonic_s']=now
        self.inject('release')
        self.transition('flight')

    def completion_outcome(self):
        return self.recovery_tracker.first_confirmation is not None, "flight_completed"

    def event(self, name, **values):
        data = {
            "event": name,
            "wall_time_ns": time.time_ns(),
            "monotonic_s": time.monotonic(),
            **values,
        }
        self.report["events"].append(data)
        self.records.write(json.dumps(data, allow_nan=False) + "\n")
        self.get_logger().info(name + " " + json.dumps(values, allow_nan=False))

    def transition(self, stage):
        self.stage = stage
        self.stage_at = time.monotonic()
        self.last_command = -math.inf
        self.good_since = None
        self.good_count = 0
        self.event("phase", phase=stage)

    def on_truth(self, msg):
        try:
            data = json.loads(msg.data)
            with self.lock:
                self.telemetry = data
                self.telemetry_at = time.monotonic()
        except ValueError:
            pass

    def on_odom(self, msg):
        self.odom = msg
        self.odom_at = time.monotonic()

    def on_local(self, msg):
        self.local = msg
        self.local_at = time.monotonic()

    def on_ack(self, msg):
        self.event("command_ack", command=int(msg.command), result=int(msg.result))

    def command(self, code, **parameters):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.command = code
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 42
        msg.source_component = 1
        msg.from_external = True
        for key, val in parameters.items():
            setattr(msg, key, float(val))
        self.cmd.publish(msg)
        self.last_command = time.monotonic()
        self.event("command", command=code, parameters=parameters)

    def inject(self, operation, values=()):
        msg = StringMsg()
        msg.data = ",".join(
            (operation, uuid.uuid4().hex[:12])
            + tuple(format(x, ".17g") for x in values)
        )
        self.gzpub.publish(msg)
        self.event("gazebo_command", operation=operation, values=list(values))

    def errors(self, obs, expected):
        return state_errors(obs, expected)

    def finish(self, passed, reason):
        self.report["recovery_passed"] = bool(passed)
        if not passed:
            if reason in (
                "stale_px4_state",
                "stale_gazebo_state",
                "lost_offboard_or_fresh_policy",
            ) or reason.startswith("workflow_timeout_"):
                self.report["failure_category"] = "communication_mode_interruption"
            elif not self.report["initialization_passed"]:
                self.report["failure_category"] = "initialization_failure"
            else:
                self.report["failure_category"] = "policy_recovery_failure"
        self.report.update(
            passed=passed,
            reason=reason,
            control_count=self.control_count,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.event("finished", passed=passed, reason=reason)
        if self.release_time is not None:
            self.transition("landing")
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            return
        self.transition("stop")
        self.command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0, param2=21196
        )

    def tick(self):
        now = time.monotonic()
        if self.stage == "stop":
            status = self.authority.vehicle_status
            if (
                status is not None
                and status.arming_state == status.ARMING_STATE_DISARMED
            ):
                self.done = True
            elif now - self.stage_at > 5:
                self.done = True
                self.report["disarm_confirmed"] = False
            elif now - self.last_command > 0.5:
                self.command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=0,
                    param2=21196,
                )
            return
        if self.stage == "landing":
            status = self.authority.vehicle_status
            if (
                status is not None
                and status.arming_state == status.ARMING_STATE_DISARMED
            ):
                self.report["native_landing_complete"] = True
                self.event("landed_and_disarmed")
                self.done = True
            elif now - self.stage_at > 35:
                self.report["native_landing_complete"] = False
                self.report["postflight_cleanup_reason"] = "native_landing_timeout"
                self.transition("stop")
                self.command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=0,
                    param2=21196,
                )
            elif now - self.last_command > 2:
                self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            return
        if now - self.started > self.args.timeout:
            self.finish(False, "workflow_timeout_" + self.stage)
            return
        if (
            self.odom is None
            or self.local is None
            or now - self.odom_at > 0.2
            or now - self.local_at > 0.3
        ):
            if self.release_requested is not None:
                self.finish(False, "stale_px4_state")
            return
        try:
            obs = message_to_observation(
                self.odom, (0, 0, 5), (0, 0, 5), 0.0, 0.0, self.args.attitude_encoding
            )
        except ValueError as e:
            self.finish(False, str(e))
            return
        with self.lock:
            truth = self.telemetry
            truth_at = self.telemetry_at
        if truth is None or now - truth_at > 0.15:
            if self.release_requested is not None:
                self.finish(False, "stale_gazebo_state")
            return
        auth = self.authority.decision(now)
        hb = OffboardControlMode()
        fill_offboard_heartbeat(hb, self.get_clock().now().nanoseconds // 1000)
        self.hb.publish(hb)
        self.control_tick_time = now
        result, error, count = self.worker.snapshot()
        source = {
            k: int(getattr(self.odom, k))
            for k in [
                "timestamp",
                "timestamp_sample",
                "reset_counter",
                "pose_frame",
                "velocity_frame",
            ]
        }
        self.worker.submit(obs, self.odom_at, source)
        self.latest_result = result
        fresh = result_is_fresh(result, now, 0.1, 0.1)
        published = False
        if fresh and auth.rates:
            msg = VehicleRatesSetpoint()
            fill_rates_setpoint(
                msg,
                self.get_clock().now().nanoseconds // 1000,
                result.rates_frd,
                result.thrust_body_frd,
            )
            self.rates.publish(msg)
            self.control_count += 1
            published = True
        trace = {
            "event": "tick",
            "wall_time_ns": time.time_ns(),
            "monotonic_s": now,
            "phase": self.stage,
            "px4_timestamp_sample": int(self.odom.timestamp_sample),
            "reset_counter": int(self.odom.reset_counter),
            "observation": list(obs),
            "truth": truth,
            "authority": auth.status,
            "ctbr_published": published,
            "local_valid": bool(
                self.local.xy_valid
                and self.local.z_valid
                and self.local.v_xy_valid
                and self.local.v_z_valid
            ),
            "result_observation": list(result.observation) if result else None,
            "raw_action": list(result.raw_action) if result else None,
            "rates_frd": list(result.rates_frd) if result else None,
            "thrust_body_frd": list(result.thrust_body_frd) if result else None,
            "policy_source_reset_counter": (
                (result.source_odometry or {}).get("reset_counter") if result else None
            ),
        }
        trace["policy_sequence"] = result.sequence if result else None
        trace["setpoint_timestamp_us"] = int(msg.timestamp) if published else None
        trace["simulation_time_s"] = truth["sim_time_s"]
        trace["physical_thrust_saturated"] = bool(
            result
            and (
                result.thrust_mapping.saturated_low
                or result.thrust_mapping.saturated_high
            )
        )
        trace["policy_action_age_ms"] = (
            (now - result.completed_at) * 1000 if result else None
        )
        trace["policy_input_age_ms"] = (
            (now - result.observation_time) * 1000 if result else None
        )
        trace.update(self.control_metadata())
        self.records.write(json.dumps(trace, allow_nan=False) + "\n")
        if now - self.last_print > 1:
            self.last_print = now
            self.get_logger().info(
                "phase=%s obs=%s authority=%s NN=%s"
                % (self.stage, tuple(round(x, 3) for x in obs), auth.status, published)
            )
        if self.stage == "heartbeat":
            if now - self.stage_at > 2 and fresh:
                self.transition("offboard")
        elif self.stage == "offboard":
            s = self.authority.vehicle_status
            if s is not None and s.nav_state == s.NAVIGATION_STATE_OFFBOARD:
                self.transition("arm")
            elif now - self.last_command > 0.5:
                self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1, param2=6)
        elif self.stage == "arm":
            if auth.rates:
                if not self.prepared_once:
                    self.prepared_once = True
                    self.transition("prepare")
                    operation,values=self.prepare_operation()
                    self.inject(operation,values)
                    return
                err = self.errors(obs, self.expected)
                if not (
                    published
                    and ready_for_release(
                        obs,
                        self.expected,
                        trace["local_valid"],
                        auth,
                        result,
                        int(self.odom.reset_counter),
                        now,
                    )
                ):
                    if now - self.stage_at > 2:
                        self.finish(False, "armed_state_not_matched")
                    return
                self.release_matched_state(obs,result,now)
            elif now - self.last_command > 0.2:
                self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1)
        elif self.stage == "prepare":
            err = self.errors(obs, self.expected)
            good = (
                published
                and truth["holding"]
                and truth["prepared"]
                and ready_for_release(
                    obs,
                    self.expected,
                    trace["local_valid"],
                    auth,
                    result,
                    int(self.odom.reset_counter),
                    now,
                )
            )
            if good:
                if self.good_since is None:
                    self.good_since = now
                self.good_count += 1
            else:
                self.good_since = None
                self.good_count = 0
            if self.preparation_ready(good,obs,truth,result,trace,now):
                self.report["handover_observation"] = list(obs)
                self.report["handover_errors"] = err
                self.report["handover_truth"] = truth
                self.report["handover_action"] = list(result.raw_action)
                self.report["handover_policy_observation"] = list(result.observation)
                self.report["prepare_observation"] = list(obs)
                self.report["prepare_errors"] = err
                self.preparation_complete(obs,result,now)
            elif now - self.stage_at > 5:
                self.finish(False, "full_state_sync_failed_" + str(err))
        elif self.stage == "flight":
            if self.release_time is None and not truth["holding"]:
                self.release_time = now
                self.first_free = truth
                self.report["first_free_truth"] = truth
                self.report["first_free_observation"] = list(obs)
                self.report["release_physics_sample"] = truth.get(
                    "first_released_state"
                )
                self.report["policy_takeover_passed"] = bool(published and auth.rates)
                self.report["first_free_ctbr_already_published"] = bool(published)
                self.release_sim_time = truth["released_sim_time_s"]
                self.report["release_sim_time_s"] = self.release_sim_time
                self.event("released", gazebo_sim_time_s=truth["sim_time_s"])
            if self.release_time is not None:
                if not auth.rates or not published:
                    self.finish(False, "lost_offboard_or_fresh_policy")
                    return
                if not (0.5 < obs[2] < 10) or math.hypot(obs[0], obs[1]) > 5:
                    self.finish(False, "flight_envelope_exceeded")
                    return
                elapsed = truth["sim_time_s"] - self.release_sim_time
                if (
                    self.recovery_tracker.last_time is None
                    or elapsed > self.recovery_tracker.last_time + 1e-9
                ):
                    metrics = self.recovery_tracker.update(
                        elapsed,
                        math.dist(obs[:3], (0, 0, 5)),
                        math.sqrt(sum(x * x for x in obs[6:9])),
                    )
                    self.report["recovery_metrics"] = metrics
                    self.report["recovery_time_s"] = metrics["recovery_time_s"]
                if elapsed >= self.args.duration:
                    self.report["final_observation"] = list(obs)
                    self.report["completed_duration_sim_s"] = elapsed
                    self.finish(*self.completion_outcome())
                    return

    def close(self):
        self.worker.close()
        self.records.close()
        self.report["disarm_confirmed"] = bool(
            self.authority.vehicle_status
            and self.authority.vehicle_status.arming_state
            == VehicleStatus.ARMING_STATE_DISARMED
        )
        self.report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (self.args.output / "trial_result.json").write_text(
            json.dumps(self.report, indent=2, allow_nan=False)
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=float, nargs=12, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5801)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--attitude-encoding", choices=("eigen_zyx", "principal_zyx"), default="eigen_zyx")
    parser.add_argument("--tracking-axis", choices=("roll", "pitch", "yaw", "thrust"))
    parser.add_argument("--tracking-amplitude", type=float, default=0.12)
    args = parser.parse_args(argv)
    args.state = validate_training_state(args.state)
    args.output.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("GZ_PARTITION", "").startswith("flightlxx_recovery_"):
        parser.error("Use run_recovery_trial.sh to launch the isolated SITL first")
    if (
        not math.isfinite(args.duration)
        or args.duration <= 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("duration and timeout must be finite and positive")
    return args


def run_trial(args, node_class=RecoveryTrial):
    rclpy.init()
    node = node_class(args)
    try:
        while not node.done:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        node.finish(False, "interrupted")
        try:
            while not node.done:
                rclpy.spin_once(node, timeout_sec=0.01)
        except KeyboardInterrupt:
            node.command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0, param2=21196
            )
            node.report["reason"] = "second_interrupt_simulation_abort"
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
    print(json.dumps(node.report, indent=2, allow_nan=False))
    return 0 if node.report["passed"] else 2


def main():
    return run_trial(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Record FlightLxx experiments without issuing any vehicle commands.

ROS callbacks enqueue snapshots; one writer thread owns JSONL/CSV files.
Ctrl+C stops recording, optionally copies this run's PX4 ULogs, and creates ZIP.
Requires the existing ROS 2 Humble environment, not the policy Conda environment.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    rclpy = None
    Node = object
    ExternalShutdownException = KeyboardInterrupt

TRACE_TOPIC = "/flightlxx/sim_to_sim/policy_trace"
TOPIC_PATTERN = re.compile(
    r"^/fmu/(?:in|out)/(?:vehicle_odometry|vehicle_local_position|vehicle_rates_setpoint|"
    r"offboard_control_mode|vehicle_status|vehicle_control_mode|"
    r"vehicle_command|vehicle_command_ack|vehicle_land_detected|"
    r"failsafe_flags|vehicle_angular_velocity|vehicle_torque_setpoint|"
    r"vehicle_thrust_setpoint|actuator_motors|actuator_outputs|esc_status|"
    r"hover_thrust_estimate)(?:_v\d+)?$"
)
OBS_NAMES = ("x", "y", "z", "yaw", "pitch", "roll", "vx", "vy", "vz", "wx", "wy", "wz")
PREFIX = ("receive_wall_ns", "receive_monotonic_ns", "receive_ros_ns", "topic")
CSV_FIELDS = {
    "odometry": PREFIX + ("timestamp", "timestamp_sample", "pose_frame", "velocity_frame", "reset_counter")
        + tuple("position_" + str(i) for i in range(3))
        + tuple("q_wxyz_" + str(i) for i in range(4))
        + tuple("velocity_" + str(i) for i in range(3))
        + tuple("angular_velocity_frd_" + str(i) for i in range(3)),
    "local_position": PREFIX + (
        "timestamp", "timestamp_sample", "xy_valid", "z_valid",
        "v_xy_valid", "v_z_valid", "x", "y", "z", "vx", "vy", "vz",
        "ax", "ay", "az", "heading",
    ),
    "rates_setpoint": PREFIX + ("timestamp", "roll", "pitch", "yaw", "thrust_x", "thrust_y", "thrust_z", "reset_integral"),
    "policy": PREFIX + (
        "controller_run_id", "tick", "status", "published", "setpoint_timestamp_us",
        "heartbeat_published", "heartbeat_timestamp_us", "heartbeat_hz",
        "control_authority", "ctbr_allowed", "px4_nav_state", "px4_arming_state",
        "vehicle_status_age_ms", "control_mode_age_ms",
        "sequence", "new_action", "inference_submitted", "inference_count",
        "target_hz", "publish_hz", "infer_hz", "action_age_ms", "state_age_ms",
        "inference_ms", "latest_odometry_age_ms", "source_timestamp_us",
        "source_timestamp_sample_us", "source_reset_counter", "source_pose_frame",
        "source_velocity_frame", "collective_log_m_s2", "error",
        "px4_thrust_fraction", "motor_output_fraction", "motor_speed_rad_s",
        "minimum_acceleration_m_s2", "maximum_acceleration_m_s2",
        "thrust_saturated_low", "thrust_saturated_high",
        "thrust_model_vehicle_mass_kg", "thrust_model_motor_count",
        "thrust_model_motor_thrust_coefficient", "thrust_model_motor_omega_min",
        "thrust_model_motor_omega_max", "thrust_model_thrust_model_factor",
        "thrust_model_max_acceleration_m_s2",
    ) + tuple("obs_" + x for x in OBS_NAMES)
        + tuple("action_" + str(i) for i in range(4))
        + ("rate_sp_roll", "rate_sp_pitch", "rate_sp_yaw", "thrust_sp_x", "thrust_sp_y", "thrust_sp_z"),
    "vehicle_status": PREFIX + (
        "timestamp", "arming_state", "nav_state", "failsafe", "accepts_offboard_setpoints",
        "pre_flight_checks_pass", "armed_time", "takeoff_time",
    ),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_json(value):
    """Preserve arrays and fields; encode ROS non-finite floats as JSON null."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "tolist"):
        return clean_json(value.tolist())
    if isinstance(value, (list, tuple, bytes, bytearray)) or hasattr(value, "__iter__"):
        return [clean_json(v) for v in value]
    raise TypeError("Unsupported ROS field type: " + type(value).__name__)


def csv_row(event):
    data = event["data"]
    row = {k: event.get(k) for k in PREFIX}
    base = re.sub(r"_v\d+$", "", event["topic"].rsplit("/", 1)[-1])
    if event["topic"] == TRACE_TOPIC:
        result = data.get("result")
        result = result if isinstance(result, dict) else {}
        row.update(data)
        row.update(result)
        source = result.get("source_odometry")
        source = source if isinstance(source, dict) else {}
        row.update({"source_" + k + ("_us" if k in ("timestamp", "timestamp_sample") else ""): v for k, v in source.items()})
        mapping = result.get("thrust_mapping")
        if isinstance(mapping, dict):
            row.update({
                "px4_thrust_fraction": mapping.get("px4_fraction"),
                "motor_output_fraction": mapping.get("motor_output_fraction"),
                "motor_speed_rad_s": mapping.get("motor_speed_rad_s"),
                "minimum_acceleration_m_s2": mapping.get(
                    "minimum_acceleration_m_s2"
                ),
                "maximum_acceleration_m_s2": mapping.get(
                    "maximum_acceleration_m_s2"
                ),
                "thrust_saturated_low": mapping.get("saturated_low"),
                "thrust_saturated_high": mapping.get("saturated_high"),
            })
        config = data.get("config")
        config = config if isinstance(config, dict) else {}
        thrust_model = config.get("thrust_model")
        if isinstance(thrust_model, dict):
            row.update({"thrust_model_" + k: v for k, v in thrust_model.items()})
        if "thrust_model_max_acceleration_m_s2" in config:
            row["thrust_model_max_acceleration_m_s2"] = config[
                "thrust_model_max_acceleration_m_s2"
            ]
        for names, values in (
            (["obs_" + x for x in OBS_NAMES], result.get("observation", [])),
            (["action_" + str(i) for i in range(4)], result.get("raw_action", [])),
            (["rate_sp_roll", "rate_sp_pitch", "rate_sp_yaw"], result.get("rates_frd", [])),
            (["thrust_sp_x", "thrust_sp_y", "thrust_sp_z"], result.get("thrust_body_frd", [])),
        ):
            if isinstance(values, (list, tuple)):
                row.update(zip(names, values))
        return "policy", row
    row.update(data)
    if base == "vehicle_odometry":
        for key, label in (("position", "position"), ("q", "q_wxyz"), ("velocity", "velocity"), ("angular_velocity", "angular_velocity_frd")):
            row.update({label + "_" + str(i): v for i, v in enumerate(data.get(key, []))})
        return "odometry", row
    if base == "vehicle_local_position":
        return "local_position", row
    if base == "vehicle_rates_setpoint":
        row.update(zip(("thrust_x", "thrust_y", "thrust_z"), data.get("thrust_body", [])))
        return "rates_setpoint", row
    if base == "vehicle_status":
        return "vehicle_status", row
    return None, None


class AsyncWriter:
    def __init__(self, directory, queue_size=4096, flush_period=1.0):
        self.directory = Path(directory)
        self.queue = queue.Queue(maxsize=queue_size)
        self.flush_period = flush_period
        self.stop_event = threading.Event()
        self.written = Counter()
        self.dropped = 0
        self.error = None
        self.configs = {}
        self.thread = threading.Thread(target=self._run, name="experiment_file_writer", daemon=True)
        self.thread.start()

    def submit(self, event):
        if self.error is not None or self.stop_event.is_set():
            self.dropped += 1
            return False
        try:
            self.queue.put_nowait(event)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=15.0)
        if self.thread.is_alive():
            self.error = self.error or "file writer did not finish within 15 seconds"
        return not self.thread.is_alive()

    def _run(self):
        files = []
        try:
            raw = (self.directory / "events.jsonl").open("w", encoding="utf-8", buffering=1024*1024)
            files.append(raw)
            csv_writers = {}
            last_flush = time.monotonic()
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    event = self.queue.get(timeout=0.1)
                except queue.Empty:
                    event = None
                if event is not None:
                    raw.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                    self.written[event["topic"]] += 1
                    if event["topic"] == TRACE_TOPIC and "config" in event["data"]:
                        self.configs[str(event["data"].get("controller_run_id", "unknown"))] = event["data"]["config"]
                    name, row = csv_row(event)
                    if name is not None:
                        if name not in csv_writers:
                            stream = (self.directory / (name + ".csv")).open("w", newline="", encoding="utf-8")
                            files.append(stream)
                            csv_writers[name] = csv.DictWriter(stream, CSV_FIELDS[name], extrasaction="ignore")
                            csv_writers[name].writeheader()
                        csv_writers[name].writerow(row)
                if time.monotonic() - last_flush >= self.flush_period:
                    for stream in files:
                        stream.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            self.error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            for stream in files:
                try:
                    stream.close()
                except Exception as exc:
                    self.error = self.error or str(exc)


def collect_ulogs(px4_directory, run_directory, start_wall_ns):
    """Copy only logs modified since this recorder started; report live writes."""
    root = Path(px4_directory).expanduser()
    roots = [root / "build/px4_sitl_default/rootfs/fs/log", root / "build/px4_sitl_default/rootfs/log"]
    results, seen = [], set()
    for log_root in roots:
        if not log_root.is_dir():
            continue
        for path in sorted(log_root.rglob("*.ulg")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                before = path.stat()
                if before.st_mtime_ns < start_wall_ns:
                    continue
                destination = Path(run_directory) / "px4_ulog" / log_root.parent.name / path.relative_to(log_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                after = path.stat()
                results.append({"source": str(path), "saved": str(destination.relative_to(run_directory)),
                                "bytes": destination.stat().st_size,
                                "changed_during_copy": (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)})
            except OSError as exc:
                results.append({"source": str(path), "error": str(exc)})
    return results


def make_zip(directory):
    directory = Path(directory)
    destination = directory.with_suffix(".zip")
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path(directory.name) / path.relative_to(directory)))
    temporary.replace(destination)
    return destination


class ExperimentRecorder(Node):
    def __init__(self, args, writer):
        super().__init__("flightlxx_experiment_recorder")
        self.args, self.writer = args, writer
        self.started = time.monotonic()
        self.subscriptions_by_topic = {}
        self.received = Counter()
        self.parse_errors = 0
        self.type_errors = {}
        self.last_display = 0.0
        self.last_dropped = 0
        self.qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.timer = self.create_timer(1.0, self.monitor)
        self.discover_topics()

    def discover_topics(self):
        for topic, types in self.get_topic_names_and_types():
            wanted = topic == TRACE_TOPIC or TOPIC_PATTERN.fullmatch(topic) or topic in self.args.extra_topic
            if not wanted or topic in self.subscriptions_by_topic or not types:
                continue
            try:
                type_name = types[0]
                cls = get_message(type_name)
                sub = self.create_subscription(cls, topic, lambda msg, t=topic, n=type_name: self.receive(t, n, msg), self.qos)
                self.subscriptions_by_topic[topic] = (sub, type_name)
                self.type_errors.pop(topic, None)
                self.get_logger().info("recording %s [%s]" % (topic, type_name))
            except Exception as exc:
                self.type_errors[topic] = str(exc)

    def receive(self, topic, type_name, message):
        self.received[topic] += 1
        try:
            event = {"receive_wall_ns": time.time_ns(), "receive_monotonic_ns": time.monotonic_ns(),
                     "receive_ros_ns": self.get_clock().now().nanoseconds, "topic": topic, "message_type": type_name}
            payload = json.loads(message.data) if topic == TRACE_TOPIC else message_to_ordereddict(message)
            if not isinstance(payload, dict):
                raise ValueError("message payload must be an object")
            event["data"] = clean_json(payload)
            self.writer.submit(event)
        except Exception as exc:
            self.parse_errors += 1
            if self.parse_errors <= 3:
                self.get_logger().warning("could not record %s: %s" % (topic, exc))

    def monitor(self):
        if self.writer.error:
            self.get_logger().error("recording stopped: " + self.writer.error)
            rclpy.try_shutdown()
            return
        self.discover_topics()
        now = time.monotonic()
        if now - self.last_display >= 5.0:
            self.last_display = now
            self.get_logger().info("received=%d topics=%d policy_trace=%d dropped=%d" %
                                   (sum(self.received.values()), len(self.received), self.received[TRACE_TOPIC], self.writer.dropped))
            if self.received[TRACE_TOPIC] == 0:
                self.get_logger().warning("No policy trace yet: start the replacement step09; old steps only provide PX4 topic data.")
        if self.writer.dropped > self.last_dropped:
            self.get_logger().warning("recording queue overflow: %d events dropped" % self.writer.dropped)
            self.last_dropped = self.writer.dropped
        if self.args.duration > 0 and now - self.started >= self.args.duration:
            rclpy.try_shutdown()


def self_test():
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name) / "example"
        directory.mkdir()
        writer = AsyncWriter(directory, queue_size=32)
        base = dict(receive_wall_ns=1, receive_monotonic_ns=2, receive_ros_ns=3)
        events = [
            dict(base, topic="/fmu/out/vehicle_odometry", data={"position":[1,2,3],"q":[1,0,0,0],"velocity":[0,0,0],"angular_velocity":[0,0,0]}),
            dict(base, topic="/fmu/out/vehicle_local_position", data={"x":1.0,"y":2.0,"z":-5.0,"vx":0.1,"vy":0.2,"vz":0.0,"xy_valid":True,"z_valid":True}),
            dict(base, topic="/fmu/out/vehicle_status_v4", data={"nav_state":14,"arming_state":2}),
            dict(base, topic=TRACE_TOPIC, data={
                "controller_run_id":"test", "published":True,
                "heartbeat_published":True, "heartbeat_hz":50.0,
                "control_authority":"offboard_control", "ctbr_allowed":True,
                "px4_nav_state":14, "px4_arming_state":2,
                "config":{
                    "thrust_model":{
                        "vehicle_mass_kg":2.0643, "motor_count":4,
                        "motor_thrust_coefficient":8.54858e-6,
                        "motor_omega_min":150.0, "motor_omega_max":1000.0,
                        "thrust_model_factor":0.0,
                    },
                    "thrust_model_max_acceleration_m_s2":16.5645,
                },
                "result":{
                    "sequence":7, "observation":list(range(12)),
                    "raw_action":[0,0,0,0],
                    "source_odometry":{"timestamp_sample":123},
                    "thrust_mapping":{
                        "px4_fraction":0.729,
                        "motor_output_fraction":0.729,
                        "motor_speed_rad_s":769.5,
                        "minimum_acceleration_m_s2":0.373,
                        "maximum_acceleration_m_s2":16.5645,
                        "saturated_low":False, "saturated_high":False,
                    },
                },
            }),
        ]
        for event in events:
            assert writer.submit(clean_json(event))
        assert writer.close() and writer.error is None
        assert sum(writer.written.values()) == len(events)
        with (directory/"local_position.csv").open() as stream:
            row = next(csv.DictReader(stream))
        assert row["z"] == "-5.0" and row["xy_valid"] == "True"
        with (directory/"policy.csv").open() as stream:
            row = next(csv.DictReader(stream))
        assert row["sequence"] == "7" and row["obs_wz"] == "11" and row["source_timestamp_sample_us"] == "123"
        assert row["px4_thrust_fraction"] == "0.729"
        assert row["thrust_saturated_high"] == "False"
        assert row["thrust_model_max_acceleration_m_s2"] == "16.5645"
        assert row["heartbeat_published"] == "True" and row["heartbeat_hz"] == "50.0"
        assert row["control_authority"] == "offboard_control" and row["px4_nav_state"] == "14"
        assert clean_json({"bad":float("nan"),"a":[float("inf")]}) == {"bad":None,"a":[None]}
        archive = make_zip(directory)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and "example/events.jsonl" in z.namelist()
        assert not writer.submit(events[0]) and writer.dropped == 1
    print("step15 self-test: PASS (JSONL, CSV, local position, policy sequence, nonfinite values, shutdown, ZIP)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"logs")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 records until Ctrl+C")
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--with-ulog", action="store_true", help="copy this run's modified PX4 .ulg files after recording")
    parser.add_argument("--px4-dir", type=Path, default=Path.home()/"PX4-Autopilot")
    parser.add_argument("--extra-topic", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not math.isfinite(args.duration) or args.duration < 0 or args.queue_size < 1:
        parser.error("duration must be finite/nonnegative; queue-size must be positive")
    if rclpy is None:
        print("ERROR: source ROS 2 Humble and px4_ros2_ws; run with /usr/bin/python3, not Conda.")
        return 2
    run_dir = args.output_dir.expanduser().resolve() / (datetime.now(timezone.utc).strftime("experiment_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True)
    start_wall_ns, start_monotonic_ns = time.time_ns(), time.monotonic_ns()
    source_dir = Path(__file__).resolve().parent
    metadata = {"schema_version":1, "started_utc":utc_now(), "start_wall_ns":start_wall_ns,
                "start_monotonic_ns":start_monotonic_ns, "python":sys.version, "arguments":vars(args).copy(),
                "nonfinite_floats":"JSON null", "age_definition":"policy state_age starts at controller odometry callback receipt",
                "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.glob("*.py")}}
    metadata["arguments"] = {k:str(v) if isinstance(v,Path) else v for k,v in metadata["arguments"].items()}
    (run_dir/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Recording directory:", run_dir, flush=True)
    writer = AsyncWriter(run_dir, args.queue_size)
    node, recorder_error = None, None
    try:
        rclpy.init()
        node = ExperimentRecorder(args, writer)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        recorder_error = "%s: %s" % (type(exc).__name__, exc)
        print("ERROR:", recorder_error, flush=True)
    finally:
        if node is not None:
            node.timer.cancel()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        writer.close()
    if writer.thread.is_alive():
        print("ERROR:", writer.error, "Data directory retained:", run_dir)
        return 1
    summary = {"finished_utc":utc_now(), "elapsed_s":(time.monotonic_ns()-start_monotonic_ns)/1e9,
               "received_counts":dict(node.received) if node is not None else {},
               "written_counts":dict(writer.written), "queue_dropped":writer.dropped,
               "parse_errors":node.parse_errors if node is not None else None, "writer_error":writer.error,
               "recorder_error":recorder_error,
               "subscribed_types":{k:v[1] for k,v in node.subscriptions_by_topic.items()} if node is not None else {},
               "type_errors":node.type_errors if node is not None else {}, "controller_configs":writer.configs}
    if args.with_ulog:
        print("Collecting this run's ULogs...", flush=True)
        summary["ulog_files"] = collect_ulogs(args.px4_dir, run_dir, start_wall_ns)
    (run_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Creating ZIP...", flush=True)
    archive = make_zip(run_dir)
    print("EXPERIMENT ZIP:", archive, flush=True)
    print("Written:", sum(writer.written.values()), "Dropped:", writer.dropped, "Writer error:", writer.error)
    if writer.written[TRACE_TOPIC] == 0:
        print("NOTE: no exact policy trace captured; check that the new step09 process was restarted.")
    if args.with_ulog and not any("saved" in x for x in summary.get("ulog_files", [])):
        print("NOTE: no new ULog found in the default locations; ROS data is still saved.")
    return 1 if writer.error or recorder_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

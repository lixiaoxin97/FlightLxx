#!/usr/bin/env python3
"""Recovery-only Gazebo truth -> PX4 EV with explicit reset events.

This is a simulation sensor input, never a replacement for EKF policy feedback.
Disable SIM_GZ_EN_ODOM so exactly one publisher owns vehicle_visual_odometry.
"""

import argparse
import json
import math
import random
import threading
import time

from gz.transport13 import Node as GazeboNode
from gz.msgs10.stringmsg_pb2 import StringMsg
from gz.msgs10.imu_pb2 import IMU
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry

from recovery_state import quaternion_multiply, rotate_vector
from ramp_initialization import initializer_sensor_reference


def convert(state):
    """Gazebo ENU/FLU -> NED/FRD, using the actual link state."""
    state=initializer_sensor_reference(state)
    # ROS generated setters require Python float scalars; Gazebo's protobuf
    # JSON encoder can emit integer 0 for an exactly-zero coordinate.
    p = [float(value) for value in state["position_enu"]]
    v = [float(value) for value in state["velocity_enu"]]
    q = state["quaternion_enu_flu_wxyz"]
    q_ne = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)
    q_fb = (0.0, -1.0, 0.0, 0.0)
    q_nb = quaternion_multiply(quaternion_multiply(q_ne, q), q_fb)
    qnorm = math.sqrt(sum(x * x for x in q_nb))
    world_rate = state["angular_velocity_world_enu"]
    body_rate = rotate_vector((q[0], -q[1], -q[2], -q[3]), world_rate)
    return {
        "position": [p[1], p[0], -p[2]],
        "q": [x / qnorm for x in q_nb],
        "velocity": [v[1], v[0], -v[2]],
        "angular_velocity": [body_rate[0], -body_rate[1], -body_rate[2]],
        "reset_counter": int(state["reset_counter"]),
    }


def condition_held_imu(msg, state, noise=random.gauss):
    """Constrain only initializer IMU; preserve native flight data byte-for-byte."""
    if not state["holding"]:
        return msg
    q = state["quaternion_enu_flu_wxyz"]
    specific_force = rotate_vector((q[0], -q[1], -q[2], -q[3]), (0.0, 0.0, 9.8))
    rates = state["prepared_body_rates_flu"] if state["prepared"] else (0.0, 0.0, 0.0)
    for axis, value in zip(("x", "y", "z"), specific_force):
        sigma = 0.00686 if axis == "z" else 0.00637
        setattr(msg.linear_acceleration, axis, value + noise(0.0, sigma))
    for axis, value in zip(("x", "y", "z"), rates):
        setattr(msg.angular_velocity, axis, value + noise(0.0, 0.0008726646))
    return msg


class VisionBridge(Node):
    def __init__(self, args):
        super().__init__("flightlxx_recovery_vision")
        self.latest = None
        self.lock = threading.Lock()
        self.states = {}
        self.pending_imu = {}
        self.last_sim_time = -1.0
        self.count = 0
        self.pub = self.create_publisher(
            VehicleOdometry,
            "/fmu/in/vehicle_visual_odometry",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.gz = GazeboNode()
        self.gz.subscribe(StringMsg, args.topic, self.receive)
        imu_base = "/world/default/model/prototype_recovery_0/link/base_link/sensor/imu_sensor/"
        self.imu_pub = self.gz.advertise(imu_base + "imu", IMU)
        self.gz.subscribe(IMU, imu_base + "imu_raw", self.receive_imu)
        self.timer = self.create_timer(0.004, self.publish)
        self.get_logger().info("EV telemetry source: " + args.topic)

    def receive(self, msg):
        try:
            state = json.loads(msg.data)
            stamp = round(state["sim_time_s"] * 1e6)
            with self.lock:
                self.latest = (state, time.monotonic())
                self.states[stamp] = state
                if len(self.states) > 64:
                    self.states.pop(next(iter(self.states)))
                imu = self.pending_imu.pop(stamp, None)
            if imu is not None:
                self.forward_imu(imu, state)
        except (ValueError, TypeError):
            pass

    def receive_imu(self, msg):
        stamp = msg.header.stamp.sec * 1000000 + msg.header.stamp.nsec // 1000
        with self.lock:
            state = self.states.get(stamp)
            if state is None:
                # Physics telemetry and IMU arrive on separate transports.
                # Pair by simulation time so a previous held-state flag cannot
                # condition the first free-flight IMU sample.
                self.pending_imu[stamp] = msg
                if len(self.pending_imu) > 64:
                    self.pending_imu.pop(next(iter(self.pending_imu)))
                return
        self.forward_imu(msg, state)

    def forward_imu(self, msg, state):
        condition_held_imu(msg, state)
        self.imu_pub.publish(msg)

    def publish(self):
        with self.lock:
            item = self.latest
        if item is None:
            return
        state, received = item
        if (
            time.monotonic() - received > 0.1
            or state["sim_time_s"] <= self.last_sim_time
        ):
            return
        data = convert(state)
        msg = VehicleOdometry()
        # The DDS client translates synchronized ROS timestamps into PX4 HRT.
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.timestamp_sample = msg.timestamp
        msg.pose_frame = msg.POSE_FRAME_NED
        msg.velocity_frame = msg.VELOCITY_FRAME_NED
        for name, value in data.items():
            setattr(msg, name, value)
        msg.position_variance = [0.0001] * 3
        msg.velocity_variance = [0.0001] * 3
        msg.orientation_variance = [0.0025] * 3
        msg.quality = 100
        self.pub.publish(msg)
        self.last_sim_time = state["sim_time_s"]
        self.count += 1
        if self.count % 250 == 0:
            self.get_logger().info(
                "EV samples=%d sim=%.3f reset=%d holding=%s prepared=%s"
                % (
                    self.count,
                    self.last_sim_time,
                    msg.reset_counter,
                    state["holding"],
                    state["prepared"],
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/world/default/model/prototype_recovery_0/flightlxx/state_ack/telemetry",
    )
    args = parser.parse_args()
    rclpy.init()
    node = VisionBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

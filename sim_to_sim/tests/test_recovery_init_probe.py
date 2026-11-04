import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
#!/usr/bin/env python3
"""Regression checks for the disarmed estimator initialization gate."""

from types import SimpleNamespace
import time
import unittest

try:
    import rclpy
    from px4_msgs.msg import VehicleOdometry, VehicleLocalPosition, VehicleStatus
    from initialization_gate import InitProbe
except ImportError:
    rclpy = None


@unittest.skipIf(rclpy is None, "source ROS 2 and px4_ros2_ws first")
class RecoveryInitGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        args = SimpleNamespace(
            timeout=5, odometry_topic="/test/odometry",
            local_position_topic="/test/local_position", status_topic="/test/status",
            ready_file=None, max_age=.25, status_max_age=1.5,
            attitude_encoding="eigen_zyx", expected=(0, 0, 5)+(0,)*9,
            position_tolerance=.06, attitude_tolerance=.06,
            velocity_tolerance=.06, rate_tolerance=.06,
            require_preflight=False, phase="hold",
            stable_duration=.1, minimum_samples=10,
        )
        self.report = {"sample_count": 0, "stale_sample_count": 0, "passed": False}
        self.node = InitProbe(args, self.report)
        now = time.monotonic()
        odom = VehicleOdometry()
        odom.position = [0., 0., -5.]
        odom.q = [1., 0., 0., 0.]
        odom.velocity = [0., 0., 0.]
        odom.angular_velocity = [0., 0., 0.]
        odom.pose_frame = odom.POSE_FRAME_NED
        odom.velocity_frame = odom.VELOCITY_FRAME_NED
        local = VehicleLocalPosition()
        local.xy_valid = local.z_valid = local.v_xy_valid = local.v_z_valid = True
        status = VehicleStatus()
        status.arming_state = status.ARMING_STATE_DISARMED
        status.nav_state = status.NAVIGATION_STATE_AUTO_LOITER
        self.node.latest = {"odometry": (odom, now), "local_position": (local, now),
                            "status": (status, now-.6)}
        self.node.good_since = now-.2
        self.node.good_samples = 10
        self.node.last_print = now

    def tearDown(self):
        self.node.destroy_node()

    def test_normal_low_rate_status_allows_continuous_hold(self):
        self.node.evaluate()
        self.assertTrue(self.node.passed)
        self.assertFalse(self.report["last_sample"]["px4_preflight_pass"])

    def test_truly_stale_status_blocks_release(self):
        msg, received = self.node.latest["status"]
        self.node.latest["status"] = (msg, received-2)
        self.node.evaluate()
        self.assertFalse(self.node.passed)
        self.assertIsNone(self.node.good_since)

    def test_estimator_reset_restarts_hold_window(self):
        self.node.previous_reset = 1
        self.node.evaluate()
        self.assertFalse(self.node.passed)
        self.assertIn("estimator_reset", self.report["last_sample"]["reasons"])

    def test_invalid_frame_breaks_previous_good_interval(self):
        self.node.latest["odometry"][0].pose_frame = 0
        self.node.evaluate()
        self.assertFalse(self.node.passed)
        self.assertIsNone(self.node.good_since)
        self.assertEqual(self.node.good_samples, 0)

    def test_armed_vehicle_is_rejected(self):
        status = self.node.latest["status"][0]
        status.arming_state = status.ARMING_STATE_ARMED
        self.node.evaluate()
        self.assertFalse(self.node.passed)
        self.assertIn("vehicle_not_disarmed", self.report["last_sample"]["reasons"])


if __name__ == "__main__":
    unittest.main()

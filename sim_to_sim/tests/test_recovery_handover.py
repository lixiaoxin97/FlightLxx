import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
"""Recovery-only IO and release-gate regression tests (source ROS first)."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import json
import threading
import unittest

try:
    from gz.msgs10.imu_pb2 import IMU
    from recovery_vision_bridge import convert, condition_held_imu, VisionBridge
    from gz.msgs10.stringmsg_pb2 import StringMsg
    from recovery_controller import ready_for_release
    from install_recovery_ekf_patch import install, OLD, NEW
except ImportError:
    IMU = None


@unittest.skipIf(IMU is None, "source ROS and px4_ros2_ws")
class RecoveryHandoverTests(unittest.TestCase):
    def setUp(self):
        self.obs = (0.0, 0.0, 5.0) + (0.0,) * 9
        self.action = SimpleNamespace(
            observation=self.obs,
            completed_at=9.99,
            observation_time=9.98,
            source_odometry={"reset_counter": 12},
        )
        self.authority = SimpleNamespace(rates=True)

    def ready(self):
        return ready_for_release(
            self.obs, self.obs, True, self.authority, self.action, 12, 10.0
        )

    def test_matched_fresh_offboard_action_allows_release(self):
        self.assertTrue(self.ready())

    def test_action_from_before_reset_is_rejected(self):
        self.action.source_odometry["reset_counter"] = 11
        self.assertFalse(self.ready())

    def test_old_state_input_is_rejected_even_if_action_just_completed(self):
        self.action.observation_time = 9.0
        self.assertFalse(self.ready())

    def test_policy_input_must_match_requested_state(self):
        self.action.observation = self.obs[:6] + (0.2,) + (0.0,) * 5
        self.assertFalse(self.ready())

    def test_px4_native_mode_cannot_release(self):
        self.authority.rates = False
        self.assertFalse(self.ready())

    def test_native_flight_imu_is_unchanged(self):
        msg = IMU()
        msg.header.stamp.sec = 42
        msg.linear_acceleration.x = 1.2
        msg.angular_velocity.y = -0.5
        original = msg.SerializeToString()
        condition_held_imu(msg, {"holding": False})
        self.assertEqual(msg.SerializeToString(), original)

    def test_release_imu_waits_for_same_simulation_timestamp(self):
        forwarded = []
        bridge = SimpleNamespace(
            lock=threading.Lock(),
            states={},
            pending_imu={},
            latest=None,
            forward_imu=lambda msg, state: forwarded.append((msg, state)),
        )
        bridge.states[1000] = {"holding": True}
        imu = IMU()
        imu.header.stamp.nsec = 2000000
        VisionBridge.receive_imu(bridge, imu)
        self.assertEqual(forwarded, [])
        state = StringMsg()
        state.data = json.dumps({"sim_time_s": 0.002, "holding": False})
        VisionBridge.receive(bridge, state)
        self.assertEqual(len(forwarded), 1)
        self.assertFalse(forwarded[0][1]["holding"])
        self.assertEqual(bridge.pending_imu, {})

    def test_held_imu_uses_requested_rates_and_retains_timestamp(self):
        msg = IMU()
        msg.header.stamp.sec = 42
        state = {
            "holding": True,
            "prepared": True,
            "quaternion_enu_flu_wxyz": [1.0, 0.0, 0.0, 0.0],
            "prepared_body_rates_flu": [0.1, -0.2, 0.3],
        }
        condition_held_imu(msg, state, noise=lambda mean, sigma: 0.0)
        self.assertEqual(msg.header.stamp.sec, 42)
        self.assertEqual(msg.linear_acceleration.z, 9.8)
        self.assertEqual(msg.angular_velocity.y, -0.2)

    def test_visual_input_preserves_reset_and_converts_world_velocity(self):
        state = {
            "position_enu": [2.0, 3.0, 4.0],
            "velocity_enu": [0.1, 0.2, 0.3],
            "quaternion_enu_flu_wxyz": [1.0, 0.0, 0.0, 0.0],
            "angular_velocity_world_enu": [0.4, 0.5, 0.6],
            "reset_counter": 19,
        }
        result = convert(state)
        self.assertEqual(result["reset_counter"], 19)
        self.assertEqual(result["position"], [3.0, 2.0, -4.0])
        self.assertEqual(result["velocity"], [0.2, 0.1, -0.3])
        self.assertEqual(result["angular_velocity"], [0.4, -0.5, -0.6])

    def test_zero_position_and_velocity_components_are_float_scalars(self):
        result = convert(
            {
                "position_enu": [0, 0, 5],
                "velocity_enu": [0, 0, 0],
                "quaternion_enu_flu_wxyz": [1.0, 0.0, 0.0, 0.0],
                "angular_velocity_world_enu": [0.0, 0.0, 0.0],
                "reset_counter": 0,
            }
        )
        self.assertTrue(all(type(value) is float for value in result["position"]))
        self.assertTrue(all(type(value) is float for value in result["velocity"]))

    def test_patch_is_idempotent_and_refuses_unknown_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = (
                Path(temp)
                / "src/modules/ekf2/EKF/aid_sources/external_vision/ev_vel_control.cpp"
            )
            path.parent.mkdir(parents=True)
            path.write_text(OLD)
            install(temp)
            self.assertEqual(path.read_text(), NEW)
            install(temp)
            self.assertEqual(path.read_text(), NEW)
            path.write_text("unrecognized user changes")
            with self.assertRaises(RuntimeError):
                install(temp)
            self.assertEqual(path.read_text(), "unrecognized user changes")


if __name__ == "__main__":
    unittest.main()

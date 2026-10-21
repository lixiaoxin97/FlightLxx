#!/usr/bin/env python3
"""Mode-transition regressions. No DDS traffic or vehicle commands are sent."""

from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

from offboard_authority import OffboardAuthority, px4_feedback_topic
import step09_px4_offboard_rates as controller


def status(nav=4, armed=True, failsafe=False):
    return SimpleNamespace(nav_state=nav, arming_state=2 if armed else 1,
                           failsafe=failsafe, NAVIGATION_STATE_OFFBOARD=14,
                           ARMING_STATE_ARMED=2)


def mode(offboard=False, armed=True, rates=True, attitude=False):
    return SimpleNamespace(flag_armed=armed, flag_control_offboard_enabled=offboard,
                           flag_control_rates_enabled=rates,
                           flag_control_attitude_enabled=attitude)


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.gate = OffboardAuthority(max_feedback_age=1.5)

    def update(self, nav=14, offboard=True, now=10., **kwargs):
        self.gate.update_status(status(nav, **kwargs), now)
        self.gate.update_control_mode(mode(offboard, armed=kwargs.get("armed", True)), now)

    def test_unknown_feedback_never_permits_output(self):
        self.assertFalse(self.gate.decision(10).heartbeat)
        self.gate.update_status(status(14), 10)
        self.assertFalse(self.gate.decision(10).rates)
        self.assertFalse(self.gate.decision(10).heartbeat)

    def test_all_native_modes_are_heartbeat_only(self):
        for nav in (0, 1, 2, 3, 4, 5, 10, 12, 17, 18, 19, 20):
            with self.subTest(nav=nav):
                self.update(nav=nav, offboard=False)
                decision = self.gate.decision(10)
                self.assertTrue(decision.heartbeat)
                self.assertFalse(decision.rates)

    def test_entry_needs_both_confirmations_in_either_order(self):
        for status_first in (True, False):
            self.update(nav=4, offboard=False)
            if status_first:
                self.gate.update_status(status(14), 10.1)
            else:
                self.gate.update_control_mode(mode(True), 10.1)
            self.assertFalse(self.gate.decision(10.1).rates)
            self.update(now=10.2)
            self.assertTrue(self.gate.decision(10.2).rates)

    def test_exit_stops_rates_on_either_feedback_first(self):
        for status_first in (True, False):
            self.update()
            self.assertTrue(self.gate.decision(10).rates)
            if status_first:
                self.gate.update_status(status(18), 10.1)
            else:
                self.gate.update_control_mode(mode(False, attitude=True), 10.1)
            self.assertFalse(self.gate.decision(10.1).rates)

    def test_disarmed_or_failsafe_never_permits_rates(self):
        self.update(armed=False)
        self.assertFalse(self.gate.decision(10).rates)
        self.update(failsafe=True)
        self.assertFalse(self.gate.decision(10).rates)
        self.assertFalse(self.gate.decision(10).heartbeat)
        self.update()
        self.gate.update_control_mode(mode(True, armed=False), 10)
        self.assertFalse(self.gate.decision(10).rates)

    def test_other_offboard_control_levels_cannot_receive_ctbr(self):
        self.update()
        self.gate.update_control_mode(mode(True, attitude=True), 10)
        self.assertFalse(self.gate.decision(10).rates)
        self.gate.update_control_mode(mode(True, rates=False), 10)
        self.assertFalse(self.gate.decision(10).rates)

    def test_each_feedback_stream_must_remain_fresh(self):
        self.update()
        self.gate.update_status(status(14), 12)
        self.assertEqual(self.gate.decision(12).status, "stale_control_mode")
        self.assertFalse(self.gate.decision(12).heartbeat)
        self.update()
        self.gate.update_control_mode(mode(True), 12)
        self.assertEqual(self.gate.decision(12).status, "stale_vehicle_status")
        self.assertFalse(self.gate.decision(12).rates)

    def test_monotonic_clock_rollback_stops_output(self):
        self.update()
        self.assertFalse(self.gate.decision(9).heartbeat)
        self.assertFalse(self.gate.decision(9).rates)

    def test_dds_versions_include_only_nonzero_suffixes(self):
        self.assertEqual(px4_feedback_topic("/fmu/out/vehicle_status", SimpleNamespace(MESSAGE_VERSION=4)),
                         "/fmu/out/vehicle_status_v4")
        self.assertEqual(px4_feedback_topic("/fmu/out/vehicle_control_mode", SimpleNamespace()),
                         "/fmu/out/vehicle_control_mode")
        self.assertEqual(px4_feedback_topic("/fmu/out/vehicle_control_mode", SimpleNamespace(MESSAGE_VERSION=0)),
                         "/fmu/out/vehicle_control_mode")


@unittest.skipUnless(controller.rclpy is not None,
                     "source ROS and px4_ros2_ws to test the actual controller publication path")
class ControllerPublicationTests(unittest.TestCase):
    def setUp(self):
        cls = controller.Px4OffboardRatesNode
        # Invoke production callbacks with recording publishers and real PX4
        # message objects. This creates no ROS nodes or network traffic.
        self.node = SimpleNamespace(
            authority=OffboardAuthority(), publish_enabled=True,
            last_authority_status=None, heartbeat_count=0, publish_count=0,
            offboard_publisher=Mock(), rates_publisher=Mock(),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
            get_logger=lambda: Mock(),
        )
        self.node._publish_control = MethodType(cls._publish_control, self.node)
        self.result = SimpleNamespace(rates_frd=(.2, -.4, .1), thrust_body_frd=(0., 0., -.73))

    def update(self, nav, offboard, now=10.):
        self.node.authority.update_status(status(nav), now)
        self.node.authority.update_control_mode(mode(offboard, attitude=not offboard), now)

    def test_takeoff_hold_offboard_land_cycle(self):
        for nav in (17, 4):
            self.update(nav, False)
            rates_ts, heartbeat_ts, result = self.node._publish_control(self.result, 10)
            self.assertIsNone(rates_ts)
            self.assertIsNotNone(heartbeat_ts)
            self.assertEqual(result, "px4_control")
        self.node.rates_publisher.publish.assert_not_called()
        hb = self.node.offboard_publisher.publish.call_args.args[0]
        self.assertTrue(hb.body_rate)
        self.assertFalse(hb.position or hb.velocity or hb.attitude)

        self.update(14, True)
        self.assertEqual(self.node._publish_control(self.result, 10)[2], "published")
        self.node.rates_publisher.publish.assert_called_once()
        msg = self.node.rates_publisher.publish.call_args.args[0]
        self.assertAlmostEqual(msg.roll, .2, places=6)
        self.assertAlmostEqual(msg.thrust_body[2], -.73, places=6)
        self.assertFalse(msg.reset_integral)

        # LAND stops CTBR even while the last control_mode still says Offboard.
        self.node.authority.update_status(status(18), 10.1)
        self.assertIsNone(self.node._publish_control(self.result, 10.1)[0])
        self.node.rates_publisher.publish.assert_called_once()
        self.assertEqual(self.node.heartbeat_count, 4)
        self.assertEqual(self.node.publish_count, 1)

    def test_no_feedback_stale_feedback_and_dry_run_send_nothing(self):
        self.node._publish_control(self.result, 10)
        self.update(14, True)
        self.node._publish_control(self.result, 12)
        self.node.publish_enabled = False
        self.node._publish_control(self.result, 10)
        self.node.offboard_publisher.publish.assert_not_called()
        self.node.rates_publisher.publish.assert_not_called()

    def test_stale_policy_does_not_keep_offboard_alive(self):
        self.update(14, True)
        n = self.node
        n.timer_count = 0
        n.worker = SimpleNamespace(snapshot=lambda: (None, None, 1), submit=lambda *args: True)
        n._update_statistics = Mock()
        n.latest_odometry = SimpleNamespace(position=(0., 0., -5.), q=(1., 0., 0., 0.),
                                             velocity=(0., 0., 0.), angular_velocity=(0., 0., 0.))
        n.latest_odometry_time = 10.
        n.max_odometry_age = n.max_action_age = .2
        n.physical_hover = n.nn_goal = (0., 0., 5.)
        n.physical_yaw_reference = n.nn_yaw_goal = 0.
        n._emit_trace, n._log_error_throttled = Mock(), Mock()
        from unittest.mock import patch
        with patch.object(controller.time, "monotonic", return_value=10.):
            controller.Px4OffboardRatesNode.control_callback(n)
        n.offboard_publisher.publish.assert_not_called()
        n.rates_publisher.publish.assert_not_called()
        self.assertEqual(n._emit_trace.call_args.args[2], "no_fresh_action")


if __name__ == "__main__":
    unittest.main(verbosity=2)

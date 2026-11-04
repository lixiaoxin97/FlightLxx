import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
#!/usr/bin/env python3
"""Mode-transition regressions. No DDS traffic or vehicle commands are sent."""

from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

from offboard_authority import OffboardAuthority, px4_feedback_topic


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



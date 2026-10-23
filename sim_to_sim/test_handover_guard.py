#!/usr/bin/env python3
"""Offline regressions for the initial policy handover guard."""

import math
import unittest

from handover_guard import evaluate_handover_candidate


class HandoverGuardTests(unittest.TestCase):
    def setUp(self):
        self.goal = (0.0, 0.0, 5.0)
        self.observation = (0.01, -0.01, 5.01, 0.1, 0.01, -0.02,
                           0.01, 0.01, -0.01, 0.0, 0.0, 0.0)
        self.action = (0.01, -0.02, 0.01, 0.01)

    def test_safe_hover_action_passes(self):
        safe, reasons = evaluate_handover_candidate(
            self.observation, self.action, self.goal, 0.1
        )
        self.assertTrue(safe)
        self.assertEqual(reasons, ())

    def test_equivalent_eigen_euler_flip_is_rejected(self):
        bad_obs = (0.0, 0.0, 5.0, math.pi, math.pi, -math.pi,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        bad_action = (0.999, 0.982, -0.71, -0.57)
        safe, reasons = evaluate_handover_candidate(
            bad_obs, bad_action, self.goal, 0.1
        )
        self.assertFalse(safe)
        self.assertIn("tilt", reasons)
        self.assertIn("yaw_branch_or_error", reasons)
        self.assertIn("rate_action", reasons)
        self.assertIn("collective_action", reasons)

    def test_state_and_each_control_axis_are_checked(self):
        cases = (
            (dict(observation=(0.5, 0, 5, .1, 0, 0, 0, 0, 0, 0, 0, 0)), "position_error"),
            (dict(observation=(0, 0, 5, .1, 0, 0, .31, 0, 0, 0, 0, 0)), "speed"),
            (dict(observation=(0, 0, 5, .1, 0, 0, 0, 0, 0, .51, 0, 0)), "body_rate"),
            (dict(observation=(0, 0, 5, .1, .36, 0, 0, 0, 0, 0, 0, 0)), "tilt"),
            (dict(raw_action=(.26, 0, 0, 0)), "rate_action"),
            (dict(raw_action=(0, 0, 0, .41)), "collective_action"),
        )
        for overrides, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                args = dict(observation=self.observation, raw_action=self.action,
                            nn_goal=self.goal, nn_yaw_goal=.1)
                args.update(overrides)
                safe, reasons = evaluate_handover_candidate(**args)
                self.assertFalse(safe)
                self.assertIn(expected_reason, reasons)

    def test_invalid_and_nonfinite_inputs_fail_closed(self):
        for obs, action in (((0.0,), self.action),
                            (self.observation, (0.0, 0.0, 0.0)),
                            (self.observation[:-1] + (float("nan"),), self.action)):
            with self.subTest(obs=obs, action=action):
                safe, reasons = evaluate_handover_candidate(
                    obs, action, self.goal, .1
                )
                self.assertFalse(safe)
                self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main(verbosity=2)

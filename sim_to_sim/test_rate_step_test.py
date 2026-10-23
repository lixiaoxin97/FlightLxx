#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

from step16_px4_rate_step_test import (
    build_step_sequence,
    fill_heartbeat,
    fill_rate_setpoint,
    step_at_elapsed,
)


class RateStepTestTests(unittest.TestCase):
    def test_sequence_uses_small_symmetric_roll_then_pitch_pulses(self):
        sequence = build_step_sequence(
            amplitude=0.12,
            pulse_duration=0.4,
            recovery_duration=1.2,
            cycles=2,
            initial_hold=2.0,
            final_hold=3.0,
        )
        self.assertEqual(sequence[0], ("initial_hold", 2.0, (0.0, 0.0, 0.0)))
        self.assertEqual(sequence[1], ("roll_1_positive", 0.4, (0.12, 0.0, 0.0)))
        self.assertEqual(sequence[2], ("roll_1_negative", 0.4, (-0.12, 0.0, 0.0)))
        self.assertEqual(sequence[7], ("pitch_1_positive", 0.4, (0.0, 0.12, 0.0)))
        self.assertEqual(sequence[-1], ("final_hold", 3.0, (0.0, 0.0, 0.0)))

    def test_elapsed_time_selects_expected_phase_and_completion_holds_zero(self):
        sequence = build_step_sequence(
            amplitude=0.1, pulse_duration=0.4, recovery_duration=1.0,
            cycles=1, initial_hold=2.0, final_hold=3.0,
        )
        self.assertEqual(step_at_elapsed(sequence, 0.0)[0], "initial_hold")
        self.assertEqual(step_at_elapsed(sequence, 2.0),
                         ("roll_1_positive", (0.1, 0.0, 0.0)))
        self.assertEqual(step_at_elapsed(sequence, 2.41),
                         ("roll_1_negative", (-0.1, 0.0, 0.0)))
        self.assertEqual(step_at_elapsed(sequence, 100.0),
                         ("complete_hold", (0.0, 0.0, 0.0)))

    def test_message_fields_preserve_px4_body_rate_contract(self):
        heartbeat = SimpleNamespace()
        fill_heartbeat(heartbeat, 123)
        self.assertTrue(heartbeat.body_rate)
        self.assertFalse(heartbeat.attitude)
        self.assertFalse(heartbeat.position)

        rates = SimpleNamespace()
        fill_rate_setpoint(rates, 456, (0.12, -0.12, 0.0), (0.0, 0.0, -0.343))
        self.assertEqual((rates.timestamp, rates.roll, rates.pitch, rates.yaw),
                         (456, 0.12, -0.12, 0.0))
        self.assertEqual(rates.thrust_body, [0.0, 0.0, -0.343])
        self.assertFalse(rates.reset_integral)


if __name__ == "__main__":
    unittest.main(verbosity=2)

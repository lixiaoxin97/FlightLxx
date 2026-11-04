import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import math
import unittest
from final_hover_metrics import evaluate


class FinalHoverTests(unittest.TestCase):
    def trace(self,positions=None,times=None,velocities=None,actions=None):
        times=times if times is not None else [i*.02 for i in range(501)]
        n=len(times)
        return evaluate(times,positions if positions is not None else [[0,0,5]]*n,
            velocities if velocities is not None else [[0,0,0]]*n,[[0,0,0]]*n,
            actions=actions if actions is not None else [[0,0,0,0]]*n)

    def test_relapse_before_final_window_does_not_veto(self):
        p=[[.6 if 1.<i*.02<5. else .1,0,5] for i in range(501)]
        result=self.trace(positions=p)
        self.assertTrue(result['relapsed_after_recovery'])
        self.assertTrue(result['final_stable'])

    def test_final_single_good_sample_is_not_enough(self):
        p=[[.5 if i<500 else 0,0,5] for i in range(501)]
        self.assertFalse(self.trace(positions=p)['final_stable'])

    def test_late_relapse_fails(self):
        p=[[.31 if i==475 else 0,0,5] for i in range(501)]
        self.assertFalse(self.trace(positions=p)['final_stable'])

    def test_truncated_tail_never_counts(self):
        r=self.trace(times=[i*.02 for i in range(301)])
        self.assertFalse(r['final_stable']);self.assertIsNone(r['tail_position_error_mean_m'])

    def test_gap_in_tail_invalidates_coverage(self):
        t=[i*.02 for i in range(501) if not 450<=i<=460]
        self.assertFalse(self.trace(times=t)['final_window_complete'])

    def test_early_gap_is_diagnostic_not_a_terminal_veto(self):
        t=[i*.02 for i in range(501) if not 50<=i<=60]
        self.assertTrue(self.trace(times=t)['final_stable'])

    def test_thresholds_are_strict_and_velocity_matters(self):
        self.assertFalse(self.trace(positions=[[.3,0,5]]*501)['final_stable'])
        self.assertFalse(self.trace(velocities=[[.3,0,0]]*501)['final_stable'])

    def test_final_time_is_fixed_not_observed_tail(self):
        r=self.trace(times=[i*.02 for i in range(491)])
        self.assertFalse(r['final_stable'])

    def test_constant_action_is_smooth(self):
        r=self.trace(actions=[[.2,.1,-.3,.15]]*501)
        self.assertEqual(r['whole_action_derivative_rms_per_s_xyz_thrust'],[0.]*4)

    def test_linear_action_derivative_has_known_units(self):
        r=self.trace(actions=[[i*.002,0,0,0] for i in range(501)])
        self.assertAlmostEqual(r['whole_action_derivative_rms_per_s_xyz_thrust'][0],.1)
        self.assertAlmostEqual(r['whole_action_total_variation_per_s_xyz_thrust'][0],.1)

    def test_position_bias_separate_from_jitter(self):
        r=self.trace(positions=[[.1,0,5.05]]*501)
        self.assertAlmostEqual(r['tail_position_bias_xyz_m'][0],.1)
        self.assertAlmostEqual(r['tail_velocity_jitter_m_s'],0.)

    def test_empty_and_nonfinite(self):
        self.assertFalse(evaluate([],[],[],[])['final_stable'])
        with self.assertRaises(ValueError):self.trace(times=[0.,math.nan])


if __name__=='__main__':unittest.main()

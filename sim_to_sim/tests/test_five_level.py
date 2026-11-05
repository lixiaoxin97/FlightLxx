import math
from pathlib import Path
import sys
import unittest
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from five_level_bank import generate,LEVELS
from full_random_initialization import preparation_spawn
from five_level_study import admissible,PRECISION_BANDS
from runner import schedule


class FiveLevelTests(unittest.TestCase):
    def test_balanced_frozen_bank_and_repeated_goal(self):
        bank=generate()
        self.assertEqual(bank,generate())
        self.assertEqual(len(bank['states']),2500)
        self.assertEqual(Counter(s['level'] for s in bank['states']),{i:500 for i in range(1,6)})
        self.assertEqual(sum(s['category']=='exact_goal' for s in bank['states']),100)
        self.assertEqual(len({tuple(s['state']) for s in bank['states']}),2401)
        for index in range(0,2500,5):
            self.assertEqual({s['level'] for s in bank['states'][index:index+5]},set(range(1,6)))

    def test_bounds_tilt_yaw_and_preparation_all_supported(self):
        for s in generate()['states']:
            c=LEVELS[s['level']-1];state=s['state']
            preparation_spawn(state)
            self.assertTrue(all(abs(v)<=c['position']+1e-12 for v in (state[0],state[1],state[2]-5)))
            self.assertTrue(all(abs(v)<=c['velocity'] for v in state[6:9]))
            self.assertTrue(all(abs(v)<=c['rate'] for v in state[9:]))
            if s['category']!='exact_goal':
                self.assertLessEqual(c['tilt_deg'][0],s['requested_body_z_tilt_deg']+1e-8)
                self.assertLessEqual(s['requested_body_z_tilt_deg'],c['tilt_deg'][1]+1e-8)
            self.assertLessEqual(abs(s['requested_yaw_error_deg']),c['yaw_deg'])

    def test_identical_initial_state_for_three_policies(self):
        plan=schedule(generate())
        self.assertEqual(len(plan),7500)
        for index in range(0,len(plan),3):
            group=plan[index:index+3]
            self.assertEqual({s['model'] for s in group},{'CTBR','CTBR+DR','CTBR+ID'})
            self.assertEqual(len({tuple(s['state']) for s in group}),1)

    def test_no_retry_for_valid_policy_failure(self):
        row=dict(valid_policy_trial=True,initialization_valid=True,effective_pid_verified=True,
            watchdog_timeout=False,failure_category='policy_recovery_failure',success=False)
        self.assertTrue(admissible(row))
        for key,value in (('valid_policy_trial',False),('initialization_valid',False),
            ('effective_pid_verified',False),('watchdog_timeout',True),('failure_category','communication_mode_interruption')):
            self.assertFalse(admissible(dict(row,**{key:value})))

    def test_pilot_seed_independent_and_precision_frozen(self):
        pilot=generate(2026110402,10,'five_level_50_pilot')
        formal=generate(2026110401,500,'five_level_2500_formal')
        p={tuple(s['state']) for s in pilot['states'] if s['category']!='exact_goal'}
        f={tuple(s['state']) for s in formal['states'] if s['category']!='exact_goal'}
        self.assertFalse(p&f)
        self.assertEqual(PRECISION_BANDS,(('basic',.3,.3),('precise',.1,.1),('high_precision',.05,.1)))


if __name__=='__main__':unittest.main()

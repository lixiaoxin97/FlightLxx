"""Ensure infrastructure makeups cannot turn policy failures into selected wins."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import five_level_study as study
from five_level_bank import generate
from result_io import save


class MakeupSelectionTests(unittest.TestCase):
    def test_same_state_makeup_and_first_valid_failure_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'sim_experiment',prefix='.makeup_test_') as folder:
            folder=Path(folder);bank=generate(123,5,'unit')
            original=next(s for s in bank['states'] if s['category']=='exact_goal')
            bank=dict(bank,states=[original],count=1)
            bank_path=folder/'bank.json';save(bank_path,bank)
            calls=[]
            def fake_run(output,px4,spec):
                calls.append(spec)
                failed_init=spec['model']=='CTBR' and spec['attempt_number']==1
                valid=not failed_init
                # CTBR's first valid recovery fails. A third try would succeed,
                # so any outcome-driven retry would be detected by the assertions.
                success=valid and spec['model']!='CTBR'
                row=dict(trial_id=spec['trial_id'],model=spec['model'],state_id=spec['state_id'],
                    spec=spec,requested_state=spec['state'],valid_policy_trial=valid,
                    initialization_valid=valid,effective_pid_verified=valid,
                    watchdog_timeout=False,success=success,metrics={},truth_metrics={},
                    failure_category='initialization_failure' if failed_init else None if success else 'policy_recovery_failure',
                    termination_reason='trajectory_release_window_missed' if failed_init else 'flight_completed')
                save(output/'trials'/spec['trial_id']/'completed.json',row)
                return row
            output=folder/'result'
            with patch.object(study,'run_one',side_effect=fake_run),patch.object(study,'frozen_files',return_value=[bank_path]):
                study.run_campaign(bank_path,output,workers=2,max_attempts=4)
                # Resume must return the same accepted results without extra flights.
                study.run_campaign(bank_path,output,workers=2,max_attempts=4,resume=True)
            manifest,rows=study.collect(output)
            self.assertEqual(Counter(s['model'] for s in calls),{'CTBR':2,'CTBR+DR':1,'CTBR+ID':1})
            self.assertTrue(all(s['state']==original['state'] for s in calls))
            self.assertEqual(len(rows),3)
            baseline=next(r for r in rows if r['model']=='CTBR')
            self.assertFalse(baseline['success'])
            self.assertEqual(baseline['accepted_attempt_number'],2)
            self.assertEqual(len(list((output/'attempts/trials').glob('*/completed.json'))),4)
            self.assertTrue((output/'completion_audit.json').is_file())


if __name__=='__main__':unittest.main()

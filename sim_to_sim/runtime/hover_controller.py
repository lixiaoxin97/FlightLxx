#!/usr/bin/env python3
"""Same flight controller and initialization; terminal 2s success endpoint."""
from pathlib import Path
from recovery_controller import RecoveryTrial, parse_args, run_trial
from analyze_final_hover import analyze
from final_hover_metrics import PROTOCOL
import os


class FinalHoverTrial(RecoveryTrial):
    def __init__(self,args):
        super().__init__(args)
        self.report['final_hover_protocol']=PROTOCOL

    def completion_outcome(self):
        # Tick trace has already been written, but trial_result is not closed yet.
        # Reuse the same analyzer as offline evaluation via an explicit snapshot.
        import json
        self.records.flush()
        path=self.args.output/'trial_result.json'
        snapshot=dict(self.report,reason='flight_completed')
        path.write_text(json.dumps(snapshot,indent=2))
        result=analyze(self.args.output,write=False)
        self.report['final_hover_metrics']=result['metrics']
        self.report['truth_final_hover_metrics']=result['truth_metrics']
        return bool(result['success']),'flight_completed'

    def finish(self,passed,reason):
        super().finish(passed,reason)
        if os.environ.get('S2S_BATCH_DISCARD_SIMULATOR')=='1':
            # Only the private recovery SITL launcher sets this. The scored
            # flight already ended; close this disposable simulator instead
            # of spending wall time on an unscored native landing.
            self.report['postflight_cleanup']='discard_private_sitl_after_scored_flight'
            self.done=True


if __name__=='__main__':raise SystemExit(run_trial(parse_args(),FinalHoverTrial))

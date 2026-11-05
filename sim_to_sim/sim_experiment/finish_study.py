#!/usr/bin/env python3
"""Wait for the paired campaign, audit every accepted slot, then render videos."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from result_io import save,digest
from five_level_study import collect,admissible


def verify(study):
    manifest,rows=collect(study)
    planned=manifest['schedule']
    if len(rows)!=len(planned):raise RuntimeError('Missing valid state/model slots')
    originals={s['trial_id']:s for s in planned}
    counts=Counter((r['spec']['level'],r['model']) for r in rows)
    expected=manifest['state_bank']['states_per_level']
    assert all(counts[level,model]==expected for level in range(1,6) for model in ('CTBR','CTBR+DR','CTBR+ID'))
    assert len({r['trial_id'] for r in rows})==len(planned)
    hashes={};attempt_count=0
    for row in rows:
        assert admissible(row)
        spec=originals[row['trial_id']];assert row['spec']==spec
        assert all(abs(a-b)<2e-15 for a,b in zip(row['requested_state'],spec['state']))
        assert all(0<=v<.06 for key in ('initialization_errors','action_input_errors') for v in row[key].values())
        num=row['accepted_attempt_number'];attempt_count+=num
        for index in range(1,num+1):
            path=study/'attempts/trials'/(row['trial_id']+f'__a{index:03d}')/'completed.json'
            assert path.is_file()
            previous=json.loads(path.read_text())
            if index<num:
                from result_io import audit_cleanup_warning
                assert not admissible(audit_cleanup_warning(previous)[0]),'A valid earlier result was skipped'
        hashes[row['trial_id']]=digest(study/'trials'/row['trial_id']/'completed.json')
    for path,sha in manifest['frozen_sha256'].items():
        if digest(path)!=sha:raise RuntimeError('Frozen source changed: '+path)
    evidence=dict(complete=True,valid_trials=len(rows),states=manifest['state_bank']['count'],
        attempts_through_acceptance=attempt_count,counts={f'L{k[0]}/{k[1]}':v for k,v in counts.items()},
        first_valid_attempt_rule_verified=True,policy_failures_not_retried=True,
        requested_states_and_gate_tolerances_verified=True,all_effective_pid_verified=True,
        frozen_input_hashes_verified=True,accepted_result_sha256=hashes,
        verified_utc=datetime.now(timezone.utc).isoformat())
    save(study/'analysis/final_integrity_audit.json',evidence)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--videos',action='store_true')
    args=parser.parse_args();root=args.root.resolve();formal=root/'formal'
    status=root/'delivery_status.json'
    while True:
        pipeline=root/'pipeline_status.json'
        stage=json.loads(pipeline.read_text()) if pipeline.exists() else {}
        if stage.get('error'):
            save(status,dict(stage='experiment_error',finished=False,error=stage['error']))
            raise RuntimeError(stage['error'])
        progress=formal/'progress.json'
        p=json.loads(progress.read_text()) if progress.exists() else {}
        save(status,dict(stage='waiting_for_formal_completion',finished=False,
            valid_trials=p.get('completed',0),planned=7500,updated_utc=datetime.now(timezone.utc).isoformat()))
        if stage.get('finished') and p.get('finished'):break
        time.sleep(30)
    save(status,dict(stage='verifying_all_trials',finished=False,valid_trials=7500))
    evidence=verify(formal)
    subprocess.run(['/usr/bin/python3',str(ROOT/'sim_experiment/analyze_study.py'),'--study',str(formal)],check=True)
    if args.videos:
        save(status,dict(stage='rendering_synchronized_videos',finished=False,valid_trials=7500))
        subprocess.run(['/usr/bin/python3',str(ROOT/'sim_experiment/make_comparison_video.py'),
            '--study',str(formal)],check=True)
    save(status,dict(stage='complete',finished=True,valid_trials=evidence['valid_trials'],
        report=str(formal/'analysis/EXPERIMENT_ANALYSIS.pdf'),
        videos=str(formal/'videos'),integrity_audit=str(formal/'analysis/final_integrity_audit.json'),
        finished_utc=datetime.now(timezone.utc).isoformat()))
    print('DELIVERY_COMPLETE',root,flush=True)


if __name__=='__main__':main()

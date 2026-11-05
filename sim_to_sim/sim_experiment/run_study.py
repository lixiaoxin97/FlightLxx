#!/usr/bin/env python3
"""Run the frozen five-level 2500-state comparison, with infrastructure makeups."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from result_io import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('generate','run','status','report','pipeline'))
    parser.add_argument('--bank',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-attempts',type=int,default=50)
    args=parser.parse_args()
    if args.action=='generate':
        from five_level_bank import generate
        for count,seed,name in ((500,2026110401,'five_level_2500_formal'),(10,2026110402,'five_level_50_pilot')):
            path=ROOT/'states'/(name+'.json');bank=generate(seed,count,name)
            if path.exists():
                if json.loads(path.read_text())!=bank:raise RuntimeError('Existing frozen bank differs')
            else:save(path,bank)
            print(path)
        return
    if args.action=='pipeline':
        if args.output is None:parser.error('--output is required')
        root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
        state=root/'pipeline_status.json'
        from five_level_study import run_campaign,collect,admissible
        from five_level_report import build_report
        for stage,bank in (('pilot','five_level_50_pilot.json'),('formal','five_level_2500_formal.json')):
            output=root/stage
            save(state,dict(stage=stage,finished=False,updated_utc=datetime.now(timezone.utc).isoformat()))
            try:
                run_campaign(ROOT/'states'/bank,output,workers=args.workers,
                    max_attempts=args.max_attempts,resume=(output/'manifest.json').exists())
                manifest,rows=collect(output)
                assert len(rows)==len(manifest['schedule']) and all(admissible(r) for r in rows)
                if stage=='pilot':
                    # Pilot pass concerns workflow completion, never a desired policy ranking.
                    save(root/'pilot_validation.json',dict(passed=True,valid_trials=len(rows),
                        all_levels_and_models_present=True,all_effective_pid_verified=True,
                        ranking_used_as_gate=False,bank=str(ROOT/'states'/bank)))
                build_report(output)
            except BaseException as exc:
                save(state,dict(stage=stage,finished=False,error=repr(exc),updated_utc=datetime.now(timezone.utc).isoformat()))
                raise
        save(state,dict(stage='complete',finished=True,formal_valid_trials=7500,
            report=str(root/'formal/analysis/FIVE_LEVEL_REPORT.pdf'),updated_utc=datetime.now(timezone.utc).isoformat()))
        print('STUDY_COMPLETE',root,flush=True)
        return
    if args.output is None:parser.error('--output is required')
    if args.action=='status':
        path=args.output/'progress.json'
        print(path.read_text() if path.exists() else 'No progress file yet')
    elif args.action=='report':
        from five_level_report import build_report
        build_report(args.output)
    else:
        if args.bank is None:parser.error('--bank is required')
        from five_level_study import run_campaign
        run_campaign(args.bank,args.output,args.workers,args.max_attempts,args.resume)


if __name__=='__main__':main()

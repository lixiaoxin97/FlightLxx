"""Paired five-level trials with immutable infrastructure attempts and makeups."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import threading
import time

from final_hover_metrics import PROTOCOL
from fixed_pid import PROFILE, PROFILES
from result_io import save, digest, audit_cleanup_warning
from runner import schedule, frozen_files, hashes, run_one

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MODELS=('CTBR','CTBR+DR','CTBR+ID')
PRECISION_BANDS=(('basic',.3,.3),('precise',.1,.1),('high_precision',.05,.1))


def admissible(result):
    """A policy failure ends a slot; only infrastructure failures get makeups."""
    return bool(result.get('valid_policy_trial') and result.get('initialization_valid')
        and result.get('effective_pid_verified') and not result.get('watchdog_timeout')
        and result.get('failure_category') in (None,'policy_recovery_failure'))


def collect(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    rows=[]
    for spec in manifest['schedule']:
        path=root/'trials'/spec['trial_id']/'completed.json'
        if path.exists():rows.append(json.loads(path.read_text()))
    return manifest,rows


def attempt_paths(root,logical_id):
    return sorted((Path(root)/'attempts/trials').glob(logical_id+'__a[0-9][0-9][0-9]/completed.json'))


def run_campaign(bank_path,output,workers=12,max_attempts=50,resume=False,progress_callback=None):
    bank_path=Path(bank_path).resolve();output=Path(output).resolve()
    if not 1<=workers<=12:raise ValueError('workers must be 1..12')
    if output.is_relative_to(Path('/tmp')):raise ValueError('Use sim_to_sim/logs, outside /tmp')
    if output.exists() and not resume:raise ValueError('Use a fresh directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.study.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bank=json.loads(bank_path.read_text());plan=schedule(bank)
    assert len(plan)==3*bank['count']
    px4=Path(os.environ.get('S2S_PX4_DIR',Path.home()/'PX4-Autopilot')).resolve()
    check=json.loads((ROOT/'config/policy_equivalence.json').read_text())
    for model in MODELS:
        if digest(ROOT.parent/'train/models'/(model+'.zip'))!=check[model]['checkpoint_sha256']:
            raise RuntimeError('Network checkpoint changed: '+model)
    seed=output/'parameter_seed'
    if not seed.exists():
        seed.mkdir()
        source=px4/'build/px4_sitl_default/rootfs/fs'
        for path in source.glob('parameters*.bson'):shutil.copy2(path,seed/path.name)
    if not list(seed.glob('parameters*.bson')):raise RuntimeError('Missing parameter seed')
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['schedule']!=plan:raise RuntimeError('State bank or paired schedule changed')
        if hashes([Path(x) for x in manifest['frozen_sha256']])!=manifest['frozen_sha256']:
            raise RuntimeError('Frozen inputs changed; use a new campaign')
    else:
        files=frozen_files(px4,bank_path,seed)
        files += [ROOT/'sim_experiment/run_study.py']
        manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),state_bank=bank,schedule=plan,
            frozen_sha256=hashes(files),protocol=PROTOCOL,profiles=[PROFILE],fixed_pid=PROFILES[PROFILE],
            precision_bands=PRECISION_BANDS,worker_count=workers,max_attempts_per_slot=max_attempts,
            scoring='First admissible attempt on each original state/model slot, including policy failures',
            infrastructure_makeups=True,policy_failure_retries=False,all_attempts_retained=True,
            original_state_replacement=False,automatic_retries=True,
            primary_denominator='500 admissible trials per level/model; 2500 per model',
            secondary_denominator='all first attempts; report all infrastructure overhead and retry counts',
            simulator_speed_factor=1.,numpy_validation=check)
        save(manifest_path,manifest);save(output/'state_bank.json',bank)
        for index,path in enumerate(files):
            target=output/'source_snapshot'/f'{index:03d}_{path.name}'
            target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    attempts=output/'attempts';attempts.mkdir(exist_ok=True)
    if not (attempts/'parameter_seed').exists():(attempts/'parameter_seed').symlink_to(seed)
    small={k:v for k,v in manifest['frozen_sha256'].items() if Path(k).stat().st_size<2_000_000}
    start=time.monotonic();live=set();mutex=threading.Lock()
    state={};accepted={}
    for spec in plan:
        dest=output/'trials'/spec['trial_id']/'completed.json'
        if dest.exists():
            result=json.loads(dest.read_text())
            if not admissible(result):raise RuntimeError('Invalid accepted result: '+str(dest))
            accepted[spec['trial_id']]=result
    all_attempts={}
    for path in (attempts/'trials').glob('*/completed.json'):
        row=json.loads(path.read_text());all_attempts[path.parent.name]=row

    def progress():
        by_level={}
        for level in range(1,6):
            by_level[str(level)]={model:dict(
                completed=sum(r['spec']['level']==level and r['model']==model for r in accepted.values()),
                successes=sum(r['spec']['level']==level and r['model']==model and r['success'] for r in accepted.values()))
                for model in MODELS}
        with mutex:active=sorted(live)
        with mutex:attempt_rows=list(all_attempts.values())
        rejected=[r for r in attempt_rows if not admissible(r)]
        save(output/'progress.json',dict(completed=len(accepted),planned=len(plan),
            attempts=len(attempt_rows),infrastructure_attempts=len(rejected),
            infrastructure_categories=dict(Counter(r.get('failure_category','unknown') for r in rejected)),
            first_attempts=sum(r.get('spec',{}).get('attempt_number')==1 for r in attempt_rows),
            successes=dict(Counter(r['model'] for r in accepted.values() if r['success'])),
            by_level=by_level,running=active,workers=workers,finished=len(accepted)==len(plan),
            wall_s_this_invocation=time.monotonic()-start,updated_utc=datetime.now(timezone.utc).isoformat()))

    def invoke(logical):
        if hashes([Path(x) for x in small])!=small:raise RuntimeError('Frozen inputs changed during experiment')
        key=logical['trial_id']
        with mutex:live.add(key)
        try:
            # Existing attempts are read in order; never pick the best outcome.
            for attempt_number in range(1,max_attempts+1):
                spec=dict(logical,trial_id=key+f'__a{attempt_number:03d}',
                    logical_trial_id=key,attempt_number=attempt_number)
                path=attempts/'trials'/spec['trial_id']/'completed.json'
                if path.exists():row=json.loads(path.read_text())
                else:
                    try:row=run_one(attempts,px4,spec)
                    except RuntimeError:
                        if not path.exists():raise
                        row=json.loads(path.read_text())
                audited,correction=audit_cleanup_warning(row)
                if correction:save(path.parent/'classification_audit.json',correction)
                with mutex:all_attempts[spec['trial_id']]=audited
                if admissible(audited):
                    chosen=dict(audited,trial_id=key,spec=logical,attempt_spec=spec,
                        accepted_attempt_number=attempt_number,accepted_attempt_file=str(path),
                        infrastructure_makeups_before_acceptance=attempt_number-1)
                    save(output/'trials'/key/'completed.json',chosen)
                    return chosen
                # Return to the shared queue after one new infrastructure
                # failure so a difficult state does not monopolize a worker.
                if spec['trial_id'] not in state:
                    state[spec['trial_id']]=True
                    return None
            raise RuntimeError('Persistent infrastructure failure after '+str(max_attempts)+' attempts on '+key)
        finally:
            with mutex:live.remove(key)

    pending=[spec for spec in plan if spec['trial_id'] not in accepted]
    from collections import deque
    queue=deque(pending);progress()
    completed_futures=0;last_print=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={}
            def refill():
                while queue and len(futures)<workers:
                    spec=queue.popleft();futures[pool.submit(invoke,spec)]=spec
            refill()
            while futures:
                finished,_=wait(futures,timeout=20,return_when=FIRST_COMPLETED)
                for future in finished:
                    logical=futures.pop(future);row=future.result();completed_futures+=1
                    if row is None:queue.append(logical)
                    else:accepted[logical['trial_id']]=row
                progress()
                if finished and (time.monotonic()-last_print>5 or len(accepted)==len(plan)):
                    last_print=time.monotonic()
                    print(f'VALID {len(accepted)}/{len(plan)} | attempts {len(all_attempts)} | '
                        f'successes '+str(dict(Counter(r['model'] for r in accepted.values() if r['success']))),flush=True)
                    if progress_callback:progress_callback(output)
                refill()
        if len(accepted)!=len(plan):raise RuntimeError('Campaign ended with missing valid slots')
        if hashes([Path(x) for x in manifest['frozen_sha256']])!=manifest['frozen_sha256']:
            raise RuntimeError('Frozen sources changed during campaign')
        save(output/'results.json',[accepted[spec['trial_id']] for spec in plan])
        save(output/'completion_audit.json',dict(complete=True,planned=len(plan),accepted=len(accepted),
            attempts=len(all_attempts),all_accepted_valid=True,all_failures_retained=True,
            fixed_inputs_verified=True,finished_utc=datetime.now(timezone.utc).isoformat()))
    finally:
        progress();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return output

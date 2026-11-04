"""Small result IO helpers; never rewrite preserved trial records."""
import copy
import hashlib
import json
from pathlib import Path

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_flight(trial):
    path = Path(trial["run_directory"]) / "trial_trace.jsonl"
    rows = []
    if not path.exists():
        return rows
    release = (trial.get("release_physics_sample") or {}).get("sim_time_s")
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "tick" or row.get("phase") != "flight":
            continue
        truth = row["truth"]
        if truth["holding"]:
            continue
        if release is None:
            release = truth["released_sim_time_s"]
        row["elapsed_s"] = truth["sim_time_s"] - release
        if -1e-7 <= row["elapsed_s"] <= 10.04:
            if not rows or row["elapsed_s"] > rows[-1]["elapsed_s"] + 1e-9:
                rows.append(row)
    return rows
def audit_cleanup_warning(row):
    """Reclassify only a proved postflight destructor artifact; retain failure.

    The frozen analyzer ignores this benign warning only for successful
    endpoints. A failed full flight can produce the same teardown artifact.
    Keep raw records and never turn an unstable endpoint into a success.
    """
    if not (row['failure_category']=='communication_mode_interruption'
            and row['termination_reason']=='flight_completed'
            and row['initialization_valid'] and row.get('effective_pid_verified')
            and row.get('baseline_restored') and row['metrics'].get('horizon_complete')
            and row['truth_metrics'].get('horizon_complete')
            and not row['metrics'].get('final_stable') and not row['truth_metrics'].get('final_stable')):
        return row, None
    if row['infrastructure_evidence'] != [dict(process_log='vision_bridge.txt', reason='unhandled_exception')]:
        return row, None
    run = Path(row['run_directory'])
    health = run/'process_health.json'
    log = (run/'vision_bridge.txt').read_text()
    marker = 'Exception ignored in: <function Future.__del__'
    if not (health.exists() and not json.loads(health.read_text()).get('unexpected_exits')
            and log.count('Traceback (most recent call last):')==1 and marker in log
            and "AttributeError: 'Task' object has no attribute '_exception'" in log
            and log.index(marker)<log.index('Traceback (most recent call last):')):
        return row, None
    ticks = read_flight(row)
    if not (len(ticks)>=490 and all(x['ctbr_published'] and x['authority']=='offboard_control' for x in ticks)):
        return row, None
    correction = dict(trial_id=row['trial_id'], original_category=row['failure_category'],
                      audited_category='policy_recovery_failure', success_unchanged=True,
                      original_success=False, audited_success=False,
                      evidence='Full 10s fresh Offboard flight; failed EKF+truth endpoint; no unexpected process exits; sole exception is ignored Future.__del__ in teardown',
                      flight_ticks=len(ticks))
    audited = copy.deepcopy(row)
    audited.update(failure_category='policy_recovery_failure', valid_policy_trial=True)
    return audited, correction

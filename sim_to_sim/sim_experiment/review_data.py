"""Additional analysis for the completed campaign; never mutate trial outcomes."""
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
import hashlib
import json
import math
import statistics

from five_level_study import MODELS
from result_io import save,digest
from five_level_report import describe,write_csv

KEYS=('final_stabilization_entry_s','peak_position_error_m','whole_position_iae_m_s',
    'tail_position_error_mean_m','tail_velocity_jitter_m_s','tail_body_rate_jitter_rad_s','action_saturation_fraction')


def extra_analysis(study,manifest,rows):
    study=Path(study);out=study/'analysis';root=study.parents[2]
    audit=json.loads((out/'final_integrity_audit.json').read_text())
    assert len(rows)==7500 and len(manifest['state_bank']['states'])==2500
    assert Counter((r['spec']['level'],r['model']) for r in rows)==Counter({(l,m):500 for l in range(1,6) for m in MODELS})
    for row in rows:
        path=study/'trials'/row['trial_id']/'completed.json'
        assert digest(path)==audit['accepted_result_sha256'][row['trial_id']]
    for path,sha in manifest['frozen_sha256'].items():assert digest(path)==sha,path
    states=manifest['state_bank']['states']
    assert len({tuple(s['state']) for s in states})==2401
    records={r['state_id']+':'+r['model']:r for r in rows}
    common=[s['id'] for s in states if all(records[s['id']+':'+m]['success'] for m in MODELS)]
    dynamic=[s for s in common if records[s+':CTBR']['spec']['level']>=2]
    nonrepeat=[r for r in rows if r['spec']['category']!='exact_goal']
    def subset_table(name,selected):
        return {m:dict(n=sum(r['model']==m for r in selected),
            successes=sum(r['model']==m and r['success'] for r in selected),
            rate=sum(r['model']==m and r['success'] for r in selected)/sum(r['model']==m for r in selected)) for m in MODELS}
    sensitivity=dict(all= subset_table('all',rows),without_goal_repeats=subset_table('random',nonrepeat),
        L2_L5=subset_table('dynamic',[r for r in rows if r['spec']['level']>=2]))
    write_csv(out/'population_sensitivity.csv',[dict(subset=key,model=m,**group) for key,value in sensitivity.items() for m,group in value.items()])
    dynamic_quality={m:{key:describe([records[s+':'+m]['metrics'][key] for s in dynamic]) for key in KEYS} for m in MODELS}
    stable=[]
    for name,selected in (('all',rows),('L2_L5',[r for r in rows if r['spec']['level']>=2])):
        for m in MODELS:
            own=[r for r in selected if r['model']==m]
            for t in (1,2,3,4,5,6,8):
                k=sum(r['success'] and r['metrics']['final_stabilization_entry_s']<=t for r in own)
                stable.append(dict(subset=name,model=m,time_s=t,stable_count=k,n=len(own),fraction=k/len(own)))
    write_csv(out/'stable_by_time.csv',stable)
    gate={};source_gate={};truth=[];bias=[];failure=[];actions=[]
    for m in MODELS:
        group=[r for r in rows if r['model']==m]
        gate[m]={k:max(r['initialization_errors'][k] for r in group) for k in group[0]['initialization_errors']}
        source_gate[m]={k:max(r['action_input_errors'][k] for r in group) for k in group[0]['action_input_errors']}
        success=[r for r in group if r['success']]
        truth.append(dict(model=m,n=len(group),px4_successes=len(success),
            px4_successes_also_truth_success=sum(bool(r['truth_final_success']) for r in group),
            independent_truth_final_stable=sum(bool(r['truth_metrics'].get('final_stable')) and r['termination_reason']=='flight_completed' for r in group),
            endpoint_disagreements=sum(bool(r['endpoint_disagreement']) for r in group),
            max_input_age_ms=max(r['metrics']['policy_input_age_max_ms'] for r in group),
            max_sample_gap_s=max(r['metrics']['maximum_sample_gap_s'] for r in group)))
        for subset,selected in (('own_success',success),('shared_success',[records[s+':'+m] for s in common])):
            for axis in range(3):bias.append(dict(model=m,subset=subset,axis='xyz'[axis],n=len(selected),
                mean_bias_m=statistics.mean(r['metrics']['tail_position_bias_xyz_m'][axis] for r in selected)))
        for level in range(1,6):
            subset=[r for r in group if r['spec']['level']==level]
            failures=Counter(r['termination_reason'] for r in subset if not r['success'])
            failure.append(dict(model=m,level=level,n=len(subset),successes=sum(r['success'] for r in subset),
                unfinished_stability=failures['flight_completed'],envelope_termination=failures['flight_envelope_exceeded']))
        for scope in ('whole_action_derivative_rms_per_s_xyz_thrust','tail_action_derivative_rms_per_s_xyz_thrust'):
            for channel in range(4):
                values=[records[s+':'+m]['metrics'][scope][channel] for s in common]
                actions.append(dict(model=m,scope=scope,channel=('roll','pitch','yaw','thrust')[channel],**describe(values)))
    write_csv(out/'truth_and_timing_audit.csv',truth)
    write_csv(out/'successful_position_bias.csv',bias)
    write_csv(out/'policy_failure_types.csv',failure)
    write_csv(out/'matched_action_smoothness.csv',actions)
    progress=json.loads((study/'progress.json').read_text())
    completion=json.loads((study/'completion_audit.json').read_text())
    local=timezone(timedelta(hours=8))
    started=datetime.fromisoformat(manifest['created_utc']).astimezone(local)
    ended=datetime.fromisoformat(completion['finished_utc']).astimezone(local)
    raw_recheck=out/'raw_trace_recheck.json'
    recheck=json.loads(raw_recheck.read_text()) if raw_recheck.exists() else None
    data=dict(report_date='2026-11-05 Asia/Shanghai',formal_valid=7500,attempts=progress['attempts'],
        initialization_failures=114,communication_failures=5,
        verified_hash_count=len(rows),frozen_input_files=len(manifest['frozen_sha256']),
        formal_started_local=started.strftime('%Y-%m-%d %H:%M:%S'),
        formal_finished_local=ended.strftime('%Y-%m-%d %H:%M:%S'),
        formal_elapsed_hours=progress['wall_s_this_invocation']/3600,
        common_success_n=len(common),L2_L5_common_success_n=len(dynamic),
        common_success_per_level={str(l):sum(records[s+':CTBR']['spec']['level']==l for s in common) for l in range(1,6)},
        population_sensitivity=sensitivity,stable_by_time=stable,dynamic_matched_quality=dynamic_quality,
        initialization_current_max=gate,initialization_action_source_max=source_gate,
        truth_and_timing=truth,position_bias=bias,policy_failures=failure,matched_actions=actions,
        stratified_raw_trace_rechecks=recheck)
    save(out/'reviewed_statistics.json',data)
    return data

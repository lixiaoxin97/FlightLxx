#!/usr/bin/env python3
"""Evaluate a run under the new terminal endpoint; never replace old results."""
import argparse
import json
import math
from pathlib import Path

from initialization_metrics import initial_errors, infrastructure_evidence, INIT_TOLERANCES
from final_hover_metrics import PROTOCOL, evaluate
from recovery_state import recovery_spawn, quaternion_conjugate, rotate_vector, quaternion_to_zyx, quaternion_multiply


def analyze(directory, spec=None, write=True):
    directory=Path(directory);spec=spec or {};path=directory/'trial_result.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    requested=report.get('requested_state',spec.get('state'))
    expected=recovery_spawn(requested,'principal_zyx')['expected_policy_observation'] if requested else None
    errors=initial_errors(report.get('release_observation'),expected) if expected else None
    source_errors=initial_errors(report.get('release_policy_observation'),expected) if expected else None
    valid_init=bool(report.get('initialization_passed') and errors and source_errors and
        all(errors[k]<v and source_errors[k]<v for k,v in INIT_TOLERANCES.items()))
    reason=report.get('reason','missing_trial_report')
    finished=[e for e in report.get('events',[]) if e.get('event')=='finished']
    if reason=='native_landing_timeout' and finished:reason=finished[-1]['reason']
    rows=[];release=report.get('release_sim_time_s');trace=directory/'trial_trace.jsonl'
    if trace.exists():
        for line in trace.open():
            r=json.loads(line)
            if r.get('event')!='tick' or r.get('phase')!='flight' or r['truth']['holding']:continue
            if release is None:release=r['truth'].get('released_sim_time_s')
            if release is None:continue
            t=r['truth']['sim_time_s']-release
            if -1e-7<=t<=PROTOCOL['duration_s']+.04:
                t=min(PROTOCOL['duration_s'],max(0.,t))
                if not rows or t>rows[-1][0]+1e-9:rows.append((t,r))
    metrics={};truth_metrics={}
    if rows:
        times=[t for t,r in rows];obs=[r['observation'] for t,r in rows]
        actions=[r['raw_action'] for t,r in rows];cmd=[[r['rates_frd'][0],-r['rates_frd'][1],-r['rates_frd'][2]] for t,r in rows]
        common=dict(actions=actions,commanded_rates=cmd,physical_saturation=[r.get('physical_thrust_saturated',False) for t,r in rows])
        metrics=evaluate(times,[x[:3] for x in obs],[x[6:9] for x in obs],[x[9:12] for x in obs],
                         attitudes=[[x[5],x[4],x[3]] for x in obs],**common)
        p=[];v=[];w=[];angles=[]
        for t,r in rows:
            g=r['truth'];enu=g['position_enu'];vel=g['velocity_enu'];q=g['quaternion_enu_flu_wxyz']
            p.append([enu[1],-enu[0],enu[2]]);v.append([vel[1],-vel[0],vel[2]])
            w.append(rotate_vector(quaternion_conjugate(q),g['angular_velocity_world_enu']))
            angles.append(quaternion_to_zyx(quaternion_multiply((math.sqrt(.5),0.,0.,-math.sqrt(.5)),q)))
        truth_metrics=evaluate(times,p,v,w,attitudes=angles,**common)
        metrics['maximum_ekf_truth_position_error_m']=max(math.dist(o[:3],g) for o,g in zip(obs,p))
        metrics['policy_input_age_max_ms']=max((r.get('policy_input_age_ms') or 0.) for t,r in rows)
    evidence=infrastructure_evidence(directory)
    # An explicitly ignored rclpy Task destructor warning during orderly
    # teardown is not a scored-flight communication outage. Retain the warning
    # but require complete, fresh Offboard flight and matching truth endpoint.
    ignored_cleanup=[]
    vision=directory/'vision_bridge.txt';health=directory/'process_health.json'
    if (report.get('passed') and reason=='flight_completed' and metrics.get('final_stable') and
        truth_metrics.get('final_stable') and report.get('completed_duration_sim_s',0)>=10. and
        len(rows)>=490 and all(r['ctbr_published'] and r['authority']=='offboard_control' for t,r in rows) and
        health.exists() and not json.loads(health.read_text()).get('unexpected_exits') and vision.exists()):
        log=vision.read_text()
        if (log.count('Traceback (most recent call last):')==1 and
            'Exception ignored in: <function Future.__del__' in log and
            "AttributeError: 'Task' object has no attribute '_exception'" in log):
            item=dict(process_log='vision_bridge.txt',reason='unhandled_exception')
            ignored_cleanup=[x for x in evidence if x==item];evidence=[x for x in evidence if x!=item]
    communication=bool(evidence or report.get('failure_category')=='communication_mode_interruption' or
        any(s in reason for s in ('stale_','lost_offboard','workflow_timeout','interrupted','missing_trial_report')))
    if communication:category='communication_mode_interruption'
    elif not valid_init:category='initialization_failure'
    elif reason!='flight_completed' or not metrics.get('final_stable'):category='policy_recovery_failure'
    else:category=None
    result=dict(metrics_version=PROTOCOL['version'],protocol=PROTOCOL,run_directory=str(directory.resolve()),
        trial_id=spec.get('trial_id',report.get('trial_id')),model=spec.get('model',Path(report.get('model','unknown')).stem),
        state_id=spec.get('state_id'),profile=spec.get('profile'),stratum=spec.get('stratum'),
        requested_state=requested,actual_handover_observation=report.get('release_observation'),
        actual_action_input_observation=report.get('release_policy_observation'),
        initialization_errors=errors,action_input_errors=source_errors,initialization_valid=valid_init,
        failure_category=category,valid_policy_trial=valid_init and not communication,
        success=category is None,final_success=category is None,
        truth_final_success=bool(category is None and truth_metrics.get('final_stable')),
        endpoint_disagreement=bool(metrics.get('final_stable')!=truth_metrics.get('final_stable')),
        termination_reason=reason,infrastructure_evidence=evidence,metrics=metrics,truth_metrics=truth_metrics)
    result['ignored_postflight_warnings']=ignored_cleanup
    if write:
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'final_hover_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    print(json.dumps(analyze(parser.parse_args().run),indent=2,ensure_ascii=False))

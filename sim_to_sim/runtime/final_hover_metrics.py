"""Versioned terminal-hover evaluation; prior relapse never vetoes success.

The primary endpoint is complete flight with every observed state in [8,10]s
inside strict position/speed bands. At 50Hz, <=25ms endpoint tolerance is
explicit; gaps >100ms invalidate coverage. Diagnostics use simulation time.
"""
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Simulation_Experiments.recovery_metrics import evaluate_trajectory, weighted_mean, vector_jitter

VERSION = 'terminal_hover_10s_tail2_v1'
PROTOCOL = dict(version=VERSION, duration_s=10., tail_s=2., position_m=.3,
                speed_m_s=.3, max_sample_gap_s=.1, endpoint_tolerance_s=.025,
                primary_state_source='PX4 VehicleOdometry',
                truth_crosscheck='Gazebo ENU converted to fixed NWU and body FLU',
                prior_relapse_veto=False)


def _mean(values, weights):
    return weighted_mean(values, weights)


def _rms(values, weights):
    value=_mean([x*x for x in values],weights)
    return math.sqrt(max(0.,value)) if value is not None else None


def _derivatives(times, vectors):
    return [[(b-a)/(t1-t0) for a,b in zip(x0,x1)]
            for t0,t1,x0,x1 in zip(times,times[1:],vectors,vectors[1:])]


def evaluate(times, positions, velocities, rates, actions=None, attitudes=None,
             commanded_rates=None, physical_saturation=None, protocol=None):
    cfg=dict(PROTOCOL,**(protocol or {}));end=cfg['duration_s'];start=end-cfg['tail_s']
    n=len(times)
    arrays=[positions,velocities,rates]+[a for a in (actions,attitudes,commanded_rates,physical_saturation) if a is not None]
    if any(len(a)!=n for a in arrays):raise ValueError('Trajectory array lengths differ')
    if any(not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Simulation times must be finite and strictly increasing')
    for array in (positions,velocities,rates,actions,attitudes,commanded_rates):
        if array is not None and any(not math.isfinite(v) for row in array for v in row):
            raise ValueError('Trajectory contains nonfinite values')
    if any(t<0 or t>end+1e-8 for t in times):raise ValueError('Clip observations to the declared horizon')
    result=evaluate_trajectory(times,positions,velocities,rates,actions,
        end_time=times[-1] if n else None,expected_duration=end,tail_duration=cfg['tail_s'],
        physical_saturation=physical_saturation,position_threshold=cfg['position_m'],speed_threshold=cfg['speed_m_s'])
    result.update(metrics_version=VERSION,final_stable=False,final_window_complete=False,
                  final_stabilization_entry_s=None,final_stabilization_confirmation_s=None,
                  tail_position_max_m=None,tail_speed_max_m_s=None,quality_window='fixed [duration-tail,duration]')
    if not n:return result
    gaps=[b-a for a,b in zip(times,times[1:])]
    complete=times[-1]>=end-cfg['endpoint_tolerance_s']
    tail_gaps=[b-a for a,b in zip(times,times[1:]) if b>start]
    coverage=complete and times[0]<=start and max(tail_gaps,default=math.inf)<=cfg['max_sample_gap_s']+1e-9
    # Include the state immediately before the boundary (zero-order hold).
    boundary=max((i for i,t in enumerate(times) if t<=start+1e-9),default=0)
    tail_indices=list(range(boundary,n))
    errors=[math.dist(p,(0,0,5)) for p in positions]
    speeds=[math.sqrt(sum(v*v for v in row)) for row in velocities]
    inside=[e<cfg['position_m'] and v<cfg['speed_m_s'] for e,v in zip(errors,speeds)]
    stable=coverage and all(inside[i] for i in tail_indices)
    weights=[b-a for a,b in zip(times,times[1:])]+[max(0.,end-times[-1]) if complete else 0.]
    tail_weights=[max(0.,min(end,t+w)-max(start,t)) for t,w in zip(times,weights)]
    result.update(final_stable=stable,final_window_complete=coverage,horizon_complete=complete,
        tail_complete=coverage,maximum_sample_gap_s=max(gaps,default=None),
        tail_position_max_m=max(errors[i] for i in tail_indices) if coverage else None,
        tail_speed_max_m_s=max(speeds[i] for i in tail_indices) if coverage else None,
        whole_position_iae_m_s=sum(e*w for e,w in zip(errors,weights)),
        whole_position_rms_m=_rms(errors,weights))
    if stable:
        last_bad=max((i for i,ok in enumerate(inside) if not ok),default=-1)
        entry=times[last_bad+1]
        result.update(final_stabilization_entry_s=entry,
                      final_stabilization_confirmation_s=entry+cfg['tail_s'])
    # Never label a partial pre-abort window as terminal quality.
    for key in tuple(result):
        if key.startswith('tail_') and key not in ('tail_complete','tail_window_s'):
            result[key]=None
    if coverage:
        means=[_mean([p[i]-(5. if i==2 else 0.) for p in positions],tail_weights) for i in range(3)]
        result.update(tail_position_bias_xyz_m=means,tail_position_bias_norm_m=math.sqrt(sum(x*x for x in means)),
            tail_position_error_mean_m=_mean(errors,tail_weights),tail_position_error_rms_m=_rms(errors,tail_weights),
            tail_position_max_m=max(errors[i] for i in tail_indices),tail_speed_max_m_s=max(speeds[i] for i in tail_indices),
            tail_speed_rms_m_s=_rms(speeds,tail_weights),tail_velocity_jitter_m_s=vector_jitter(velocities,tail_weights),
            tail_body_rate_jitter_rad_s=vector_jitter(rates,tail_weights),
            tail_body_rate_rms_rad_s=_rms([math.sqrt(sum(v*v for v in r)) for r in rates],tail_weights))
        if attitudes is not None:
            # Attitude is roll,pitch,yaw. Circular centering avoids yaw-wrap artifacts.
            centers=[math.atan2(_mean([math.sin(x[i]) for x in attitudes],tail_weights),
                               _mean([math.cos(x[i]) for x in attitudes],tail_weights)) for i in range(3)]
            delta=[[math.atan2(math.sin(x[i]-centers[i]),math.cos(x[i]-centers[i])) for i in range(3)] for x in attitudes]
            result['tail_attitude_jitter_xyz_rad']=[_rms([x[i] for x in delta],tail_weights) for i in range(3)]
        if commanded_rates is not None:
            err=[math.dist(a,b) for a,b in zip(rates,commanded_rates)]
            result['tail_rate_tracking_error_rms_rad_s']=_rms(err,tail_weights)
    if actions is not None and n>1:
        derivatives=_derivatives(times,actions);intervals=gaps
        valid=[i for i,dt in enumerate(intervals) if dt<=cfg['max_sample_gap_s']+1e-9]
        for name,window in [('whole',0.),('tail',start)]:
            if name=='tail' and not coverage:continue
            w=[max(0.,min(times[i+1],end)-max(times[i],window)) if i in valid else 0. for i in range(n-1)]
            elapsed=sum(w)
            result[name+'_action_derivative_rms_per_s_xyz_thrust']=[_rms([d[j] for d in derivatives],w) for j in range(4)]
            result[name+'_action_total_variation_per_s_xyz_thrust']=[sum(abs(d[j])*dt for d,dt in zip(derivatives,w))/elapsed if elapsed else None for j in range(4)]
        result['action_saturation_fraction_per_axis']=[_mean([float(abs(x[j])>=.98) for x in actions],weights) for j in range(4)]
    # Low-frequency trajectory changes are separated from high-frequency rate
    # power. This spectral diagnostic is descriptive, not a success criterion.
    if coverage and n>1:
        try:
            import numpy as np
            grid=np.arange(start,end+1e-8,.02)
            interp=np.stack([np.interp(grid,times,[r[j] for r in rates]) for j in range(3)],axis=1)
            signal=(interp-interp.mean(axis=0))*np.hanning(len(grid))[:,None]
            power=np.abs(np.fft.rfft(signal,axis=0))**2;power=power.sum(axis=1)
            freq=np.fft.rfftfreq(len(grid),.02);power[0]=0.;total=power.sum()
            result['tail_rate_power_3_10hz_fraction']=float(power[(freq>=3)&(freq<=10)].sum()/total) if total>1e-20 else 0.
            result['tail_rate_dominant_hz']=float(freq[np.argmax(power)]) if total>1e-20 else None
        except ImportError:pass
    return result

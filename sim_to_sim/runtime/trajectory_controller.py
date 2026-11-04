#!/usr/bin/env python3
"""Observe a consistent moving lead-in, then score the same terminal endpoint."""
import math
import os
from recovery_controller import ready_for_release,state_errors,parse_args,run_trial
from hover_controller import FinalHoverTrial
from recovery_state import quaternion_multiply,quaternion_to_zyx,quaternion_conjugate,rotate_vector
from ramp_initialization import initializer_sensor_reference


def telemetry_observation(truth):
    truth=initializer_sensor_reference(truth)
    q=truth['quaternion_enu_flu_wxyz']
    nwu=quaternion_multiply((math.sqrt(.5),0.,0.,-math.sqrt(.5)),q)
    roll,pitch,yaw=quaternion_to_zyx(nwu);p=truth['position_enu'];v=truth['velocity_enu']
    w=rotate_vector(quaternion_conjugate(q),truth['angular_velocity_world_enu'])
    return [p[1],-p[0],p[2],yaw,pitch,roll,v[1],-v[0],v[2],*w]


class TrajectoryInitTrial(FinalHoverTrial):
    def __init__(self,args):
        self.trajectory_matches=0
        super().__init__(args)
        self.report['initialization_protocol']='consistent_moving_lead_in_v1'
        self.report['trajectory_lead_s']=float(os.environ.get('S2S_TRAJECTORY_LEAD_S','.5'))
        self.report['held_sensor_contract']='trajectory-consistent twist for held EV/IMU; raw post-physics state retained; native sensors after release'

    def prepare_operation(self):
        return 'prepare_trajectory',(*self.args.state,self.report['trajectory_lead_s'])

    def preparation_ready(self,good,obs,truth,result,trace,now):
        target=truth.get('trajectory_target_sim_time_s')
        if target is None:return False
        moving=telemetry_observation(truth)
        synchronized=bool(truth['holding'] and truth['prepared'] and result and trace['local_valid'] and
            all(x<.06 for x in state_errors(obs,moving).values()))
        self.trajectory_matches=self.trajectory_matches+1 if synchronized else 0
        remaining=target-truth['sim_time_s']
        if remaining<-.04:
            self.report['trajectory_last_errors']=state_errors(obs,moving)
            self.finish(False,'trajectory_release_window_missed')
            return False
        ready=bool(-.004<=remaining<=.025 and self.trajectory_matches>=3 and good)
        if ready:
            self.report['trajectory_matched_samples']=self.trajectory_matches
            self.report['scheduled_release_sim_time_s']=target
            self.report['trajectory_remaining_at_release_request_s']=remaining
            self.report['trajectory_tracking_errors_at_request']=state_errors(obs,moving)
        return ready

    def preparation_complete(self,obs,result,now):
        # All release gates were just evaluated on this tick. Do not insert a
        # second arm-stage tick while the requested moving state passes by.
        self.release_matched_state(obs,result,now)


if __name__=='__main__':raise SystemExit(run_trial(parse_args(),TrajectoryInitTrial))

"""Kinematically consistent lead-in ending at the requested release state."""
import argparse
import math
from recovery_state import quaternion_from_zyx,quaternion_to_zyx,quaternion_multiply,validate_training_state,rotate_vector


def initializer_sensor_reference(truth):
    """Consistent constrained-motion twist, only before free physics starts.

    Raw post-physics telemetry is retained unchanged by the recorder. During
    a forced pose trajectory, motor forces perturb the post-step twist that
    will be overwritten on the next constrained step. EV and held IMU must
    describe the derivative of the imposed trajectory instead.
    """
    if not (truth.get('holding') and truth.get('prepared') and truth.get('trajectory_target_sim_time_s') is not None):
        return truth
    result=dict(truth)
    result['velocity_enu']=list(truth['prepared_world_velocity_enu'])
    result['angular_velocity_world_enu']=rotate_vector(truth['quaternion_enu_flu_wxyz'],truth['prepared_body_rates_flu'])
    return result


def body_rotation(rates,dt):
    length=math.sqrt(sum(w*w for w in rates))
    if length<1e-12:return (1.,0.,0.,0.)
    a=.5*length*dt;s=math.sin(a)/length
    return (math.cos(a),*(w*s for w in rates))


def at_offset(state,dt):
    q=quaternion_multiply(quaternion_from_zyx(*state[3:6]),body_rotation(state[9:12],dt))
    angles=quaternion_to_zyx(q)
    return [*(p+v*dt for p,v in zip(state[:3],state[6:9])),angles[0],angles[1],angles[2]%(2*math.pi),*state[6:]]


def spawn_state(state,lead=.5):
    result=at_offset(state,-lead);result[6:]=[0.]*6
    validate_training_state(result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',type=float,nargs=12,required=True)
    p.add_argument('--lead',type=float,default=.5);p.add_argument('--shell',action='store_true');args=p.parse_args()
    if not .1<=args.lead<=2.:p.error('Lead must be .1..2 seconds')
    state=spawn_state(validate_training_state(args.state),args.lead)
    # argparse's negative-number detection rejects values like -2e-05 when
    # consumed by the next nargs=12 option. Fixed decimals preserve precision.
    print(' '.join(format(x,'.17f') for x in state))

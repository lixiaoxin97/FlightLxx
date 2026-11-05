"""Outcome-independent, stratified five-level recovery and hover state banks."""
import math
import random

from recovery_state import quaternion_from_zyx, validate_training_state

MODELS = ['CTBR', 'CTBR+DR', 'CTBR+ID']
LEVELS = (
    dict(level=1, name='hover', position=.10, tilt_deg=[0.,10.], yaw_deg=10., velocity=.10, rate=.10),
    dict(level=2, name='mild', position=.25, tilt_deg=[10.,30.], yaw_deg=45., velocity=.25, rate=.25),
    dict(level=3, name='moderate', position=.50, tilt_deg=[30.,60.], yaw_deg=90., velocity=.50, rate=.50),
    dict(level=4, name='large', position=.75, tilt_deg=[60.,120.], yaw_deg=180., velocity=.75, rate=.75),
    dict(level=5, name='inverted', position=1., tilt_deg=[120.,180.], yaw_deg=180., velocity=1., rate=1.),
)


def latin_hypercube(count, dimensions, rng):
    columns=[]
    for _ in range(dimensions):
        values=[(i+rng.random())/count for i in range(count)]
        rng.shuffle(values)
        columns.append(values)
    return list(zip(*columns))


def generate(seed=2026110401, per_level=500, split='five_level_formal'):
    if per_level < 5 or per_level % 5:
        raise ValueError('per_level must be a positive multiple of 5, at least 5')
    rng=random.Random(seed)
    levels=[]
    for cfg in LEVELS:
        count_exact=per_level//5 if cfg['level']==1 else 0
        sampled=per_level-count_exact
        states=[]
        for index in range(count_exact):
            states.append(dict(id=f'{split}_L1_goal_{index+1:04d}',state=[0.,0.,5.]+[0.]*9,
                level=1,stratum='L1_hover',category='exact_goal',repeat_group='exact_goal',
                requested_body_z_tilt_deg=0.,requested_yaw_error_deg=0.,bounds=cfg))
        for index,u in enumerate(latin_hypercube(sampled,12,rng),1):
            p=[cfg['position']*(2*x-1) for x in u[:3]]
            v=[cfg['velocity']*(2*x-1) for x in u[3:6]]
            w=[cfg['rate']*(2*x-1) for x in u[6:9]]
            lo,hi=map(math.radians,cfg['tilt_deg'])
            cos_t=math.cos(lo)+(math.cos(hi)-math.cos(lo))*u[9]
            theta=math.acos(max(-1.,min(1.,cos_t)));phi=2*math.pi*u[10]
            # R31=-sin(pitch), R32=cos(pitch)*sin(roll), R33=cos(theta).
            # Sample the world vertical expressed in body axes, then choose
            # the ZYX yaw independently; this enforces physical tilt exactly.
            pitch=math.asin(max(-1.,min(1.,-math.sin(theta)*math.cos(phi))))
            roll=math.atan2(math.sin(theta)*math.sin(phi),math.cos(theta))
            yaw_error=math.radians(cfg['yaw_deg'])*(2*u[11]-1)
            yaw=yaw_error % (2*math.pi)
            state=list(validate_training_state([p[0],p[1],5+p[2],roll,pitch,yaw,*v,*w]))
            q=quaternion_from_zyx(roll,pitch,yaw)
            measured=math.degrees(math.acos(max(-1.,min(1.,1-2*(q[1]**2+q[2]**2)))))
            assert cfg['tilt_deg'][0]-1e-8<=measured<=cfg['tilt_deg'][1]+1e-8
            states.append(dict(id=f'{split}_L{cfg["level"]}_{index:04d}',state=state,
                level=cfg['level'],stratum=f'L{cfg["level"]}_{cfg["name"]}',
                category='near_goal' if cfg['level']==1 else 'combined_recovery',
                repeat_group=None,requested_body_z_tilt_deg=measured,
                requested_yaw_error_deg=math.degrees(yaw_error),bounds=cfg))
        rng.shuffle(states)
        levels.append(states)
    # Every consecutive five-state block contains all levels, in random order.
    ordered=[]
    for index in range(per_level):
        order=list(range(5));rng.shuffle(order)
        ordered.extend(levels[level][index] for level in order)
    return dict(schema_version=3,split=split,seed=seed,count=5*per_level,states=ordered,
        models=MODELS,levels=list(LEVELS),states_per_level=per_level,
        exact_goal_repeats=per_level//5,distinct_requested_states=5*per_level-per_level//5+1,
        target_position_m=[0.,0.,5.],initializer='trajectory',trajectory_lead_s=.35,
        sampling='Independent randomized Latin hypercube per level: position, velocity, body rates, cos(tilt), vertical azimuth, signed ZYX yaw. No outcome selection.',
        orientation='Physical tilt from body-z and world vertical; cos(tilt) stratified within each band; ZYX yaw independently stratified. Not component-uniform Euler or globally Haar-uniform.',
        position_definition='Errors from [0,0,5]; stored z is absolute height.',
        order='Randomly permuted balanced five-level blocks; model order rotates across six permutations.',
        pilot=split.endswith('pilot'),selection_uses_policy_outcomes=False)

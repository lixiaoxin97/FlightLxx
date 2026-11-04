#!/usr/bin/env python3
"""Unstratified independent component-uniform random states, with no rejection.

Matches the distributions in Simulation_Experiments/disturbances/__init__.py:
position errors +/-1m, Euler roll/pitch +/-pi, yaw 0..2pi, velocities and body
rates +/-1. This explicitly is NOT a Haar-uniform distribution over SO(3).
"""
import argparse
import json
import math
from pathlib import Path
import random

from recovery_state import quaternion_from_zyx, validate_training_state

MODELS = ["CTBR", "CTBR+DR", "CTBR+ID"]


def generate(seed, count=1000, split="full_random_20261030_v1"):
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    states = []
    for index in range(1, count + 1):
        position = [rng.uniform(-1., 1.) for _ in range(3)]
        angles = [rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi), rng.uniform(0., 2*math.pi)]
        velocity = [rng.uniform(-1., 1.) for _ in range(3)]
        rates = [rng.uniform(-1., 1.) for _ in range(3)]
        state = list(validate_training_state([position[0], position[1], 5.+position[2], *angles, *velocity, *rates]))
        w, x, y, z = quaternion_from_zyx(*angles)
        tilt = math.degrees(math.acos(max(-1., min(1., 1.-2*(x*x+y*y)))))
        states.append(dict(id=f"{split}_{index:04d}", state=state, stratum="full_random_uniform",
            requested_body_z_tilt_deg=tilt,
            initially_inside_position_speed_band=(math.sqrt(sum(p*p for p in position))<.3
                and math.sqrt(sum(v*v for v in velocity))<.3)))
    return dict(schema_version=1, count=count, seed=seed, split=split, states=states, models=MODELS,
        initializer="trajectory", trajectory_lead_s=.35, target_position_m=[0., 0., 5.],
        sampling="12 independent scalar uniform draws for each state; no strata, difficulty sorting, rejection, resampling or policy-outcome selection.",
        position_error_each_axis_m=[-1.,1.], requested_position_z_m=[4.,6.],
        euler_ranges_rad=dict(roll=[-math.pi,math.pi],pitch=[-math.pi,math.pi],yaw=[0.,2*math.pi]),
        orientation_distribution="Independent uniform Euler angles, as explicitly requested; not uniform in physical rotation space.",
        velocity_each_axis_m_s=[-1.,1.], body_rate_each_axis_rad_s=[-1.,1.],
        rejected=0, generated_draws=12*count,
        initially_inside_band_kept=sum(s["initially_inside_position_speed_band"] for s in states),
        position_definition="All position errors relative to (0,0,5m); stored state z=5+dz.",
        observation_contract="Requested order xyz,roll,pitch,yaw,vxyz,wxyz; deployed unchanged principal_zyx observes xyz,yaw,pitch,roll,vxyz,wxyz. Compare attitude with quaternions, not componentwise Euler aliases.",
        endpoint="Complete10s; every final2s observation position error<0.3m and speed<0.3m/s; prior relapse allowed.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--count",type=int,default=1000)
    parser.add_argument("--split",default="full_random_20261030_v1")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("Choose a new immutable bank path")
    data=generate(args.seed,args.count,args.split)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    print(args.output, data["count"], "rejected=0")

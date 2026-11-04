#!/usr/bin/env python3
"""Allow the backward PREPARATION pose to leave the release-state cube.

For p in +/-1 and v in +/-1, p(-0.35)=p-v*0.35 lies in +/-1.35.
Release requests still use the unchanged strict training-state validator.
No flight-state bounds, gate tolerances, trajectory, sensors or physics change.
"""
import argparse
from contextlib import contextmanager
import math
import sys

import recovery_state
from ramp_initialization import at_offset

STRICT_RELEASE_VALIDATOR = recovery_state.validate_training_state
MAX_LEAD = .35


def validate_preparation_state(state):
    state = tuple(float(x) for x in state)
    if len(state)!=12 or not all(math.isfinite(x) for x in state):
        raise ValueError("Preparation must contain twelve finite numbers")
    limit=1.+MAX_LEAD
    if any(abs(state[i]-(5. if i==2 else 0.))>limit+1e-12 for i in range(3)):
        raise ValueError("Preparation is outside the deterministic backward envelope")
    # Reuse the original angle/twist validation, without widening release bounds.
    STRICT_RELEASE_VALIDATOR((0.,0.,5.,*state[3:]))
    return state


def preparation_spawn(state,lead=.35):
    target=STRICT_RELEASE_VALIDATOR(state)
    if not .1<=lead<=MAX_LEAD:
        raise ValueError("Preparation lead must be in [0.1,0.35] seconds")
    spawn=at_offset(target,-lead)
    spawn[6:]=[0.]*6
    return list(validate_preparation_state(spawn))


@contextmanager
def preparation_validation():
    old=recovery_state.validate_training_state
    recovery_state.validate_training_state=validate_preparation_state
    try:
        yield
    finally:
        recovery_state.validate_training_state=old


def main():
    if len(sys.argv)<2 or sys.argv[1] not in ("spawn","pose","probe"):
        raise SystemExit("Usage: full_random_initialization.py spawn|pose|probe ...")
    mode=sys.argv.pop(1)
    if mode=="spawn":
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--state",type=float,nargs=12,required=True)
        parser.add_argument("--lead",type=float,default=.35)
        parser.add_argument("--shell",action="store_true")
        args=parser.parse_args()
        spawn=preparation_spawn(args.state,args.lead)
        print(" ".join(format(x,".17f") for x in spawn))
        return 0
    # This CLI process handles only a disarmed pre-release pose. Neither
    # step24's release validator nor the scoring analyzer imports this override.
    with preparation_validation():
        if mode=="pose":
            recovery_state.main()
            return 0
        import initialization_gate as probe
        probe.validate_training_state=validate_preparation_state
        return probe.main()


if __name__=="__main__":raise SystemExit(main())

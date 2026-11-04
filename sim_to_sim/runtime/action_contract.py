#!/usr/bin/env python3
"""FlightLxx PPO action scaling."""

import math


FLIGHTLXX_PI = 3.1415926
GRAVITY_M_S2 = 9.81


def raw_action_to_ctbr(raw_action):
    """Convert one raw PPO action into FlightLxx CTBR quantities."""
    values = tuple(float(value) for value in raw_action)

    if len(values) != 4:
        raise ValueError("raw_action must contain exactly 4 values")

    if not all(math.isfinite(value) for value in values):
        raise ValueError("raw_action contains NaN or Inf")

    return (
        2.0 * FLIGHTLXX_PI * values[0],
        2.0 * FLIGHTLXX_PI * values[1],
        FLIGHTLXX_PI * values[2],
        GRAVITY_M_S2 * (values[3] + 1.0),
    )

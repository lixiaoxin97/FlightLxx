#!/usr/bin/env python3
"""Step 1: verify the FlightLxx PPO action conversion."""

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Check FlightLxx raw action -> CTBR conversion"
    )

    parser.add_argument(
        "action",
        nargs=4,
        type=float,
        metavar=("A0", "A1", "A2", "A3"),
        help="raw PPO action, normally each value is in [-1, 1]",
    )

    args = parser.parse_args()
    ctbr = raw_action_to_ctbr(args.action)

    print("raw action              :", tuple(args.action))
    print(
        "body rates [rad/s]      :",
        tuple(round(value, 6) for value in ctbr[:3]),
    )
    print(
        "collective thrust [m/s²]:",
        round(ctbr[3], 6),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
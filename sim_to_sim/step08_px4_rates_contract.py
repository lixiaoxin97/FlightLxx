#!/usr/bin/env python3
"""Step 8: verify FlightLxx CTBR -> PX4 body-rate setpoint conversion."""

import argparse
import math
import sys

from step01_action_contract import raw_action_to_ctbr


def raw_action_to_px4_rates_setpoint(raw_action):
    """Convert FlightLxx raw action to PX4 FRD rate/thrust fields.

    FlightLxx:
        +X forward, +Y left, +Z up

    PX4:
        +X forward, +Y right, +Z down
    """
    values = tuple(float(value) for value in raw_action)

    if len(values) != 4:
        raise ValueError(
            "raw_action must contain exactly 4 values"
        )

    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "raw_action contains NaN or Inf"
        )

    if any(abs(value) > 1.0 + 1.0e-6 for value in values):
        raise ValueError(
            "raw_action must be within [-1, 1]"
        )

    wx_flu, wy_flu, wz_flu, collective_thrust = (
        raw_action_to_ctbr(values)
    )

    # FLU body rates -> FRD body rates.
    rates_frd = (
        wx_flu,
        -wy_flu,
        -wz_flu,
    )

    # a3=-1 -> zero thrust
    # a3= 1 -> maximum normalized thrust
    # PX4 upward thrust uses negative body-Z.
    thrust_fraction = (values[3] + 1.0) / 2.0

    thrust_body_frd = (
        0.0,
        0.0,
        -thrust_fraction,
    )

    return (
        rates_frd,
        thrust_body_frd,
        collective_thrust,
    )


def _assert_close(actual, expected, tolerance=1.0e-6):
    if len(actual) != len(expected):
        raise AssertionError((actual, expected))

    for actual_value, expected_value in zip(actual, expected):
        if abs(actual_value - expected_value) > tolerance:
            raise AssertionError((actual, expected))


def run_self_test():
    rates, thrust, collective = (
        raw_action_to_px4_rates_setpoint(
            (0.0, 0.0, 0.0, 0.0)
        )
    )

    _assert_close(
        rates,
        (0.0, 0.0, 0.0),
    )

    _assert_close(
        thrust,
        (0.0, 0.0, -0.5),
    )

    assert abs(collective - 9.81) < 1.0e-6

    rates, thrust, _ = (
        raw_action_to_px4_rates_setpoint(
            (1.0, 1.0, 1.0, 1.0)
        )
    )

    _assert_close(
        rates,
        (
            2.0 * 3.1415926,
            -2.0 * 3.1415926,
            -3.1415926,
        ),
    )

    _assert_close(
        thrust,
        (0.0, 0.0, -1.0),
    )

    rates, thrust, _ = (
        raw_action_to_px4_rates_setpoint(
            (-1.0, -1.0, -1.0, -1.0)
        )
    )

    _assert_close(
        rates,
        (
            -2.0 * 3.1415926,
            2.0 * 3.1415926,
            3.1415926,
        ),
    )

    _assert_close(
        thrust,
        (0.0, 0.0, 0.0),
    )

    for bad_action in (
        (0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 0.0),
    ):
        try:
            raw_action_to_px4_rates_setpoint(bad_action)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "invalid action was accepted"
            )

    print("step08 self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert FlightLxx raw action "
            "to PX4 setpoint fields"
        )
    )

    parser.add_argument(
        "action",
        nargs=4,
        type=float,
        metavar=("A0", "A1", "A2", "A3"),
        help="raw FlightLxx action",
    )

    args = parser.parse_args()

    rates, thrust, collective = (
        raw_action_to_px4_rates_setpoint(args.action)
    )

    print("raw action:")
    print(tuple(args.action))

    print("PX4 rates FRD [rad/s]:")
    print(tuple(round(value, 6) for value in rates))

    print("PX4 thrust_body FRD:")
    print(tuple(round(value, 6) for value in thrust))

    print("FlightLxx collective [m/s^2]:")
    print(round(collective, 6))

    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(run_self_test())

    raise SystemExit(main())
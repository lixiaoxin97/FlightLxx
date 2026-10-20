#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math

from step06_policy_client import PolicyClient


DEFAULT_OBSERVATION = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def validate_observation(values):
    values = tuple(float(value) for value in values)

    if len(values) != 12:
        raise ValueError(
            "observation must contain exactly 12 values"
        )

    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "observation contains NaN or Inf"
        )

    return values


def run_self_test() -> None:
    assert validate_observation(
        DEFAULT_OBSERVATION
    ) == DEFAULT_OBSERVATION

    assert len(validate_observation([0.0] * 12)) == 12

    try:
        validate_observation([0.0] * 11)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "wrong observation length was accepted"
        )

    try:
        validate_observation(
            [0.0] * 11 + [float("nan")]
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "NaN observation was accepted"
        )

    print("step13 self-test: PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Probe the FlightLxx policy with one "
            "explicit observation"
        )
    )

    parser.add_argument(
        "--policy-host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--policy-port",
        type=int,
        default=5800,
    )

    parser.add_argument(
        "--obs",
        type=float,
        nargs=12,
        default=DEFAULT_OBSERVATION,
        metavar="VALUE",
        help=(
            "12 values: x y z yaw pitch roll "
            "vx vy vz wx wy wz"
        ),
    )

    parser.add_argument(
        "--sequence",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.sequence < 0:
        parser.error("--sequence must be >= 0")

    try:
        args.obs = validate_observation(args.obs)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main() -> int:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return 0

    client = PolicyClient(
        args.policy_host,
        args.policy_port,
        timeout=1.0,
    )

    try:
        print("policy server ping:", client.ping())

        action = client.infer(
            args.obs,
            sequence=args.sequence,
        )

        print("observation:")
        print(
            [round(value, 6) for value in args.obs]
        )

        print("raw action:")
        print(
            [round(float(value), 6) for value in action]
        )
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
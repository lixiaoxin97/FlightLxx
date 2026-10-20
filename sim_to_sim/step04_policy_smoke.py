#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np


def flightlxx_root(explicit_root=None):
    if explicit_root:
        return os.path.abspath(os.path.expanduser(explicit_root))

    if os.environ.get("FlightLxx_PATH"):
        return os.path.abspath(
            os.path.expanduser(os.environ["FlightLxx_PATH"])
        )

    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )


def load_policy(root, model_path):
    if root not in sys.path:
        sys.path.insert(0, root)

    from rl.lxx_baselines.ppo.ppo2 import PPO2

    print("loading model:", model_path)
    return PPO2.load(model_path)


def predict_action(model, observation):
    observation = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1)

    if observation.size != 12:
        raise ValueError(
            "observation must contain exactly 12 values"
        )

    if not np.all(np.isfinite(observation)):
        raise ValueError(
            "observation contains NaN or Inf"
        )

    action, _ = model.predict(
        observation.reshape(1, 12),
        deterministic=True,
    )

    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    if action.size != 4:
        raise RuntimeError(
            "policy returned %d action values"
            % action.size
        )

    if not np.all(np.isfinite(action)):
        raise RuntimeError(
            "policy returned NaN or Inf"
        )

    return action


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load FlightLxx PPO2 and infer one "
            "deterministic action"
        )
    )

    parser.add_argument(
        "--flightlxx-path",
        default=None,
    )

    parser.add_argument(
        "--model",
        default="Simulation_Experiments/models/CTBR.zip",
        help=(
            "model path relative to FlightLxx root "
            "or an absolute path"
        ),
    )

    parser.add_argument(
        "--observation",
        type=float,
        nargs=12,
        default=(
            0.0, 0.0, 5.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ),
        metavar=(
            "X", "Y", "Z",
            "YAW", "PITCH", "ROLL",
            "VX", "VY", "VZ",
            "WX", "WY", "WZ",
        ),
    )

    args = parser.parse_args()

    root = flightlxx_root(
        args.flightlxx_path
    )

    model_path = os.path.expanduser(
        args.model
    )

    if not os.path.isabs(model_path):
        model_path = os.path.join(
            root,
            model_path,
        )

    model = load_policy(
        root,
        model_path,
    )

    observation = np.asarray(
        args.observation,
        dtype=np.float32,
    )

    action = predict_action(
        model,
        observation,
    )

    print("observation:")
    print(
        np.array2string(
            observation,
            precision=6,
            separator=", ",
        )
    )

    print("raw action:")
    print(
        np.array2string(
            action,
            precision=6,
            separator=", ",
        )
    )

    print(
        "action range:",
        float(np.min(action)),
        float(np.max(action)),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
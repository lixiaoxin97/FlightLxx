#!/usr/bin/env python3
"""
Train the three policies for Paper1 — Ideal Dynamics.

  CTBR   : exact prototype parameters
  CTBR+DR: domain randomization (7 parameters per episode)
  CTBR+ID: ideal dynamics (skip motor model, direct kinematics)

Usage:
    python3 Simulation_Experiments/train.py --model CTBR
    python3 Simulation_Experiments/train.py --model CTBR+DR
    python3 Simulation_Experiments/train.py --model CTBR+ID
"""
import os, sys, argparse
import numpy as np
import tensorflow as tf
from ruamel.yaml import YAML, dump, RoundTripDumper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines import logger
from rl.lxx_baselines.common.policies import MlpPolicy
from rl.lxx_baselines.ppo.ppo2 import PPO2
from rl.lxx_baselines.envs import vec_env_wrapper as wrapper
import rl.lxx_baselines.common.util as U
from flightgym import QuadrotorEnv_v1

# All models train on Proto platform parameters
PLATFORM = "prototype"
TOTAL_TIMESTEPS = 250_000_000  # 5000 iterations × 50,000 steps
N_ENVS = 100
N_THREADS = 10


def create_env(render=False):
    cfg = YAML().load(open(os.environ["FlightLxx_PATH"] +
                            "/libs/config/vec_env.yaml", 'r'))
    cfg["env"]["num_envs"] = N_ENVS
    cfg["env"]["num_threads"] = N_THREADS
    cfg["env"]["render"] = "yes" if render else "no"

    env = wrapper.FlightEnvVec(
        QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))

    # Switch to prototype dynamics
    proto_path = os.path.join(os.environ["FlightLxx_PATH"],
                              "libs", "config", "prototype.yaml")
    with open(proto_path) as f:
        env.setDynamics(f.read())

    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['CTBR', 'CTBR+DR', 'CTBR+ID'])
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    out_dir = os.path.join(os.path.dirname(__file__), "training_data", args.model)
    os.makedirs(out_dir, exist_ok=True)
    saver = U.ConfigurationSaver(log_dir=out_dir)

    print(f"\n{'='*60}")
    print(f"Training {args.model} on {PLATFORM}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs:   {N_ENVS}")
    print(f"  Output dir:      {out_dir}")
    print(f"{'='*60}\n")

    env = create_env(args.render)
    env.seed(args.seed)

    # ---- Model-specific setup ----
    if args.model == "CTBR+DR":
        env.enableDynamicRandomization(True)
        print("[CTBR+DR] Domain Randomization ENABLED — randomize 7 params per episode\n")
    elif args.model == "CTBR+ID":
        env.enableIdealDynamics(True)
        print("[CTBR+ID] Ideal Dynamics ENABLED — skip motor model & P-controller\n")

    # ---- PPO2 ----
    model = PPO2(
        tensorboard_log=saver.data_dir,
        policy=MlpPolicy,
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            act_fun=tf.nn.tanh),
        env=env,
        lam=0.95,
        gamma=0.99,
        n_steps=500,
        ent_coef=0.0001,
        learning_rate=3e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        nminibatches=10,
        noptepochs=10,
        cliprange=0.2,
        verbose=1,
    )

    logger.configure(folder=saver.data_dir)
    model.learn(
        total_timesteps=int(TOTAL_TIMESTEPS),
        log_dir=saver.data_dir,
        logger=logger,
    )

    # Save to models/
    save_path = os.path.join(os.path.dirname(__file__), "models", f"{args.model}.zip")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()

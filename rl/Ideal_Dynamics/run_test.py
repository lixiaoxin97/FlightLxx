#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

import os
import argparse
import numpy as np
import tensorflow as tf

from rl.lxx_baselines.common.policies import MlpPolicy
from rl.lxx_baselines.ppo.ppo2 import PPO2
from rl.Ideal_Dynamics.test_neural_network import test_model
from rl.lxx_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str,
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight_1', type=str, default='./CTBR.zip',
                        help='trained weight path (CTBR)')
    parser.add_argument('-v', '--weight_2', type=str, default='./CTBR+DR.zip',
                        help='trained weight path (CTBR+DR)')
    parser.add_argument('-y', '--weight_3', type=str, default='./CTBR+ID.zip',
                        help='trained weight path (CTBR+ID)')
    return parser


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FlightLxx_PATH"] +
                           "/libs/config/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    cfg["env"]["render"] = "yes" if args.render else "no"

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    configure_random_seed(args.seed, env=env)

    if args.train:
        # ---- Training mode ----
        from stable_baselines import logger
        import rl.lxx_baselines.common.util as U

        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root
        saver = U.ConfigurationSaver(log_dir=log_dir)

        model = PPO2(
            tensorboard_log=saver.data_dir,
            policy=MlpPolicy,
            policy_kwargs=dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])], act_fun=tf.nn.tanh),
            env=env,
            lam=0.95,
            gamma=0.99,
            n_steps=500,
            ent_coef=0.00,
            learning_rate=3e-4,
            vf_coef=0.5,
            max_grad_norm=0.5,
            nminibatches=1,
            noptepochs=10,
            cliprange=0.2,
            verbose=1,
        )

        logger.configure(folder=saver.data_dir)
        model.learn(
            total_timesteps=int(250000000),
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)

    else:
        # ---- Testing mode ----
        model_1 = PPO2.load(args.weight_1)
        model_2 = PPO2.load(args.weight_2)
        model_3 = PPO2.load(args.weight_3)
        test_model(env, model_1, model_2, model_3, render=args.render)


if __name__ == "__main__":
    main()

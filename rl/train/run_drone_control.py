#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import sys
import math
import argparse
import warnings
from collections import OrderedDict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings(
    "ignore",
    message="`tf.layers.*` is deprecated.*",
    category=UserWarning,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("FlightLxx_PATH", PROJECT_ROOT)

from rl.runtime_libs import ensure_libstdcxx_preloaded

ensure_libstdcxx_preloaded()

import numpy as np
import tensorflow as tf
from rl.tf_compat import enable_tf1_compat

enable_tf1_compat()

#
from stable_baselines import logger
from stable_baselines.common.base_class import BaseRLModel

#
from rl.lxx_baselines.common.policies import MlpPolicy, HistoryCnnPolicy
from rl.lxx_baselines.ppo.ppo2 import PPO2
from rl.lxx_baselines.ppo.ppo2_test import test_model
from rl.lxx_baselines.envs.environment_policy import LearnableEnvironmentPolicy
from rl.lxx_baselines.envs import vec_env_wrapper as wrapper
import rl.lxx_baselines.common.util as U
#
from flightgym import QuadrotorEnv_v1


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def make_policy_kwargs(hidden_size, policy_type="history_cnn",
                       history_len=50, frame_dim=25):
    hidden_size = int(hidden_size)
    if policy_type == "history_cnn":
        return dict(
            history_len=int(history_len),
            frame_dim=int(frame_dim),
            short_history_len=5,
            cnn_features=64,
            hidden_size=hidden_size,
            act_fun=tf.nn.tanh,
        )
    return dict(
        net_arch=[dict(pi=[hidden_size, hidden_size],
                       vf=[hidden_size, hidden_size])],
        act_fun=tf.nn.tanh,
    )


def resolve_policy_class(policy_type):
    if policy_type == "mlp":
        return MlpPolicy
    if policy_type == "history_cnn":
        return HistoryCnnPolicy
    raise ValueError("Unsupported policy_type {}".format(policy_type))


def make_env_policy_callback(
    env,
    env_policy,
    frequency,
    warmup_updates,
    checkpoint_freq_updates=0,
    checkpoint_dir=None,
):
    def env_policy_callback(local_vars, _global_vars):
        update = local_vars.get("update", 0)
        if (
            env_policy is not None and
            frequency > 0 and
            update >= warmup_updates and
            update % frequency == 0
        ):
            env_policy.update_and_apply(env, logger=logger)
        if (
            checkpoint_freq_updates > 0
            and checkpoint_dir is not None
            and update > 0
            and update % checkpoint_freq_updates == 0
        ):
            model = local_vars.get("self")
            if model is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_path = os.path.join(
                    checkpoint_dir, "ppo_update_{:06d}".format(update))
                model.save(model_path)
                if env_policy is not None:
                    env_policy.save(os.path.join(
                        checkpoint_dir,
                        "environment_policy_update_{:06d}".format(update),
                    ))
                print("[Checkpoint] saved {}".format(model_path))
        return True

    return env_policy_callback


def load_pretrained_parameters(model, weight_path, reset_logstd=True):
    """Use an existing policy as initialization without carrying optimizer state."""
    _, loaded_params = BaseRLModel._load_from_file(weight_path, load_data=False)
    loaded_params = dict(loaded_params)
    current_params = model.get_parameters()

    params_to_load = OrderedDict()
    skipped_logstd = []
    skipped_shape = []
    skipped_missing = []
    for name, value in loaded_params.items():
        if name not in current_params:
            skipped_missing.append(name)
            continue
        if reset_logstd and "/pi/logstd:" in name:
            skipped_logstd.append(name)
            continue
        if current_params[name].shape != value.shape:
            skipped_shape.append(
                (name, value.shape, current_params[name].shape))
            continue
        params_to_load[name] = value

    if len(params_to_load) == 0:
        raise RuntimeError(
            "No compatible pretrained parameters were found in {}".format(
                weight_path))

    model.load_parameters(params_to_load, exact_match=False)
    print(
        "[Pretrain] loaded {} tensors from {}; skipped {} logstd, {} shape, {} missing".format(
            len(params_to_load),
            weight_path,
            len(skipped_logstd),
            len(skipped_shape),
            len(skipped_missing),
        )
    )
    if skipped_shape:
        preview = ", ".join(
            "{} {}->{}".format(name, old_shape, new_shape)
            for name, old_shape, new_shape in skipped_shape[:5])
        print("[Pretrain] shape-skipped preview: {}".format(preview))


def resolve_env_policy_checkpoint(path):
    if not path:
        return None
    if os.path.isdir(path):
        fallback = os.path.join(path, "model")
        if os.path.exists(fallback + ".index"):
            return fallback
        checkpoint = tf.train.latest_checkpoint(path)
        if checkpoint is not None and os.path.exists(checkpoint + ".index"):
            return checkpoint
        return None
    if os.path.exists(path + ".index"):
        return path
    if os.path.exists(path):
        return path
    return None


def infer_env_policy_weight_path(policy_weight_path):
    if not policy_weight_path:
        return ""

    base_path = policy_weight_path[:-4] if policy_weight_path.endswith(
        ".zip") else policy_weight_path
    parent_dir, name = os.path.split(base_path)
    candidates = []

    if name.startswith("ppo_update_"):
        suffix = name[len("ppo_update_"):]
        candidates.append(
            os.path.join(parent_dir, "environment_policy_update_" + suffix))

    candidates.append(os.path.join(base_path, "environment_policy"))
    candidates.append(os.path.join(parent_dir, "environment_policy"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if resolve_env_policy_checkpoint(candidate) is not None:
            return candidate
    return ""


def load_pretrained_environment_policy(env_policy, env_policy_path,
                                       reset_optimizer=True):
    if env_policy is None:
        return
    env_policy.load(env_policy_path, reset_optimizer=reset_optimizer)


def parser():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the FlightLxx path-following policy.")
    hidden = argparse.SUPPRESS
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('--total_timesteps', type=int, default=2500000000,
                        help="Total PPO samples to train")
    parser.add_argument('--n_steps', type=int, default=1000,
                        help="PPO rollout length per environment")
    parser.add_argument('--env_policy_freq', type=int, default=25,
                        help="Update learnable environment policy every N PPO updates; <=0 disables it")
    parser.add_argument('--env_policy_warmup_updates', type=int, default=0,
                        help="Number of PPO updates before the environment policy starts changing paths")
    parser.add_argument('--pretrained_weight', '--init_weight', type=str,
                        default="",
                        help="Optional PPO checkpoint used only to initialize compatible model weights before training")
    parser.add_argument('--pretrained_env_policy', '--init_env_policy', type=str,
                        default="",
                        help="Optional environment-policy checkpoint directory/prefix used to initialize the path generator")
    parser.add_argument('--checkpoint_freq_updates', type=int, default=500,
                        help="Save PPO and environment-policy checkpoints every N PPO updates; <=0 disables")
    parser.add_argument('--num_envs', type=int, default=None,
                        help="Override number of vectorized environments")
    parser.add_argument('--num_threads', type=int, default=None,
                        help="Override number of OpenMP worker threads")
    parser.add_argument('--path_mode', type=str,
                        default=os.environ.get("FlightLxx_PATH_MODE", "default"),
                        choices=["default", "straight", "figure8", "figure_eight"],
                        help="Path generator mode; use straight for a reset-time straight-line test")
    parser.add_argument('--test_path_length', type=float, default=20.0,
                        help="Straight-test path length in meters when --path_mode straight is used")
    parser.add_argument('--test_wind', type=str, default="none",
                        choices=["none", "realistic", "strong", "extreme"],
                        help="Deterministic wind preset for evaluation; does not affect training unless explicitly set")
    parser.add_argument('-w', '--weight', type=str,
                        default=os.path.join(PROJECT_ROOT, 'rl/train/2026-06-05-01.zip'),
                        help='trained weight path')

    # Advanced knobs are intentionally hidden from --help. They remain accepted
    # for old shell snippets and can still be overridden when needed.
    parser.add_argument('--save_dir', type=str,
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help=hidden)
    parser.add_argument('--env_policy_lr', type=float, default=1e-4,
                        help=hidden)
    parser.add_argument('--env_policy_entropy_coef', type=float, default=1.0,
                        help=hidden)
    parser.add_argument('--env_policy_action_clip', type=float, default=0.25,
                        help=hidden)
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help=hidden)
    parser.add_argument('--ent_coef', type=float, default=1e-3,
                        help=hidden)
    parser.add_argument('--noptepochs', type=int, default=5,
                        help=hidden)
    parser.add_argument('--nminibatches', type=int, default=50,
                        help=hidden)
    parser.add_argument('--cliprange', type=float, default=0.12,
                        help=hidden)
    parser.add_argument('--target_kl', type=float, default=0.02,
                        help=hidden)
    parser.add_argument('--reset_on_rollout_end', type=int, default=0,
                        help=hidden)
    parser.add_argument('--policy_hidden_size', type=int, default=512,
                        help=hidden)
    parser.add_argument('--policy_type', type=str, default="history_cnn",
                        choices=["history_cnn", "mlp"],
                        help=hidden)
    parser.add_argument('--obs_history_len', type=int, default=50,
                        help=hidden)
    parser.add_argument('--reset_pretrained_logstd', type=int, default=1,
                        help=hidden)
    parser.add_argument('--reset_pretrained_env_optimizer', type=int, default=1,
                        help=hidden)
    return parser


def main():
    args = parser().parse_args()
    os.environ["FlightLxx_PATH_MODE"] = args.path_mode
    if args.path_mode == "straight":
        os.environ["FlightLxx_TEST_PATH_LENGTH"] = str(float(args.test_path_length))
    else:
        os.environ.pop("FlightLxx_TEST_PATH_LENGTH", None)
    os.environ["FlightLxx_TEST_WIND_MODE"] = args.test_wind
    cfg = YAML().load(open(os.environ.get("FlightLxx_PATH", PROJECT_ROOT) +
                           "/libs/config/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
    if args.num_envs is not None:
        cfg["env"]["num_envs"] = args.num_envs
    if args.num_threads is not None:
        cfg["env"]["num_threads"] = args.num_threads

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    obs_history_len = args.obs_history_len if args.policy_type == "history_cnn" else 1
    env = wrapper.FlightEnvVec(
        QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False),
        obs_history_len=obs_history_len)
    policy_class = resolve_policy_class(args.policy_type)
    env_policy = None
    if args.train and args.env_policy_freq > 0:
        env_policy = LearnableEnvironmentPolicy(
            obs_dim=env.env_policy_obs_dim,
            act_dim=env.env_policy_act_dim,
            learning_rate=args.env_policy_lr,
            entropy_coef=args.env_policy_entropy_coef,
            action_clip=args.env_policy_action_clip,
            seed=args.seed,
        )

    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root
        saver = U.ConfigurationSaver(log_dir=log_dir)
        model = PPO2(
            tensorboard_log=saver.data_dir,
            policy=policy_class,  # check activation function
            policy_kwargs=make_policy_kwargs(
                args.policy_hidden_size,
                policy_type=args.policy_type,
                history_len=obs_history_len,
                frame_dim=env.base_num_obs,
            ),
            env=env,
            lam=0.95,
            gamma=0.99,  # lower 0.9 ~ 0.99
            # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            learning_rate=args.learning_rate,
            vf_coef=0.5,
            max_grad_norm=0.5,
            nminibatches=args.nminibatches,
            noptepochs=args.noptepochs,
            cliprange=args.cliprange,
            target_kl=args.target_kl if args.target_kl > 0.0 else None,
            reset_on_rollout_end=bool(args.reset_on_rollout_end),
            verbose=1,
        )
        if args.pretrained_weight:
            load_pretrained_parameters(
                model,
                args.pretrained_weight,
                reset_logstd=bool(args.reset_pretrained_logstd),
            )
        env_policy_weight = args.pretrained_env_policy
        if env_policy_weight:
            load_pretrained_environment_policy(
                env_policy,
                env_policy_weight,
                reset_optimizer=bool(args.reset_pretrained_env_optimizer),
            )
        elif args.pretrained_weight and env_policy is not None:
            print(
                "[EnvPolicy] starting from scratch; old environment-policy checkpoints are not compatible with the closed-curve generator")

        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        logger.configure(folder=saver.data_dir, format_strs=["log", "csv"])
        print("[Log] compact console metrics enabled")
        print("[Log] full metrics are saved to {}".format(saver.data_dir))
        model.learn(
            total_timesteps=int(args.total_timesteps),
            log_dir=saver.data_dir, logger=logger,
            callback=make_env_policy_callback(
                env,
                env_policy,
                args.env_policy_freq,
                args.env_policy_warmup_updates,
                args.checkpoint_freq_updates,
                os.path.join(saver.data_dir, "checkpoints"),
            )
            if (env_policy is not None or args.checkpoint_freq_updates > 0)
            else None)
        model.save(saver.data_dir)
        if env_policy is not None:
            env_policy.save(os.path.join(saver.data_dir, "environment_policy"))

    # # Testing mode with a trained weight
    else:
        custom_objects = {
            "learning_rate": 3e-4,
            "cliprange": 0.2,
            "policy": policy_class,
            "policy_kwargs": make_policy_kwargs(
                args.policy_hidden_size,
                policy_type=args.policy_type,
                history_len=obs_history_len,
                frame_dim=env.base_num_obs,
            ),
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }
        model = PPO2.load(args.weight, env=env, custom_objects=custom_objects)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()

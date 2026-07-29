#!/usr/bin/env python3
"""
Recovery-from-random-state experiment for Paper1 — Ideal Dynamics.

Tests 3 policies (CTBR / CTBR+DR / CTBR+ID) on 3 platforms
(Prototype / Iris / Crazyflie). Each trial resets the quadrotor
to a random aggressive initial state and measures recovery performance.

Usage:
    python3 Simulation_Experiments/run_experiment.py --trials 100 --seed 42
"""
import os
import sys
import argparse
import time
import numpy as np
from collections import defaultdict

from ruamel.yaml import YAML, dump, RoundTripDumper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.lxx_baselines.ppo.ppo2 import PPO2
from rl.lxx_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1

from Simulation_Experiments.configs import get_platform_config, get_model_weight
from Simulation_Experiments.disturbances import random_initial_state
from Simulation_Experiments.metrics import TrialResult, ExperimentResults, compute_metrics


# ---------------------------------------------------------------------------
#  Environment & model helpers
# ---------------------------------------------------------------------------

def create_env(platform: str, render: bool = False):
    cfg = YAML().load(open(os.environ["FlightLxx_PATH"] +
                            "/libs/config/vec_env.yaml", 'r'))
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = "yes" if render else "no"

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env.setDynamics(get_platform_config(platform))
    if render:
        env.connectUnity()
    return env


def load_model(path: str):
    return PPO2.load(path)


# ---------------------------------------------------------------------------
#  Single trial
# ---------------------------------------------------------------------------

def run_recovery_trial(env, model, model_name: str, platform: str,
                       trial_id: int) -> TrialResult:
    """Reset to random state, run 500 steps (10s), compute recovery metrics."""
    max_steps = 500

    state = random_initial_state()
    obs = np.zeros((1, 12), dtype=np.float32)
    env.resetToState(state, obs)

    pos_list, vel_list, euler_list, omega_list = [], [], [], []
    actions_list = []

    for _ in range(max_steps):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(act)

        pos_list.append(obs[0, 0:3].copy())
        vel_list.append(obs[0, 6:9].copy())
        euler_list.append(obs[0, 3:6].copy())
        omega_list.append(obs[0, 9:12].copy())
        actions_list.append(act[0, :].copy())

        if done:
            break

    metrics = compute_metrics({
        'pos': pos_list, 'vel': vel_list, 'euler': euler_list,
        'omega': omega_list, 'actions': actions_list,
    })

    return TrialResult(
        platform=platform,
        model_name=model_name,
        disturbance="random_state",
        trial_id=trial_id,
        success=metrics['success'],
        recovery_time=metrics['recovery_time'],
        max_pos_error=metrics['max_pos_error'],
        max_attitude_error=metrics['max_attitude_error'],
        max_speed=metrics['max_speed'],
        control_cost=metrics['control_cost'],
        trajectory_jerk=metrics['trajectory_jerk'],
        episode_length=metrics['episode_length'],
        disturbance_velocity=0.0,
    )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recovery-from-random-state experiment")
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--platform', type=str, default=None,
                        choices=['prototype', 'iris', 'crazyflie'],
                        help='Single platform to test (default: all three)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['CTBR', 'CTBR+DR', 'CTBR+ID'],
                        help='Single model to test (default: all three)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'results.csv'))
    args = parser.parse_args()

    np.random.seed(args.seed)

    all_models = {
        "CTBR":    get_model_weight("CTBR"),
        "CTBR+DR": get_model_weight("CTBR+DR"),
        "CTBR+ID": get_model_weight("CTBR+ID"),
    }
    all_platforms = ["prototype", "iris", "crazyflie"]

    models = {args.model: all_models[args.model]} if args.model else all_models
    platforms = [args.platform] if args.platform else all_platforms

    t_start = time.time()
    results = ExperimentResults()

    for platform in platforms:
        for model_name, model_path in models.items():
            print(f"\n{'='*60}")
            print(f"Platform: {platform:12s} | Model: {model_name}")
            print(f"{'='*60}")

            env = create_env(platform, args.render)
            model = load_model(model_path)

            for trial in range(args.trials):
                result = run_recovery_trial(
                    env, model, model_name, platform, trial)
                results.add(result)
                status = "✓" if result.success else "✗"
                print(f"  Trial {trial+1:2d}/{args.trials} {status} "
                      f"rec={result.recovery_time:.2f}s "
                      f"err={result.max_pos_error:.2f}m "
                      f"att={result.max_attitude_error:.0f}°")

    # ---- Summary ----
    df = results.to_dataframe()
    df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print("RECOVERY EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    grouped = defaultdict(list)
    for r in results.results:
        grouped[(r.platform, r.model_name)].append(r)

    print(f"  {'Platform':12s} {'Model':8s} {'Success':>7s} {'Rec(s)':>7s} "
          f"{'MaxErr(m)':>9s} {'MaxAtt':>6s} | n")
    print(f"  {'-'*60}")
    for plat in platforms:
        for model_name in models:
            trials = grouped[(plat, model_name)]
            rate = sum(1 for t in trials if t.success) / len(trials) * 100
            recs = [t.recovery_time for t in trials if t.success]
            avg_rec = np.mean(recs) if recs else float('inf')
            avg_err = np.mean([t.max_pos_error for t in trials])
            avg_att = np.mean([t.max_attitude_error for t in trials])
            print(f"  {plat:12s} {model_name:8s} {rate:6.1f}% {avg_rec:6.2f}s "
                  f"{avg_err:8.2f}m {avg_att:5.0f}° | {len(trials)}")

    print(f"\nResults saved to: {args.output}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

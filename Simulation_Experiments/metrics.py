"""
Metrics collection for simulation experiments.

Recovery criterion: pos_error < 0.3 m AND speed < 0.3 m/s, sustained for 0.3 s.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrialResult:
    """Results from a single trial."""
    platform: str
    model_name: str
    disturbance: str
    trial_id: int
    success: bool = True
    recovery_time: float = 0.0        # seconds from disturbance to recovery
    max_pos_error: float = 0.0        # maximum position deviation (m)
    max_attitude_error: float = 0.0   # maximum attitude deviation (deg)
    max_speed: float = 0.0            # maximum speed (m/s)
    control_cost: float = 0.0         # ∫|ω_cmd| dt
    trajectory_jerk: float = 0.0      # mean jerk (m/s³)
    episode_length: int = 0           # total steps before termination
    disturbance_velocity: float = 0.0 # magnitude of applied disturbance

    def summary(self) -> dict:
        return {
            "platform": self.platform,
            "model": self.model_name,
            "disturbance": self.disturbance,
            "trial": self.trial_id,
            "success": self.success,
            "recovery_time_s": round(self.recovery_time, 3),
            "max_pos_error_m": round(self.max_pos_error, 3),
            "max_att_error_deg": round(self.max_attitude_error, 1),
            "max_speed_ms": round(self.max_speed, 3),
            "control_cost": round(self.control_cost, 3),
            "jerk_ms3": round(self.trajectory_jerk, 3),
            "ep_len": self.episode_length,
        }


@dataclass
class ExperimentResults:
    """Aggregated results for a full experiment."""
    results: List[TrialResult] = field(default_factory=list)

    def add(self, result: TrialResult):
        self.results.append(result)

    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def avg_recovery_time(self) -> float:
        successes = [r.recovery_time for r in self.results if r.success]
        return float(np.mean(successes)) if successes else float('inf')

    def avg_metric(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.results if r.success]
        return float(np.mean(vals)) if vals else 0.0

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([r.summary() for r in self.results])


def compute_metrics(trajectory: dict, dt: float = 0.02,
                    pos_goal: np.ndarray = np.array([0.0, 0.0, 5.0]),
                    recovery_threshold_pos: float = 0.3,
                    recovery_threshold_spd: float = 0.3,
                    recovery_sustain_s: float = 0.3) -> dict:
    """Compute recovery metrics from a rollout trajectory.

    Recovery criterion:
      pos_error < 0.3 m AND speed < 0.3 m/s, sustained for 0.3 s.

    Args:
        trajectory: dict with keys 'pos', 'vel', 'euler', 'omega', 'actions'
        dt: simulation timestep (s)
        pos_goal: target hover position [x, y, z]
        recovery_threshold_pos: position error threshold for recovery (m)
        recovery_threshold_spd: speed threshold for recovery (m/s)
        recovery_sustain_s: required sustained time for recovery (s)

    Returns:
        dict with success, recovery_time, max_* metrics
    """
    pos = np.asarray(trajectory['pos'])
    vel = np.asarray(trajectory['vel'])
    euler = np.asarray(trajectory['euler'])
    omega = np.asarray(trajectory['omega'])
    actions = np.asarray(trajectory['actions'])

    n_steps = pos.shape[0]
    sustain_steps = int(recovery_sustain_s / dt)

    # Position error (distance to goal)
    pos_err = np.linalg.norm(pos - pos_goal, axis=1)

    # Speed
    spd = np.linalg.norm(vel, axis=1)

    # Attitude error (magnitude of euler angles, in degrees)
    att_err = np.linalg.norm(euler, axis=1) * 180 / np.pi

    # Recovery detection: walk backwards to find last sustained recovery window
    # Uses pos < 0.3m AND speed < 0.3 m/s
    recovery_step = n_steps
    recovered = False
    for i in range(n_steps - sustain_steps, -1, -1):
        window_pos = pos_err[i:i + sustain_steps]
        window_spd = spd[i:i + sustain_steps]
        if np.all(window_pos < recovery_threshold_pos) and \
           np.all(window_spd < recovery_threshold_spd):
            recovery_step = i
            recovered = True
            break

    # Jerk: derivative of acceleration (approximated from velocity)
    if n_steps > 3:
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        mean_jerk = float(np.mean(np.linalg.norm(jerk, axis=1)))
    else:
        mean_jerk = 0.0

    # Control cost: ∫|ω_cmd|
    control_cost = float(np.sum(np.linalg.norm(actions[:, :3], axis=1)) * dt)

    return {
        'success': recovered,
        'recovery_time': recovery_step * dt if recovered else n_steps * dt,
        'max_pos_error': float(np.max(pos_err)),
        'max_attitude_error': float(np.max(att_err)),
        'max_speed': float(np.max(spd)),
        'control_cost': control_cost,
        'trajectory_jerk': mean_jerk,
        'episode_length': n_steps,
    }

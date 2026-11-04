"""Shared recovery definitions for FlightGym and PX4/Gazebo experiments.

A recovery time is the first *confirmation* time of a continuous 0.3 s
position/speed window, measured from release in simulation seconds.
The entry time is also returned explicitly. A single out-of-band sample after
confirmation is a relapse; a 0.3 s out-of-band run is a sustained relapse.
"""

import math

METRICS_VERSION = "first_sustained_recovery_simtime_v1"
DEFAULT_THRESHOLDS = {
    "position_m": 0.3,
    "speed_m_s": 0.3,
    "sustain_s": 0.3,
    "relapse_sustain_s": 0.3,
    "max_sample_gap_s": 0.1,
    "tail_duration_s": 2.0,
    "action_saturation_threshold": 0.98,
}


def norm(vector):
    return math.sqrt(sum(float(value) ** 2 for value in vector))


class RecoveryTracker:
    def __init__(
        self,
        position_threshold=0.3,
        speed_threshold=0.3,
        sustain_s=0.3,
        relapse_sustain_s=0.3,
        max_gap_s=0.1,
    ):
        self.position_threshold = position_threshold
        self.speed_threshold = speed_threshold
        self.sustain_s = sustain_s
        self.relapse_sustain_s = relapse_sustain_s
        self.max_gap_s = max_gap_s
        self.good_since = None
        self.bad_since = None
        self.first_entry = None
        self.first_confirmation = None
        self.first_relapse = None
        self.first_sustained_relapse = None
        self.last_time = None
        self.previous_inside = None
        self.relapse_count = 0
        self.invalid_gap_count = 0

    def update(self, timestamp, position_error, speed):
        values = (timestamp, position_error, speed)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Recovery observations must be finite")
        if self.last_time is not None:
            if timestamp <= self.last_time:
                raise ValueError("Simulation timestamps must increase strictly")
            if timestamp - self.last_time > self.max_gap_s + 1e-9:
                self.good_since = None
                self.bad_since = None
                self.invalid_gap_count += 1
        inside = (
            position_error < self.position_threshold and speed < self.speed_threshold
        )
        if inside:
            self.bad_since = None
            if self.good_since is None:
                self.good_since = timestamp
            if (
                self.first_confirmation is None
                and timestamp - self.good_since >= self.sustain_s - 1e-9
            ):
                self.first_entry = self.good_since
                self.first_confirmation = timestamp
        else:
            self.good_since = None
            if self.first_confirmation is not None:
                if self.first_relapse is None:
                    self.first_relapse = timestamp
                if self.previous_inside is not False:
                    self.relapse_count += 1
                if self.bad_since is None:
                    self.bad_since = timestamp
                if (
                    self.first_sustained_relapse is None
                    and timestamp - self.bad_since >= self.relapse_sustain_s - 1e-9
                ):
                    self.first_sustained_relapse = timestamp
        self.previous_inside = inside
        self.last_time = timestamp
        return self.summary()

    def summary(self):
        return {
            "metrics_version": METRICS_VERSION,
            "recovered": self.first_confirmation is not None,
            "recovery_time_s": self.first_confirmation,
            "recovery_entry_time_s": self.first_entry,
            "relapsed_after_recovery": self.first_relapse is not None,
            "first_relapse_time_s": self.first_relapse,
            "sustained_relapse": self.first_sustained_relapse is not None,
            "first_sustained_relapse_time_s": self.first_sustained_relapse,
            "relapse_episode_count": self.relapse_count,
            "invalid_time_gap_count": self.invalid_gap_count,
            "final_inside_band": self.previous_inside,
        }


def weighted_mean(values, weights):
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total if total else None


def vector_jitter(vectors, weights):
    """sqrt(sum(axis population variances)), with temporal weighting."""
    total = sum(weights)
    if len(vectors) == 0 or total <= 0:
        return None
    means = [
        sum(v[axis] * w for v, w in zip(vectors, weights)) / total for axis in range(3)
    ]
    variance = (
        sum(
            w * sum((v[axis] - means[axis]) ** 2 for axis in range(3))
            for v, w in zip(vectors, weights)
        )
        / total
    )
    return math.sqrt(max(0.0, variance))


def evaluate_trajectory(
    times,
    positions,
    velocities,
    rates,
    actions=None,
    pos_goal=(0.0, 0.0, 5.0),
    end_time=None,
    expected_duration=10.0,
    tail_duration=2.0,
    physical_saturation=None,
    position_threshold=0.3,
    speed_threshold=0.3,
    sustain_s=0.3,
):
    n = len(times)
    if not (n == len(positions) == len(velocities) == len(rates)):
        raise ValueError("Trajectory arrays have different lengths")
    if actions is not None and len(actions) != n:
        raise ValueError("Action count differs from state count")
    tracker = RecoveryTracker(position_threshold, speed_threshold, sustain_s)
    errors = [math.dist(p, pos_goal) for p in positions]
    speeds = [norm(v) for v in velocities]
    for timestamp, error, speed in zip(times, errors, speeds):
        tracker.update(timestamp, error, speed)
    result = tracker.summary()
    result["sample_count"] = n
    if not n:
        return result
    end_time = float(times[-1] if end_time is None else end_time)
    if end_time < times[-1] - 1e-8:
        raise ValueError("End time precedes final sample")
    # The last sample covers only until the known end. State statistics use
    # duration weights so a slower or jittery callback does not gain influence.
    weights = [times[i + 1] - times[i] for i in range(n - 1)] + [
        max(0.0, end_time - times[-1])
    ]
    tail_start = max(0.0, end_time - tail_duration)
    tail_weights = [
        max(0.0, min(end_time, times[i] + weights[i]) - max(tail_start, times[i]))
        for i in range(n)
    ]
    tail_complete = end_time >= expected_duration - 0.04
    result.update(
        {
            "observed_duration_s": end_time,
            "horizon_complete": tail_complete,
            "peak_position_error_m": max(errors),
            "peak_speed_m_s": max(speeds),
            "peak_body_rate_rad_s": max(norm(v) for v in rates),
            "final_position_error_m": errors[-1],
            "final_speed_m_s": speeds[-1],
            "tail_window_s": tail_duration,
            "tail_complete": tail_complete,
            "tail_position_error_mean_m": weighted_mean(errors, tail_weights),
            "tail_position_error_rms_m": (
                math.sqrt(weighted_mean([x * x for x in errors], tail_weights))
                if sum(tail_weights)
                else None
            ),
            "tail_speed_rms_m_s": (
                math.sqrt(weighted_mean([x * x for x in speeds], tail_weights))
                if sum(tail_weights)
                else None
            ),
            "tail_velocity_jitter_m_s": vector_jitter(velocities, tail_weights),
            "tail_body_rate_jitter_rad_s": vector_jitter(rates, tail_weights),
            "whole_velocity_jitter_m_s": vector_jitter(velocities, weights),
            "whole_body_rate_jitter_rad_s": vector_jitter(rates, weights),
        }
    )
    if actions is not None and len(actions):
        saturated = [
            float(any(abs(value) >= 0.98 for value in action)) for action in actions
        ]
        rate_saturated = [
            float(any(abs(value) >= 0.98 for value in action[:3])) for action in actions
        ]
        thrust_saturated = [float(abs(action[3]) >= 0.98) for action in actions]
        result.update(
            {
                "action_saturation_fraction": weighted_mean(saturated, weights),
                "rate_action_saturation_fraction": weighted_mean(
                    rate_saturated, weights
                ),
                "thrust_action_saturation_fraction": weighted_mean(
                    thrust_saturated, weights
                ),
                "action_saturation_sample_fraction": sum(saturated) / n,
            }
        )
    if physical_saturation is not None:
        result["physical_thrust_saturation_fraction"] = weighted_mean(
            [float(value) for value in physical_saturation], weights
        )
    return result

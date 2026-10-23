#!/usr/bin/env python3
"""Conservative pre-handover checks for the hover-trained neural policy."""

import math


def evaluate_handover_candidate(
    observation,
    raw_action,
    nn_goal,
    nn_yaw_goal,
    max_position_error_m=0.30,
    max_speed_m_s=0.30,
    max_body_rate_rad_s=0.50,
    max_tilt_rad=0.35,
    max_yaw_error_rad=0.35,
    max_rate_action=0.25,
    max_collective_offset=0.40,
):
    """Return (safe, reasons) for a stationary hover-to-policy handover.

    This guard is only for the initial handover. It checks that the frozen PPO
    policy recognizes the current level hover and requests a moderate command;
    it is not an in-flight state or action limiter.
    """
    try:
        obs = tuple(float(value) for value in observation)
        action = tuple(float(value) for value in raw_action)
        goal = tuple(float(value) for value in nn_goal)
        yaw_goal = float(nn_yaw_goal)
    except (TypeError, ValueError, OverflowError):
        return False, ("malformed_observation_or_action",)

    if len(obs) != 12 or len(action) != 4 or len(goal) != 3:
        return False, ("invalid_observation_action_or_goal_size",)
    if not all(math.isfinite(x) for x in obs + action + goal + (yaw_goal,)):
        return False, ("nonfinite_observation_or_action",)

    reasons = []
    position_error = math.sqrt(sum((obs[i] - goal[i])**2 for i in range(3)))
    if position_error > max_position_error_m:
        reasons.append("position_error")

    speed = math.sqrt(sum(obs[i]**2 for i in range(6, 9)))
    if speed > max_speed_m_s:
        reasons.append("speed")
    if math.sqrt(sum(obs[i]**2 for i in range(9, 12))) > max_body_rate_rad_s:
        reasons.append("body_rate")

    # observation order is yaw, pitch, roll after position.
    if max(abs(obs[4]), abs(obs[5])) > max_tilt_rad:
        reasons.append("tilt")
    yaw_error = (obs[3] - yaw_goal + math.pi) % (2.0 * math.pi) - math.pi
    if abs(yaw_error) > max_yaw_error_rad:
        reasons.append("yaw_branch_or_error")

    if max(abs(action[0]), abs(action[1]), abs(action[2])) > max_rate_action:
        reasons.append("rate_action")
    if abs(action[3]) > max_collective_offset:
        reasons.append("collective_action")

    return not reasons, tuple(reasons)

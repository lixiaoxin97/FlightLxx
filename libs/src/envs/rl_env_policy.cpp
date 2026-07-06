#include "rl_env.hpp"

namespace flightlib {

namespace {
constexpr Scalar kSmallNumber = 1.0e-4f;
}

bool QuadrotorEnv::getEnvPolicyObs(Ref<Vector<>> obs) const {
  if (obs.rows() != pathenv::kNEnvObs || obs.cols() != 1) {
    return false;
  }

  const Scalar path_error_ratio =
    clampScalar(last_episode_path_error_ /
                  std::max(max_tracking_error_, kSmallNumber),
                0.0, 1.0);
  const Scalar lookahead_error_ratio =
    clampScalar(last_episode_lookahead_error_ /
                  static_cast<Scalar>(4.0),
                0.0, 1.0);
  const Scalar tracking_error_ratio =
    clampScalar(static_cast<Scalar>(0.65) * path_error_ratio +
                  static_cast<Scalar>(0.35) * lookahead_error_ratio,
                0.0, 1.0);

  for (int i = 0; i < checkpoint_count_; i++) {
    const Vector<3> p = checkpointPosition(i);
    const int base = pathenv::kCheckpointState + 3 * i;
    obs(base + 0) =
      clampScalar(p.x() / std::max(max_path_length_, kSmallNumber),
                  -1.0, 1.0);
    obs(base + 1) =
      clampScalar(p.y() / std::max(max_checkpoint_xy_offset_, kSmallNumber),
                  -1.0, 1.0);
    obs(base + 2) =
      clampScalar((p.z() - base_z_) /
                    std::max(max_checkpoint_z_offset_, kSmallNumber),
                  -1.0, 1.0);
    obs(pathenv::kPassErrors + i) = tracking_error_ratio;
  }
  obs(pathenv::kPassErrors) = path_error_ratio;
  obs(pathenv::kPassErrors + 1) = lookahead_error_ratio;

  obs(pathenv::kEpisodeProgress) = last_episode_progress_;
  obs(pathenv::kCompleted) = last_episode_completed_ ? 1.0 : 0.0;
  obs(pathenv::kCrashed) = last_episode_crashed_ ? 1.0 : 0.0;
  obs(pathenv::kEnvScore) = clampScalar(last_episode_score_ / 100.0,
                                        -2.0, 2.0);
  obs(pathenv::kInfeasibleCount) =
    infeasible_reset_patience_ > 0
      ? clampScalar(static_cast<Scalar>(infeasible_counter_) /
                      static_cast<Scalar>(infeasible_reset_patience_),
                    0.0, 1.0)
      : 0.0;

  const int wind_base = pathenv::kWindState;
  obs(wind_base + 0) =
    clampScalar(wind_mean_.x() / std::max(max_wind_mean_xy_, kSmallNumber),
                -1.0, 1.0);
  obs(wind_base + 1) =
    clampScalar(wind_mean_.y() / std::max(max_wind_mean_xy_, kSmallNumber),
                -1.0, 1.0);
  obs(wind_base + 2) =
    clampScalar(wind_mean_.z() / std::max(max_wind_mean_z_, kSmallNumber),
                -1.0, 1.0);
  obs(wind_base + 3) =
    clampScalar(wind_gust_amp_.x() /
                  std::max(max_wind_gust_xy_, kSmallNumber),
                0.0, 1.0);
  obs(wind_base + 4) =
    clampScalar(wind_gust_amp_.y() /
                  std::max(max_wind_gust_xy_, kSmallNumber),
                0.0, 1.0);
  obs(wind_base + 5) =
    clampScalar(wind_gust_amp_.z() /
                  std::max(max_wind_gust_z_, kSmallNumber),
                0.0, 1.0);
  for (int i = 0; i < 3; i++) {
    obs(wind_base + 6 + i) =
      normalizeRange(wind_gust_freq_(i), min_wind_gust_freq_,
                     max_wind_gust_freq_);
    obs(wind_base + 9 + i) = std::sin(wind_gust_phase_(i));
  }
  obs(wind_base + 12) =
    clampScalar(wind_burst_amp_.x() /
                  std::max(max_wind_burst_xy_, kSmallNumber),
                -1.0, 1.0);
  obs(wind_base + 13) =
    clampScalar(wind_burst_amp_.y() /
                  std::max(max_wind_burst_xy_, kSmallNumber),
                -1.0, 1.0);
  obs(wind_base + 14) =
    clampScalar(wind_burst_amp_.z() /
                  std::max(max_wind_burst_z_, kSmallNumber),
                -1.0, 1.0);
  obs(wind_base + 15) = clampScalar(wind_burst_start_ratio_, 0.0, 1.0);
  obs(wind_base + 16) =
    normalizeRange(wind_burst_duration_, min_wind_burst_duration_,
                   max_wind_burst_duration_);
  return true;
}

Scalar QuadrotorEnv::getEnvironmentScore() const {
  return last_episode_score_;
}

int QuadrotorEnv::getEnvPolicyObsDim() const {
  return pathenv::kNEnvObs;
}

int QuadrotorEnv::getEnvPolicyActDim() const {
  return pathenv::kNEnvAct;
}

void QuadrotorEnv::applyEnvPolicyAction(const Ref<Vector<>> act) {
  if (!adaptive_path_enabled_ || useFigureEightPath() ||
      act.rows() < pathenv::kNEnvAct)
    return;

  const bool has_episode =
    last_episode_progress_ > 0.0 || last_episode_path_error_ > 0.0 ||
    last_episode_completed_ || last_episode_crashed_;
  const Scalar path_error_ratio =
    clampScalar(last_episode_path_error_ /
                  std::max(max_tracking_error_, kSmallNumber),
                0.0, 1.0);
  const Scalar lookahead_error_ratio =
    clampScalar(last_episode_lookahead_error_ /
                  static_cast<Scalar>(4.0),
                0.0, 1.0);
  const Scalar tracking_error_ratio =
    clampScalar(static_cast<Scalar>(0.65) * path_error_ratio +
                  static_cast<Scalar>(0.35) * lookahead_error_ratio,
                0.0, 1.0);
  const Scalar tracking_score =
    static_cast<Scalar>(1.0) - tracking_error_ratio;
  if (has_episode) {
    if (last_episode_completed_ || last_episode_progress_ > env_reset_min_progress_ ||
        tracking_score > static_cast<Scalar>(0.35)) {
      infeasible_counter_ = 0;
    } else {
      infeasible_counter_ += 1;
    }
  }

  if (infeasible_counter_ >= infeasible_reset_patience_) {
    resetPathToInitialDifficulty();
    resetWindToInitialDifficulty();
    infeasible_counter_ = 0;
    current_env_rank_reward_ = -1.0;
    return;
  }

  const Scalar action_clip = clampScalar(env_action_clip_, 0.0, 1.0);
  const Scalar action_denom = std::max(action_clip, kSmallNumber);
  const auto normalized_action = [&](const int index) {
    if (index < 0 || index >= act.rows()) return static_cast<Scalar>(0.0);
    return clampScalar(act(index) / action_denom, -1.0, 1.0);
  };

  const Scalar length_action = normalized_action(pathenv::kLengthAction);
  const Scalar radius_action = normalized_action(pathenv::kRadiusAction);
  const Scalar vertical_action = normalized_action(pathenv::kVerticalAction);
  const Scalar wave_action = normalized_action(pathenv::kWaveAction);
  const Scalar closed_action = normalized_action(pathenv::kClosedAction);
  const Scalar phase_action = normalized_action(pathenv::kPhaseAction);
  const auto clamp_wind_component = [&](const Scalar value, const int axis,
                                        const Scalar xy_limit,
                                        const Scalar z_limit) {
    const Scalar limit = axis == 2 ? z_limit : xy_limit;
    return clampScalar(value, -limit, limit);
  };

  const Scalar adaptive_length_limit =
    std::min(max_path_length_, static_cast<Scalar>(20.0));
  const Scalar closed_delta_scale =
    closed_action >= 0.0 ? static_cast<Scalar>(1.0)
                         : static_cast<Scalar>(0.25);
  path_shape_mix_ =
    clampScalar(path_shape_mix_ +
                  env_shape_step_ * closed_delta_scale * closed_action,
                0.0, max_path_shape_mix_);
  const Scalar closed_mix =
    clampScalar(path_shape_mix_, 0.0, max_path_shape_mix_);

  Scalar next_length =
    clampScalar(path_length_ + env_length_step_ * length_action,
                min_path_length_, adaptive_length_limit);
  next_length =
    (static_cast<Scalar>(1.0) -
     static_cast<Scalar>(0.15) * closed_mix) *
      next_length +
    static_cast<Scalar>(0.15) * closed_mix *
      clampScalar(figure_eight_length_, min_path_length_,
                  adaptive_length_limit);
  path_length_ =
    clampScalar(next_length, min_path_length_, adaptive_length_limit);

  Scalar next_lateral =
    clampScalar(lateral_amp_ + env_lateral_step_ * radius_action,
                0.0, max_lateral_amp_);
  const Scalar closed_radius_floor =
    closed_mix * clampScalar(figure_eight_radius_, 0.0, max_lateral_amp_);
  lateral_amp_ =
    clampScalar(std::max(next_lateral, closed_radius_floor), 0.0,
                max_lateral_amp_);

  Scalar next_vertical =
    clampScalar(vertical_amp_ + env_vertical_step_ * vertical_action,
                0.0, max_vertical_amp_);
  const Scalar closed_vertical_floor =
    static_cast<Scalar>(0.25) * closed_mix * max_vertical_amp_;
  vertical_amp_ =
    clampScalar(std::max(next_vertical, closed_vertical_floor), 0.0,
                max_vertical_amp_);

  wave_count_ =
    clampScalar(wave_count_ + env_wave_step_ * wave_action,
                min_wave_count_, max_wave_count_);
  phase_y_ += static_cast<Scalar>(0.5) * env_wave_step_ * phase_action;
  phase_z_ += env_wave_step_ * phase_action;

  if (wind_enabled_) {
    const Scalar wind_action_scale = static_cast<Scalar>(1.0);
    for (int axis = 0; axis < 3; axis++) {
      const Scalar mean_action =
        normalized_action(pathenv::kWindMeanAction + axis);
      wind_mean_(axis) = clamp_wind_component(
        wind_mean_(axis) +
          env_wind_mean_step_ * wind_action_scale * mean_action,
        axis, max_wind_mean_xy_, max_wind_mean_z_);

      const Scalar gust_action =
        normalized_action(pathenv::kWindGustAmpAction + axis);
      const Scalar gust_limit =
        axis == 2 ? max_wind_gust_z_ : max_wind_gust_xy_;
      wind_gust_amp_(axis) =
        clampScalar(wind_gust_amp_(axis) +
                      env_wind_gust_step_ * wind_action_scale * gust_action,
                    0.0, gust_limit);

      const Scalar freq_action =
        normalized_action(pathenv::kWindGustFreqAction + axis);
      wind_gust_freq_(axis) =
        clampScalar(wind_gust_freq_(axis) +
                      env_wind_freq_step_ * wind_action_scale * freq_action,
                    min_wind_gust_freq_, max_wind_gust_freq_);

      const Scalar phase_wind_action =
        normalized_action(pathenv::kWindGustPhaseAction + axis);
      wind_gust_phase_(axis) +=
        static_cast<Scalar>(0.50) * EIGEN_PI * wind_action_scale *
        phase_wind_action;

      const Scalar burst_action =
        normalized_action(pathenv::kWindBurstAmpAction + axis);
      wind_burst_amp_(axis) = clamp_wind_component(
        wind_burst_amp_(axis) +
          env_wind_burst_step_ * wind_action_scale * burst_action,
        axis, max_wind_burst_xy_, max_wind_burst_z_);
    }

    wind_burst_start_ratio_ =
      clampScalar(wind_burst_start_ratio_ +
                    env_wind_burst_time_step_ * wind_action_scale *
                      normalized_action(pathenv::kWindBurstStartAction),
                  static_cast<Scalar>(0.10), static_cast<Scalar>(0.85));
    wind_burst_duration_ =
      clampScalar(wind_burst_duration_ +
                    (max_wind_burst_duration_ -
                     min_wind_burst_duration_) *
                      env_wind_burst_time_step_ * wind_action_scale *
                      normalized_action(pathenv::kWindBurstDurationAction),
                  min_wind_burst_duration_, max_wind_burst_duration_);
    enforceWindBudgets();
    updateWindField();
  }

  const Scalar nominal_spacing =
    path_length_ / static_cast<Scalar>(std::max(checkpoint_count_, 1));
  const Scalar spacing_scale = static_cast<Scalar>(1.0);
  const Scalar lateral_budget = max_checkpoint_xy_offset_;
  const Scalar vertical_budget = max_checkpoint_z_offset_;
  const Scalar local_x_budget =
    static_cast<Scalar>(0.35) * nominal_spacing;
  const Scalar closed_x_budget =
    static_cast<Scalar>(0.35) * nominal_spacing;
  const Scalar closed_y_budget =
    static_cast<Scalar>(0.25) * max_checkpoint_xy_offset_;
  const Scalar closed_z_budget =
    static_cast<Scalar>(0.25) * max_checkpoint_z_offset_;
  const Scalar min_segment =
    std::max(static_cast<Scalar>(0.60) * checkpoint_radius_,
             static_cast<Scalar>(0.25) * nominal_spacing * spacing_scale);

  Matrix<3, pathenv::kNCheckpoints> desired_offsets =
    Matrix<3, pathenv::kNCheckpoints>::Zero();
  Scalar previous_x = 0.0;

  for (int i = 0; i < checkpoint_count_; i++) {
    const int base = 3 * i;
    const Scalar ax = normalized_action(base + 0);
    const Scalar ay = normalized_action(base + 1);
    const Scalar az = normalized_action(base + 2);

    const Vector<3> base_pos = checkpointBasePosition(i);
    Scalar desired_x =
      nominal_spacing * static_cast<Scalar>(i + 1) * spacing_scale +
      ax * local_x_budget;
    desired_x =
      clampScalar(desired_x, previous_x + min_segment,
                  adaptive_length_limit);
    previous_x = desired_x;

    const Scalar desired_y = ay * lateral_budget;
    const Scalar desired_z = base_z_ + az * vertical_budget;

    const Scalar open_offset_x = desired_x - base_pos.x();
    const Scalar open_offset_y = desired_y - base_pos.y();
    const Scalar open_offset_z = desired_z - base_pos.z();
    const Scalar closed_offset_x = ax * closed_x_budget;
    const Scalar closed_offset_y = ay * closed_y_budget;
    const Scalar closed_offset_z = az * closed_z_budget;

    desired_offsets(0, i) =
      clampScalar((static_cast<Scalar>(1.0) - closed_mix) * open_offset_x +
                    closed_mix * closed_offset_x,
                  -max_checkpoint_x_offset_,
                  max_checkpoint_x_offset_);
    desired_offsets(1, i) =
      clampScalar((static_cast<Scalar>(1.0) - closed_mix) * open_offset_y +
                    closed_mix * closed_offset_y,
                  -max_checkpoint_xy_offset_,
                  max_checkpoint_xy_offset_);
    desired_offsets(2, i) =
      clampScalar((static_cast<Scalar>(1.0) - closed_mix) * open_offset_z +
                    closed_mix * closed_offset_z,
                  -max_checkpoint_z_offset_,
                  max_checkpoint_z_offset_);
  }

  const Scalar apply_rate = static_cast<Scalar>(0.55);
  checkpoint_offsets_ =
    (static_cast<Scalar>(1.0) - apply_rate) * checkpoint_offsets_ +
    apply_rate * desired_offsets;
  rebuildSampledPath();

  for (int i = 0; i < 3 &&
                  sampled_path_length_ >
                    adaptive_length_limit + static_cast<Scalar>(0.05);
       i++) {
    const Scalar scale =
      clampScalar(adaptive_length_limit /
                    std::max(sampled_path_length_, kSmallNumber),
                  static_cast<Scalar>(0.65), static_cast<Scalar>(0.98));
    path_length_ =
      clampScalar(path_length_ * scale, min_path_length_,
                  adaptive_length_limit);
    lateral_amp_ =
      clampScalar(lateral_amp_ * scale, 0.0, max_lateral_amp_);
    vertical_amp_ =
      clampScalar(vertical_amp_ * scale, 0.0, max_vertical_amp_);
    wave_count_ =
      clampScalar(static_cast<Scalar>(1.0) +
                    (wave_count_ - static_cast<Scalar>(1.0)) * scale,
                  min_wave_count_, max_wave_count_);
    checkpoint_offsets_ *= scale;
    rebuildSampledPath();
  }

  const Scalar frontier_center = 0.60;
  const Scalar frontier_width = 0.55;
  const Scalar follow_score =
    clampScalar(static_cast<Scalar>(0.65) * tracking_score +
                  static_cast<Scalar>(0.35) * last_episode_progress_,
                0.0, 1.0);
  current_env_rank_reward_ =
    has_episode
      ? clampScalar(static_cast<Scalar>(1.0) -
                      std::abs(follow_score - frontier_center) /
                        std::max(frontier_width, kSmallNumber),
                    static_cast<Scalar>(-1.0), static_cast<Scalar>(1.0))
      : 0.0;
}

void QuadrotorEnv::applyPathEnvironmentPolicy(const int rank, const int num_envs,
                                              const int lower_rank,
                                              const int upper_rank) {
  if (!adaptive_path_enabled_ || useFigureEightPath()) return;

  if (infeasible_counter_ >= infeasible_reset_patience_) {
    resetPathToInitialDifficulty();
    resetWindToInitialDifficulty();
    infeasible_counter_ = 0;
    current_env_rank_reward_ = 0.0;
    return;
  }

  const int last_rank = std::max(0, num_envs - 1);
  Scalar rank_action = 0.0;

  if (num_envs <= 1) {
    rank_action = env_policy_noise_ * uniform_dist_(random_gen_);
    current_env_rank_reward_ = env_policy_reward_scale_;
  } else if (rank < lower_rank) {
    const Scalar denom = std::max(1, lower_rank);
    rank_action = env_policy_step_ * (1.0 - static_cast<Scalar>(rank) / denom);
    current_env_rank_reward_ =
      env_policy_reward_scale_ * static_cast<Scalar>(rank) / denom;
  } else if (rank > upper_rank) {
    const Scalar denom = std::max(1, last_rank - upper_rank);
    rank_action =
      -env_policy_step_ *
      (static_cast<Scalar>(rank - upper_rank) / static_cast<Scalar>(denom));
    current_env_rank_reward_ =
      env_policy_reward_scale_ *
      static_cast<Scalar>(last_rank - rank) / static_cast<Scalar>(denom);
  } else {
    rank_action = 0.20 * env_policy_step_ * uniform_dist_(random_gen_);
    current_env_rank_reward_ = env_policy_reward_scale_;
  }

  const Scalar noise = env_policy_noise_ * uniform_dist_(random_gen_);
  for (int i = 1; i < checkpoint_count_; i++) {
    const Scalar sign = (i % 2 == 0) ? 1.0 : -1.0;
    checkpoint_offsets_(1, i) =
      clampScalar(checkpoint_offsets_(1, i) +
                    sign * env_checkpoint_xy_step_ * rank_action + noise,
                  -max_checkpoint_xy_offset_, max_checkpoint_xy_offset_);
    checkpoint_offsets_(2, i) =
      clampScalar(checkpoint_offsets_(2, i) +
                    env_checkpoint_z_step_ * 0.5 * rank_action,
                  -max_checkpoint_z_offset_, max_checkpoint_z_offset_);
  }

  phase_y_ += 0.25 * uniform_dist_(random_gen_);
  phase_z_ += 0.25 * uniform_dist_(random_gen_);
  rebuildSampledPath();
}

}  // namespace flightlib

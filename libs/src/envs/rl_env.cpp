#include "rl_env.hpp"

namespace flightlib {

// Main reset/step/reward logic. Path geometry is in rl_env_path.cpp, YAML
// loading is in rl_env_config.cpp, and environment-policy actions are in
// rl_env_policy.cpp.

namespace {
constexpr Scalar kSmallNumber = 1.0e-4f;
}

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(std::string(getenv("FlightLxx_PATH") ? getenv("FlightLxx_PATH")
                                                       : ".") +
                 "/libs/config/rl_env.yaml") {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase() {
  cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();

  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  nominal_dynamics_ = dynamics;
  const char *verbose_dynamics = std::getenv("FlightLxx_VERBOSE_DYNAMICS");
  if (verbose_dynamics && std::string(verbose_dynamics) == "1") {
    dynamics.showQuadrotorDynamicsParams();
  }
  quadrotor_ptr_->updateDynamics(dynamics);

  loadParam(cfg_);
  resetPathToInitialDifficulty();
  resetWindToInitialDifficulty();

  world_box_ << -10.0, max_path_length_ + 10.0, -max_lateral_amp_ - 10.0,
    max_lateral_amp_ + 10.0, 0.0, base_z_ + max_vertical_amp_ + 10.0;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set world box");
  }

  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;

  const Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<quadenv::kNAct>::Zero();
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 2.0;

  quad_obs_.setZero();
  quad_act_.setZero();
  prev_act_norm_.setZero();
  prev_prev_act_norm_.setZero();
  delayed_act_norm_.setZero();
  fillActionDelayBuffer();
  fillObservationDelayBuffer();
  checkpoint_pass_errors_ *= max_tracking_error_;
  last_checkpoint_pass_errors_ *= max_tracking_error_;
  updateExtraInfo();
}

QuadrotorEnv::~QuadrotorEnv() {}

void QuadrotorEnv::setSeed(const int seed) {
  std::srand(seed);
  random_gen_.seed(seed);
}

void QuadrotorEnv::resetEpisodeState() {
  elapsed_time_ = 0.0;
  progress_s_ = 0.0;
  previous_progress_s_ = 0.0;
  current_pos_error_ = 0.0;
  current_target_error_ = 0.0;
  current_forward_speed_ = 0.0;
  current_approach_speed_ = 0.0;
  current_alive_reward_ = 0.0;
  current_advance_reward_ = 0.0;
  current_path_reward_ = 0.0;
  current_lookahead_reward_ = 0.0;
  current_speed_reward_ = 0.0;
  current_progress_reward_ = 0.0;
  current_progress_overspeed_reward_ = 0.0;
  current_pass_reward_ = 0.0;
  current_body_rate_reward_ = 0.0;
  current_act_reward_ = 0.0;
  current_act_second_delta_reward_ = 0.0;
  current_thrust_switch_reward_ = 0.0;
  current_reversal_deadzone_reward_ = 0.0;
  current_crash_reward_ = 0.0;
  current_lookahead_error_ = 0.0;
  current_track_gate_ = 0.0;
  current_delta_s_ = 0.0;
  previous_lookahead_error_ = 0.0;
  current_speed_over_ = 0.0;
  episode_pos_error_sum_ = 0.0;
  episode_lookahead_error_sum_ = 0.0;
  episode_advance_reward_sum_ = 0.0;
  episode_progress_ = 0.0;
  previous_target_distance_ = 0.0;
  previous_position_.setZero();
  previous_lookahead_position_.setZero();
  episode_steps_ = 0;
  current_checkpoint_idx_ = 0;
  passed_checkpoint_count_ = 0;
  checkpoint_pass_errors_.setOnes();
  checkpoint_pass_errors_ *= max_tracking_error_;
  completed_ = false;
  crashed_ = false;
  timed_out_ = false;
  path_error_terminal_ = false;
  ground_crash_ = false;
}

void QuadrotorEnv::computeObservation(Vector<quadenv::kNObs> *obs) {
  quadrotor_ptr_->getState(&quad_state_);

  const Scalar obs_s = clampScalar(progress_s_, 0.0, sampled_path_length_);
  const Vector<3> closest_pos = pathPosition(obs_s);
  const Vector<3> lookahead_near =
    pathPosition(obs_s + reward_cfg_.lookahead_near_distance);
  const Vector<3> tangent = pathTangent(obs_s);
  const Matrix<3, 3> rot = quad_state_.R();

  *obs << rot.col(0), rot.col(1), quad_state_.v, quad_state_.w,
    prev_act_norm_, closest_pos - quad_state_.p,
    lookahead_near - quad_state_.p, tangent;
}

void QuadrotorEnv::fillObservationDelayBuffer() {
  computeObservation(&quad_obs_);
  for (auto &buffer_obs : obs_delay_buffer_) {
    buffer_obs = quad_obs_;
  }
}

void QuadrotorEnv::pushObservationDelayBuffer() {
  computeObservation(&quad_obs_);
  for (int i = kMaxDelaySteps; i > 0; --i) {
    obs_delay_buffer_[i] = obs_delay_buffer_[i - 1];
  }
  obs_delay_buffer_[0] = quad_obs_;
}

void QuadrotorEnv::fillActionDelayBuffer() {
  for (auto &buffer_act : act_delay_buffer_) {
    buffer_act = prev_act_norm_;
  }
}

Vector<quadenv::kNAct> QuadrotorEnv::delayedAction(
  const Vector<quadenv::kNAct> &policy_act) {
  if (episode_steps_ == 0) {
    for (auto &buffer_act : act_delay_buffer_) {
      buffer_act = policy_act;
    }
  } else {
    for (int i = kMaxDelaySteps; i > 0; --i) {
      act_delay_buffer_[i] = act_delay_buffer_[i - 1];
    }
    act_delay_buffer_[0] = policy_act;
  }
  const int delay =
    std::max(0, std::min(current_action_delay_steps_, kMaxDelaySteps));
  return act_delay_buffer_[delay];
}

void QuadrotorEnv::randomizeDynamicsForEpisode(const bool random) {
  current_mass_scale_ = 1.0;
  current_inertia_scale_ = 1.0;
  current_thrust_scale_ = 1.0;
  current_negative_thrust_scale_ = 1.0;
  current_motor_tau_scale_ = 1.0;
  current_motor_reversal_dead_time_scale_ = 1.0;
  current_motor_deadzone_omega_scale_ = 1.0;
  current_drag_scale_ = 1.0;
  current_action_delay_steps_ = 0;
  current_obs_delay_steps_ = 0;

  QuadrotorDynamics dynamics = nominal_dynamics_;
  if (domain_randomization_enabled_ && random) {
    const auto sample_scale = [&](const Scalar half_width) {
      return std::max(static_cast<Scalar>(0.05),
                      static_cast<Scalar>(1.0) +
                        half_width * uniform_dist_(random_gen_));
    };
    current_mass_scale_ = sample_scale(dr_mass_scale_);
    current_inertia_scale_ = sample_scale(dr_inertia_scale_);
    current_thrust_scale_ = sample_scale(dr_thrust_scale_);
    current_negative_thrust_scale_ =
      sample_scale(dr_negative_thrust_scale_);
    current_motor_tau_scale_ = sample_scale(dr_motor_tau_scale_);
    current_motor_reversal_dead_time_scale_ =
      sample_scale(dr_motor_reversal_dead_time_scale_);
    current_motor_deadzone_omega_scale_ =
      sample_scale(dr_motor_deadzone_omega_scale_);
    current_drag_scale_ = sample_scale(dr_drag_scale_);

    dynamics.setMass(nominal_dynamics_.getMass() * current_mass_scale_);
    dynamics.scaleInertia(Vector<3>(
      sample_scale(dr_inertia_scale_), sample_scale(dr_inertia_scale_),
      sample_scale(dr_inertia_scale_)));
    dynamics.scaleThrustMap(current_thrust_scale_,
                            current_negative_thrust_scale_);
    dynamics.setMotortauInv(
      nominal_dynamics_.getMotorTauInv() /
      std::max(current_motor_tau_scale_, kSmallNumber));
    dynamics.setMotorReversalDeadTime(
      nominal_dynamics_.getMotorReversalDeadTime() *
      current_motor_reversal_dead_time_scale_);
    dynamics.setMotorDeadzoneOmega(
      nominal_dynamics_.getMotorDeadzoneOmega() *
      current_motor_deadzone_omega_scale_);

    if (max_action_delay_steps_ > 0) {
      std::uniform_int_distribution<int> action_delay_dist(
        0, std::min(max_action_delay_steps_, kMaxDelaySteps));
      current_action_delay_steps_ = action_delay_dist(random_gen_);
    }
    if (max_obs_delay_steps_ > 0) {
      std::uniform_int_distribution<int> obs_delay_dist(
        0, std::min(max_obs_delay_steps_, kMaxDelaySteps));
      current_obs_delay_steps_ = obs_delay_dist(random_gen_);
    }
  }

  quadrotor_ptr_->updateDynamics(dynamics);
  const Scalar mass = quadrotor_ptr_->getMass();
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 2.0;
}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  evaluatePreviousEpisode();

  randomizeDynamicsForEpisode(random);
  quad_state_.setZero();
  quad_act_.setZero();
  prev_act_norm_.setZero();
  prev_prev_act_norm_.setZero();
  delayed_act_norm_.setZero();
  fillActionDelayBuffer();
  resetEpisodeState();
  episode_wind_sign_ =
    random && wind_enabled_ && uniform_dist_(random_gen_) < 0.0 ? -1.0 : 1.0;
  updateWindField();

  if (useFigureEightPath()) {
    resetPathToInitialDifficulty();
    resetWindToInitialDifficulty();
  } else if (random) {
    phase_y_ += 0.10 * uniform_dist_(random_gen_);
    phase_z_ += 0.10 * uniform_dist_(random_gen_);
    rebuildSampledPath();
  }

  const Scalar start_s = 0.0;
  progress_s_ = start_s;
  previous_progress_s_ = start_s;
  episode_progress_ = 0.0;

  const Vector<3> tangent = pathTangent(start_s);
  const Vector<3> world_up(0.0, 0.0, 1.0);
  Vector<3> side = world_up.cross(tangent);
  if (side.norm() <= kSmallNumber) {
    side = Vector<3>(0.0, 1.0, 0.0);
  } else {
    side.normalize();
  }

  const Vector<3> start_pos = pathPosition(start_s);
  const Scalar lateral_offset =
    random ? reset_position_xy_noise_ * uniform_dist_(random_gen_) : 0.0;
  const Scalar vertical_offset =
    random ? reset_position_z_noise_ * uniform_dist_(random_gen_) : 0.0;
  quad_state_.p = start_pos + lateral_offset * side +
                  vertical_offset * world_up;

  const Scalar forward_speed =
    std::max(static_cast<Scalar>(0.0),
             target_speed_ +
               (random ? reset_forward_speed_noise_ *
                           uniform_dist_(random_gen_)
                       : static_cast<Scalar>(0.0)));
  quad_state_.v =
    tangent * forward_speed +
    (random ? reset_lateral_speed_noise_ * uniform_dist_(random_gen_) : 0.0) *
      side +
    (random ? reset_vertical_speed_noise_ * uniform_dist_(random_gen_) : 0.0) *
      world_up;

  const Scalar yaw =
    std::atan2(tangent.y(), tangent.x()) +
    (random ? reset_yaw_noise_ * uniform_dist_(random_gen_) : 0.0);
  const Scalar pitch =
    random ? reset_roll_pitch_noise_ * uniform_dist_(random_gen_) : 0.0;
  const Scalar roll =
    random ? reset_roll_pitch_noise_ * uniform_dist_(random_gen_) : 0.0;
  const Quaternion attitude =
    Eigen::AngleAxis<Scalar>(yaw, Vector<3>::UnitZ()) *
    Eigen::AngleAxis<Scalar>(pitch, Vector<3>::UnitY()) *
    Eigen::AngleAxis<Scalar>(roll, Vector<3>::UnitX());
  quad_state_.q(attitude.normalized());

  quad_state_.w << (random ? reset_body_rate_noise_ *
                                uniform_dist_(random_gen_)
                            : 0.0),
    (random ? reset_body_rate_noise_ * uniform_dist_(random_gen_) : 0.0),
    (random ? reset_body_rate_noise_ * uniform_dist_(random_gen_) : 0.0);

  quadrotor_ptr_->reset(quad_state_);
  updateWindField();
  previous_position_ = quad_state_.p;
  previous_target_distance_ = (quad_state_.p - pathPosition(progress_s_)).norm();
  previous_lookahead_position_ =
    pathPosition(progress_s_ + reward_cfg_.lookahead_near_distance);
  const Scalar initial_lookahead_distance =
    std::max(reward_cfg_.lookahead_near_distance, kSmallNumber);
  previous_lookahead_error_ =
    (quad_state_.p - previous_lookahead_position_).norm() /
    initial_lookahead_distance;
  current_lookahead_error_ = previous_lookahead_error_;

  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  fillObservationDelayBuffer();
  getObs(obs);
  updateExtraInfo();
  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  const int delay =
    std::max(0, std::min(current_obs_delay_steps_, kMaxDelaySteps));
  obs.segment<quadenv::kNObs>(quadenv::kObs) = obs_delay_buffer_[delay];
  return true;
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  const Scalar near_distance =
    std::max(reward_cfg_.lookahead_near_distance, kSmallNumber);

  const Vector<quadenv::kNAct> policy_act = act;
  delayed_act_norm_ = delayedAction(policy_act);
  quad_act_ = delayed_act_norm_.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

  updateWindField();
  quadrotor_ptr_->run(cmd_, sim_dt_);
  elapsed_time_ += sim_dt_;

  quadrotor_ptr_->getState(&quad_state_);

  const Scalar closest_s = closestPathProgress(quad_state_.p);
  progress_s_ = std::max(previous_progress_s_, closest_s);
  progress_s_ = clampScalar(progress_s_, 0.0, sampled_path_length_);

  const Vector<3> closest_pos = pathPosition(progress_s_);
  const Vector<3> lookahead_near =
    pathPosition(progress_s_ + reward_cfg_.lookahead_near_distance);
  const Vector<3> tangent = pathTangent(progress_s_);
  const Vector<3> pos_error = quad_state_.p - closest_pos;

  current_pos_error_ = pos_error.norm();
  current_target_error_ = current_pos_error_;
  const Scalar signed_forward_speed = quad_state_.v.dot(tangent);
  current_forward_speed_ =
    std::max(static_cast<Scalar>(0.0), signed_forward_speed);
  current_speed_over_ =
    std::max(static_cast<Scalar>(0.0),
             current_forward_speed_ - reward_cfg_.speed_limit);
  current_crash_reward_ =
    quad_state_.x(QS::POSZ) <= 0.2 ? reward_cfg_.crash_penalty : 0.0;

  const Scalar dt = std::max(sim_dt_, kSmallNumber);
  current_delta_s_ =
    std::max(static_cast<Scalar>(0.0), progress_s_ - previous_progress_s_);
  current_approach_speed_ = current_delta_s_ / dt;

  const Scalar path_error_n =
    current_pos_error_ / std::max(reward_cfg_.path_error_scale, kSmallNumber);
  current_lookahead_error_ =
    (quad_state_.p - lookahead_near).norm() / near_distance;

  const Scalar body_rate_n =
    quad_state_.w.norm() /
    std::max(reward_cfg_.body_rate_scale, kSmallNumber);
  const Scalar act_delta_n =
    (policy_act - prev_act_norm_).norm() /
    std::max(reward_cfg_.act_delta_scale, kSmallNumber);
  const Scalar act_second_delta_n =
    (policy_act - static_cast<Scalar>(2.0) * prev_act_norm_ +
     prev_prev_act_norm_)
      .norm() /
    std::max(reward_cfg_.act_second_delta_scale, kSmallNumber);
  int thrust_switch_count = 0;
  for (int i = 0; i < quadenv::kNAct; ++i) {
    if (std::abs(policy_act(i)) > static_cast<Scalar>(0.05) &&
        std::abs(prev_act_norm_(i)) > static_cast<Scalar>(0.05) &&
        policy_act(i) * prev_act_norm_(i) < 0.0) {
      thrust_switch_count += 1;
    }
  }
  const Scalar thrust_switch_n =
    static_cast<Scalar>(thrust_switch_count) /
    static_cast<Scalar>(quadenv::kNAct);
  const Scalar speed_over_n =
    current_speed_over_ / std::max(reward_cfg_.speed_limit, kSmallNumber);
  const Scalar max_penalty =
    std::max(reward_cfg_.max_penalty, static_cast<Scalar>(0.0));

  current_alive_reward_ = reward_cfg_.alive_reward;
  const Scalar advance_ref =
    std::max(reward_cfg_.advance_ref_speed, kSmallNumber) * dt;
  const Scalar advance_n =
    clampScalar(current_delta_s_ / std::max(advance_ref, kSmallNumber),
                0.0, 1.0);
  const Scalar track_gate_sigma =
    std::max(reward_cfg_.track_gate_sigma, kSmallNumber);
  current_track_gate_ =
    std::exp(-current_pos_error_ * current_pos_error_ /
             (track_gate_sigma * track_gate_sigma));
  current_advance_reward_ =
    reward_cfg_.advance_coeff * advance_n * current_track_gate_;
  current_path_reward_ =
    -reward_cfg_.path_coeff *
    clampScalar(path_error_n * path_error_n, 0.0, max_penalty);
  current_lookahead_reward_ = 0.0;
  current_speed_reward_ =
    -reward_cfg_.speed_coeff *
    clampScalar(speed_over_n * speed_over_n, 0.0, max_penalty);
  current_progress_reward_ = 0.0;
  current_progress_overspeed_reward_ = 0.0;
  current_pass_reward_ = 0.0;
  current_body_rate_reward_ =
    -reward_cfg_.body_rate_coeff *
    clampScalar(body_rate_n * body_rate_n, 0.0, max_penalty);
  current_act_reward_ =
    -reward_cfg_.act_coeff *
    clampScalar(act_delta_n * act_delta_n, 0.0, max_penalty);
  current_act_second_delta_reward_ =
    -reward_cfg_.act_second_delta_coeff *
    clampScalar(act_second_delta_n * act_second_delta_n, 0.0, max_penalty);
  current_thrust_switch_reward_ =
    -reward_cfg_.thrust_switch_coeff * thrust_switch_n;
  const Scalar reversal_active_ratio =
    quadrotor_ptr_ ? quadrotor_ptr_->getMotorReversalActiveRatio() : 0.0;
  current_reversal_deadzone_reward_ =
    -reward_cfg_.reversal_deadzone_coeff * reversal_active_ratio;

  const Scalar total_reward =
    current_alive_reward_ + current_advance_reward_ + current_path_reward_ +
    current_body_rate_reward_ +
    current_act_reward_ + current_act_second_delta_reward_ +
    current_thrust_switch_reward_ + current_reversal_deadzone_reward_ +
    current_speed_reward_ + current_crash_reward_;

  episode_progress_ =
    clampScalar(progress_s_ / std::max(sampled_path_length_, kSmallNumber),
                0.0, 1.0);
  passed_checkpoint_count_ =
    std::min(checkpoint_count_,
             static_cast<int>(std::floor(episode_progress_ *
                                         static_cast<Scalar>(
                                           checkpoint_count_))));
  current_checkpoint_idx_ =
    std::min(passed_checkpoint_count_, checkpoint_count_ - 1);
  completed_ =
    progress_s_ >= sampled_path_length_ - finish_margin_ &&
    current_pos_error_ <= finish_error_;

  prev_prev_act_norm_ = prev_act_norm_;
  prev_act_norm_ = policy_act;
  previous_progress_s_ = progress_s_;
  episode_pos_error_sum_ += current_pos_error_;
  episode_lookahead_error_sum_ += current_lookahead_error_;
  episode_advance_reward_sum_ += current_advance_reward_;
  episode_steps_ += 1;
  previous_position_ = quad_state_.p;
  previous_target_distance_ = current_pos_error_;
  previous_lookahead_error_ = current_lookahead_error_;
  previous_lookahead_position_ =
    pathPosition(progress_s_ + reward_cfg_.lookahead_near_distance);

  pushObservationDelayBuffer();
  getObs(obs);
  updateExtraInfo();
  return total_reward;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  reward = 0.0;

  if (completed_) {
    reward = reward_cfg_.success_reward;
    return true;
  }

  if (quad_state_.x(QS::POSZ) <= 0.2) {
    crashed_ = true;
    ground_crash_ = true;
    reward =
      current_crash_reward_ < 0.0 ? 0.0 : reward_cfg_.crash_penalty;
    return true;
  }

  if (current_target_error_ > max_tracking_error_) {
    crashed_ = true;
    path_error_terminal_ = true;
    reward = reward_cfg_.terminal_penalty;
    return true;
  }

  if (elapsed_time_ >= max_t_) {
    timed_out_ = true;
    reward = 0.0;
    return true;
  }

  return false;
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorEnv::updateExtraInfo() {
  Scalar checkpoint_x_min = 0.0;
  Scalar checkpoint_x_max = 0.0;
  Scalar checkpoint_y_min = 0.0;
  Scalar checkpoint_y_max = 0.0;
  Scalar checkpoint_z_min = 0.0;
  Scalar checkpoint_z_max = 0.0;
  Scalar checkpoint_turn_sum = 0.0;
  int checkpoint_turn_count = 0;
  int checkpoint_reverse_count = 0;
  int checkpoint_segment_count = 0;
  Vector<3> previous_checkpoint = Vector<3>::Zero();
  Vector<3> previous_direction = Vector<3>::Zero();
  bool has_previous_checkpoint = false;
  bool has_previous_direction = false;

  for (int i = 0; i < checkpoint_count_; i++) {
    const Vector<3> checkpoint = checkpointPosition(i);
    if (i == 0) {
      checkpoint_x_min = checkpoint.x();
      checkpoint_x_max = checkpoint.x();
      checkpoint_y_min = checkpoint.y();
      checkpoint_y_max = checkpoint.y();
      checkpoint_z_min = checkpoint.z();
      checkpoint_z_max = checkpoint.z();
    } else {
      checkpoint_x_min = std::min(checkpoint_x_min, checkpoint.x());
      checkpoint_x_max = std::max(checkpoint_x_max, checkpoint.x());
      checkpoint_y_min = std::min(checkpoint_y_min, checkpoint.y());
      checkpoint_y_max = std::max(checkpoint_y_max, checkpoint.y());
      checkpoint_z_min = std::min(checkpoint_z_min, checkpoint.z());
      checkpoint_z_max = std::max(checkpoint_z_max, checkpoint.z());
    }

    if (has_previous_checkpoint) {
      const Vector<3> segment = checkpoint - previous_checkpoint;
      const Scalar segment_norm = segment.norm();
      if (segment_norm > kSmallNumber) {
        checkpoint_segment_count += 1;
        if (segment.x() < -static_cast<Scalar>(0.02)) {
          checkpoint_reverse_count += 1;
        }
        const Vector<3> direction = segment / segment_norm;
        if (has_previous_direction) {
          checkpoint_turn_sum += (direction - previous_direction).norm();
          checkpoint_turn_count += 1;
        }
        previous_direction = direction;
        has_previous_direction = true;
      }
    }

    previous_checkpoint = checkpoint;
    has_previous_checkpoint = true;
  }

  const Scalar checkpoint_x_span = checkpoint_x_max - checkpoint_x_min;
  const Scalar checkpoint_y_span = checkpoint_y_max - checkpoint_y_min;
  const Scalar checkpoint_z_span = checkpoint_z_max - checkpoint_z_min;
  const Scalar checkpoint_reverse_ratio =
    checkpoint_segment_count > 0
      ? static_cast<Scalar>(checkpoint_reverse_count) /
          static_cast<Scalar>(checkpoint_segment_count)
      : 0.0;
  const Scalar checkpoint_closure_error =
    (checkpointPosition(checkpoint_count_ - 1) - basePathPosition(0.0)).norm();
  const Scalar checkpoint_closure_score =
    static_cast<Scalar>(1.0) -
    clampScalar(checkpoint_closure_error /
                  std::max(figure_eight_radius_ + max_vertical_amp_,
                           kSmallNumber),
                0.0, 1.0);
  const Scalar checkpoint_mean_turn =
    checkpoint_turn_count > 0
      ? checkpoint_turn_sum / static_cast<Scalar>(checkpoint_turn_count)
      : 0.0;
  const Scalar checkpoint_complexity =
    clampScalar(
      static_cast<Scalar>(0.22) *
          clampScalar(checkpoint_y_span /
                        std::max(max_checkpoint_xy_offset_, kSmallNumber),
                      0.0, 1.0) +
        static_cast<Scalar>(0.14) *
          clampScalar(checkpoint_z_span /
                        std::max(max_checkpoint_z_offset_, kSmallNumber),
                      0.0, 1.0) +
        static_cast<Scalar>(0.32) *
          clampScalar(checkpoint_mean_turn / static_cast<Scalar>(0.75),
                      0.0, 1.0) +
        static_cast<Scalar>(0.20) *
          clampScalar(checkpoint_reverse_ratio / static_cast<Scalar>(0.35),
                      0.0, 1.0) +
        static_cast<Scalar>(0.12) * checkpoint_closure_score *
          clampScalar(path_shape_mix_, 0.0, 1.0),
      0.0, 1.0);

  extra_info_["path_error"] = current_pos_error_;
  extra_info_["lookahead_error"] = current_lookahead_error_;
  extra_info_["track_gate"] = current_track_gate_;
  extra_info_["delta_s"] = current_delta_s_;
  extra_info_["forward_speed"] = current_forward_speed_;
  extra_info_["approach_speed"] = current_approach_speed_;
  extra_info_["progress"] = episode_progress_;
  extra_info_["episode_steps"] = static_cast<Scalar>(episode_steps_);
  extra_info_["completed"] = completed_ ? 1.0 : 0.0;
  extra_info_["crashed"] = crashed_ ? 1.0 : 0.0;
  extra_info_["timed_out"] = timed_out_ ? 1.0 : 0.0;
  extra_info_["path_error_terminal"] = path_error_terminal_ ? 1.0 : 0.0;
  extra_info_["ground_crash"] = ground_crash_ ? 1.0 : 0.0;
  extra_info_["path_length"] = sampled_path_length_;
  extra_info_["checkpoint_x_span"] = checkpoint_x_span;
  extra_info_["checkpoint_y_span"] = checkpoint_y_span;
  extra_info_["checkpoint_z_span"] = checkpoint_z_span;
  extra_info_["checkpoint_mean_turn"] = checkpoint_mean_turn;
  extra_info_["checkpoint_reverse_ratio"] = checkpoint_reverse_ratio;
  extra_info_["checkpoint_closure_error"] = checkpoint_closure_error;
  extra_info_["checkpoint_closure_score"] = checkpoint_closure_score;
  extra_info_["checkpoint_complexity"] = checkpoint_complexity;
  extra_info_["lateral_amp"] = lateral_amp_;
  extra_info_["vertical_amp"] = vertical_amp_;
  extra_info_["wave_count"] = wave_count_;
  extra_info_["shape_mix"] = path_shape_mix_;
  extra_info_["figure_eight_mode"] = useFigureEightPath() ? 1.0 : 0.0;
  extra_info_["wind_x"] = current_wind_.x();
  extra_info_["wind_y"] = current_wind_.y();
  extra_info_["wind_z"] = current_wind_.z();
  const Vector<3> signed_wind_mean = episode_wind_sign_ * wind_mean_;
  const Vector<3> signed_wind_burst_amp = episode_wind_sign_ * wind_burst_amp_;
  extra_info_["wind_mean_x"] = signed_wind_mean.x();
  extra_info_["wind_mean_y"] = signed_wind_mean.y();
  extra_info_["wind_mean_z"] = signed_wind_mean.z();
  extra_info_["wind_gust_amp_x"] = wind_gust_amp_.x();
  extra_info_["wind_gust_amp_y"] = wind_gust_amp_.y();
  extra_info_["wind_gust_amp_z"] = wind_gust_amp_.z();
  extra_info_["wind_gust_x"] = current_wind_gust_.x();
  extra_info_["wind_gust_y"] = current_wind_gust_.y();
  extra_info_["wind_gust_z"] = current_wind_gust_.z();
  extra_info_["wind_burst_amp_x"] = signed_wind_burst_amp.x();
  extra_info_["wind_burst_amp_y"] = signed_wind_burst_amp.y();
  extra_info_["wind_burst_amp_z"] = signed_wind_burst_amp.z();
  extra_info_["wind_burst_x"] = current_wind_burst_.x();
  extra_info_["wind_burst_y"] = current_wind_burst_.y();
  extra_info_["wind_burst_z"] = current_wind_burst_.z();
  extra_info_["wind_total_norm"] = current_wind_.norm();
  extra_info_["wind_mean_norm"] = wind_mean_.norm();
  extra_info_["wind_gust_norm"] = wind_gust_amp_.norm();
  extra_info_["wind_burst_norm"] = wind_burst_amp_.norm();
  extra_info_["wind_acc_norm"] = current_wind_acc_.norm();
  extra_info_["wind_difficulty"] = current_wind_difficulty_;
  extra_info_["wind_sign"] = episode_wind_sign_;
  Vector<quadenv::kNAct> motor_thrusts = Vector<quadenv::kNAct>::Zero();
  if (quadrotor_ptr_ && quadrotor_ptr_->getMotorThrusts(motor_thrusts)) {
    int negative_count = 0;
    for (int i = 0; i < quadenv::kNAct; ++i) {
      if (motor_thrusts(i) < 0.0) negative_count += 1;
    }
    extra_info_["motor_negative_thrust_ratio"] =
      static_cast<Scalar>(negative_count) /
      static_cast<Scalar>(quadenv::kNAct);
    extra_info_["motor_min_thrust"] = motor_thrusts.minCoeff();
    extra_info_["motor_max_thrust"] = motor_thrusts.maxCoeff();
    extra_info_["motor_reversal_active_ratio"] =
      quadrotor_ptr_->getMotorReversalActiveRatio();
  }
  extra_info_["crash_reward"] = current_crash_reward_;
  extra_info_["alive_reward"] = current_alive_reward_;
  extra_info_["advance_reward"] = current_advance_reward_;
  extra_info_["avg_advance_reward"] =
    episode_steps_ > 0
      ? episode_advance_reward_sum_ /
          static_cast<Scalar>(std::max(episode_steps_, 1))
      : 0.0;
  extra_info_["path_reward"] = current_path_reward_;
  extra_info_["speed_reward"] = current_speed_reward_;
  extra_info_["body_rate_reward"] = current_body_rate_reward_;
  extra_info_["act_reward"] = current_act_reward_;
  extra_info_["act_second_delta_reward"] = current_act_second_delta_reward_;
  extra_info_["thrust_switch_reward"] = current_thrust_switch_reward_;
  extra_info_["reversal_deadzone_reward"] =
    current_reversal_deadzone_reward_;
  const Scalar hover_thrust_per_motor =
    quadrotor_ptr_ ? quadrotor_ptr_->getMass() * (-Gz) /
                       static_cast<Scalar>(quadenv::kNAct)
                   : 0.0;
  extra_info_["hover_action_norm"] =
    act_std_.cwiseAbs().minCoeff() > kSmallNumber
      ? hover_thrust_per_motor / act_std_.cwiseAbs().minCoeff()
      : 0.0;
  extra_info_["action_delay_steps"] =
    static_cast<Scalar>(current_action_delay_steps_);
  extra_info_["obs_delay_steps"] = static_cast<Scalar>(current_obs_delay_steps_);
  extra_info_["dr_mass_scale"] = current_mass_scale_;
  extra_info_["dr_inertia_scale"] = current_inertia_scale_;
  extra_info_["dr_thrust_scale"] = current_thrust_scale_;
  extra_info_["dr_negative_thrust_scale"] = current_negative_thrust_scale_;
  extra_info_["dr_motor_tau_scale"] = current_motor_tau_scale_;
  extra_info_["dr_motor_reversal_dead_time_scale"] =
    current_motor_reversal_dead_time_scale_;
  extra_info_["dr_motor_deadzone_omega_scale"] =
    current_motor_deadzone_omega_scale_;
  extra_info_["dr_drag_scale"] = current_drag_scale_;
  extra_info_["env_score"] = last_episode_score_;
  extra_info_["env_rank_reward"] = current_env_rank_reward_;
}

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Path Following Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "path_length =        [" << quad_env.path_length_ << "]\n"
     << "target_speed =       [" << quad_env.target_speed_ << "]\n"
     << "lateral_amp =        [" << quad_env.lateral_amp_ << "]\n"
     << "vertical_amp =       [" << quad_env.vertical_amp_ << "]\n"
     << "wave_count =         [" << quad_env.wave_count_ << "]\n"
     << "shape_mix =          [" << quad_env.path_shape_mix_ << "]\n"
     << "wind_enabled =       [" << quad_env.wind_enabled_ << "]\n"
     << "wind_difficulty =    [" << quad_env.current_wind_difficulty_ << "]\n"
     << "wind =               [" << quad_env.current_wind_.transpose()
     << "]\n"
     << "sampled_path_length =[" << quad_env.sampled_path_length_ << "]\n"
     << "checkpoint_count =   [" << quad_env.checkpoint_count_ << "]\n"
     << "checkpoint_radius =  [" << quad_env.checkpoint_radius_ << "]\n"
     << "reward.alive =       ["
     << quad_env.reward_cfg_.alive_reward << "]\n"
     << "reward.advance =     ["
     << quad_env.reward_cfg_.advance_coeff << "]\n"
     << "reward.path =        ["
     << quad_env.reward_cfg_.path_coeff << "]\n"
     << "reward.lookahead =   ["
     << quad_env.reward_cfg_.lookahead_coeff << "]\n"
     << "reward.body_rate =   ["
     << quad_env.reward_cfg_.body_rate_coeff << "]\n"
     << "reward.act =         [" << quad_env.reward_cfg_.act_coeff << "]\n"
     << "reward.act_second =  ["
     << quad_env.reward_cfg_.act_second_delta_coeff << "]\n"
     << "reward.switch =      ["
     << quad_env.reward_cfg_.thrust_switch_coeff << "]\n"
     << "reward.deadzone =    ["
     << quad_env.reward_cfg_.reversal_deadzone_coeff << "]\n"
     << "reward.speed =       [" << quad_env.reward_cfg_.speed_coeff << "]\n"
     << "reward.crash =       ["
     << quad_env.reward_cfg_.crash_penalty << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib

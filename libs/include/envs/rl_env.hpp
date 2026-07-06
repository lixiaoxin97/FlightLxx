#pragma once

// std lib
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "unity_bridge.hpp"
#include "command.hpp"
#include "logger.hpp"
#include "quad_state.hpp"
#include "types.hpp"
#include "env_base.hpp"
#include "quadrotor.hpp"

namespace flightlib {

namespace quadenv {

enum Ctl : int {
  // observations
  kObs = 0,
  kRot = 0,
  kNRot = 6,
  kLinVel = 6,
  kNLinVel = 3,
  kAngVel = 9,
  kNAngVel = 3,
  kPrevAct = 12,
  kNPrevAct = 4,
  kClosestDelta = 16,
  kNClosestDelta = 3,
  kTargetDelta = kClosestDelta,
  kNTargetDelta = kNClosestDelta,
  kLookaheadNearDelta = 19,
  kNLookaheadNearDelta = 3,
  kNextTargetDelta = kLookaheadNearDelta,
  kNNextTargetDelta = kNLookaheadNearDelta,
  kPathTangent = 22,
  kNPathTangent = 3,
  kNObs = 25,
  // control actions
  kAct = 0,
  kNAct = 4,
};
};

namespace pathenv {
enum EnvPolicy : int {
  kNCheckpoints = 8,
  kMaxPathSamples = 256,
  kCheckpointState = 0,
  kNCheckpointState = 3 * kNCheckpoints,
  kPassErrors = kCheckpointState + kNCheckpointState,
  kNPassErrors = kNCheckpoints,
  kEpisodeProgress = kPassErrors + kNPassErrors,
  kCompleted = kEpisodeProgress + 1,
  kCrashed = kCompleted + 1,
  kEnvScore = kCrashed + 1,
  kInfeasibleCount = kEnvScore + 1,
  kCheckpointAction = 0,
  kNCheckpointAction = 3 * kNCheckpoints,
  kPathGlobalAction = kCheckpointAction + kNCheckpointAction,
  kLengthAction = kPathGlobalAction,
  kRadiusAction = kLengthAction + 1,
  kVerticalAction = kRadiusAction + 1,
  kWaveAction = kVerticalAction + 1,
  kClosedAction = kWaveAction + 1,
  kPhaseAction = kClosedAction + 1,
  kNPathGlobalAction = 6,
  kWindAction = kPathGlobalAction + kNPathGlobalAction,
  kWindMeanAction = kWindAction,
  kWindGustAmpAction = kWindMeanAction + 3,
  kWindGustFreqAction = kWindGustAmpAction + 3,
  kWindGustPhaseAction = kWindGustFreqAction + 3,
  kWindBurstAmpAction = kWindGustPhaseAction + 3,
  kWindBurstStartAction = kWindBurstAmpAction + 3,
  kWindBurstDurationAction = kWindBurstStartAction + 1,
  kNWindAction = 17,
  kWindState = kInfeasibleCount + 1,
  kNWindState = kNWindAction,
  kNEnvObs = kWindState + kNWindState,
  kNEnvAct = kNCheckpointAction + kNPathGlobalAction + kNWindAction,
};
};

struct PathRewardConfig {
  // Continuous-path-following reward. Loaded from path_following.reward.
  Scalar alive_reward{0.0};
  Scalar advance_coeff{1.0};
  Scalar advance_ref_speed{1.0};
  Scalar track_gate_sigma{1.0};
  Scalar path_coeff{1.0};
  Scalar path_error_scale{1.0};
  Scalar lookahead_coeff{0.30};
  Scalar lookahead_near_distance{0.40};
  Scalar lookahead_far_distance{0.80};
  Scalar lookahead_near_weight{0.70};
  Scalar lookahead_far_weight{0.30};
  Scalar lookahead_improvement_eps{0.01};
  Scalar body_rate_coeff{0.04};
  Scalar body_rate_scale{4.0};
  Scalar act_coeff{0.05};
  Scalar act_delta_scale{2.0};
  Scalar act_second_delta_coeff{0.0};
  Scalar act_second_delta_scale{2.0};
  Scalar thrust_switch_coeff{0.0};
  Scalar reversal_deadzone_coeff{0.0};
  Scalar speed_limit{2.0};
  Scalar speed_coeff{0.20};
  Scalar max_penalty{4.0};
  Scalar crash_penalty{-30.0};
  Scalar terminal_penalty{-10.0};
  Scalar success_reward{20.0};
  // Legacy fields kept for config compatibility with earlier experiments.
  Scalar progress_coeff{0.0};
  Scalar pass_reward{0.0};
  Scalar progress_clip{1.0};
  Scalar progress_speed_limit{2.0};
  Scalar progress_overspeed_coeff{0.02};
  Scalar pos_coeff{-0.20};
  Scalar ang_vel_coeff{-0.005};
  Scalar survival_reward{0.01};
  Scalar pos_sigma{0.60};
};

class QuadrotorEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QuadrotorEnv();
  QuadrotorEnv(const std::string &cfg_path);
  ~QuadrotorEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);
  void setSeed(const int seed);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;
  bool getEnvPolicyObs(Ref<Vector<>> obs) const;
  Scalar getEnvironmentScore() const;
  int getEnvPolicyObsDim() const;
  int getEnvPolicyActDim() const;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void updateExtraInfo() override;
  void applyEnvPolicyAction(const Ref<Vector<>> act);
  void applyPathEnvironmentPolicy(const int rank, const int num_envs,
                                  const int lower_rank,
                                  const int upper_rank);
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"QaudrotorEnv"};

  PathRewardConfig reward_cfg_;
  QuadrotorDynamics nominal_dynamics_;

  // observations and actions (for RL)
  Vector<quadenv::kNObs> quad_obs_;
  Vector<quadenv::kNAct> quad_act_;
  Vector<quadenv::kNAct> prev_act_norm_;
  Vector<quadenv::kNAct> prev_prev_act_norm_;
  Vector<quadenv::kNAct> delayed_act_norm_;

  // episode bookkeeping
  void resetEpisodeState();
  Scalar elapsed_time_{0.0};
  Scalar progress_s_{0.0};
  Scalar previous_progress_s_{0.0};
  Scalar current_pos_error_{0.0};
  Scalar current_target_error_{0.0};
  Scalar current_forward_speed_{0.0};
  Scalar current_approach_speed_{0.0};
  Scalar current_alive_reward_{0.0};
  Scalar current_advance_reward_{0.0};
  Scalar current_path_reward_{0.0};
  Scalar current_lookahead_reward_{0.0};
  Scalar current_speed_reward_{0.0};
  Scalar current_progress_reward_{0.0};
  Scalar current_progress_overspeed_reward_{0.0};
  Scalar current_pass_reward_{0.0};
  Scalar current_body_rate_reward_{0.0};
  Scalar current_act_reward_{0.0};
  Scalar current_act_second_delta_reward_{0.0};
  Scalar current_thrust_switch_reward_{0.0};
  Scalar current_reversal_deadzone_reward_{0.0};
  Scalar current_crash_reward_{0.0};
  Scalar current_env_rank_reward_{0.0};
  Scalar current_lookahead_error_{0.0};
  Scalar current_track_gate_{0.0};
  Scalar current_delta_s_{0.0};
  Scalar previous_lookahead_error_{0.0};
  Scalar current_speed_over_{0.0};
  Scalar episode_pos_error_sum_{0.0};
  Scalar episode_lookahead_error_sum_{0.0};
  Scalar episode_advance_reward_sum_{0.0};
  Scalar episode_progress_{0.0};
  Scalar last_episode_score_{0.0};
  Scalar last_episode_path_error_{0.0};
  Scalar last_episode_pass_error_{0.0};
  Scalar last_episode_lookahead_error_{0.0};
  Scalar last_episode_progress_{0.0};
  Scalar previous_target_distance_{0.0};
  Vector<3> previous_position_{Vector<3>::Zero()};
  Vector<3> previous_lookahead_position_{Vector<3>::Zero()};
  int episode_steps_{0};
  int checkpoint_count_{pathenv::kNCheckpoints};
  int current_checkpoint_idx_{0};
  int passed_checkpoint_count_{0};
  bool completed_{false};
  bool crashed_{false};
  bool last_episode_completed_{false};
  bool last_episode_crashed_{false};
  bool timed_out_{false};
  bool path_error_terminal_{false};
  bool ground_crash_{false};

  // path geometry
  void computeObservation(Vector<quadenv::kNObs> *obs);
  void fillObservationDelayBuffer();
  void pushObservationDelayBuffer();
  void fillActionDelayBuffer();
  Vector<quadenv::kNAct> delayedAction(
    const Vector<quadenv::kNAct> &policy_act);
  void randomizeDynamicsForEpisode(const bool random);
  Vector<3> basePathPosition(const Scalar s) const;
  Vector<3> basePathDerivative(const Scalar s) const;
  Vector<3> controlPointPosition(const int index) const;
  Vector<3> interpolateControlPath(const Scalar u) const;
  void rebuildSampledPath();
  Vector<3> pathPosition(const Scalar s) const;
  Vector<3> pathDerivative(const Scalar s) const;
  Vector<3> pathTangent(const Scalar s) const;
  Vector<3> checkpointBasePosition(const int index) const;
  Vector<3> checkpointPosition(const int index) const;
  Scalar checkpointProgress(const int index) const;
  Scalar segmentDistanceToPoint(const Vector<3> &a, const Vector<3> &b,
                                const Vector<3> &p) const;
  Scalar closestPathProgress(const Vector<3> &position) const;
  void resetWindToInitialDifficulty();
  void applyTestWindPreset();
  Vector<3> limitWindVectorNorm(const Vector<3> &value,
                                const Scalar max_norm) const;
  void enforceWindBudgets();
  Scalar smoothPulse(const Scalar time, const Scalar start,
                     const Scalar duration) const;
  void updateWindField();
  Scalar windDifficulty() const;
  Scalar clampScalar(const Scalar value, const Scalar min_value,
                     const Scalar max_value) const;
  Scalar normalizeRange(const Scalar value, const Scalar min_value,
                        const Scalar max_value) const;
  bool useFigureEightPath() const;

  std::string path_mode_{"default"};
  Scalar base_z_{5.0};
  Scalar path_length_{18.0};
  Scalar target_speed_{2.0};
  Scalar lateral_amp_{0.0};
  Scalar vertical_amp_{0.0};
  Scalar wave_count_{1.0};
  Scalar path_shape_mix_{0.0};
  Scalar phase_y_{0.0};
  Scalar phase_z_{0.0};
  Scalar sampled_path_length_{18.0};
  int path_sample_count_{201};
  Matrix<3, pathenv::kMaxPathSamples> sampled_path_ =
    Matrix<3, pathenv::kMaxPathSamples>::Zero();
  Vector<pathenv::kMaxPathSamples> sampled_s_ =
    Vector<pathenv::kMaxPathSamples>::Zero();
  Scalar initial_path_length_{18.0};
  Scalar initial_target_speed_{2.0};
  Scalar initial_lateral_amp_{0.0};
  Scalar initial_vertical_amp_{0.0};
  Scalar initial_wave_count_{1.0};
  Scalar initial_path_shape_mix_{0.0};
  Scalar figure_eight_length_{18.0};
  Scalar figure_eight_radius_{2.4};
  Scalar figure_eight_vertical_amp_{0.0};
  Scalar figure_eight_speed_{2.0};
  Scalar min_path_length_{10.0};
  Scalar max_path_length_{20.0};
  Scalar min_target_speed_{1.0};
  Scalar max_target_speed_{5.0};
  Scalar max_lateral_amp_{5.0};
  Scalar max_vertical_amp_{2.0};
  Scalar min_wave_count_{0.5};
  Scalar max_wave_count_{3.0};
  Scalar max_path_shape_mix_{1.0};
  Scalar lookahead_distance_{2.0};
  Scalar max_tracking_error_{5.0};
  Scalar finish_margin_{0.5};
  Scalar finish_error_{1.0};
  Scalar closest_search_backward_{1.0};
  Scalar closest_search_forward_{3.0};
  Scalar checkpoint_radius_{0.8};
  Matrix<3, pathenv::kNCheckpoints> checkpoint_offsets_ =
    Matrix<3, pathenv::kNCheckpoints>::Zero();
  Vector<pathenv::kNCheckpoints> checkpoint_pass_errors_ =
    Vector<pathenv::kNCheckpoints>::Ones();
  Vector<pathenv::kNCheckpoints> last_checkpoint_pass_errors_ =
    Vector<pathenv::kNCheckpoints>::Ones();

  // Reset-state randomization. Position noise is intentionally larger than
  // attitude/rate noise to avoid unrealistic side-flip initial states.
  Scalar reset_progress_range_{1.0};
  Scalar reset_position_xy_noise_{0.5};
  Scalar reset_position_z_noise_{0.25};
  Scalar reset_forward_speed_noise_{0.2};
  Scalar reset_lateral_speed_noise_{0.2};
  Scalar reset_vertical_speed_noise_{0.1};
  Scalar reset_roll_pitch_noise_{0.12};
  Scalar reset_yaw_noise_{0.25};
  Scalar reset_body_rate_noise_{0.2};

  // Real-hardware robustness: randomized dynamics and sampled I/O delays.
  static constexpr int kMaxDelaySteps = 8;
  std::array<Vector<quadenv::kNObs>, kMaxDelaySteps + 1> obs_delay_buffer_;
  std::array<Vector<quadenv::kNAct>, kMaxDelaySteps + 1> act_delay_buffer_;
  bool domain_randomization_enabled_{false};
  Scalar dr_mass_scale_{0.0};
  Scalar dr_inertia_scale_{0.0};
  Scalar dr_thrust_scale_{0.0};
  Scalar dr_negative_thrust_scale_{0.0};
  Scalar dr_motor_tau_scale_{0.0};
  Scalar dr_motor_reversal_dead_time_scale_{0.0};
  Scalar dr_motor_deadzone_omega_scale_{0.0};
  Scalar dr_drag_scale_{0.0};
  int max_action_delay_steps_{0};
  int max_obs_delay_steps_{0};
  int current_action_delay_steps_{0};
  int current_obs_delay_steps_{0};
  Scalar current_mass_scale_{1.0};
  Scalar current_inertia_scale_{1.0};
  Scalar current_thrust_scale_{1.0};
  Scalar current_negative_thrust_scale_{1.0};
  Scalar current_motor_tau_scale_{1.0};
  Scalar current_motor_reversal_dead_time_scale_{1.0};
  Scalar current_motor_deadzone_omega_scale_{1.0};
  Scalar current_drag_scale_{1.0};

  // previous-episode evaluation for adaptive path generation
  void evaluatePreviousEpisode();
  void resetPathToInitialDifficulty();

  bool adaptive_path_enabled_{true};
  Scalar env_policy_step_{0.25};
  Scalar env_policy_noise_{0.04};
  Scalar env_policy_reward_scale_{1.0};
  Scalar env_lateral_step_{0.35};
  Scalar env_vertical_step_{0.18};
  Scalar env_wave_step_{0.18};
  Scalar env_shape_step_{0.10};
  Scalar env_length_step_{1.0};
  Scalar env_action_clip_{0.25};
  Scalar env_recovery_action_{0.60};
  Scalar env_guard_min_progress_{0.12};
  Scalar env_guard_max_error_ratio_{0.65};
  Scalar env_checkpoint_xy_step_{0.50};
  Scalar env_checkpoint_z_step_{0.25};
  Scalar max_checkpoint_x_offset_{18.0};
  Scalar max_checkpoint_xy_offset_{4.0};
  Scalar max_checkpoint_z_offset_{2.0};
  int infeasible_reset_patience_{3};
  int infeasible_counter_{0};
  Scalar infeasible_score_threshold_{-30.0};
  Scalar env_reset_min_progress_{0.05};

  // Wind field generated by the environment policy. The main policy does not
  // observe these values directly.
  bool wind_enabled_{true};
  Scalar wind_linear_drag_{0.18};
  Scalar wind_quadratic_drag_{0.04};
  Scalar max_wind_total_speed_{12.0};
  Scalar max_wind_mean_norm_{6.0};
  Scalar max_wind_gust_norm_{4.0};
  Scalar max_wind_burst_norm_{2.0};
  Scalar max_wind_mean_xy_{6.0};
  Scalar max_wind_mean_z_{6.0};
  Scalar max_wind_gust_xy_{4.0};
  Scalar max_wind_gust_z_{4.0};
  Scalar min_wind_gust_freq_{0.04};
  Scalar max_wind_gust_freq_{0.45};
  Scalar max_wind_burst_xy_{2.0};
  Scalar max_wind_burst_z_{2.0};
  Scalar min_wind_burst_duration_{0.35};
  Scalar max_wind_burst_duration_{3.0};
  Scalar env_wind_mean_step_{0.70};
  Scalar env_wind_gust_step_{0.50};
  Scalar env_wind_freq_step_{0.06};
  Scalar env_wind_burst_step_{0.80};
  Scalar env_wind_burst_time_step_{0.08};
  Vector<3> wind_mean_{Vector<3>::Zero()};
  Vector<3> wind_gust_amp_{Vector<3>::Zero()};
  Vector<3> wind_gust_freq_{Vector<3>::Zero()};
  Vector<3> wind_gust_phase_{Vector<3>::Zero()};
  Vector<3> wind_burst_amp_{Vector<3>::Zero()};
  Vector<3> current_wind_gust_{Vector<3>::Zero()};
  Vector<3> current_wind_burst_{Vector<3>::Zero()};
  Vector<3> current_wind_{Vector<3>::Zero()};
  Vector<3> current_wind_acc_{Vector<3>::Zero()};
  Scalar episode_wind_sign_{1.0};
  Scalar wind_burst_start_ratio_{0.50};
  Scalar wind_burst_duration_{1.0};
  Scalar current_wind_difficulty_{0.0};

  // action and observation normalization (for learning)
  Vector<quadenv::kNAct> act_mean_;
  Vector<quadenv::kNAct> act_std_;
  Vector<quadenv::kNObs> obs_mean_ = Vector<quadenv::kNObs>::Zero();
  Vector<quadenv::kNObs> obs_std_ = Vector<quadenv::kNObs>::Ones();

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib

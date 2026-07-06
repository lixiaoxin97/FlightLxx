#include "rl_env.hpp"

namespace flightlib {

// All QuadrotorEnv YAML loading lives here. Main policy reward coefficients are
// read from path_following.reward in libs/config/rl_env.yaml.

namespace {

class ConfigReader {
 public:
  explicit ConfigReader(const YAML::Node &node) : node_(node) {}

  explicit operator bool() const { return static_cast<bool>(node_); }

  template <typename T>
  ConfigReader &read(const char *key, T &target) {
    if (node_ && node_[key]) {
      target = node_[key].as<T>();
    }
    return *this;
  }

 private:
  YAML::Node node_;
};

void readRewardConfig(ConfigReader &reader, PathRewardConfig &reward) {
  reader.read("alive_reward", reward.alive_reward)
    .read("survival_reward", reward.alive_reward)
    .read("advance_coeff", reward.advance_coeff)
    .read("advance_ref_speed", reward.advance_ref_speed)
    .read("track_gate_sigma", reward.track_gate_sigma)
    .read("path_coeff", reward.path_coeff)
    .read("path_error_scale", reward.path_error_scale)
    .read("lookahead_coeff", reward.lookahead_coeff)
    .read("lookahead_near_distance", reward.lookahead_near_distance)
    .read("lookahead_far_distance", reward.lookahead_far_distance)
    .read("lookahead_near_weight", reward.lookahead_near_weight)
    .read("lookahead_far_weight", reward.lookahead_far_weight)
    .read("lookahead_improvement_eps", reward.lookahead_improvement_eps)
    .read("body_rate_coeff", reward.body_rate_coeff)
    .read("bodyrate_coeff", reward.body_rate_coeff)
    .read("body_rate_scale", reward.body_rate_scale)
    .read("act_coeff", reward.act_coeff)
    .read("act_delta_scale", reward.act_delta_scale)
    .read("act_second_delta_coeff", reward.act_second_delta_coeff)
    .read("act_second_delta_scale", reward.act_second_delta_scale)
    .read("thrust_switch_coeff", reward.thrust_switch_coeff)
    .read("reversal_deadzone_coeff", reward.reversal_deadzone_coeff)
    .read("speed_limit", reward.speed_limit)
    .read("speed_coeff", reward.speed_coeff)
    .read("max_penalty", reward.max_penalty)
    .read("terminal_penalty", reward.terminal_penalty)
    .read("crash_penalty", reward.crash_penalty)
    .read("success_reward", reward.success_reward)
    .read("progress_coeff", reward.progress_coeff)
    .read("pass_reward", reward.pass_reward)
    .read("progress_clip", reward.progress_clip)
    .read("progress_speed_limit", reward.progress_speed_limit)
    .read("progress_overspeed_coeff", reward.progress_overspeed_coeff)
    .read("pos_coeff", reward.pos_coeff)
    .read("ang_vel_coeff", reward.ang_vel_coeff)
    .read("survival_reward", reward.survival_reward)
    .read("pos_sigma", reward.pos_sigma);
}

}  // namespace

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  ConfigReader quad_env(cfg["quadrotor_env"]);
  if (!quad_env) return false;

  quad_env.read("sim_dt", sim_dt_)
    .read("max_t", max_t_);

  const YAML::Node path_node = cfg["path_following"];
  ConfigReader path(path_node);
  path.read("path_mode", path_mode_)
    .read("base_z", base_z_)
    .read("initial_length", initial_path_length_)
    .read("min_length", min_path_length_)
    .read("max_length", max_path_length_)
    .read("initial_speed", initial_target_speed_)
    .read("min_speed", min_target_speed_)
    .read("max_speed", max_target_speed_)
    .read("initial_lateral_amp", initial_lateral_amp_)
    .read("initial_vertical_amp", initial_vertical_amp_)
    .read("max_lateral_amp", max_lateral_amp_)
    .read("max_vertical_amp", max_vertical_amp_)
    .read("initial_wave_count", initial_wave_count_)
    .read("min_wave_count", min_wave_count_)
    .read("max_wave_count", max_wave_count_)
    .read("initial_shape_mix", initial_path_shape_mix_)
    .read("max_shape_mix", max_path_shape_mix_)
    .read("lookahead_distance", lookahead_distance_)
    .read("max_tracking_error", max_tracking_error_)
    .read("finish_margin", finish_margin_)
    .read("finish_error", finish_error_)
    .read("closest_search_backward", closest_search_backward_)
    .read("closest_search_forward", closest_search_forward_)
    .read("path_samples", path_sample_count_)
    .read("checkpoint_count", checkpoint_count_)
    .read("checkpoint_radius", checkpoint_radius_)
    .read("figure_eight_length", figure_eight_length_)
    .read("figure_eight_radius", figure_eight_radius_)
    .read("figure_eight_vertical_amp", figure_eight_vertical_amp_)
    .read("figure_eight_speed", figure_eight_speed_);

  // Backward compatibility for old flat path_following reward keys.
  readRewardConfig(path, reward_cfg_);
  ConfigReader reward(path_node["reward"]);
  readRewardConfig(reward, reward_cfg_);

  ConfigReader reset(path_node["reset"]);
  reset.read("progress_range", reset_progress_range_)
    .read("position_xy_noise", reset_position_xy_noise_)
    .read("position_z_noise", reset_position_z_noise_)
    .read("forward_speed_noise", reset_forward_speed_noise_)
    .read("lateral_speed_noise", reset_lateral_speed_noise_)
    .read("vertical_speed_noise", reset_vertical_speed_noise_)
    .read("roll_pitch_noise", reset_roll_pitch_noise_)
    .read("yaw_noise", reset_yaw_noise_)
    .read("body_rate_noise", reset_body_rate_noise_);

  ConfigReader domain_randomization(cfg["domain_randomization"]);
  domain_randomization.read("enabled", domain_randomization_enabled_)
    .read("mass_scale", dr_mass_scale_)
    .read("inertia_scale", dr_inertia_scale_)
    .read("thrust_scale", dr_thrust_scale_)
    .read("negative_thrust_scale", dr_negative_thrust_scale_)
    .read("motor_tau_scale", dr_motor_tau_scale_)
    .read("motor_reversal_dead_time_scale",
          dr_motor_reversal_dead_time_scale_)
    .read("motor_deadzone_omega_scale", dr_motor_deadzone_omega_scale_)
    .read("drag_scale", dr_drag_scale_)
    .read("max_action_delay_steps", max_action_delay_steps_)
    .read("max_obs_delay_steps", max_obs_delay_steps_);
  max_action_delay_steps_ =
    std::max(0, std::min(max_action_delay_steps_, kMaxDelaySteps));
  max_obs_delay_steps_ =
    std::max(0, std::min(max_obs_delay_steps_, kMaxDelaySteps));

  const char *path_mode_override = getenv("FlightLxx_PATH_MODE");
  if (path_mode_override && path_mode_override[0] != '\0') {
    path_mode_ = std::string(path_mode_override);
  }
  const char *path_length_override = getenv("FlightLxx_TEST_PATH_LENGTH");
  if (path_length_override && path_length_override[0] != '\0') {
    initial_path_length_ =
      clampScalar(static_cast<Scalar>(std::atof(path_length_override)),
                  min_path_length_, max_path_length_);
  }

  ConfigReader env_policy(cfg["environment_policy"]);
  env_policy.read("enabled", adaptive_path_enabled_)
    .read("step", env_policy_step_)
    .read("noise", env_policy_noise_)
    .read("rank_reward_scale", env_policy_reward_scale_)
    .read("lateral_step", env_lateral_step_)
    .read("vertical_step", env_vertical_step_)
    .read("wave_step", env_wave_step_)
    .read("shape_step", env_shape_step_)
    .read("length_step", env_length_step_)
    .read("action_clip", env_action_clip_)
    .read("recovery_action", env_recovery_action_)
    .read("guard_min_progress", env_guard_min_progress_)
    .read("guard_max_error_ratio", env_guard_max_error_ratio_)
    .read("checkpoint_xy_step", env_checkpoint_xy_step_)
    .read("checkpoint_z_step", env_checkpoint_z_step_)
    .read("max_checkpoint_x_offset", max_checkpoint_x_offset_)
    .read("max_checkpoint_xy_offset", max_checkpoint_xy_offset_)
    .read("max_checkpoint_z_offset", max_checkpoint_z_offset_)
    .read("infeasible_reset_patience", infeasible_reset_patience_)
    .read("infeasible_score_threshold", infeasible_score_threshold_)
    .read("reset_min_progress", env_reset_min_progress_);

  ConfigReader wind(cfg["wind"]);
  wind.read("enabled", wind_enabled_)
    .read("linear_drag", wind_linear_drag_)
    .read("quadratic_drag", wind_quadratic_drag_)
    .read("max_total_speed", max_wind_total_speed_)
    .read("max_mean_norm", max_wind_mean_norm_)
    .read("max_gust_norm", max_wind_gust_norm_)
    .read("max_burst_norm", max_wind_burst_norm_)
    .read("max_mean_xy", max_wind_mean_xy_)
    .read("max_mean_z", max_wind_mean_z_)
    .read("max_gust_xy", max_wind_gust_xy_)
    .read("max_gust_z", max_wind_gust_z_)
    .read("min_gust_freq", min_wind_gust_freq_)
    .read("max_gust_freq", max_wind_gust_freq_)
    .read("max_burst_xy", max_wind_burst_xy_)
    .read("max_burst_z", max_wind_burst_z_)
    .read("min_burst_duration", min_wind_burst_duration_)
    .read("max_burst_duration", max_wind_burst_duration_)
    .read("mean_step", env_wind_mean_step_)
    .read("gust_step", env_wind_gust_step_)
    .read("freq_step", env_wind_freq_step_)
    .read("burst_step", env_wind_burst_step_)
    .read("burst_time_step", env_wind_burst_time_step_);

  checkpoint_count_ =
    std::max(2, std::min(checkpoint_count_,
                         static_cast<int>(pathenv::kNCheckpoints)));

  return true;
}

}  // namespace flightlib

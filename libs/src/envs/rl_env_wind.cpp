#include "rl_env.hpp"

namespace flightlib {

namespace {
constexpr Scalar kSmallNumber = 1.0e-4f;
}

void QuadrotorEnv::resetWindToInitialDifficulty() {
  wind_mean_.setZero();
  wind_gust_amp_.setZero();
  wind_gust_freq_.setConstant(min_wind_gust_freq_);
  wind_gust_phase_.setZero();
  wind_burst_amp_.setZero();
  current_wind_gust_.setZero();
  current_wind_burst_.setZero();
  current_wind_.setZero();
  current_wind_acc_.setZero();
  wind_burst_start_ratio_ = 0.50;
  episode_wind_sign_ = 1.0;
  wind_burst_duration_ =
    clampScalar(static_cast<Scalar>(1.0), min_wind_burst_duration_,
                max_wind_burst_duration_);
  applyTestWindPreset();
  current_wind_difficulty_ = 0.0;
  if (quadrotor_ptr_) {
    quadrotor_ptr_->setWindDrag(wind_linear_drag_ * current_drag_scale_,
                                wind_quadratic_drag_ * current_drag_scale_);
    quadrotor_ptr_->setWindVelocity(current_wind_);
  }
}

Vector<3> QuadrotorEnv::limitWindVectorNorm(const Vector<3> &value,
                                            const Scalar max_norm) const {
  const Scalar safe_limit = std::max(max_norm, static_cast<Scalar>(0.0));
  const Scalar norm = value.norm();
  if (safe_limit <= kSmallNumber || norm <= safe_limit) {
    return value;
  }
  return value * (safe_limit / std::max(norm, kSmallNumber));
}

void QuadrotorEnv::enforceWindBudgets() {
  const Scalar total_limit =
    std::max(max_wind_total_speed_, static_cast<Scalar>(kSmallNumber));
  Scalar mean_limit =
    std::max(max_wind_mean_norm_, static_cast<Scalar>(0.0));
  Scalar gust_limit =
    std::max(max_wind_gust_norm_, static_cast<Scalar>(0.0));
  Scalar burst_limit =
    std::max(max_wind_burst_norm_, static_cast<Scalar>(0.0));
  const Scalar budget_sum = mean_limit + gust_limit + burst_limit;
  if (budget_sum > total_limit) {
    const Scalar scale = total_limit / std::max(budget_sum, kSmallNumber);
    mean_limit *= scale;
    gust_limit *= scale;
    burst_limit *= scale;
  }

  wind_mean_ = limitWindVectorNorm(wind_mean_, mean_limit);
  wind_gust_amp_ = limitWindVectorNorm(wind_gust_amp_, gust_limit);
  wind_burst_amp_ = limitWindVectorNorm(wind_burst_amp_, burst_limit);
}

void QuadrotorEnv::applyTestWindPreset() {
  const char *mode_env = getenv("FlightLxx_TEST_WIND_MODE");
  if (!mode_env || mode_env[0] == '\0') return;

  const std::string mode(mode_env);
  if (mode == "none" || mode == "0" || mode == "off") return;

  wind_enabled_ = true;

  if (mode == "extreme") {
    wind_mean_ = Vector<3>(
      clampScalar(static_cast<Scalar>(6.8), -max_wind_mean_xy_,
                  max_wind_mean_xy_),
      clampScalar(static_cast<Scalar>(4.8), -max_wind_mean_xy_,
                  max_wind_mean_xy_),
      clampScalar(static_cast<Scalar>(1.4), -max_wind_mean_z_,
                  max_wind_mean_z_));
    wind_gust_amp_ = Vector<3>(
      clampScalar(static_cast<Scalar>(4.8), 0.0, max_wind_gust_xy_),
      clampScalar(static_cast<Scalar>(4.2), 0.0, max_wind_gust_xy_),
      clampScalar(static_cast<Scalar>(1.8), 0.0, max_wind_gust_z_));
    wind_burst_amp_ = Vector<3>(
      clampScalar(static_cast<Scalar>(7.5), -max_wind_burst_xy_,
                  max_wind_burst_xy_),
      clampScalar(static_cast<Scalar>(-6.0), -max_wind_burst_xy_,
                  max_wind_burst_xy_),
      clampScalar(static_cast<Scalar>(2.6), -max_wind_burst_z_,
                  max_wind_burst_z_));
    wind_burst_duration_ =
      clampScalar(static_cast<Scalar>(2.2), min_wind_burst_duration_,
                  max_wind_burst_duration_);
  } else if (mode == "strong") {
    wind_mean_ = Vector<3>(3.6, 2.0, 0.4);
    wind_gust_amp_ = Vector<3>(1.8, 1.4, 0.6);
    wind_burst_amp_ = Vector<3>(2.6, -1.8, 0.9);
    wind_burst_duration_ =
      clampScalar(static_cast<Scalar>(1.8), min_wind_burst_duration_,
                  max_wind_burst_duration_);
  } else {
    wind_mean_ = Vector<3>(2.5, 1.2, 0.2);
    wind_gust_amp_ = Vector<3>(1.0, 0.8, 0.35);
    wind_burst_amp_ = Vector<3>(1.8, -1.2, 0.6);
    wind_burst_duration_ =
      clampScalar(static_cast<Scalar>(1.6), min_wind_burst_duration_,
                  max_wind_burst_duration_);
  }

  wind_gust_freq_ = Vector<3>(
    clampScalar(static_cast<Scalar>(0.18), min_wind_gust_freq_,
                max_wind_gust_freq_),
    clampScalar(static_cast<Scalar>(0.32), min_wind_gust_freq_,
                max_wind_gust_freq_),
    clampScalar(static_cast<Scalar>(0.43), min_wind_gust_freq_,
                max_wind_gust_freq_));
  wind_gust_phase_ = Vector<3>(0.0, 2.1, 4.0);
  wind_burst_start_ratio_ = 0.45;
  enforceWindBudgets();
}

Scalar QuadrotorEnv::smoothPulse(const Scalar time, const Scalar start,
                                 const Scalar duration) const {
  const Scalar safe_duration = std::max(duration, kSmallNumber);
  if (time < start || time > start + safe_duration) {
    return 0.0;
  }
  const Scalar alpha = clampScalar((time - start) / safe_duration, 0.0, 1.0);
  return std::sin(EIGEN_PI * alpha);
}

Scalar QuadrotorEnv::windDifficulty() const {
  const auto component_score = [&](const Vector<3> &value, const Scalar xy_max,
                                   const Scalar z_max) {
    const Scalar x =
      value.x() / std::max(xy_max, static_cast<Scalar>(kSmallNumber));
    const Scalar y =
      value.y() / std::max(xy_max, static_cast<Scalar>(kSmallNumber));
    const Scalar z =
      value.z() / std::max(z_max, static_cast<Scalar>(kSmallNumber));
    return clampScalar(Vector<3>(x, y, z).norm(), 0.0, 1.0);
  };

  const Scalar mean_score =
    component_score(wind_mean_, max_wind_mean_xy_, max_wind_mean_z_);
  const Scalar gust_score =
    component_score(wind_gust_amp_, max_wind_gust_xy_, max_wind_gust_z_);
  const Scalar burst_score =
    component_score(wind_burst_amp_, max_wind_burst_xy_, max_wind_burst_z_);
  Scalar freq_score = 0.0;
  for (int i = 0; i < 3; i++) {
    freq_score += normalizeRange(wind_gust_freq_(i), min_wind_gust_freq_,
                                 max_wind_gust_freq_);
  }
  freq_score /= 3.0;
  const Scalar duration_score =
    normalizeRange(wind_burst_duration_, min_wind_burst_duration_,
                   max_wind_burst_duration_);

  return clampScalar(static_cast<Scalar>(0.35) * mean_score +
                       static_cast<Scalar>(0.20) * gust_score +
                       static_cast<Scalar>(0.10) * freq_score +
                       static_cast<Scalar>(0.30) * burst_score +
                       static_cast<Scalar>(0.05) * duration_score,
                     0.0, 1.0);
}

void QuadrotorEnv::updateWindField() {
  if (!wind_enabled_) {
    current_wind_gust_.setZero();
    current_wind_burst_.setZero();
    current_wind_.setZero();
    current_wind_acc_.setZero();
    current_wind_difficulty_ = 0.0;
    if (quadrotor_ptr_) {
      quadrotor_ptr_->setWindVelocity(current_wind_);
    }
    return;
  }

  current_wind_gust_.setZero();
  for (int i = 0; i < 3; i++) {
    current_wind_gust_(i) =
      wind_gust_amp_(i) *
      std::sin(static_cast<Scalar>(2.0) * EIGEN_PI * wind_gust_freq_(i) *
                 elapsed_time_ +
               wind_gust_phase_(i));
  }

  const Scalar burst_start = wind_burst_start_ratio_ * max_t_;
  const Scalar burst_pulse =
    smoothPulse(elapsed_time_, burst_start, wind_burst_duration_);
  current_wind_burst_ = burst_pulse * wind_burst_amp_;
  current_wind_gust_ *= episode_wind_sign_;
  current_wind_burst_ *= episode_wind_sign_;
  current_wind_ =
    episode_wind_sign_ * wind_mean_ + current_wind_gust_ + current_wind_burst_;
  const Scalar wind_norm = current_wind_.norm();
  if (wind_norm > max_wind_total_speed_) {
    current_wind_ *= max_wind_total_speed_ / std::max(wind_norm, kSmallNumber);
  }
  current_wind_difficulty_ = windDifficulty();

  quadrotor_ptr_->getState(&quad_state_);
  const Vector<3> relative_air_velocity = quad_state_.v - current_wind_;
  const Vector<3> wind_force =
    -(wind_linear_drag_ * current_drag_scale_) * relative_air_velocity -
    (wind_quadratic_drag_ * current_drag_scale_) *
      relative_air_velocity.norm() *
      relative_air_velocity;
  current_wind_acc_ = wind_force / std::max(quadrotor_ptr_->getMass(),
                                            static_cast<Scalar>(kSmallNumber));

  quadrotor_ptr_->setWindDrag(wind_linear_drag_ * current_drag_scale_,
                              wind_quadratic_drag_ * current_drag_scale_);
  quadrotor_ptr_->setWindVelocity(current_wind_);
}

}  // namespace flightlib

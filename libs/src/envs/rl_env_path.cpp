#include "rl_env.hpp"

namespace flightlib {

namespace {
constexpr Scalar kSmallNumber = 1.0e-4f;
}

Scalar QuadrotorEnv::clampScalar(const Scalar value, const Scalar min_value,
                                 const Scalar max_value) const {
  return std::max(min_value, std::min(value, max_value));
}

Scalar QuadrotorEnv::normalizeRange(const Scalar value, const Scalar min_value,
                                    const Scalar max_value) const {
  const Scalar denom = std::max(max_value - min_value, kSmallNumber);
  return clampScalar((value - min_value) / denom, 0.0, 1.0);
}

bool QuadrotorEnv::useFigureEightPath() const {
  return path_mode_ == "figure8" || path_mode_ == "figure_eight";
}

Vector<3> QuadrotorEnv::basePathPosition(const Scalar s) const {
  const Scalar s_clamped = clampScalar(s, 0.0, path_length_);

  if (useFigureEightPath()) {
    const Scalar radius = std::max(lateral_amp_, kSmallNumber);
    const Scalar u =
      2.0 * EIGEN_PI * s_clamped / std::max(path_length_, kSmallNumber);
    const Scalar x = radius * std::sin(u);
    const Scalar y = 0.5 * radius * std::sin(2.0 * u);
    const Scalar z = base_z_ + vertical_amp_ * std::sin(u + phase_z_);
    return Vector<3>(x, y, z);
  }

  const Scalar shape_mix = clampScalar(path_shape_mix_, 0.0,
                                       max_path_shape_mix_);
  const Scalar omega =
    2.0 * EIGEN_PI * wave_count_ / std::max(path_length_, kSmallNumber);
  const Scalar y =
    lateral_amp_ * std::sin(omega * s_clamped + phase_y_) +
    0.35 * lateral_amp_ * std::sin(1.7 * omega * s_clamped + 0.5 * phase_y_);
  const Scalar z =
    base_z_ + vertical_amp_ * std::sin(0.8 * omega * s_clamped + phase_z_);

  const Vector<3> forward_path(s_clamped, y, z);
  if (shape_mix <= kSmallNumber) {
    return forward_path;
  }

  const Scalar closed_radius_floor =
    clampScalar(figure_eight_radius_ *
                  (static_cast<Scalar>(0.35) +
                   static_cast<Scalar>(0.65) * shape_mix),
                static_cast<Scalar>(0.25), max_lateral_amp_);
  const Scalar radius =
    clampScalar(std::max(lateral_amp_, closed_radius_floor),
                static_cast<Scalar>(0.25), max_lateral_amp_);
  const Scalar u =
    2.0 * EIGEN_PI * s_clamped / std::max(path_length_, kSmallNumber);
  const Scalar curve_wave =
    std::round(clampScalar(wave_count_, 1.0, max_wave_count_));
  const Vector<3> figure_eight_path(
    radius * std::sin(u) +
      static_cast<Scalar>(0.15) * radius * shape_mix *
        std::sin((curve_wave + static_cast<Scalar>(1.0)) * u + phase_y_),
    static_cast<Scalar>(0.5) * radius * std::sin(2.0 * u) +
      static_cast<Scalar>(0.10) * radius * shape_mix *
        std::sin((curve_wave + static_cast<Scalar>(2.0)) * u +
                 static_cast<Scalar>(0.5) * phase_y_),
    base_z_ +
      vertical_amp_ *
        (static_cast<Scalar>(0.75) * std::sin(u + phase_z_) +
         static_cast<Scalar>(0.25) * shape_mix *
           std::sin((curve_wave + static_cast<Scalar>(1.0)) * u +
                    phase_y_)));

  return (static_cast<Scalar>(1.0) - shape_mix) * forward_path +
         shape_mix * figure_eight_path;
}

Vector<3> QuadrotorEnv::basePathDerivative(const Scalar s) const {
  const Scalar s_clamped = clampScalar(s, 0.0, path_length_);

  if (useFigureEightPath()) {
    const Scalar radius = std::max(lateral_amp_, kSmallNumber);
    const Scalar du_ds =
      2.0 * EIGEN_PI / std::max(path_length_, kSmallNumber);
    const Scalar u = du_ds * s_clamped;
    const Scalar dx = radius * std::cos(u) * du_ds;
    const Scalar dy = radius * std::cos(2.0 * u) * du_ds;
    const Scalar dz = vertical_amp_ * std::cos(u + phase_z_) * du_ds;
    return Vector<3>(dx, dy, dz);
  }

  const Scalar shape_mix = clampScalar(path_shape_mix_, 0.0,
                                       max_path_shape_mix_);
  const Scalar omega =
    2.0 * EIGEN_PI * wave_count_ / std::max(path_length_, kSmallNumber);
  const Scalar dy =
    lateral_amp_ * omega * std::cos(omega * s_clamped + phase_y_) +
    0.35 * lateral_amp_ * 1.7 * omega *
      std::cos(1.7 * omega * s_clamped + 0.5 * phase_y_);
  const Scalar dz =
    vertical_amp_ * 0.8 * omega *
    std::cos(0.8 * omega * s_clamped + phase_z_);

  const Vector<3> forward_derivative(1.0, dy, dz);
  if (shape_mix <= kSmallNumber) {
    return forward_derivative;
  }

  const Scalar closed_radius_floor =
    clampScalar(figure_eight_radius_ *
                  (static_cast<Scalar>(0.35) +
                   static_cast<Scalar>(0.65) * shape_mix),
                static_cast<Scalar>(0.25), max_lateral_amp_);
  const Scalar radius =
    clampScalar(std::max(lateral_amp_, closed_radius_floor),
                static_cast<Scalar>(0.25), max_lateral_amp_);
  const Scalar du_ds =
    2.0 * EIGEN_PI / std::max(path_length_, kSmallNumber);
  const Scalar u = du_ds * s_clamped;
  const Scalar curve_wave =
    std::round(clampScalar(wave_count_, 1.0, max_wave_count_));
  const Vector<3> figure_eight_derivative(
    radius * std::cos(u) * du_ds +
      static_cast<Scalar>(0.15) * radius * shape_mix *
        (curve_wave + static_cast<Scalar>(1.0)) *
        std::cos((curve_wave + static_cast<Scalar>(1.0)) * u + phase_y_) *
        du_ds,
    radius * std::cos(2.0 * u) * du_ds +
      static_cast<Scalar>(0.10) * radius * shape_mix *
        (curve_wave + static_cast<Scalar>(2.0)) *
        std::cos((curve_wave + static_cast<Scalar>(2.0)) * u +
                 static_cast<Scalar>(0.5) * phase_y_) *
        du_ds,
    vertical_amp_ *
      (static_cast<Scalar>(0.75) * std::cos(u + phase_z_) * du_ds +
       static_cast<Scalar>(0.25) * shape_mix *
         (curve_wave + static_cast<Scalar>(1.0)) *
         std::cos((curve_wave + static_cast<Scalar>(1.0)) * u + phase_y_) *
         du_ds));

  return (static_cast<Scalar>(1.0) - shape_mix) * forward_derivative +
         shape_mix * figure_eight_derivative;
}

Vector<3> QuadrotorEnv::controlPointPosition(const int index) const {
  const int clamped_index = std::max(0, std::min(index, checkpoint_count_));
  if (clamped_index == 0) {
    return basePathPosition(0.0);
  }
  return checkpointPosition(clamped_index - 1);
}

Vector<3> QuadrotorEnv::interpolateControlPath(const Scalar u) const {
  const Scalar u_clamped =
    clampScalar(u, 0.0, static_cast<Scalar>(checkpoint_count_));
  const int segment =
    std::min(static_cast<int>(std::floor(u_clamped)), checkpoint_count_ - 1);
  const Scalar t = u_clamped - static_cast<Scalar>(segment);

  const Vector<3> p0 = controlPointPosition(segment - 1);
  const Vector<3> p1 = controlPointPosition(segment);
  const Vector<3> p2 = controlPointPosition(segment + 1);
  const Vector<3> p3 = controlPointPosition(segment + 2);
  const Scalar t2 = t * t;
  const Scalar t3 = t2 * t;

  return static_cast<Scalar>(0.5) *
         ((static_cast<Scalar>(2.0) * p1) + (-p0 + p2) * t +
          (static_cast<Scalar>(2.0) * p0 -
           static_cast<Scalar>(5.0) * p1 +
           static_cast<Scalar>(4.0) * p2 - p3) *
            t2 +
          (-p0 + static_cast<Scalar>(3.0) * p1 -
           static_cast<Scalar>(3.0) * p2 + p3) *
            t3);
}

void QuadrotorEnv::rebuildSampledPath() {
  path_sample_count_ =
    std::max(2, std::min(path_sample_count_,
                         static_cast<int>(pathenv::kMaxPathSamples)));

  sampled_path_.setZero();
  sampled_s_.setZero();
  sampled_path_.col(0) = interpolateControlPath(0.0);
  sampled_s_(0) = 0.0;

  for (int i = 1; i < path_sample_count_; i++) {
    const Scalar alpha =
      static_cast<Scalar>(i) / static_cast<Scalar>(path_sample_count_ - 1);
    const Scalar u = alpha * static_cast<Scalar>(checkpoint_count_);
    sampled_path_.col(i) = interpolateControlPath(u);
    sampled_s_(i) =
      sampled_s_(i - 1) +
      (sampled_path_.col(i) - sampled_path_.col(i - 1)).norm();
  }

  sampled_path_length_ =
    std::max(sampled_s_(path_sample_count_ - 1), kSmallNumber);
}

Vector<3> QuadrotorEnv::pathPosition(const Scalar s) const {
  if (path_sample_count_ < 2) {
    return basePathPosition(clampScalar(s, 0.0, path_length_));
  }

  const Scalar s_clamped = clampScalar(s, 0.0, sampled_path_length_);
  if (s_clamped <= 0.0) return sampled_path_.col(0);
  if (s_clamped >= sampled_path_length_) {
    return sampled_path_.col(path_sample_count_ - 1);
  }

  for (int i = 0; i < path_sample_count_ - 1; i++) {
    const Scalar s0 = sampled_s_(i);
    const Scalar s1 = sampled_s_(i + 1);
    if (s_clamped <= s1 || i == path_sample_count_ - 2) {
      const Scalar t = (s_clamped - s0) / std::max(s1 - s0, kSmallNumber);
      return (static_cast<Scalar>(1.0) - t) * sampled_path_.col(i) +
             t * sampled_path_.col(i + 1);
    }
  }

  return sampled_path_.col(path_sample_count_ - 1);
}

Vector<3> QuadrotorEnv::pathDerivative(const Scalar s) const {
  const Scalar eps =
    std::max(static_cast<Scalar>(0.02),
             sampled_path_length_ /
               static_cast<Scalar>(std::max(path_sample_count_ - 1, 1)));
  const Scalar s0 = clampScalar(s - eps, 0.0, sampled_path_length_);
  const Scalar s1 = clampScalar(s + eps, 0.0, sampled_path_length_);
  if (s1 - s0 <= kSmallNumber) {
    return basePathDerivative(clampScalar(s, 0.0, path_length_));
  }
  return (pathPosition(s1) - pathPosition(s0)) / (s1 - s0);
}

Vector<3> QuadrotorEnv::pathTangent(const Scalar s) const {
  Vector<3> tangent = pathDerivative(s);
  const Scalar norm = tangent.norm();
  if (norm <= kSmallNumber) {
    return Vector<3>(1.0, 0.0, 0.0);
  }
  return tangent / norm;
}

Scalar QuadrotorEnv::checkpointProgress(const int index) const {
  const int clamped_index = std::max(0, std::min(index, checkpoint_count_));
  return sampled_path_length_ * static_cast<Scalar>(clamped_index) /
         static_cast<Scalar>(std::max(checkpoint_count_, 1));
}

Vector<3> QuadrotorEnv::checkpointBasePosition(const int index) const {
  const int clamped_index = std::max(0, std::min(index, checkpoint_count_ - 1));
  const Scalar s =
    path_length_ * static_cast<Scalar>(clamped_index + 1) /
    static_cast<Scalar>(std::max(checkpoint_count_, 1));
  return basePathPosition(s);
}

Vector<3> QuadrotorEnv::checkpointPosition(const int index) const {
  const int clamped_index = std::max(0, std::min(index, checkpoint_count_ - 1));
  return checkpointBasePosition(clamped_index) +
         checkpoint_offsets_.col(clamped_index);
}

Scalar QuadrotorEnv::segmentDistanceToPoint(const Vector<3> &a,
                                            const Vector<3> &b,
                                            const Vector<3> &p) const {
  const Vector<3> ab = b - a;
  const Scalar denom = std::max(ab.squaredNorm(), kSmallNumber);
  const Scalar t = clampScalar((p - a).dot(ab) / denom, 0.0, 1.0);
  return (a + t * ab - p).norm();
}

Scalar QuadrotorEnv::closestPathProgress(const Vector<3> &position) const {
  const auto search_segments = [&](const Scalar lower_bound,
                                   const Scalar upper_bound,
                                   bool *found) {
    Scalar best_s = clampScalar(progress_s_, 0.0, sampled_path_length_);
    Scalar best_dist = (position - pathPosition(best_s)).squaredNorm();
    *found = false;

    const Scalar lower = clampScalar(lower_bound, 0.0, sampled_path_length_);
    const Scalar upper = clampScalar(upper_bound, 0.0, sampled_path_length_);
    for (int i = 0; i < path_sample_count_ - 1; i++) {
      const Scalar s0 = sampled_s_(i);
      const Scalar s1 = sampled_s_(i + 1);
      if (s1 < lower || s0 > upper) continue;

      const Vector<3> a = sampled_path_.col(i);
      const Vector<3> b = sampled_path_.col(i + 1);
      const Vector<3> ab = b - a;
      const Scalar t =
        clampScalar((position - a).dot(ab) /
                      std::max(ab.squaredNorm(), kSmallNumber),
                    0.0, 1.0);
      const Vector<3> candidate = a + t * ab;
      const Scalar dist = (position - candidate).squaredNorm();
      if (dist < best_dist || !*found) {
        best_dist = dist;
        best_s = s0 + t * (s1 - s0);
        *found = true;
      }
    }

    return best_s;
  };

  const Scalar seed = clampScalar(progress_s_, 0.0, sampled_path_length_);
  bool found = false;
  Scalar closest =
    search_segments(seed - closest_search_backward_,
                    seed + closest_search_forward_, &found);
  if (!found) {
    closest = search_segments(0.0, sampled_path_length_, &found);
  }
  return clampScalar(closest, 0.0, sampled_path_length_);
}

void QuadrotorEnv::evaluatePreviousEpisode() {
  if (episode_steps_ <= 0) return;

  const Scalar inv_steps = 1.0 / static_cast<Scalar>(episode_steps_);
  const Scalar avg_pos_error = episode_pos_error_sum_ * inv_steps;
  const Scalar avg_lookahead_error = episode_lookahead_error_sum_ * inv_steps;
  const Scalar progress_ratio = clampScalar(episode_progress_, 0.0, 1.0);

  last_episode_path_error_ = avg_pos_error;
  last_episode_lookahead_error_ = avg_lookahead_error;
  last_episode_progress_ = progress_ratio;
  last_episode_completed_ = completed_;
  last_episode_crashed_ = crashed_;
  last_checkpoint_pass_errors_.setOnes();
  last_checkpoint_pass_errors_ *=
    clampScalar(avg_pos_error, 0.0, max_tracking_error_);
  last_episode_pass_error_ = avg_pos_error;
  const Scalar path_error_ratio =
    clampScalar(avg_pos_error / std::max(max_tracking_error_, kSmallNumber),
                0.0, 1.0);
  const Scalar lookahead_error_ratio =
    clampScalar(avg_lookahead_error / static_cast<Scalar>(4.0), 0.0, 1.0);
  const Scalar tracking_score =
    static_cast<Scalar>(1.0) -
    clampScalar(static_cast<Scalar>(0.65) * path_error_ratio +
                  static_cast<Scalar>(0.35) * lookahead_error_ratio,
                0.0, 1.0);

  last_episode_score_ =
    static_cast<Scalar>(80.0) * progress_ratio +
    static_cast<Scalar>(70.0) * tracking_score -
    static_cast<Scalar>(60.0);
  if (completed_) last_episode_score_ += 60.0;
  if (crashed_) last_episode_score_ -= 80.0;
  if (timed_out_) last_episode_score_ -= 20.0 * (1.0 - progress_ratio);

  if (!std::isfinite(last_episode_score_)) {
    last_episode_score_ = -100.0;
  }

  // The infeasible reset counter is updated when the environment policy is
  // evaluated, not at every episode reset. This matches the paper's
  // evaluation-level reset rule and avoids over-counting short episodes.
}

void QuadrotorEnv::resetPathToInitialDifficulty() {
  if (useFigureEightPath()) {
    path_length_ =
      clampScalar(figure_eight_length_, min_path_length_, max_path_length_);
    target_speed_ =
      clampScalar(figure_eight_speed_, min_target_speed_, max_target_speed_);
    lateral_amp_ = clampScalar(figure_eight_radius_, 0.0, max_lateral_amp_);
    vertical_amp_ =
      clampScalar(figure_eight_vertical_amp_, 0.0, max_vertical_amp_);
    wave_count_ = 1.0;
    path_shape_mix_ = 1.0;
    phase_y_ = 0.0;
    phase_z_ = 0.0;
    checkpoint_offsets_.setZero();
    rebuildSampledPath();
    return;
  }

  path_length_ =
    clampScalar(initial_path_length_, min_path_length_, max_path_length_);
  target_speed_ =
    clampScalar(initial_target_speed_, min_target_speed_, max_target_speed_);
  lateral_amp_ = clampScalar(initial_lateral_amp_, 0.0, max_lateral_amp_);
  vertical_amp_ = clampScalar(initial_vertical_amp_, 0.0, max_vertical_amp_);
  wave_count_ =
    clampScalar(initial_wave_count_, min_wave_count_, max_wave_count_);
  path_shape_mix_ =
    clampScalar(initial_path_shape_mix_, 0.0, max_path_shape_mix_);
  phase_y_ = 0.0;
  phase_z_ = 0.5 * EIGEN_PI;
  checkpoint_offsets_.setZero();
  rebuildSampledPath();
}

}  // namespace flightlib

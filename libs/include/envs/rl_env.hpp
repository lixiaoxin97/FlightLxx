#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>

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
  //
  kPos = 0,
  kNPos = 3,
  kOri = 3,
  kNOri = 3,
  kLinVel = 6,
  kNLinVel = 3,
  kAngVel = 9,
  kNAngVel = 3,
  kNObs = 12,
  // control actions
  kAct = 0,
  kNAct = 4,
};
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

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  std::shared_ptr<Quadrotor> target_quadrotor_ptr_;
  std::shared_ptr<Quadrotor> std_quadrotor_ptr_;
  QuadState quad_state_;
  QuadState target_quad_state_;
  Command cmd_;
  Logger logger_{"QaudrotorEnv"};

  // Define reward for training
  Scalar pos_coeff_, ori_coeff_, lin_vel_coeff_, ang_vel_coeff_, act_coeff_;

  // target position
  Vector<3> target_pos_;

  // postion before and after, distance to target_position
  Vector<3> position_before_;
  Vector<3> position_after_;
  Scalar distance_to_target_position_;

  // distance before and after , coefficient
  Scalar d_before_, d_after_, d_coeff_;

  // observations and actions (for RL)
  Vector<quadenv::kNObs> quad_obs_;
  Vector<quadenv::kNAct> quad_act_;

  // initial position
  Vector<3> position_initial_;

  // reward function design (for model-free reinforcement learning)
  Vector<quadenv::kNObs> goal_state_;

  // action and observation normalization (for learning)
  Vector<quadenv::kNAct> act_mean_;
  Vector<quadenv::kNAct> act_std_;
  Vector<quadenv::kNObs> obs_mean_ = Vector<quadenv::kNObs>::Zero();
  Vector<quadenv::kNObs> obs_std_ = Vector<quadenv::kNObs>::Ones();

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

Scalar distanceToSegment(const Vector<3>& A, 
                        const Vector<3>& B,
                        const Vector<3>& P);

}  // namespace flightlib
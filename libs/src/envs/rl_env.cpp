#include "rl_env.hpp"

namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FlightLxx_PATH") +
                 std::string("/libs/config/rl_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase(),
    pos_coeff_(0.0),
    ori_coeff_(0.0),
    lin_vel_coeff_(0.0),
    ang_vel_coeff_(0.0),
    act_coeff_(0.0),
    d_coeff_(-1.0),
    goal_state_((Vector<quadenv::kNObs>() << 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0)
                  .finished()) {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  
  //###################### Dynamic Randominzation ##############################
  //############################################################################
  // update dynamics
  // QuadrotorDynamics dynamics;
  // dynamics.updateParams(cfg_);
  // dynamics.showQuadrotorDynamicsParams();
  // quadrotor_ptr_->updateDynamics(dynamics);
  //############################################################################

  // define a bounding box
  world_box_ << -20, 20, -20, 20, 0, 20;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;

  Scalar mass = quadrotor_ptr_->getMass();
  //############################################################################
  //############################## Action ######################################
  act_mean_ = Vector<quadenv::kNAct>::Ones() * (-Gz) ;
  act_mean_.segment<3>(0).setZero();
  act_std_ = Vector<quadenv::kNAct>::Ones() * 2 * 3.1415926;
  act_std_(2) = 3.1415926;
  act_std_(3) = (-Gz*1);
  //############################################################################
  //############################################################################

  // load parameters
  loadParam(cfg_);
}

QuadrotorEnv::~QuadrotorEnv() {}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  target_pos_.setZero();
  quad_act_.setZero();

  if (random) {
    // randomly reset the quadrotor state
    //############################################################################
    //################################ State #####################################
    // | P | Q | V | W |
    // reset position #############################################
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_) * 20;
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_) * 20;
    quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) * 5 + 10;
    if (quad_state_.x(QS::POSZ) < -0.0)
      quad_state_.x(QS::POSZ) = -quad_state_.x(QS::POSZ);
    // quad_state_.x(QS::POSX) = 0;
    // quad_state_.x(QS::POSY) = 0;
    // quad_state_.x(QS::POSZ) = 3;
    // if (quad_state_.x(QS::POSZ) < -0.0)
    //   quad_state_.x(QS::POSZ) = -quad_state_.x(QS::POSZ);
    // reset linear velocity ######################################
    quad_state_.x(QS::VELX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELZ) = uniform_dist_(random_gen_);
    // quad_state_.x(QS::VELX) = 0;
    // quad_state_.x(QS::VELY) = 0;
    // quad_state_.x(QS::VELZ) = 0;
    // reset orientation ##########################################
    quad_state_.x(QS::ATTW) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_);
    quad_state_.qx /= quad_state_.qx.norm();
    // quad_state_.x(QS::ATTW) = 1;
    // quad_state_.x(QS::ATTX) = 0;
    // quad_state_.x(QS::ATTY) = 0;
    // quad_state_.x(QS::ATTZ) = 0;
    // quad_state_.qx /= quad_state_.qx.norm();
    // reset body rate ############################################
    quad_state_.x(QS::OMEX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::OMEY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::OMEZ) = uniform_dist_(random_gen_);
    // quad_state_.x(QS::OMEX) = 0;
    // quad_state_.x(QS::OMEY) = 0;
    // quad_state_.x(QS::OMEZ) = 0;
    // reset target position ######################################
    target_pos_(0) = 0;
    target_pos_(1) = 0;
    target_pos_(2) = 10;
    // target_pos_(0) = uniform_dist_(random_gen_) * 5;
    // target_pos_(1) = uniform_dist_(random_gen_) * 5;
    // target_pos_(2) = uniform_dist_(random_gen_) * 1.5 + 5;
    //############################################################################
    //############################################################################
  }


  //############################################################################
  //###################### Dynamic Randominzation ##############################
  //############################################################################
  YAML::Node cfg_ = YAML::LoadFile(getenv("FlightLxx_PATH") + std::string("/libs/config/rl_env.yaml"));
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  dynamics.dynamicRandomization();
  dynamics.showQuadrotorDynamicsParams();
  quadrotor_ptr_->updateDynamics(dynamics);
  //############################################################################
  //############################################################################
  //############################################################################


  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  // cmd_.thrusts.setZero();
  cmd_.collective_thrust = 0.0;
  cmd_.omega.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  // quaternionToEuler(quad_state_.q(), euler);
  quad_obs_ << quad_state_.p - target_pos_, euler_zyx, quad_state_.v, quad_state_.w;

  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  return true;
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  //############################################################################
  //############################## Action ######################################
  // | Wx | Wy | Wz| CT |
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.omega = quad_act_.segment<3>(0);
  cmd_.collective_thrust = quad_act_(3);
  //############################################################################
  //############################################################################

  d_before_ = (quad_state_.p - target_pos_).norm();

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  d_after_ = (quad_state_.p - target_pos_).norm();

  Matrix<3, 3> rot = quad_state_.q().toRotationMatrix();

  // ---------------------- reward function design
  //############################################################################
  //################################# Reward ###################################
  // | pos | ori | lin | ang | act |
  // // - position tracking
  // Scalar pos_reward =
  //   pos_coeff_ * (quad_obs_.segment<quadenv::kNPos>(quadenv::kPos) -
  //                 goal_state_.segment<quadenv::kNPos>(quadenv::kPos))
  //                  .squaredNorm();
  // // - orientation tracking
  // Scalar ori_reward =
  //   ori_coeff_ * (quad_obs_.segment<quadenv::kNOri>(quadenv::kOri) -
  //                 goal_state_.segment<quadenv::kNOri>(quadenv::kOri))
  //                  .squaredNorm();
  // // - linear velocity tracking
  // Scalar lin_vel_reward =
  //   lin_vel_coeff_ * (quad_obs_.segment<quadenv::kNLinVel>(quadenv::kLinVel) -
  //                     goal_state_.segment<quadenv::kNLinVel>(quadenv::kLinVel))
  //                      .squaredNorm();
  // // - angular velocity tracking
  Scalar ang_vel_reward =
    ang_vel_coeff_ * (quad_obs_.segment<quadenv::kNAngVel>(quadenv::kAngVel) -
                      goal_state_.segment<quadenv::kNAngVel>(quadenv::kAngVel))
                       .squaredNorm();
  // 
  // // - control action penalty
  // Scalar act_reward = act_coeff_ * act.cast<Scalar>().norm();

  Scalar total_reward = d_coeff_ * (d_after_ - d_before_) + ang_vel_reward;

  // // survival reward
  // total_reward += 0.1;

  //############################################################################
  //############################################################################

  return total_reward;
}

//############################################################################
//################################# Reward ###################################
// | crash |
bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  if (quad_state_.x(QS::POSZ) <= 0.0) {
    reward = 0.0;
    return true;
  }
  if ((quad_state_.p - target_pos_).norm() <= 0.2) {
    reward = 0.0;
    return true;
  }
  reward = 0.0;
  return false;
}
//############################################################################
//############################################################################

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    pos_coeff_ = cfg["rl"]["pos_coeff"].as<Scalar>();
    ori_coeff_ = cfg["rl"]["ori_coeff"].as<Scalar>();
    lin_vel_coeff_ = cfg["rl"]["lin_vel_coeff"].as<Scalar>();
    ang_vel_coeff_ = cfg["rl"]["ang_vel_coeff"].as<Scalar>();
    act_coeff_ = cfg["rl"]["act_coeff"].as<Scalar>();
  } else {
    return false;
  }
  return true;
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

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
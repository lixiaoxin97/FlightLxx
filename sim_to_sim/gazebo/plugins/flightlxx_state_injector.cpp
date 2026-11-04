#include <cmath>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gz/math/Helpers.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/plugin/Register.hh>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <gz/transport/Node.hh>

namespace flightlxx::sim
{

// Holds a recovery pose, then releases a 12D state over two physics updates.
// Input state: [x,y,z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz], where position and
// velocity are in the FlightLxx NWU world frame, orientation is body-FLU to
// that world frame, and angular velocity is body-FLU. The policy frame is
// related to this physical frame by a fixed yaw calibration.
class StateInjector final : public gz::sim::System,
                            public gz::sim::ISystemConfigure,
                            public gz::sim::ISystemConfigurePriority,
                            public gz::sim::ISystemPreUpdate,
                            public gz::sim::ISystemPostUpdate
{
  public: gz::sim::System::PriorityType ConfigurePriority() override
  {
    // Install pose/velocity commands before Gazebo's Physics system consumes
    // and clears the one-step command components.
    return -100;
  }

  public: void Configure(const gz::sim::Entity &_entity,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         gz::sim::EntityComponentManager &_ecm,
                         gz::sim::EventManager &) override
  {
    this->model_ = gz::sim::Model(_entity);
    const std::string linkName =
        _sdf->Get<std::string>("link_name", "base_link").first;
    this->link_ = gz::sim::Link(this->model_.LinkByName(_ecm, linkName));
    this->commandTopic_ = _sdf->Get<std::string>(
        "command_topic", "/world/default/model/prototype/flightlxx/set_state").first;
    this->ackTopic_ = _sdf->Get<std::string>(
        "ack_topic", "/world/default/model/prototype/flightlxx/state_ack").first;
    this->releaseGravity_ = _sdf->Get<bool>("release_gravity", false).first;
    this->holding_ = _sdf->Get<bool>("hold_on_start", false).first;
    this->statePublisher_ = this->node_.Advertise<gz::msgs::StringMsg>(this->ackTopic_ + "/telemetry");
    if (!this->model_.Entity() || !this->link_.Entity()) {
      gzerr << "FlightLxx state injector could not resolve model/link\n";
      return;
    }
    this->link_.EnableVelocityChecks(_ecm);
    this->holdModelPose_ = gz::sim::worldPose(this->model_.Entity(), _ecm);
    this->ackPublisher_ = this->node_.Advertise<gz::msgs::StringMsg>(this->ackTopic_);
    this->subscribed_ = this->node_.Subscribe(
        this->commandTopic_, &StateInjector::OnCommand, this);
    if (!this->subscribed_) {
      gzerr << "FlightLxx state injector could not subscribe to "
            << this->commandTopic_ << "\n";
    } else {
      gzmsg << "FlightLxx state injector listening on " << this->commandTopic_
            << " (ack: " << this->ackTopic_ << ")\n";
    }
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &_info,
                         gz::sim::EntityComponentManager &_ecm) override
  {
    this->simulationTime_ = std::chrono::duration<double>(_info.simTime).count();
    // Physics clears these components' data to zero, but does not remove
    // them. Leaving the zero commands attached would override free dynamics
    // on EVERY later step. They belong to this injector for ONE step only.
    if (this->commandsApplied_ && !_info.paused) {
      _ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(this->link_.Entity());
      _ecm.RemoveComponent<gz::sim::components::AngularVelocityCmd>(this->link_.Entity());
      _ecm.RemoveComponent<gz::sim::components::WorldPoseCmd>(this->model_.Entity());
      this->commandsApplied_ = false;
    }
    // The physics system consumes WorldPoseCmd before velocity commands. Wait
    // one simulation tick so the ECM pose is the requested pose, then convert
    // the desired world velocity into the current link frame.
    if (this->pendingVelocity_ && !_info.paused) {
      const auto currentPose = gz::sim::worldPose(this->model_.Entity(), _ecm);
      this->link_.SetLinearVelocity(
          _ecm, currentPose.Rot().RotateVectorReverse(this->desiredWorldVelocity_));
      this->link_.SetAngularVelocity(_ecm, this->desiredBodyRates_);
      if (this->releaseGravity_) {
        this->link_.SetGravityEnabled(_ecm, true);
      }
      this->pendingVelocity_ = false;
      this->commandsApplied_ = true;
      this->releasedSimulationTime_ = this->simulationTime_;
      this->releaseStepSeconds_ = std::chrono::duration<double>(_info.dt).count();
      this->captureRelease_ = true;
    }
    std::optional<Request> request;
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      request.swap(this->pending_);
    }
    if (request) {
      if (request->operation == "anchor") {
        this->CaptureAnchor(*request, _ecm);
      } else if (request->operation == "set" || request->operation == "prepare"
                 || request->operation == "prepare_trajectory") {
        if (_info.paused) {
          this->PublishAck(request->id + ",error,unpause_before_injection");
        } else {
          this->ApplyState(*request, _ecm);
          if (request->operation == "prepare" || request->operation == "prepare_trajectory") {
            this->pendingVelocity_ = false;
            this->holding_ = true;
            this->prepared_ = true;
            if (request->operation == "prepare_trajectory") {
              this->trajectoryTargetTime_ = this->simulationTime_ + request->values[12];
              this->trajectoryTargetModelPose_ = this->holdModelPose_;
              this->trajectoryReleaseArmed_ = false;
            }
          }
        }
      } else if (request->operation == "release") {
        if (!this->prepared_ || _info.paused) {
          this->PublishAck(request->id + ",error,prepare_and_unpause_required");
        } else {
          if (this->trajectoryTargetTime_) {
            this->trajectoryReleaseArmed_ = true;
          } else {
            this->holding_ = false;
            this->pendingVelocity_ = true;
          }
          this->releaseWallTimeNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count();
          this->PublishAck(request->id + ",ok,release_queued");
        }
      } else if (request->operation == "read") {
        this->PublishMeasuredState(*request, _ecm);
      } else {
        this->PublishAck(request->id + ",error,unknown_operation");
      }
    }
    if (this->holding_ && this->trajectoryTargetTime_ && !_info.paused) {
      const double remaining = *this->trajectoryTargetTime_ - this->simulationTime_;
      const auto targetLink = this->trajectoryTargetModelPose_ * this->modelToLink_;
      const double angle = -remaining * this->desiredBodyRates_.Length();
      const gz::math::Quaterniond turn(this->desiredBodyRates_, angle);
      const gz::math::Pose3d movingLink(
          targetLink.Pos() - this->desiredWorldVelocity_ * remaining,
          targetLink.Rot() * turn);
      this->holdModelPose_ = movingLink * this->modelToLink_.Inverse();
      if (remaining <= 1e-9 && this->trajectoryReleaseArmed_) {
        // At this boundary, pose and twist already match the requested state.
        // Keep only the one-step twist command, then remove it on the next tick.
        this->holding_ = false;
        if (this->releaseGravity_) {
          this->link_.SetGravityEnabled(_ecm, true);
        }
        this->link_.SetLinearVelocity(_ecm, movingLink.Rot().RotateVectorReverse(this->desiredWorldVelocity_));
        this->link_.SetAngularVelocity(_ecm, this->desiredBodyRates_);
        this->commandsApplied_ = true;
        this->releasedSimulationTime_ = this->simulationTime_;
        this->releaseStepSeconds_ = std::chrono::duration<double>(_info.dt).count();
        this->captureRelease_ = true;
      }
    }
    if (this->holding_ && !_info.paused) {
      this->HoldPose(_ecm);
    }
  }

  public: void PostUpdate(const gz::sim::UpdateInfo &_info,
                          const gz::sim::EntityComponentManager &_ecm) override
  {
    if (this->captureRelease_ && !_info.paused) {
      this->releaseSnapshot_ = this->MeasuredStateJson(_ecm);
      this->captureRelease_ = false;
    }
    if (!_info.paused) {
      gz::msgs::StringMsg state;
      const auto current = this->MeasuredStateJson(_ecm);
      state.set_data(current.substr(0, current.size()-1) +
          ",\"release_physics_step_s\":" + std::to_string(this->releaseStepSeconds_) +
          ",\"first_released_state\":" +
          (this->releaseSnapshot_.empty() ? "null" : this->releaseSnapshot_) + "}");
      this->statePublisher_.Publish(state);
    }
  }

  private: void HoldPose(gz::sim::EntityComponentManager &_ecm)
  {
    // Gravity suppression alone does not constrain rotor/joint disturbances.
    this->link_.SetGravityEnabled(_ecm, false);
    this->model_.SetWorldPoseCmd(_ecm, this->holdModelPose_);
    const auto pose = gz::sim::worldPose(this->link_.Entity(), _ecm);
    this->link_.SetLinearVelocity(_ecm, this->prepared_
        ? pose.Rot().RotateVectorReverse(this->desiredWorldVelocity_)
        : gz::math::Vector3d::Zero);
    this->link_.SetAngularVelocity(_ecm, this->prepared_
        ? this->desiredBodyRates_ : gz::math::Vector3d::Zero);
    this->commandsApplied_ = true;
  }

  private: struct Request
  {
    std::string operation;
    std::string id;
    std::vector<double> values;
  };

  private: static bool Parse(const std::string &_payload, Request &_request)
  {
    std::stringstream stream(_payload);
    std::string token;
    if (!std::getline(stream, _request.operation, ',') ||
        !std::getline(stream, _request.id, ',')) {
      return false;
    }
    while (std::getline(stream, token, ',')) {
      try {
        std::size_t consumed = 0;
        const double value = std::stod(token, &consumed);
        if (consumed != token.size() || !std::isfinite(value)) {
          return false;
        }
        _request.values.push_back(value);
      } catch (...) {
        return false;
      }
    }
    return !_request.id.empty();
  }

  private: void OnCommand(const gz::msgs::StringMsg &_message)
  {
    Request request;
    if (!Parse(_message.data(), request)) {
      this->PublishAck("unknown,error,malformed_request");
      return;
    }
    const std::size_t expected = request.operation == "prepare_trajectory" ? 13 :
                                 request.operation == "anchor" ? 11 :
                                 (request.operation == "set" || request.operation == "prepare") ? 12 :
                                 (request.operation == "read" || request.operation == "release") ? 0 :
                                 std::numeric_limits<std::size_t>::max();
    if (request.values.size() != expected) {
      this->PublishAck(request.id + ",error,wrong_field_count");
      return;
    }
    if (request.operation == "prepare_trajectory"
        && (request.values[12] < 0.1 || request.values[12] > 2.0)) {
      this->PublishAck(request.id + ",error,trajectory_lead_out_of_bounds");
      return;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->pending_ = std::move(request);
  }

  private: void CaptureAnchor(const Request &_request,
                              gz::sim::EntityComponentManager &_ecm)
  {
    this->anchorLinkPose_ = gz::sim::worldPose(this->link_.Entity(), _ecm);
    const auto modelPose = gz::sim::worldPose(this->model_.Entity(), _ecm);
    this->modelToLink_ = modelPose.Inverse() * this->anchorLinkPose_;
    this->anchorPhysicalPosition_ = gz::math::Vector3d(
        _request.values[0], _request.values[1], _request.values[2]);
    this->physicalHover_ = gz::math::Vector3d(
        _request.values[3], _request.values[4], _request.values[5]);
    this->physicalYawReference_ = _request.values[6];
    this->nnGoal_ = gz::math::Vector3d(
        _request.values[7], _request.values[8], _request.values[9]);
    this->nnYawGoal_ = _request.values[10];
    this->anchorReady_ = true;
    this->holdModelPose_ = modelPose;
    this->holding_ = true;
    this->prepared_ = false;
    this->pendingVelocity_ = false;
    this->releasedSimulationTime_.reset();
    this->captureRelease_ = false;
    this->releaseWallTimeNs_ = 0;
    this->releaseStepSeconds_ = 0.0;
    this->releaseSnapshot_.clear();
    this->trajectoryTargetTime_.reset();
    this->trajectoryReleaseArmed_ = false;
    this->PublishAck(_request.id + ",ok,anchor_captured_holding");
  }

  private: void ApplyState(const Request &_request,
                           gz::sim::EntityComponentManager &_ecm)
  {
    if (!this->anchorReady_) {
      this->PublishAck(_request.id + ",error,anchor_required");
      return;
    }
    const auto &v = _request.values;
    const double alpha = this->nnYawGoal_ - this->physicalYawReference_;
    const auto rotateZ = [](const gz::math::Vector3d &_value, double _angle) {
      const double c = std::cos(_angle);
      const double s = std::sin(_angle);
      return gz::math::Vector3d(
          c * _value.X() - s * _value.Y(),
          s * _value.X() + c * _value.Y(), _value.Z());
    };

    const gz::math::Vector3d targetNetworkPosition(v[0], v[1], v[2]);
    const gz::math::Vector3d targetPhysicalPosition = this->physicalHover_ +
        rotateZ(targetNetworkPosition - this->nnGoal_, -alpha);
    const gz::math::Vector3d deltaPhysicalPosition =
        targetPhysicalPosition - this->anchorPhysicalPosition_;
    // FlightLxx NWU -> Gazebo ENU: Rz(+pi/2).
    const gz::math::Vector3d deltaGazeboPosition(
        -deltaPhysicalPosition.Y(), deltaPhysicalPosition.X(),
        deltaPhysicalPosition.Z());

    const gz::math::Quaterniond networkAttitude(v[3], v[4], v[5]);
    const gz::math::Quaterniond policyToPhysical(0.0, 0.0, -alpha);
    const gz::math::Quaterniond nwuToEnu(0.0, 0.0, GZ_PI_2);
    const gz::math::Quaterniond gazeboAttitude =
        nwuToEnu * policyToPhysical * networkAttitude;
    const gz::math::Pose3d targetLinkPose(
        this->anchorLinkPose_.Pos() + deltaGazeboPosition, gazeboAttitude);
    const auto targetModelPose = targetLinkPose * this->modelToLink_.Inverse();
    this->holdModelPose_ = targetModelPose;
    this->holding_ = false;
    this->prepared_ = false;
    this->model_.SetWorldPoseCmd(_ecm, targetModelPose);
    this->resetCounter_ = (this->resetCounter_ + 1) % 256;

    const gz::math::Vector3d networkVelocity(v[6], v[7], v[8]);
    const gz::math::Vector3d physicalVelocity = rotateZ(networkVelocity, -alpha);
    const gz::math::Vector3d gazeboWorldVelocity(
        -physicalVelocity.Y(), physicalVelocity.X(), physicalVelocity.Z());
    this->desiredWorldVelocity_ = gazeboWorldVelocity;
    this->desiredBodyRates_ = gz::math::Vector3d(v[9], v[10], v[11]);
    this->pendingVelocity_ = true;
    this->releasedSimulationTime_.reset();
    this->releaseSnapshot_.clear();
    this->trajectoryTargetTime_.reset();
    this->trajectoryReleaseArmed_ = false;
    this->releaseWallTimeNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    // This only acknowledges queued ECM commands, NEVER estimator validity.
    std::ostringstream response;
    response.precision(17);
    response << _request.id << ",ok,commands_queued,{\"sim_time_s\":"
             << this->simulationTime_ << '}';
    this->PublishAck(response.str());
  }

  private: std::string MeasuredStateJson(
      const gz::sim::EntityComponentManager &_ecm)
  {
    const auto linear = this->link_.WorldLinearVelocity(_ecm);
    const auto angular = this->link_.WorldAngularVelocity(_ecm);
    const auto pose = gz::sim::worldPose(this->link_.Entity(), _ecm);
    if (!linear || !angular) {
      return "null";
    }
    std::ostringstream response;
    response.precision(17);
    response << "{\"sim_time_s\":" << this->simulationTime_
             << ",\"position_enu\":[" << pose.Pos().X() << ','
             << pose.Pos().Y() << ',' << pose.Pos().Z()
             << "],\"quaternion_enu_flu_wxyz\":[" << pose.Rot().W() << ','
             << pose.Rot().X() << ',' << pose.Rot().Y() << ',' << pose.Rot().Z()
             << "],\"velocity_enu\":[" << linear->X() << ','
             << linear->Y() << ',' << linear->Z()
             << "],\"angular_velocity_world_enu\":[" << angular->X() << ','
             << angular->Y() << ',' << angular->Z()
             << "],\"holding\":" << (this->holding_ ? "true" : "false")
             << ",\"prepared\":" << (this->prepared_ ? "true" : "false")
             << ",\"prepared_world_velocity_enu\":[" << this->desiredWorldVelocity_.X()
             << ',' << this->desiredWorldVelocity_.Y() << ',' << this->desiredWorldVelocity_.Z()
             << "],\"prepared_body_rates_flu\":[" << this->desiredBodyRates_.X()
             << ',' << this->desiredBodyRates_.Y() << ',' << this->desiredBodyRates_.Z() << ']'
             << ",\"reset_counter\":" << this->resetCounter_
             << ",\"trajectory_target_sim_time_s\":";
    if (this->trajectoryTargetTime_) {
      response << *this->trajectoryTargetTime_;
    } else {
      response << "null";
    }
    response << ",\"trajectory_release_armed\":" << (this->trajectoryReleaseArmed_ ? "true" : "false")
             << ",\"release_pending\":" << (this->pendingVelocity_ ? "true" : "false")
             << ",\"released_sim_time_s\":";
    if (this->releasedSimulationTime_) {
      response << *this->releasedSimulationTime_;
    } else {
      response << "null";
    }
    response << ",\"velocity_command_attached\":"
             << ((_ecm.Component<gz::sim::components::LinearVelocityCmd>(this->link_.Entity())
                  || _ecm.Component<gz::sim::components::AngularVelocityCmd>(this->link_.Entity()))
                 ? "true" : "false") << '}';
    return response.str();
  }

  private: void PublishMeasuredState(
      const Request &_request,
      const gz::sim::EntityComponentManager &_ecm)
  {
    const auto state = this->MeasuredStateJson(_ecm);
    if (state == "null") {
      this->PublishAck(_request.id + ",error,velocity_unavailable");
      return;
    }
    // Retain the first post-physics sample; a new CLI process may discover the
    // transport too late to observe the actual release tick directly.
    this->PublishAck(_request.id + ",ok,read," + state.substr(0, state.size()-1) +
        ",\"release_request_wall_time_ns\":" + std::to_string(this->releaseWallTimeNs_) +
        ",\"release_physics_step_s\":" + std::to_string(this->releaseStepSeconds_) +
        ",\"first_released_state\":" +
        (this->releaseSnapshot_.empty() ? "null" : this->releaseSnapshot_) + "}");
  }

  private: void PublishAck(const std::string &_data)
  {
    if (!this->ackPublisher_) {
      return;
    }
    gz::msgs::StringMsg message;
    message.set_data(_data);
    this->ackPublisher_.Publish(message);
  }

  private: gz::sim::Model model_;
  private: gz::sim::Link link_;
  private: gz::transport::Node node_;
  private: gz::transport::Node::Publisher ackPublisher_;
  private: gz::transport::Node::Publisher statePublisher_;
  private: std::string commandTopic_;
  private: std::string ackTopic_;
  private: bool subscribed_{false};
  private: std::mutex mutex_;
  private: std::optional<Request> pending_;
  private: bool anchorReady_{false};
  private: gz::math::Pose3d anchorLinkPose_;
  private: gz::math::Pose3d modelToLink_;
  private: gz::math::Vector3d anchorPhysicalPosition_;
  private: gz::math::Vector3d physicalHover_;
  private: gz::math::Vector3d nnGoal_;
  private: double physicalYawReference_{0.0};
  private: double nnYawGoal_{0.0};
  private: bool releaseGravity_{false};
  private: bool holding_{false};
  private: bool prepared_{false};
  private: std::optional<double> trajectoryTargetTime_;
  private: gz::math::Pose3d trajectoryTargetModelPose_;
  private: bool trajectoryReleaseArmed_{false};
  private: unsigned resetCounter_{0};
  private: gz::math::Pose3d holdModelPose_;
  private: std::optional<double> releasedSimulationTime_;
  private: bool captureRelease_{false};
  private: std::string releaseSnapshot_;
  private: int64_t releaseWallTimeNs_{0};
  private: double releaseStepSeconds_{0.0};
  private: bool commandsApplied_{false};
  private: bool pendingVelocity_{false};
  private: gz::math::Vector3d desiredWorldVelocity_;
  private: gz::math::Vector3d desiredBodyRates_;
  private: double simulationTime_{0.0};
};

}  // namespace flightlxx::sim

GZ_ADD_PLUGIN(flightlxx::sim::StateInjector,
              gz::sim::System,
              gz::sim::ISystemConfigure,
              gz::sim::ISystemConfigurePriority,
              gz::sim::ISystemPreUpdate,
              gz::sim::ISystemPostUpdate)


#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include <gz/msgs/stringmsg.pb.h>
#include <gz/transport/Node.hh>

class Client
{
  public: bool Run(const std::string &_commandTopic,
                   const std::string &_ackTopic,
                   const std::string &_payload,
                   const std::string &_requestId)
  {
    if (!this->node_.Subscribe(_ackTopic, &Client::OnAck, this)) {
      std::cerr << "Failed to subscribe to " << _ackTopic << '\n';
      return false;
    }
    auto publisher = this->node_.Advertise<gz::msgs::StringMsg>(_commandTopic);
    const auto connectionDeadline = std::chrono::steady_clock::now() +
                                    std::chrono::seconds(5);
    while (!publisher.HasConnections() &&
           std::chrono::steady_clock::now() < connectionDeadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!publisher.HasConnections()) {
      std::cerr << "No Gazebo subscriber found on " << _commandTopic << '\n';
      return false;
    }
    gz::msgs::StringMsg message;
    message.set_data(_payload);
    if (!publisher.Publish(message)) {
      std::cerr << "Failed to publish state request\n";
      return false;
    }

    std::unique_lock<std::mutex> lock(this->mutex_);
    const bool received = this->condition_.wait_for(
        lock, std::chrono::seconds(5), [&]() {
          return this->ack_.rfind(_requestId + ",", 0) == 0;
        });
    if (!received) {
      std::cerr << "Timed out waiting for state-injector acknowledgement\n";
      return false;
    }
    std::cout << this->ack_ << '\n';
    return this->ack_.find(",ok,") != std::string::npos;
  }

  private: void OnAck(const gz::msgs::StringMsg &_message)
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->ack_ = _message.data();
    this->condition_.notify_all();
  }

  private: gz::transport::Node node_;
  private: std::mutex mutex_;
  private: std::condition_variable condition_;
  private: std::string ack_;
};

int main(int _argc, char **_argv)
{
  if (_argc != 5) {
    std::cerr << "Usage: flightlxx_state_injector_cli COMMAND_TOPIC ACK_TOPIC "
                 "REQUEST_ID PAYLOAD\n";
    return 2;
  }
  Client client;
  return client.Run(_argv[1], _argv[2], _argv[4], _argv[3]) ? 0 : 1;
}

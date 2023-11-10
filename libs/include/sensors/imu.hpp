#pragma once

#include "types.hpp"
#include "sensor_base.hpp"

namespace flightlib {

class IMU : SensorBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IMU();
  ~IMU();

 private:
};
}  // namespace flightlib

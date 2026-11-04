cmake_minimum_required(VERSION 3.16)
project(flightlxx_state_injector LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GZ_SIM REQUIRED IMPORTED_TARGET gz-sim8)
pkg_check_modules(GZ_TRANSPORT REQUIRED IMPORTED_TARGET gz-transport13)
pkg_check_modules(GZ_MSGS REQUIRED IMPORTED_TARGET gz-msgs10)

add_library(flightlxx_state_injector SHARED flightlxx_state_injector.cpp)
target_compile_features(flightlxx_state_injector PRIVATE cxx_std_17)
target_link_libraries(flightlxx_state_injector PRIVATE
  PkgConfig::GZ_SIM
  PkgConfig::GZ_TRANSPORT
  PkgConfig::GZ_MSGS
)

add_executable(flightlxx_state_injector_cli state_injector_cli.cpp)
target_compile_features(flightlxx_state_injector_cli PRIVATE cxx_std_17)
target_link_libraries(flightlxx_state_injector_cli PRIVATE
  PkgConfig::GZ_TRANSPORT
  PkgConfig::GZ_MSGS
)

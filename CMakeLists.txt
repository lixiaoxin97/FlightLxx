cmake_minimum_required(VERSION 3.7.7)
project(FlightLxx)

# ----C++17
set(CMAKE_CXX_STANDARD 17)

# ----OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -fPIC -Wall -DNDEBUG  -fopenmp")

# ----OpenCV
# sudo apt install libopencv-dev
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
endif(OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})

# ----Eigen3
# sudo apt install libeigen3-dev
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
  message(STATUS "Eigen3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}")
endif(Eigen3_FOUND)
add_definitions(${EIGEN3_DEFINITIONS})
include_directories(${EIGEN3_INCLUDE_DIRS})

# ----yaml-cpp
# sudo apt install libyaml-cpp-dev
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
  message(STATUS "YAML_CPP_INCLUDE_DIR = ${YAML_CPP_INCLUDE_DIR}")
  message(STATUS "YAML_CPP_LIBRARIES = ${YAML_CPP_LIBRARIES}")
endif(yaml-cpp_FOUND)
include_directories(${YAML_CPP_INCLUDE_DIR})

# ----pybind11
# sudo apt install pybind11-dev
find_package(pybind11 REQUIRED)
if(pybind11_FOUND)
  message(STATUS "pybind11_INCLUDE_DIR = ${pybind11_INCLUDE_DIR}")
endif(pybind11_FOUND)
include_directories(${PYBIND11_INCLUDE_DIR})

# ----ZeroMQ zmqpp
# sudo apt install libsodium-dev
# git clone git://github.com/zeromq/libzmq.git
# cd libzmq
# ./autogen.sh 
# ./configure --with-libsodium && make
# sudo make install
# sudo ldconfig
# cd ../
# git clone git://github.com/zeromq/zmqpp.git
# cd zmqpp
# make
# make check
# sudo make install
# make installcheck

# ----flightlib include
include_directories(${PROJECT_SOURCE_DIR}/libs/include/bridges)
include_directories(${PROJECT_SOURCE_DIR}/libs/include/common)
include_directories(${PROJECT_SOURCE_DIR}/libs/include/dynamics)
include_directories(${PROJECT_SOURCE_DIR}/libs/include/envs)
include_directories(${PROJECT_SOURCE_DIR}/libs/include/json)
include_directories(${PROJECT_SOURCE_DIR}/libs/include/objects)
include_directories(${PROJECT_SOURCE_DIR}/libs/include/sensors)

# ----flightlib sources
file(GLOB_RECURSE FLIGHTLIB_SOURCES
  ${PROJECT_SOURCE_DIR}/libs/src/bridges/*.cpp
  ${PROJECT_SOURCE_DIR}/libs/src/dynamics/*.cpp
  ${PROJECT_SOURCE_DIR}/libs/src/objects/*.cpp
  ${PROJECT_SOURCE_DIR}/libs/src/sensors/*.cpp
  ${PROJECT_SOURCE_DIR}/libs/src/envs/*.cpp
  ${PROJECT_SOURCE_DIR}/libs/src/common/*.cpp
)

# ----flightlib
add_library(flightlib ${FLIGHTLIB_SOURCES})

target_link_libraries(flightlib PRIVATE
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
  zmq
  zmqpp
  stdc++fs)

set_target_properties(flightlib PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# ----flightgym source
file(GLOB_RECURSE FLIGHTLIB_GYM_SOURCE
  ${PROJECT_SOURCE_DIR}/libs/src/wrapper/*.cpp 
)

# flightgym
pybind11_add_module(flightgym MODULE 
  ${FLIGHTLIB_GYM_SOURCE})

target_include_directories(flightgym PRIVATE
  ${PYBIND11_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/libs/include
  ${EIGEN3_INCLUDE_DIR})  

target_link_libraries(flightgym PRIVATE flightlib)
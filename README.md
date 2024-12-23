# FlightLxx
**FlightLxx** is a quadrotor simulator with deep reinforcement learning.   
This idea was originally from rpg/flightmare. I made several changes.  
MIT License

# Installation
## FlightLxx_PATH
echo "export FlightLxx_PATH=~/Desktop/FlightLxx" >> ~/.bashrc  
source ~/.bashrc  

## CMake
sudo apt install cmake  
## build-essential
sudo apt install build-essential  
## OpenCV
sudo apt install libopencv-dev  
## Eigen3
sudo apt install libeigen3-dev  
## yaml-cpp
sudo apt install libyaml-cpp-dev  
## pybind11
sudo apt install pybind11-dev  
## ZeroMQ zmqpp
sudo apt install libsodium-dev  
git clone git://github.com/zeromq/libzmq.git  
cd libzmq  
./autogen.sh  
./configure --with-libsodium && make  
sudo make install  
sudo ldconfig  
cd ../  
git clone git://github.com/zeromq/zmqpp.git  
cd zmqpp  
make  
make check  
sudo make install  
make installcheck  

## Anaconda
bash Anaconda3-2023.09-0-Linux-x86_64.sh  
conda config --set auto_activate_base false  
## conda environment
conda create --name flightlxx3.8 python=3.8.10  
conda activate flightlxx3.8  
pip install --user nvidia-pyindex  
conda install -c conda-forge openmpi  
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/flightlxx3.8/lib/" >> ~/.bashrc  
pip install --user nvidia-tensorflow[horovod]  
## test FlightLxx environment
import tensorflow as tf  
print(tf.__version__)
print(tf.test.is_gpu_available())  

## use FlightLxx
conda activate flightlxx3.8  
cd FlightLxx  
pip install .  
cd rl/train  
python3 run_drone_control.py --train 1 --render 0  
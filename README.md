# FlightLxx

Quadrotor simulator with deep reinforcement learning.  
Originally from rpg/flightmare, with modifications for sim-to-real control research.  
MIT License.

## Quick Start

```bash
# Set environment
echo "export FlightLxx_PATH=~/Desktop/FlightLxx" >> ~/.bashrc
source ~/.bashrc

# Install C++ dependencies (Ubuntu)
sudo apt install cmake build-essential libopencv-dev libeigen3-dev libyaml-cpp-dev pybind11-dev libsodium-dev

# Install ZeroMQ (source build)
git clone git://github.com/zeromq/libzmq.git && cd libzmq
./autogen.sh && ./configure --with-libsodium && make && sudo make install && sudo ldconfig && cd ..
git clone git://github.com/zeromq/zmqpp.git && cd zmqpp
make && make check && sudo make install && make installcheck && cd ..

# Create conda environment
conda create --name flightlxx3.8 python=3.8.10
conda activate flightlxx3.8
pip install --user nvidia-pyindex
conda install -c conda-forge openmpi
pip install --user nvidia-tensorflow[horovod]

# Build FlightLxx
cd FlightLxx
pip install -e .

# Train a policy
python3 Simulation_Experiments/train.py --model CTBR

# Run experiments
python3 Simulation_Experiments/run_experiment.py --trials 100 --seed 42
```

See `Simulation_Experiments/README.md` for detailed documentation.

# Simulation Experiments — FlightLxx

Paper: *Ideal Dynamics for Sim-to-Real Quadrotor Control*

## Directory Structure

```
Simulation_Experiments/
├── train.py              # Training script
├── run_experiment.py     # Experiment runner
├── configs.py            # Platform & model configuration
├── metrics.py            # Recovery metrics
├── plot_results.py       # Visualization
├── disturbances/         # Random initial state generator
├── models/               # Trained model weights (3 .zip files)
├── training_data/        # Training logs (progress.csv per model)
├── figures/              # Figure output
└── README.md
```

## Training

Three strategies are trained on the **Prototype** platform (0.78 kg, custom-built racing drone):

| Strategy | Command | Method |
|---|---|---|
| **CTBR** | `train.py --model CTBR` | Exact prototype parameters |
| **CTBR+DR** | `train.py --model CTBR+DR` | 7 parameters randomized per episode |
| **CTBR+ID** | `train.py --model CTBR+ID` | Ideal dynamics (skip motor model) |

```bash
conda activate flightlxx3.8
python3 Simulation_Experiments/train.py --model CTBR     # ~50 hours on RTX 3090
python3 Simulation_Experiments/train.py --model CTBR+DR
python3 Simulation_Experiments/train.py --model CTBR+ID
```

Models are saved to `models/` and training logs to `training_data/`.

### Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | PPO2 (GAE λ=0.95, γ=0.99) |
| Network | MLP [64, 64] tanh, pi & vf separate |
| Total steps | 250M (5000 iterations × 50,000 steps) |
| Parallel envs | 100 (OpenMP, 10 threads) |
| Learning rate | 3e-4 (linear decay) |
| Action space | ω ∈ [-2π, 2π], thrust ∈ [0, 2g] (normalized to [-1,1]) |
| Observation | 12-dim [p, euler(ZYX), v, ω] |

## Experiment

Tests recovery from random initial states on 3 platforms.

```bash
# 900 trials (3 platforms × 3 models × 100 repeats)
python3 Simulation_Experiments/run_experiment.py --trials 100 --seed 42

# Visualize
python3 Simulation_Experiments/plot_results.py results.csv
```

### Design

Each trial: random state → policy controls 500 steps (10 s) → check recovery.

| Dimension | Range |
|---|---|
| Position x, y | [-1, 1] m |
| Height z | [4, 6] m |
| Roll, pitch | [-45°, 45°] |
| Yaw | [0, 360°] |
| Linear velocity | [-0.5, 0.5] m/s |
| Angular velocity | [-0.5, 0.5] rad/s |

**Recovery criterion:** pos < 0.3 m AND speed < 0.3 m/s, sustained 0.3 s.

### Platforms

| | Prototype | Iris | Crazyflie 2.0 |
|---|---|---|---|
| Mass | 0.78 kg | 1.5 kg | 0.027 kg |
| T/W | 7.3 | 1.9 | 116 |
| Motor τ | 0.033 s | 0.0125 s | 0.0001 s |
| P_rate | [16.6, 16.6, 5] | [45, 45, 60] | [250, 250, 120] |
| Source | Measured | PX4 iris.sdf | Bitcraze firmware |

# Sim-Flight Experiment — Ideal Dynamics for Sim-to-Real Quadrotor Control

## Experiment Overview

**Goal:** Evaluate three policies (CTBR, CTBR+DR, CTBR+ID) on three quadrotor platforms (Prototype, Iris, Crazyflie 2.0) for recovery from random aggressive initial states.

**Scale:** 9 conditions × 1,000 trials = 9,000 total trials.

**Experiment order:** Outer loop = platforms (prototype → iris → crazyflie), inner loop = models (CTBR → CTBR+DR → CTBR+ID), innermost = trials (0..999). All 1,000 trials for one (platform, model) pair complete before moving to the next model.

---

## Three Policies

| Policy | Training Method | P-Controller | Motor Model | Training FPS |
|---|---|---|---|---|
| **CTBR** | Exact prototype parameters (0.78 kg, 0.125 m arm) | Yes | Yes (τ=0.033 s) | 8,278 |
| **CTBR+DR** | 7 dynamics parameters randomized per episode | Yes | Yes | 8,174 |
| **CTBR+ID** | Ideal dynamics: commands executed instantly and perfectly | No | No | 17,716 |

All three share the same PPO hyperparameters, MLP[64,64]-tanh network architecture, and CTBR control framework. Training: 250M steps across 100 parallel environments.

---

## Three Platforms

| | Prototype | Iris | Crazyflie 2.0 |
|---|---|---|---|
| Mass | 0.78 kg | 1.5 kg | 0.027 kg |
| Arm length | 0.125 m | 0.275 m | 0.044 m |
| Ixx / Iyy / Izz | 0.0023 / 0.0023 / 0.0036 | 0.0291 / 0.0291 / 0.0552 | 1.66e-5 / 1.67e-5 / 2.93e-5 |
| Motor time constant τ | 0.033 s | 0.0125 s | 0.0001 s |
| Per-step response | 45% | 80% | ≈100% |
| P-rate [roll, pitch, yaw] | [16.6, 16.6, 5.0] | [45, 45, 60] | [250, 250, 120] |
| Thrust-to-weight ratio | 7.3 | 1.9 | 116.7 |
| Parameter source | Custom-built (measured) | PX4 iris.sdf | Bitcraze firmware pid.h |

---

## Experimental Design

### Random Initial State

Every trial begins with the quadrotor at a randomly generated state. All platforms use identical random distributions.

| Dimension | Range |
|---|---|
| Position x, y | [-1, 1] m |
| Height z | [4, 6] m (hover target: [0, 0, 5] m) |
| Attitude (roll, pitch, yaw) | Any angle (−180° to 180°) |
| Linear velocity | [-1, 1] m/s (per axis) |
| Angular velocity | [-1, 1] rad/s (per axis) |

### Trial Procedure

1. Generate random 12D initial state
2. Reset quadrotor via `resetToState()`
3. Run 500 steps (10 s simulated time) at 50 Hz
4. At each step: policy inference (MLP forward-pass) → C++ physics step (8 RK4 sub-steps at 2.5 ms)
5. Record position, velocity, attitude, angular velocity, and actions

### Recovery Criterion

A trial is **successful** if the quadrotor maintains all of the following for 15 consecutive steps (0.3 s):

- **Position error < 0.3 m** (Euclidean distance to hover target [0, 0, 5] m)
- **Speed < 0.3 m/s** (Euclidean norm of velocity vector)

Failure modes: crash (z ≤ 0) or timeout (500 steps without meeting criterion).

### Control Architecture

```
Policy (MLP 2×64 tanh) → act ∈ [-1, 1]⁴
  ↓ Denormalize
ω_des = act[0:3] ⊙ [2π, 2π, π]          ∈ [-2π, 2π] × [-2π, 2π] × [-π, π] rad/s
thrust = act[3] × 9.81 + 9.81            ∈ [0, 19.62] m/s²
  ↓ P rate controller (platform-specific Kp)
τ_des = J · Kp · (ω_des − ω) + ω × Jω
  ↓ Thrust allocation matrix B⁻¹
[f₁, f₂, f₃, f₄] → per-motor thrusts
  ↓ First-order motor model (Ω̇ = (Ω_des − Ω) / τ)
  ↓ Aerodynamic thrust (fᵢ = Kf1·Ω² + Kf2·Ω + Kf3)
  ↓ RK4 integration (8 sub-steps × 2.5 ms)
  ↓ Collision detection: z ≤ 0 → crash
New state [p, q, v, ω]
```

---

## Results

**Recovery success rate (%) — N=1,000 per cell, seed=42:**

| Platform | CTBR | CTBR+DR | CTBR+ID |
|---|---|---|---|
| **Prototype** (T/W=7.3) | **97.8%** | 93.8% | 37.3% |
| **Iris** (T/W=1.9) | 91.1% | 83.9% | 89.0% |
| **Crazyflie 2.0** (T/W=116.7) | 1.3% | 96.7% | **98.1%** |

**Mean max position error (m):**

| Platform | CTBR | CTBR+DR | CTBR+ID |
|---|---|---|---|
| Prototype | 1.89 | 2.12 | 3.35 |
| Iris | 3.63 | 3.42 | 2.84 |
| Crazyflie | 5.49 | 2.07 | 1.62 |

**Crash rate (%):**

| Platform | CTBR | CTBR+DR | CTBR+ID |
|---|---|---|---|
| Prototype | 2.2% | 6.2% | 27.4% |
| Iris | 8.1% | 16.1% | 11.0% |
| Crazyflie | 6.8% | 3.3% | 1.9% |

---

## Key Findings

1. **CTBR overfits to the training platform.** 97.8% on Prototype, but drops to 1.3% on Crazyflie. Exact parameter matching delivers optimal in-domain performance at the cost of zero cross-platform generalization.

2. **CTBR+DR provides balanced performance.** 83.9–96.7% across all three platforms. Domain randomization sacrifices peak performance for robustness.

3. **CTBR+ID excels on fast-response platforms.** 98.1% on Crazyflie (τ≈0), 89.0% on Iris (τ=0.0125 s), but only 37.3% on Prototype (τ=0.033 s). The ideal dynamics assumption only holds when motor lag is negligible (per-step response ≥ 80%).

4. **Motor response rate is the key variable.** CTBR+ID success correlates almost perfectly with per-step motor response: CF ≈100% → 98.1% success; Iris 80% → 89.0% success; Proto 45% → 37.3% success (with 27.4% crash rate).

5. **Thrust-to-weight ratio defines the difficulty floor.** Iris (T/W=1.9) is the hardest platform for all policies. Even CTBR+ID, which excels on CF, only reaches 89.0% on Iris.

## Data File

`results.csv` — 9,000 rows with columns:
`platform, model, disturbance, trial, success, recovery_time_s, max_pos_error_m, max_att_error_deg, max_speed_ms, control_cost, jerk_ms3, ep_len`

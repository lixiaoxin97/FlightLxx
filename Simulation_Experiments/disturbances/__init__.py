"""Random initial state generation for recovery experiments."""
import numpy as np


def random_initial_state(difficulty: str = "default"):
    return np.array([
        np.random.uniform(-1.0, 1.0),                          # px
        np.random.uniform(-1.0, 1.0),                          # py
        np.random.uniform(4.0, 6.0),                           # pz
        np.random.uniform(-np.pi, np.pi),                      # roll  any
        np.random.uniform(-np.pi, np.pi),                      # pitch any
        np.random.uniform(0, 2 * np.pi),                       # yaw   any
        np.random.uniform(-1.0, 1.0),                          # vx
        np.random.uniform(-1.0, 1.0),                          # vy
        np.random.uniform(-1.0, 1.0),                          # vz
        np.random.uniform(-1.0, 1.0),                          # ωx
        np.random.uniform(-1.0, 1.0),                          # ωy
        np.random.uniform(-1.0, 1.0),                          # ωz
    ], dtype=np.float32)

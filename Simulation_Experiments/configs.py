"""
Platform dynamics configuration management.
Provides easy switching between Prototype, Iris, and Crazyflie platforms.
"""
import os

FLIGHTLXX_PATH = os.environ.get("FlightLxx_PATH", os.path.expanduser("~/Desktop/FlightLxx"))

PLATFORM_CONFIGS = {
    "prototype": os.path.join(FLIGHTLXX_PATH, "libs", "config", "prototype.yaml"),
    "iris":      os.path.join(FLIGHTLXX_PATH, "libs", "config", "iris.yaml"),
    "crazyflie": os.path.join(FLIGHTLXX_PATH, "libs", "config", "crazyflie.yaml"),
}

PLATFORM_PARAMS = {
    "prototype": {
        "mass": 0.78, "arm_l": 0.125,
        "inertia": [0.0023, 0.0023, 0.0036],
        "K_tau": 0.01, "K_alpha": 0.033,
        "P_rate": [16.6, 16.6, 5.0],
    },
    "iris": {
        "mass": 1.5, "arm_l": 0.275,
        "inertia": [0.029125, 0.029125, 0.055225],
        "K_tau": 0.06, "K_alpha": 0.0125,
        "P_rate": [45, 45, 60],
    },
    "crazyflie": {
        "mass": 0.027, "arm_l": 0.044,
        "inertia": [1.657e-05, 1.666e-05, 2.926e-05],
        "K_tau": 0.005964552, "K_alpha": 0.0001,
        "P_rate": [250, 250, 120],
    },
}

MODEL_WEIGHTS = {
    "CTBR":    os.path.join(FLIGHTLXX_PATH, "Simulation_Experiments", "models", "CTBR.zip"),
    "CTBR+DR": os.path.join(FLIGHTLXX_PATH, "Simulation_Experiments", "models", "CTBR+DR.zip"),
    "CTBR+ID": os.path.join(FLIGHTLXX_PATH, "Simulation_Experiments", "models", "CTBR+ID.zip"),
}


def get_platform_config(name: str) -> str:
    """Read a platform YAML config file and return it as a string."""
    path = PLATFORM_CONFIGS[name]
    with open(path, 'r') as f:
        return f.read()


def get_model_weight(name: str) -> str:
    """Return the path to a trained model's .zip weight file."""
    return MODEL_WEIGHTS[name]

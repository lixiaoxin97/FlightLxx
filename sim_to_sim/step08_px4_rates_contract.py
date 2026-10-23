#!/usr/bin/env python3
"""Convert FlightLxx CTBR actions to PX4 body-rate/thrust setpoints.

The default vehicle profile is FlightLxx's Gazebo Prototype. PX4's
``VehicleRatesSetpoint.thrust_body`` is a normalized collective demand, not an
acceleration. This module converts FlightLxx's mass-normalized collective
thrust through the selected vehicle's motor model to that normalized demand.
"""

import argparse
from dataclasses import dataclass
import math
import sys

from step01_action_contract import raw_action_to_ctbr


# Values from PX4-Autopilot's 4001_gz_x500 airframe, x500 SDF and ULog.
# motor_omega_min/max are the effective SIM_GZ_EC output limits in rad/s;
# they must also be compatible with the SDF maxRotVelocity.
X500_THRUST_MODEL = {
    "vehicle_mass_kg": 2.0643076923076915,
    "motor_count": 4,
    "motor_thrust_coefficient": 8.54858e-6,
    "motor_omega_min": 150.0,
    "motor_omega_max": 1000.0,
    "thrust_model_factor": 0.0,
}

# PX4's SIM_GZ_EC output parameters are limited to [0, 1000]. Scale the
# Prototype's RPM range into that range and choose the quadratic coefficient
# that minimizes the worst relative thrust error against prototype.yaml over
# the normalized [idle, maximum] motor-command interval.
PROTOTYPE_THRUST_MODEL = {
    "vehicle_mass_kg": 0.78,
    "motor_count": 4,
    # Gazebo's T=k_f*omega^2 approximation; max relative fit error is ~1.1%.
    "motor_thrust_coefficient": 1.4131562456735816e-5,
    # PX4 constrains SIM_GZ_EC_MIN/MAX to [0, 1000]. These are the
    # corresponding command-speed limits used by the Gazebo motor model.
    "motor_omega_min": 38.0,
    "motor_omega_max": 1000.0,
    "thrust_model_factor": 0.0,
}
VEHICLE_THRUST_MODELS = {
    "prototype": PROTOTYPE_THRUST_MODEL,
    "x500": X500_THRUST_MODEL,
}
DEFAULT_VEHICLE_PROFILE = "prototype"
DEFAULT_THRUST_MODEL = VEHICLE_THRUST_MODELS[DEFAULT_VEHICLE_PROFILE]


@dataclass(frozen=True)
class ThrustMapping:
    """Physical and normalized values for one collective-thrust request."""

    requested_acceleration_m_s2: float
    px4_thrust_fraction: float
    motor_output_fraction: float
    motor_speed_rad_s: float
    minimum_acceleration_m_s2: float
    maximum_acceleration_m_s2: float
    saturated_low: bool
    saturated_high: bool


def map_collective_acceleration(
    collective_acceleration_m_s2,
    vehicle_mass_kg=DEFAULT_THRUST_MODEL["vehicle_mass_kg"],
    motor_count=DEFAULT_THRUST_MODEL["motor_count"],
    motor_thrust_coefficient=DEFAULT_THRUST_MODEL["motor_thrust_coefficient"],
    motor_omega_min=DEFAULT_THRUST_MODEL["motor_omega_min"],
    motor_omega_max=DEFAULT_THRUST_MODEL["motor_omega_max"],
    thrust_model_factor=DEFAULT_THRUST_MODEL["thrust_model_factor"],
):
    """Map mass-normalized thrust [m/s^2] to normalized PX4 collective thrust.

    The Gazebo motor model uses ``T = k_f * omega^2``.  PX4's motor output
    function maps a normalized thrust demand ``u`` to motor output ``x`` using
    ``u = f*x^2 + (1-f)*x`` (``THR_MDL_FAC=f``).  The physical speed limits
    are applied after this output curve, so invert in this order:

      1. requested specific thrust -> rotor speed using k_f and vehicle mass;
      2. rotor speed -> normalized PX4 motor output using omega min/max;
      3. normalized motor output -> PX4 thrust demand using THR_MDL_FAC.

    PX4's control allocator normalizes the collective thrust axis against the
    full actuator range.  ``CA_ROTOR*_CT`` is therefore not the Gazebo k_f and
    must not be substituted for ``motor_thrust_coefficient`` here.

    When the armed motor output has a nonzero minimum, thrust below that idle
    thrust is unreachable: this function sends the minimum command and reports
    ``saturated_low``.  Set the SITL output minimum to zero if the experiment
    must reproduce FlightLxx's exact zero-thrust endpoint.
    """
    acceleration = float(collective_acceleration_m_s2)
    mass = float(vehicle_mass_kg)
    count_value = float(motor_count)
    k_f = float(motor_thrust_coefficient)
    omega_min = float(motor_omega_min)
    omega_max = float(motor_omega_max)
    curve_factor = float(thrust_model_factor)

    values = (acceleration, mass, count_value, k_f, omega_min, omega_max, curve_factor)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("thrust model values must all be finite")
    if acceleration < 0.0:
        raise ValueError("collective acceleration must be nonnegative")
    if mass <= 0.0:
        raise ValueError("vehicle_mass_kg must be positive")
    if count_value < 1.0 or not count_value.is_integer():
        raise ValueError("motor_count must be a positive integer")
    if k_f <= 0.0:
        raise ValueError("motor_thrust_coefficient must be positive")
    if omega_min < 0.0 or omega_max <= omega_min:
        raise ValueError("motor speeds must satisfy 0 <= min < max")
    if not 0.0 <= curve_factor <= 1.0:
        raise ValueError("thrust_model_factor must be in [0, 1]")

    count = int(count_value)
    maximum_acceleration = count * k_f * omega_max**2 / mass
    minimum_acceleration = count * k_f * omega_min**2 / mass
    saturated_high = acceleration > maximum_acceleration
    saturated_low = acceleration < minimum_acceleration

    # Clamp to the thrust physically available from the configured motor range.
    requested_force_n = mass * min(acceleration, maximum_acceleration)
    rotor_speed = math.sqrt(requested_force_n / (count * k_f))
    motor_output_fraction = (
        (rotor_speed - omega_min) / (omega_max - omega_min)
    )
    motor_output_fraction = min(1.0, max(0.0, motor_output_fraction))

    # Invert FunctionMotors' model: rel_thrust = f*x^2 + (1-f)*x.
    px4_thrust_fraction = (
        curve_factor * motor_output_fraction**2
        + (1.0 - curve_factor) * motor_output_fraction
    )
    px4_thrust_fraction = min(1.0, max(0.0, px4_thrust_fraction))

    # Report the speed that results from the clipped actuator request.
    commanded_speed = omega_min + motor_output_fraction * (omega_max - omega_min)

    return ThrustMapping(
        requested_acceleration_m_s2=acceleration,
        px4_thrust_fraction=px4_thrust_fraction,
        motor_output_fraction=motor_output_fraction,
        motor_speed_rad_s=commanded_speed,
        minimum_acceleration_m_s2=minimum_acceleration,
        maximum_acceleration_m_s2=maximum_acceleration,
        saturated_low=saturated_low,
        saturated_high=saturated_high,
    )


def convert_raw_action_to_px4(raw_action, **thrust_model):
    """Return ``(rates_frd, thrust_body_frd, collective, mapping)``."""
    values = tuple(float(value) for value in raw_action)

    if len(values) != 4:
        raise ValueError("raw_action must contain exactly 4 values")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("raw_action contains NaN or Inf")
    if any(abs(value) > 1.0 + 1.0e-6 for value in values):
        raise ValueError("raw_action must be within [-1, 1]")
    # Bound only the roundoff tolerated above; reject material violations.
    values = tuple(min(1.0, max(-1.0, value)) for value in values)

    wx_flu, wy_flu, wz_flu, collective_acceleration = raw_action_to_ctbr(values)

    # FLU body rates -> FRD body rates.
    rates_frd = (wx_flu, -wy_flu, -wz_flu)
    mapping = map_collective_acceleration(collective_acceleration, **thrust_model)

    # PX4 multicopter upward thrust points along negative body-Z in FRD.
    thrust_body_frd = (0.0, 0.0, -mapping.px4_thrust_fraction)
    return rates_frd, thrust_body_frd, collective_acceleration, mapping


def raw_action_to_px4_rates_setpoint(raw_action, **thrust_model):
    """Backward-compatible three-value conversion interface."""
    rates, thrust, collective, _mapping = convert_raw_action_to_px4(
        raw_action, **thrust_model
    )
    return rates, thrust, collective


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def unit_interval_float(value):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1]")
    return value


def add_thrust_model_arguments(parser):
    """Add a named default profile with optional parameter overrides."""
    parser.add_argument(
        "--vehicle-profile", choices=tuple(VEHICLE_THRUST_MODELS),
        default=DEFAULT_VEHICLE_PROFILE,
        help="PX4/Gazebo vehicle profile (default: %s)" % DEFAULT_VEHICLE_PROFILE,
    )
    parser.add_argument(
        "--vehicle-mass-kg", type=positive_float,
        default=None, help="all-up SDF mass; overrides the selected profile",
    )
    parser.add_argument(
        "--motor-count", type=int, default=None,
        help="number of equal rotors; overrides the selected profile",
    )
    parser.add_argument(
        "--motor-thrust-coefficient", type=positive_float,
        default=None, help="Gazebo motorConstant k_f in N/(rad/s)^2",
    )
    parser.add_argument(
        "--motor-omega-min", type=nonnegative_float,
        default=None, help="effective armed SIM_GZ_EC_MIN in rad/s",
    )
    parser.add_argument(
        "--motor-omega-max", type=positive_float,
        default=None,
        help="SIM_GZ_EC_MAX in rad/s; must not exceed SDF maxRotVelocity",
    )
    parser.add_argument(
        "--thrust-model-factor", type=unit_interval_float,
        default=None, help="PX4 THR_MDL_FAC in [0, 1]",
    )


def thrust_model_from_args(args):
    profile = getattr(args, "vehicle_profile", DEFAULT_VEHICLE_PROFILE)
    if profile not in VEHICLE_THRUST_MODELS:
        raise ValueError("unknown vehicle profile: %s" % profile)
    model = dict(VEHICLE_THRUST_MODELS[profile])
    for name in model:
        override = getattr(args, name, None)
        if override is not None:
            model[name] = override
    # Validate cross-field constraints once before starting control.
    map_collective_acceleration(0.0, **model)
    return model


def _assert_close(actual, expected, tolerance=1.0e-6):
    if len(actual) != len(expected):
        raise AssertionError((actual, expected))
    for actual_value, expected_value in zip(actual, expected):
        if abs(actual_value - expected_value) > tolerance:
            raise AssertionError((actual, expected))


def run_self_test():
    # The default Prototype profile should map a 1 g request near hover.
    rates, thrust, collective, mapping = convert_raw_action_to_px4((0, 0, 0, 0))
    _assert_close(rates, (0.0, 0.0, 0.0))
    _assert_close(thrust, (0.0, 0.0, -0.3429551081), tolerance=1.0e-6)
    assert abs(collective - 9.81) < 1.0e-6
    assert abs(mapping.motor_speed_rad_s - 367.9228140) < 1.0e-6
    assert not mapping.saturated_high and not mapping.saturated_low

    # Keep an explicit regression for the selectable X500 profile.
    x500 = VEHICLE_THRUST_MODELS["x500"]
    _rates, x500_thrust, x500_collective, x500_mapping = convert_raw_action_to_px4(
        (0, 0, 0, 0), **x500
    )
    _assert_close(x500_thrust, (0.0, 0.0, -0.7288988027), tolerance=1.0e-6)
    assert abs(x500_collective - 9.81) < 1.0e-6
    assert abs(x500_mapping.motor_speed_rad_s - 769.5639823) < 1.0e-5

    class NoProfileArgs:
        vehicle_mass_kg = motor_count = motor_thrust_coefficient = None
        motor_omega_min = motor_omega_max = thrust_model_factor = None

    assert thrust_model_from_args(NoProfileArgs()) == PROTOTYPE_THRUST_MODEL

    # Prototype model's upper policy endpoint is within motor authority.
    _rates, proto_max_thrust, _collective, proto_max = convert_raw_action_to_px4(
        (0, 0, 0, 1)
    )
    _assert_close(proto_max_thrust, (0.0, 0.0, -0.5013736315), tolerance=1.0e-6)
    assert not proto_max.saturated_high

    # The X500 model cannot produce the policy's 2 g upper endpoint.
    _rates, x500_max_thrust, _collective, x500_max = convert_raw_action_to_px4(
        (0, 0, 0, 1), **x500
    )
    _assert_close(x500_max_thrust, (0.0, 0.0, -1.0))
    assert x500_max.saturated_high
    assert abs(x500_max.maximum_acceleration_m_s2 - 16.5645461321) < 1.0e-8

    # Prototype model should map hover-level thrust using its configured fit.
    class PrototypeArgs:
        vehicle_profile = "prototype"
        vehicle_mass_kg = motor_count = motor_thrust_coefficient = None
        motor_omega_min = motor_omega_max = thrust_model_factor = None

    prototype = thrust_model_from_args(PrototypeArgs())
    proto_hover = map_collective_acceleration(9.81, **prototype)
    assert prototype["vehicle_mass_kg"] == 0.78
    assert prototype["motor_omega_min"] == 38.0
    assert prototype["motor_omega_max"] == 1000.0
    assert abs(proto_hover.px4_thrust_fraction - 0.3429551081) < 1e-8

    # Verify the configured nonlinear PX4 curve and exact zero-minimum case.
    custom = {
        "vehicle_mass_kg": 1.0,
        "motor_count": 4,
        "motor_thrust_coefficient": 1.0,
        "motor_omega_min": 0.0,
        "motor_omega_max": 2.0,
        "thrust_model_factor": 1.0,
    }
    quarter_max = map_collective_acceleration(4.0, **custom)
    assert abs(quarter_max.motor_output_fraction - 0.5) < 1.0e-12
    assert abs(quarter_max.px4_thrust_fraction - 0.25) < 1.0e-12

    linear_curve = map_collective_acceleration(
        4.0, **dict(custom, thrust_model_factor=0.0)
    )
    assert abs(linear_curve.px4_thrust_fraction - 0.5) < 1.0e-12

    # Reconstruct acceleration through PX4's forward output curve and the
    # Gazebo quadratic motor model for both supported vehicle profiles.
    for profile in VEHICLE_THRUST_MODELS.values():
        for omega_min in (0.0, profile["motor_omega_min"]):
            for factor in (0.0, 0.3, 1.0):
                model = dict(profile, motor_omega_min=omega_min,
                             thrust_model_factor=factor)
                for acceleration in (1.0, 4.905, 9.81, 16.0, 19.62):
                    mapped = map_collective_acceleration(acceleration, **model)
                    demand = mapped.px4_thrust_fraction
                    if factor == 0.0:
                        motor_output = demand
                    else:
                        motor_output = (
                            math.sqrt((1.0 - factor)**2 + 4.0 * factor * demand)
                            - (1.0 - factor)
                        ) / (2.0 * factor)
                    speed = omega_min + motor_output * (
                        model["motor_omega_max"] - omega_min
                    )
                    reconstructed = (
                        model["motor_count"] * model["motor_thrust_coefficient"]
                        * speed**2 / model["vehicle_mass_kg"]
                    )
                    expected = min(acceleration, mapped.maximum_acceleration_m_s2)
                    assert math.isclose(reconstructed, expected, rel_tol=1.0e-12)

    # X500's 150 rad/s armed minimum cannot reproduce FlightLxx zero thrust.
    idle = map_collective_acceleration(0.0, **x500)
    assert idle.px4_thrust_fraction == 0.0
    assert idle.saturated_low
    zero_min = map_collective_acceleration(0.0, motor_omega_min=0.0)
    assert zero_min.motor_speed_rad_s == 0.0 and not zero_min.saturated_low

    # Tiny policy roundoff at the endpoint must not create negative thrust.
    _rates, thrust, collective = raw_action_to_px4_rates_setpoint(
        (0.0, 0.0, 0.0, -1.0000001)
    )
    assert collective == 0.0 and thrust[2] == 0.0

    # Axis/rate conversion remains unchanged.
    rates, _thrust, _collective = raw_action_to_px4_rates_setpoint((1, 1, 1, 0))
    _assert_close(rates, (2.0 * 3.1415926, -2.0 * 3.1415926, -3.1415926))

    for bad_action in (
        (0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.1),
    ):
        try:
            raw_action_to_px4_rates_setpoint(bad_action)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid action was accepted")

    print("step08 self-test: PASS")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FlightLxx raw action using x500 or Prototype model"
    )
    parser.add_argument(
        "action", nargs=4, type=float,
        metavar="VALUE",
        help="raw FlightLxx action",
    )
    add_thrust_model_arguments(parser)
    args = parser.parse_args()
    model = thrust_model_from_args(args)

    rates, thrust, collective, mapping = convert_raw_action_to_px4(
        args.action, **model
    )

    print("raw action:", tuple(args.action))
    print("PX4 rates FRD [rad/s]:", tuple(round(value, 6) for value in rates))
    print("PX4 thrust_body FRD:", tuple(round(value, 6) for value in thrust))
    print("FlightLxx collective [m/s^2]:", round(collective, 6))
    print("%s model max [m/s^2]:" % args.vehicle_profile,
          round(mapping.maximum_acceleration_m_s2, 6))
    print("commanded rotor speed [rad/s]:", round(mapping.motor_speed_rad_s, 3))
    if mapping.saturated_high:
        print("WARNING: requested collective exceeds model maximum; thrust is saturated")
    if mapping.saturated_low:
        print("WARNING: requested collective is below armed motor idle thrust")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        run_self_test()
        raise SystemExit(0)

    raise SystemExit(main())

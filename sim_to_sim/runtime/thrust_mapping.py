#!/usr/bin/env python3
"""Convert FlightLxx CTBR actions to PX4 body-rate/thrust setpoints.

The default vehicle profile is FlightLxx's Gazebo Prototype. PX4's
``VehicleRatesSetpoint.thrust_body`` is a normalized collective demand, not an
acceleration. This module converts FlightLxx's mass-normalized collective
thrust through the selected vehicle's motor model to that normalized demand.
"""

from dataclasses import dataclass
import math

from action_contract import raw_action_to_ctbr


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
DEFAULT_THRUST_MODEL = PROTOTYPE_THRUST_MODEL


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


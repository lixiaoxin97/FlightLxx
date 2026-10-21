"""PX4 feedback determines when an external body-rate setpoint may be sent.

No ROS imports: the same decisions are tested offline and used by step09.
"""

from dataclasses import dataclass


def px4_feedback_topic(base, message_class):
    """PX4 DDS appends _vN for messages with a nonzero MESSAGE_VERSION."""
    version = int(getattr(message_class, "MESSAGE_VERSION", 0))
    return base + ("_v%d" % version if version else "")


@dataclass(frozen=True)
class PublicationAuthority:
    status: str
    heartbeat: bool = False
    rates: bool = False


class OffboardAuthority:
    """Require fresh, agreeing status and controller feedback for NN CTBR.

    Keep heartbeat separate: PX4 needs OffboardControlMode before entering
    Offboard, but VehicleRatesSetpoint must not be sent during native modes.
    Receive times are monotonic, independent of PX4/ROS clock synchronization.
    """

    def __init__(self, max_feedback_age=1.5):
        self.max_feedback_age = max_feedback_age
        self.vehicle_status = None
        self.status_received_at = None
        self.control_mode = None
        self.control_mode_received_at = None

    def update_status(self, message, now):
        self.vehicle_status = message
        self.status_received_at = now

    def update_control_mode(self, message, now):
        self.control_mode = message
        self.control_mode_received_at = now

    def decision(self, now):
        for name, message, received_at in (
            ("vehicle_status", self.vehicle_status, self.status_received_at),
            ("control_mode", self.control_mode, self.control_mode_received_at),
        ):
            if message is None:
                return PublicationAuthority("waiting_" + name)
            if not 0 <= now - received_at <= self.max_feedback_age:
                return PublicationAuthority("stale_" + name)

        status, control = self.vehicle_status, self.control_mode
        if status.nav_state != status.NAVIGATION_STATE_OFFBOARD:
            return PublicationAuthority("px4_control", heartbeat=True)
        if status.failsafe:
            return PublicationAuthority("px4_failsafe")
        if status.arming_state != status.ARMING_STATE_ARMED or not control.flag_armed:
            return PublicationAuthority("offboard_disarmed", heartbeat=True)
        if (not control.flag_control_offboard_enabled
                or not control.flag_control_rates_enabled
                or control.flag_control_attitude_enabled):
            return PublicationAuthority("offboard_control_pending", heartbeat=True)
        return PublicationAuthority("offboard_control", heartbeat=True, rates=True)

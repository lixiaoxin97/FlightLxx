"""PX4 odometry to the deployed policy observation."""
from state_contract import vehicle_odometry_to_observation

def message_to_observation(
    message,
    physical_hover,
    nn_goal,
    physical_yaw_reference=0.0,
    nn_yaw_goal=0.0,
    attitude_encoding="principal_zyx",
):
    """Convert one PX4 VehicleOdometry message to policy input."""

    return vehicle_odometry_to_observation(
        position_ned=message.position,
        quaternion_ned_frd_wxyz=message.q,
        velocity_ned=message.velocity,
        angular_velocity_frd=message.angular_velocity,
        pose_frame=getattr(message, "pose_frame", 1),
        velocity_frame=getattr(message, "velocity_frame", 1),
        physical_hover=physical_hover,
        nn_goal=nn_goal,
        physical_yaw_reference=physical_yaw_reference,
        nn_yaw_goal=nn_yaw_goal,
        attitude_encoding=attitude_encoding,
    )

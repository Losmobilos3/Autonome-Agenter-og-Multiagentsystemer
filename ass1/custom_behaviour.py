import numpy as np

from irsim.lib import register_behavior
from irsim.util.util import WrapToPi, relative_position

# @register_behavior("diff", "RL_Circle")
# def beh_diff_RL_circle(ego_object, radius=0.5, external_objects = None, **kwargs):
#     """
#     Utilized reinforcement learning to get the object to move in a circle, around the goal point

#     Args:
#         state (np.array): Current state [x, y, theta] (3x1).
#         center_point (np.array): Center point [x, y, theta] (3x1).
#         radius (float): Radius to circle the center point at (default 0.5)

#     Returns:
#         np.array: Velocity [linear, angular] (2x1).
#     """

#     state = ego_object.state
#     center = ego_object.goal

#     # Pick a random angle (-5 / +5) (velocity is constant)

#     # Calculate if the robot is getting futher away from the desired radius.
#     # If yes: punish
#     # If no: Good!

#     # Profit!
#     linear = 1 # Scale linear velocity
#     angular = 1 # Change in angular velocity pr. frame


#     # Return velocity and angle
#     return np.array([[linear], [angular]])

# def judge_angle(angle):
#     pass


@register_behavior("diff", "circle_basic")
def beh_diff_dash(ego_object, external_objects=None, i=0, **kwargs):
    if external_objects is None:
        external_objects = []
    # print("This is a custom behavior example for differential drive with dash2")

    state = ego_object.state
    center_point = ego_object.goal
    radius = ego_object.goal_threshold
    _, max_vel = ego_object.get_vel_range()
    angle_tolerance = kwargs.get("angle_tolerance", 0.1)
    radius = 1
    # return DiffDash2(state, goal, max_vel, goal_threshold, angle_tolerance)
    return DiffCircleBasic(state, center_point, 0.5, radius)


def DiffCircleBasic(state, center_point, vel, radius):
    distance, angle = relative_position(state, center_point)

    # TODO: PID-regulator
    # Implementing just P-regulator for now
    # regulator for linear speed. using distance here
    p_linear = 0.25
    # i = 1
    # d = 1
    e_linear = distance - radius
    # print(distance, e)
    linear_diff = p_linear * e_linear
    # linear_diff = WrapToPi(wanted_angle_diff)

    # regulator for angular speed. using angle here
    p_angle = 0.6
    e_angle = WrapToPi(angle - np.pi/2)  # Want to be perpendicular to the center?
    # print(angle, state[2, 0], e_angle)
    wanted_angle_diff = p_angle * e_angle
    angle_diff = WrapToPi(wanted_angle_diff)
    # print(angle_diff)

    return np.array([[linear_diff], [wanted_angle_diff]])

def DiffDash2(state, goal, max_vel, goal_threshold=0.1, angle_tolerance=0.2):
    """
    Calculate the differential drive velocity to reach a goal.

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        goal (np.array): Goal position [x, y, theta] (3x1).
        max_vel (np.array): Maximum velocity [linear, angular] (2x1).
        goal_threshold (float): Distance threshold to consider goal reached (default 0.1).
        angle_tolerance (float): Allowable angular deviation (default 0.2).

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    distance, radian = relative_position(state, goal)

    if distance < goal_threshold:
        return np.zeros((2, 1))

    diff_radian = WrapToPi(radian - state[2, 0])
    linear = max_vel[0, 0] * np.cos(diff_radian)

    if abs(diff_radian) < angle_tolerance:
        angular = 0
    else:
        angular = max_vel[1, 0] * np.sign(diff_radian)

    return np.array([[linear], [angular]])
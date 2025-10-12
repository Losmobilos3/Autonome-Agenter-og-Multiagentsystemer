import numpy as np
from irsim.lib import register_behavior

@register_behavior("diff", "follower")
def beh_follower(ego_object, external_objects=None, **kwargs):
    """
    Employs the boid behavior pattern, following a specified leader

    Args:
        external_objects: External objects in the scene
        ego_object:

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    if external_objects is None:
        external_objects = []

    radius = kwargs["radius"]

    neighbors = [robot for robot in external_objects if distance_to_object(ego_object.position, robot.position) < radius]
    leader = next((robot for robot in neighbors if robot.name == "leader"), None)

    # Separation: Each agent should steer to avoid crowding amongst other agents
    separation = compute_separation(ego_object, neighbors)

    # Alignment: Each agent should steer towards the heading of the leader
    alignment = compute_alignment(ego_object, leader)

    # Cohesion: Each agent should steer towards the leader
    cohesion = compute_cohesion(leader)

    # Compute new heading
    angular = separation + alignment + cohesion

    # Return velocity and angle
    return np.array([[1], [angular]])

def distance_to_object(obj_pos_a, obj_pos_b):
    return np.linalg.norm(obj_pos_a - obj_pos_b)

def compute_separation(ego_object, neighbors):
    angular = 0
    for neighbor in neighbors:
        angular -= ego_object.heading - neighbor.heading
    return angular

def compute_alignment(ego_object, leader):
    if not leader:
        return 0

    return ego_object.heading - leader.heading

def compute_cohesion(leader):
    if not leader:
        return 0

    return leader.heading
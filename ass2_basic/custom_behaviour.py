import numpy as np
from irsim.lib import register_behavior

@register_behavior("omni", "follower")
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
    separation_strength = kwargs["separation_strength"]
    alignment_strength = kwargs["alignment_strength"]
    cohesion_strength = kwargs["cohesion_strength"]
    separation_radius = kwargs["separation_radius"]
    turn_factor = kwargs["turn_factor"]

    neighbors = [robot for robot in external_objects if distance_to_object(ego_object.position, robot.position) < radius]
    leader = next((robot for robot in neighbors if robot.name == "leader"), None)

    # Separation: Each agent should steer to avoid crowding amongst other agents
    separation = compute_separation(ego_object, neighbors, separation_radius)

    # Alignment: Each agent should steer towards the heading of the leader
    alignment = compute_alignment(ego_object, leader)

    # Cohesion: Each agent should steer towards the leader
    cohesion = compute_cohesion(ego_object, leader)

    # Compute new vector
    steering = separation_strength * separation + alignment_strength * alignment + cohesion_strength * cohesion

    # Add wall avoidance
    steering += turn_factor * avoid_wall(ego_object)

    return steering

def distance_to_object(obj_pos_a, obj_pos_b):
    return np.linalg.norm(obj_pos_a - obj_pos_b)

def compute_separation(ego_object, neighbors, radius):
    closest_neighbors = [neighbor for neighbor in neighbors if distance_to_object(ego_object.position, neighbor.position) < radius]

    # No change if no neighbor is too close
    if len(closest_neighbors) == 0:
        return ego_object.velocity

    steering = 0
    for neighbor in closest_neighbors:
        steering += ego_object.position - neighbor.position

    return steering / len(closest_neighbors)

def compute_alignment(ego_object, leader):
    if not leader:
        return ego_object.velocity

    return leader.velocity

def compute_cohesion(ego_object, leader):
    if not leader:
        return ego_object.velocity

    steering = leader.position - ego_object.position
    return steering

def avoid_wall(ego_object):
    steering = 0

    # Avoid left wall
    if ego_object.position[0] < 5:
        steering += np.array([[1], [0]])

    # Avoid right wall
    if ego_object.position[0] > 45:
        steering += np.array([[-1], [0]])

    # Avoid top wall
    if ego_object.position[1] > 45:
        steering += np.array([[0], [-1]])

    # Avoid right wall
    if ego_object.position[1] < 5:
        steering += np.array([[0], [1]])

    return steering

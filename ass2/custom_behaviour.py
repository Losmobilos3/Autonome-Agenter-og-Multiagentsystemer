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
    protect_radius = kwargs["protect_radius"]
    S = kwargs["S"]
    A = kwargs["A"]
    C = kwargs["C"]

    neighbors = [robot for robot in external_objects if distance_to_object(ego_object.position, robot.position) < radius]
    close_neighbors = [robot for robot in external_objects if distance_to_object(ego_object.position, robot.position) < protect_radius]

    if len(neighbors) == 0 or np.linalg.norm(ego_object.velocity_xy) == 0:
        return np.array([[1], [0]])

    # Separation: Each agent should steer to avoid crowding amongst other agents
    dv_sep = compute_separation(ego_object, close_neighbors)

    # Alignment: Each agent should steer towards the heading of the leader
    dv_align = compute_alignment(ego_object, neighbors)

    # Cohesion: Each agent should steer towards the leader
    dv_cohesion = compute_cohesion(ego_object, neighbors)

    dv_stay_on_screen = stay_on_screen(ego_object)

    # Compute new heading
    dv_total = S * dv_sep + A * dv_align + C * dv_cohesion
    linear = np.linalg.norm(dv_total)
    angular = ego_object.velocity_xy.T @ dv_total / (np.linalg.norm(ego_object.velocity_xy) * np.linalg.norm(dv_total))

    # Return velocity and angle
    # TODO: Skal vi droppe linear speed aspektet?
    return np.array([[linear], [angular.item()]])

def distance_to_object(obj_pos_a, obj_pos_b):
    return np.linalg.norm(obj_pos_a - obj_pos_b)

# See https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html
def compute_separation(ego_object, neighbors):
    dv = 0
    for neighbor in neighbors:
        dv += ego_object.position - neighbor.position
    return dv

def compute_alignment(ego_object, neighbors):
    avg_vel = np.average([n.velocity_xy for n in neighbors], axis=0)
    dv = avg_vel - ego_object.velocity_xy
    return dv

def compute_cohesion(ego_object, neighbors):
    avg_position = np.average([n.position for n in neighbors], axis=0)
    dv = avg_position - ego_object.position 
    return dv

def stay_on_screen(ego_object):
    # TODO
    return np.array([0, 0])
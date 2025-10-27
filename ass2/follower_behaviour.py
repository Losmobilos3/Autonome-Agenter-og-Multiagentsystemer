import numpy as np
from irsim.lib import register_behavior
import irsim

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
    protect_radius = kwargs["protect_radius"]
    S = kwargs["S"]
    A = kwargs["A"]
    C = kwargs["C"]
    leader_prio = kwargs["leader_prio"]

    leaders = [robot for robot in external_objects if distance_to_object(ego_object.position, robot.position) < radius and robot.name == "leader"]
    neighbors = [robot for robot in external_objects if distance_to_object(ego_object.position, robot.position) < radius]
    close_neighbors = [robot for robot in external_objects if distance_to_object(ego_object.position, robot.position) < protect_radius]

    dv_stay_on_screen = stay_on_screen(ego_object)

    if len(neighbors) == 0 or np.linalg.norm(ego_object.velocity_xy) == 0:
        if np.linalg.norm(ego_object.velocity_xy) > 0:
            return ego_object.velocity_xy + dv_stay_on_screen
        else:
            return np.random.rand(2, 1)

    # Separation: Each agent should steer to avoid crowding amongst other agents
    dv_sep = compute_separation(ego_object, close_neighbors)

    # Alignment: Each agent should steer towards the heading of the leader
    dv_align = compute_alignment(ego_object, neighbors)

    # Cohesion: Each agent should steer towards the leader
    dv_cohesion = compute_cohesion(ego_object, neighbors)

    dv_leader = np.zeros((2, 1))
    if (len(leaders) > 0):
        dv_sep_leader = compute_separation(ego_object, leaders)

        dv_align_leader = compute_alignment(ego_object, leaders)

        dv_cohesion_leader = compute_cohesion(ego_object, leaders)

        dv_leader = S * dv_sep_leader + A * dv_align_leader + C * dv_cohesion_leader

    # Compute new heading
    dv_total = (1 - leader_prio) * (S * dv_sep + A * dv_align + C * dv_cohesion) + 10* dv_stay_on_screen + leader_prio * dv_leader
    # linear = np.linalg.norm(dv_total)
    # angular = ego_object.velocity_xy.T @ dv_total / (np.linalg.norm(ego_object.velocity_xy) * np.linalg.norm(dv_total))

    # Return velocity and angle
    # TODO: Skal vi droppe linear speed aspektet?
    #return np.array([[linear], [angular.item()]])
    return dv_total

def distance_to_object(obj_pos_a, obj_pos_b):
    return np.linalg.norm(obj_pos_a - obj_pos_b)

# See https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html
def compute_separation(ego_object, neighbors):
    dv = 0
    for neighbor in neighbors:
        dv += ego_object.position - neighbor.position
    return dv

def compute_alignment(ego_object, neighbors):
    # Align with leader and not others
    avg_v = np.mean([n.velocity_xy for n in neighbors])
    dv = avg_v
    return dv

def compute_cohesion(ego_object, neighbors):
    avg_position = np.average([n.position for n in neighbors], axis=0)
    dv = avg_position - ego_object.position
    return dv

def stay_on_screen(ego_object):
    dv = np.zeros((2, 1))
    if ego_object.position[0] < 5:
        dv += np.eye(2)[:, [0]]
    elif ego_object.position[0] > 45:
        dv -= np.eye(2)[:, [0]]
    if ego_object.position[1] < 5:
        dv += np.eye(2)[:, [1]]
    elif ego_object.position[1] > 45:
        dv -= np.eye(2)[:, [1]]
    return dv
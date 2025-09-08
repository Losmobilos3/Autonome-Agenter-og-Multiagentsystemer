import numpy as np

from irsim.lib import register_behavior
from irsim.util.util import WrapToPi, relative_position

@register_behavior("diff", "RL_Circle")
def beh_diff_RL_circle(ego_object, radius=0.5, external_objects = None, **kwargs):
    """
    Utilized reinforcement learning to get the object to move in a circle, around the goal point

    Args:
        state (np.array): Current state [x, y, theta] (3x1).
        center_point (np.array): Center point [x, y, theta] (3x1).
        radius (float): Radius to circle the center point at (default 0.5)

    Returns:
        np.array: Velocity [linear, angular] (2x1).
    """
    
    learner = Learner()

    alpha = 0.1   # learning rate
    gamma = 0.9   # discount factor

    # TODO: implement q-learning (TD-learning + exploration)
    # States: i goldylock zone, too close to center, too far from center

    

    state = ego_object.state
    center = ego_object.goal

    # Pick a random angle (-5 / +5) (velocity is constant)

    # Calculate if the robot is getting futher away from the desired radius.
    # If yes: punish
    # If no: Good!

    # Profit!
    linear = 1 # Scale linear velocity
    angular = 1 # Change in angular velocity pr. frame


    # Return velocity and angle
    return np.array([[linear], [angular]])

def judge_angle(angle):
    pass


class Learner(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(Learner, cls).__new__(cls)
      cls.Q = np.ndarray([2, 3]) # 3 strates, 3 actions (right, left, straight)
      
    return cls.instance

  
import numpy as np

from irsim.lib import register_behavior
from irsim.util.util import WrapToPi, relative_position

@register_behavior("diff", "RL_Circle")
def beh_diff_RL_circle(ego_object, radius=2, external_objects = None, **kwargs):
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
    distFromCenter, _ = relative_position(ego_object.goal, ego_object.state)
    if distFromCenter < radius - 0.05:
       currState = 0
    elif distFromCenter > radius + 0.05:
       currState = 0
    else:
       currState = 1

    action = learner.update_Q(currState, distFromCenter, radius)

    # TODO: implement q-learning (TD-learning + exploration)
    # States: i goldylock zone, too close to center, too far from center

    # Profit!
    linear = 1 # Scale linear velocity
    angular = -1 if action == 0 else 1 if action == 2 else 0


    # Return velocity and angle
    return np.array([[linear], [angular]])

def judge_angle(angle):
    pass


class Learner(object):
  # INITIALIZE LEARNER
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(Learner, cls).__new__(cls)
      cls.Q = np.random.rand(2, 3) # 3 states, 3 actions (right, left, straight)
      cls.Q[1, :] = np.zeros(3)
      cls.alpha = 0.1   # learning rate
      cls.gamma = 0.9   # discount factor

      # Last values
      cls.lastState = None
      cls.lastAction = None
      cls.lastR = None
    return cls.instance
  
  # DEFINE LEARNING/DECISION FUNCTION
  # TODO: Maybe split into two functions
  def update_Q(self, currState, distance, radius=0.5):
    # Call with first currState different from 1
    if (self.lastState is None) or (self.lastAction is None) or (self.lastR is None):
        print("SE HER - First call, initializing values")
        self.lastState = currState
        action = np.argmax(self.Q[currState, :])
        self.lastAction = action
        self.lastR = -abs(distance - radius)  # Reward is higher the closer we are to the desired radius
        return action
    else:
        # Update Q-table based on previous experience
        print("SE HER - Updating Q-table:", self.lastState, self.lastAction, self.lastR)
        self.Q[self.lastState, self.lastAction] = self.Q[self.lastState, self.lastAction] + self.alpha * (self.lastR + self.gamma * np.max(self.Q[currState, :]) - self.Q[self.lastState, self.lastAction])
        
        # If in terminal state, reset
        if currState == 1:
            self.lastState = None
            self.lastAction = None
            self.lastR = None
            return 1
            
        # Update for next iteration
        self.lastState = currState
        action = np.argmax(self.Q[currState, :])
        self.lastAction = action
        self.lastR = -abs(distance - radius)
        return action

  
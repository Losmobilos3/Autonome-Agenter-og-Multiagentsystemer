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
    
    learner = Learner(ego_object.id)
    currState, distFromCenter = get_state(ego_object, radius, ego_object.goal)

    action = learner.update_Q(currState, distFromCenter, radius)

    # TODO: implement q-learning (TD-learning + exploration)
    # States: i goldylock zone, too close to center, too far from center

    # Profit!
    linear = 1 # Scale linear velocity
    angular = -1 if action == 0 else 1 if action == 2 else 0


    # Return velocity and angle
    return np.array([[linear], [angular]])

def get_state(ego_object, radius, goal):
    dist_from_center, angle_to_center = relative_position(goal, ego_object.state)
    print("Angle: ", angle_to_center)
    # Discretize distance (3 zones)
    if dist_from_center < radius - 0.1:
        dist_state = 0  # too close
    elif dist_from_center > radius + 0.1:
        dist_state = 2  # too far  
    else:
        dist_state = 1  # good distance
    
    # Discretize angular velocity direction (4 directions)
    robot_heading = ego_object.state[2]
    desired_tangent = angle_to_center + np.pi/2  # perpendicular to radius
    heading_error = WrapToPi(desired_tangent - robot_heading)
    
    if heading_error < -np.pi/8:
        heading_state = 0
    elif heading_error < np.pi/8:
        heading_state = 1
    else:
        heading_state = 2
        
    return dist_state * 3 + heading_state, dist_from_center  # Combined state (0-8)


class Learner(object):
  # INITIALIZE LEARNER
  def __new__(cls, id):
    if not hasattr(cls, 'instances'):
      cls.instances = {}
    if id not in cls.instances:
      cls.instances[id] = super(Learner, cls).__new__(cls)
      cls.Q = np.random.rand(9, 3) # 9 states, 3 actions (right, left, straight)
      cls.alpha = 0.1   # learning rate
      cls.gamma = 0.4   # discount factor

      # Last values
      cls.lastState = None
      cls.lastAction = None
      cls.lastR = None
    return cls.instances[id]
  
  # DEFINE LEARNING/DECISION FUNCTION
  def update_Q(self, currState, distance, radius=0.5):

    # Call with first currState different from 1
    if self.is_first():
        return self.take_action(currState, distance, radius)

    # Update Q-table based on previous experience
    self.Q[self.lastState, self.lastAction] = self.Q[self.lastState, self.lastAction] + self.alpha * (self.lastR + self.gamma * np.max(self.Q[currState, :]) - self.Q[self.lastState, self.lastAction])

    return self.take_action(currState, distance, radius)
    
  def take_action(self, currState, distance, radius):

        # Determine the action based on the Q-table
        action = np.argmax(self.Q[currState, :])

        # Update for next iteration
        self.lastState = currState
        self.lastAction = action
        self.lastR = -abs(distance - radius) # Reward is higher the closer we are to the desired radius

        return action
    
  def is_first(self):
     return self.lastState is None or self.lastAction is None or self.lastR is None
     
  
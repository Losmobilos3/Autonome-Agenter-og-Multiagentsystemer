import numpy as np
import time 
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
    
    agent = Agent(ego_object, radius)
    currState, distFromCenter = get_state(ego_object, radius)

    action = agent.learner.update_Q(currState, distFromCenter, radius)

    # TODO: implement q-learning (TD-learning + exploration)
    # States: i goldylock zone, too close to center, too far from center

    # Profit!
    linear = 1 # Scale linear velocity
    angular = -1 if action == 0 else 1 if action == 2 else 0

    agent.update_metrics()

    # Return velocity and angle
    return np.array([[linear], [angular]])

def get_state(ego_object, radius):
    dist_from_center, angle_to_center = relative_position(ego_object.goal, ego_object.state)
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
        heading_state = 1 # Good angle
    else:
        heading_state = 2
        
    return dist_state * 3 + heading_state, dist_from_center  # Combined state (0-8)


class Agent(object):
    def __new__(cls, ego_object, desired_radius=2):
        if not hasattr(cls, 'instances'):
            cls.instances = {}
        if ego_object.id not in cls.instances:
            agent = super(Agent, cls).__new__(cls)
            agent.learner = Learner()
            agent.metrics = {}
            agent.ego_object = ego_object
            agent.r = desired_radius
            cls.instances[ego_object.id] = agent
        return cls.instances[ego_object.id]
    
    def update_metrics(self):
        self._update_mse()
        self._update_time_crashed()

    def _update_mse(self):
        if 'mse_stats' not in self.metrics:
            self.metrics['mse_stats'] = {"cumulated_error": 0, "count": 0}
        d, _ = relative_position(self.ego_object.goal, self.ego_object.state)
        self.metrics['mse_stats']["cumulated_error"] += (self.r-d)**2
        self.metrics['mse_stats']["count"] += 1

    def read_mse(self):
        if 'mse_stats' not in self.metrics:
            print("ERROR: NO MSE AVAILABLE!")
            return None
        return self.metrics['mse_stats']["cumulated_error"] / self.metrics['mse_stats']["count"]
    
    def _update_time_crashed(self):
        if 'crash_stats' not in self.metrics:
            self.metrics['crash_stats'] = {"crashed_time": 0, "total_time": time.time()}
        if self.ego_object.collision_flag:
            self.metrics['crashed_time'] += time.time() - self.metrics['crash_stats']['total_time'] # TODO: set to time

    def read_crashed(self):
        return self.ego_object.collision_flag
    
    def read_crash_time(self):
        if 'crash_stats' not in self.metrics:
            print("ERROR: NO CRASH STATS AVAILABLE!")
            return None
        return self.metrics['crash_stats']['crashed_time']

class Learner:
  def __init__(self):
      self.Q = np.random.rand(9, 3) # 9 states, 3 actions (right, left, straight)
      self.alpha = 0.1   # learning rate
      self.gamma = 0.4   # discount factor

      # Last values
      self.lastState = None
      self.lastAction = None
      self.lastR = None
  
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
     
  
import copy
import random
from typing import Tuple

import numpy as np

from Assignment_3_Grid.Action import Action
from Assignment_3_Grid.State import State
from Assignment_3_Grid.settings import SIMULATION_SIZE


class Agent():
    """Agent class

    Attributes:
        position: agent position in the grid
        level: agent level
        collection: collection flag
        state_action_history: history of state actions

    """
    def __init__(self):
        """Initialize an agent"""

        # Initialize with a random position in the grid
        self.position: np.ndarray = np.array([
            random.randint(0, SIMULATION_SIZE - 1),
            random.randint(0, SIMULATION_SIZE - 1)
        ])

        self.level: int = 1 # Initialize a level 1

        self.collecting: bool = False # Collection flag

        self.state_action_history: np.ndarray = np.array([])

    def act(
            self,
            current_state: State
    ) -> Action:
        """The Agent observes the game state, and takes an action

        :arg current_state: current state of the agent

        :returns: Agent action
        """

        # Convert the current state to relative positions instead of absolute positions
        current_state: State = copy.deepcopy(current_state)
        for agent in current_state.agents:
            # Skip self
            if self == agent:
                continue

            agent.position = np.subtract(agent.position, self.position)

        for fruit in current_state.fruits:
            fruit.position = np.subtract(fruit.position, self.position)

        # Add the state to the state action history
        np.append(self.state_action_history, current_state)

        # Utilize RL to determine the best action
        action: Action = self.learn()

        # Take the action
        self.collecting = False # Reset collection flag
        match action:
            case Action.UP:
                self.position = np.add(self.position, [0, 1])
            case Action.DOWN:
                self.position = np.add(self.position, [0, -1])
            case Action.LEFT:
                self.position = np.add(self.position, [-1, 0])
            case Action.RIGHT:
                self.position = np.add(self.position, [1, 0])
            case Action.DOWN:
                self.collecting = True
            case Action.NOOP:
                pass

        # Add the action to the state action history
        np.append(self.state_action_history, action)

        return action

    def learn(self) -> Action:
        """Agent policy (Used to determine the best action)

        :returns: Agent action
        """
        pass

    def reward(self):
        """Agent reward function

        :return:
        """
        # NOTE: The states and actions needed to compute the reward should be in the state action history
        pass
import copy
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from Assignment_3_Grid.Action import Action
from Assignment_3_Grid.Fruit import Fruit
from Assignment_3_Grid.Model import Model
from Assignment_3_Grid.State import State
from Assignment_3_Grid.settings import SIMULATION_SIZE, N_AGENTS, N_FRUITS, EPSILON, DISCOUNT_FACTOR


class Agent:
    """Agent class

    Attributes:
        position: agent position in the grid
        level: agent level
        collecting: collection flag
        state_action_history: history of state actions
        reward_history: history of rewards
        q_history: history of Q values
        model: learning model
        optimizer: optimizer object
        discount_factor: discount factor
    """
    def __init__(
            self,
            level: int = 1
    ):
        """Initialize an agent

        :arg level: Agent level
        """

        # Initialize with a random position in the grid
        self.position: np.ndarray = np.array([
            random.randint(0, SIMULATION_SIZE - 1),
            random.randint(0, SIMULATION_SIZE - 1)
        ])

        self.level: int = level # Initialize a level 1

        self.collecting: bool = False # Collection flag

        self.state_action_history: List[Tuple[State, Action]] = [] # State action history

        self.reward_history: List[float] = [] # Reward history

        self.q_history: List[Tensor] = [] # Q value history

        self.model: Model = Model(
            # x- and -y position, level, and collection flag for the agents
            # x- and y- position, level, and picked flag for the fruits
            input_size= 4 * (N_AGENTS - 1) + 4 * N_FRUITS
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, current_state: State) -> Action:
        """The Agent observes the game state, and takes an action

        :arg current_state: current game state

        :returns: Agent action
        """

        # Convert the current state to relative positions instead of absolute positions
        current_state: State = copy.deepcopy(current_state)

        # Remove self and update positions
        current_state.agents.remove(self)
        for agent in current_state.agents:
            agent.position = np.subtract(agent.position, self.position)

        for fruit in current_state.fruits:
            fruit.position = np.subtract(fruit.position, self.position)

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

        # Add the state and action to the state action history
        self.state_action_history.append((current_state, action))

        return action

    def learn(self) -> Action:
        """Agent policy (Used to determine the best action)

        :returns: Agent action
        """

        # Compute Q-values and append to history
        self.q_history.append(
            self.model(
                self.state_action_history[-1][0] # Get the current state
            )
        )

        # Utilize epsilon to sometimes explorer randomly
        if np.random.random() < EPSILON:
            action_id = torch.tensor(np.random.randint(0, 5))
        else:
            action_id = torch.argmax(self.q_history[-1])

        if len(self.state_action_history) > 1: # Check if there exists prior history
            previous_reward = self.reward_history[-1]

            target = previous_reward + DISCOUNT_FACTOR * torch.max(self.q_history[-1]).detach()
            loss = F.mse_loss(self.q_history[-2], target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return Action(action_id)

    def reward(self) -> None:
        """Agent reward function

        :return:
        """
        # NOTE: This is where we are... write the reward function

        # NOTE: The states and actions needed to compute
        # the reward should be in the state action history

        remaining_fruits: List[Fruit] = [
            fruit for fruit in self.state_action_history[-1][0].fruits if not fruit.collected
        ]
        if not remaining_fruits:
            return

        # Give reward based on distance to nearest fruit
        distance_to_fruits = np.array([
            np.linalg.norm(fruit.position - self.position) for fruit in remaining_fruits
        ])
        nearest_fruit = remaining_fruits[np.argmin(distance_to_fruits)]

        reward = shortest_distance

        pass

    def reset(self) -> None:
        """Reset the agent position, level and collecting flag"""
        self.position: np.ndarray = np.array([
            random.randint(0, SIMULATION_SIZE - 1),
            random.randint(0, SIMULATION_SIZE - 1)
        ])
        self.level: int = 1
        self.collecting = False

    def to_tensor(self) -> np.ndarray:
        return np.concatenate(
            self.position,
            self.level,
            self.collecting,
            axis=None
        )
from typing import List

import numpy as np

from Assignment_3_Grid.Agent import Agent
from Assignment_3_Grid.Fruit import Fruit


class State:
    """Stochastic game state

    Each state can be viewed as a non-repeated normal form game

    Attributes:
        agents: list of agents
        fruits (List[Fruit]): list of fruits
    """
    def __init__(
            self,
            agents: List[Agent],
            fruits: List[Fruit],
    ):
        """ Initialise state

        :param agents:
        :param fruits:
        """
        self.agents = agents
        self.fruits = fruits

    def to_tensor(self):
        """Flatten the state to be used in training

        :returns 1D array state tensor
        """
        agents: np.ndarray = np.array(
            agent.to_tensor() for agent in self.agents
        )
        fruits: np.ndarray = np.array(
            fruit.to_tensor() for fruit in self.fruits
        )

        return np.concatenate((agents, fruits))
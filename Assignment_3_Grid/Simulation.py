import copy
import random
from typing import Tuple, List

import numpy as np

from Assignment_3_Grid.Agent import Agent
from Assignment_3_Grid.Fruit import Fruit
from Assignment_3_Grid.State import State
from Assignment_3_Grid.settings import SIMULATION_SIZE, STEPS_PER_EPISODE, N_EPISODES


class Simulation:
    """LBF Simulation class

    """
    def __init__(
            self,
            n_agents: int,
            n_fruits: int,
    ):
        """ Initialize LBF Simulation

        :param n_agents: number of agents
        :param n_fruits: number of fruits

        """

        # Initialize n agents
        self.agents: List[Agent] = [
            Agent() for _ in range(n_agents)
        ]

        # Initialize n fruits
        self.fruits: List[Fruit] = [
            Fruit() for _ in range(n_fruits)
        ]

        # Initialize the first game state (Important: Use copies)
        self.states: List[State] = [
            State(
                copy.deepcopy(self.agents),
                copy.deepcopy(self.fruits)
            )
        ]

    def reset(self) -> None:
        """Reset the simulation"""
        for agent in self.agents:
            agent.reset()

        for fruit in self.fruits:
            fruit.reset()

    def step(self) -> None:
        """Take one simulation step

        :return:
        """

        # All agents take an action
        for agent in self.agents:
            agent.act(self.states[-1])

        # Append the new state to the state list
        self.states.append(
            State(
                copy.deepcopy(self.agents),
                copy.deepcopy(self.fruits)
            )
        )

        # Reward the agents
        for agent in self.agents:
            agent.reward()

    def run(self, n_episodes, n_steps) -> None:
        """Run the simulation

        :param n_episodes: number of episodes
        :param n_steps: number of steps

        """
        for i in range(n_episodes):
            print(f"Episode: {i} of {n_episodes}")
            for _ in range(n_steps):
                self.step()
            self.reset()
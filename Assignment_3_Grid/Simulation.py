import copy
from typing import Tuple, List

from Assignment_3_Grid.Agent import Agent
from Assignment_3_Grid.Fruit import Fruit
from Assignment_3_Grid.State import State


class Simulation:
    """LBF Simulation class

    """
    def __init__(
            self,
            simulation_size: Tuple[int, int],
            n_agents: int,
            n_fruits: int,
            steps: int,
    ):
        """ Initialize LBF Simulation

        :param simulation_size: size of simulation
        :param n_agents: number of agents
        :param n_fruits: number of fruits
        :param steps: number of steps in the simulation

        """

        # Set internal variables
        self.simulation_size = simulation_size
        self.steps = steps

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

    def step(self):
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

    def run(self):
        for _ in range(self.steps):
            self.step()


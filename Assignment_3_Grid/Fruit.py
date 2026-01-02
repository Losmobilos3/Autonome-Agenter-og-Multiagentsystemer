import random
import numpy as np

from Assignment_3_Grid.settings import SIMULATION_SIZE


class Fruit:
    """Fruit class

    Attributes:
        position: fruit position in the grid
        level: fruit level
        collected: flag denoting the collected state

    """
    def __init__(self):
        """Initialize a new fruit in a random position"""

        # Initialize with a random position within the bounds of the simulation
        self.position: np.ndarray = np.array([
            random.randint(0,SIMULATION_SIZE - 1),
            random.randint(0,SIMULATION_SIZE - 1)
        ])

        self.level: int = 0 # Always initialize at level 1 (Can be updated later)

        self.collected: bool = False # Flag denoting the collected state
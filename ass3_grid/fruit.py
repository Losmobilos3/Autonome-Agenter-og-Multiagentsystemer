import numpy as np

class Fruit:
    def __init__(self, position: np.ndarray, level: int = 1):
        self.pos = position
        self.level = level
        self.picked = 0
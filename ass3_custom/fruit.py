import numpy as np

class Fruit:
    def __init__(self, position: np.ndarray, level: int):
        self.pos = position
        self.level = level
        self.picked = 0
        
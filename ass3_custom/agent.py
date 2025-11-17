import numpy as np
from config import MAX_VEL

S, A, C, L = 0.05, 0.005, 0.005, 0.005

class Agent:
    def __init__(self, sim_ref: "Simulation"):
        self.sim_ref = sim_ref
        self.pos = np.random.rand(2, 1) * self.sim_ref.size
        self.vel = np.random.rand(2, 1) * 3
        self.avg_dist_to_leader_data = []


    def update(self):
        # Limit velocity
        self.vel *= MAX_VEL
        self.pos += self.vel

    def act(self):
        pass  # To be implemented in subclasses


class Worker(Agent):
    def __init__(self, sim_ref: "Simulation", level: int):
        super().__init__(sim_ref)
        self.level = level

    def act(self):
        pass
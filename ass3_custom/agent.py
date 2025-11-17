import numpy as np
from config import MAX_VEL, TURN_FACTOR

S, A, C, L = 0.05, 0.005, 0.005, 0.005

class Agent:
    def __init__(self, view_distance, protect_distance, sim_ref: "Simulation"):
        self.sim_ref = sim_ref
        self.view_distance = view_distance
        self.protect_distance = protect_distance
        self.pos = np.random.rand(2, 1) * self.sim_ref.size
        self.vel = np.random.rand(2, 1) * 3
        self.avg_dist_to_leader_data = []


    def update(self):
        # Limit velocity
        self.vel *= MAX_VEL / np.linalg.norm(self.vel) * (2/3 if isinstance(self, Leader) else 1)
        self.pos += self.vel

    def render(self, ax):
        return 0

    def act(self):
        pass  # To be implemented in subclasses


class Follower(Agent):
    def __init__(self, view_distance, protect_distance, sim_ref: "Simulation"):
        super().__init__(view_distance, protect_distance, sim_ref)

    def act(self):
        neighbors = [a for a in self.sim_ref.agents if np.linalg.norm(a.pos - self.pos) < self.view_distance]

        near_neighbors = [a for a in neighbors if np.linalg.norm(a.pos - self.pos) < self.protect_distance]

        if len(neighbors) == 0:
            return

        separation = self.compute_separation(near_neighbors)

        alignment = self.compute_alignment(neighbors)

        cohesion = self.compute_cohesion(neighbors)

        leader_cohesion = self.compute_leader_cohesion(neighbors)

        # Apply boid principles
        self.vel += S * separation + A * alignment + C * cohesion

        # Apply leader force
        self.vel += L * leader_cohesion

        # Avoid the wall
        self.vel += TURN_FACTOR * self.stay_on_screen()


    def compute_separation(self, neighbors: list['Agent']):
        if len(neighbors) == 0:
            return 0
        dv = 0
        for neighbor in neighbors:
            dv += self.pos - neighbor.pos
        return dv 

    def compute_alignment(self, neighbors: list['Agent']):
        if len(neighbors) == 0:
            return 0
        return np.mean([n.vel for n in neighbors], axis=0)

    def compute_cohesion(self, neighbors: list['Agent']):
        if len(neighbors) == 0:
            return 0
        avg_position = np.mean([n.pos for n in neighbors], axis=0)
        dv = avg_position - self.pos
        return dv

    def compute_leader_cohesion(self, neighbors: list['Agent']):
        if len(neighbors) == 0:
            return 0
        leaders = [a for a in neighbors if isinstance(a, Leader)]
        if leaders:
            return leaders[0].pos - self.pos
        return 0

    def stay_on_screen(self):
        dv = np.zeros((2,1))
        if self.pos[0] < self.sim_ref.size[0] * (1 - self.sim_ref.screen_ratio):
            dv[0] = 1
        elif self.pos[0] > self.sim_ref.size[0] * self.sim_ref.screen_ratio:
            dv[0] = -1
        if self.pos[1] < self.sim_ref.size[1] * (1 - self.sim_ref.screen_ratio):
            dv[1] = 1
        elif self.pos[1] > self.sim_ref.size[1] * self.sim_ref.screen_ratio:
            dv[1] = -1
        return dv
        
class Leader(Agent):
    def __init__(self, view_distance, protect_distance, sim_ref: "Simulation"):
        super().__init__(view_distance, protect_distance, sim_ref)

    def act(self):
        self.vel += TURN_FACTOR * self.stay_on_screen() + (np.random.rand(2,1) - 0.5) * 0.1

    def stay_on_screen(self):
        dv = np.zeros((2,1))
        if self.pos[0] < self.sim_ref.size[0] * (1 - self.sim_ref.screen_ratio):
            dv[0] = 1
        elif self.pos[0] > self.sim_ref.size[0] * self.sim_ref.screen_ratio:
            dv[0] = -1
        if self.pos[1] < self.sim_ref.size[1] * (1 - self.sim_ref.screen_ratio):
            dv[1] = 1
        elif self.pos[1] > self.sim_ref.size[1] * self.sim_ref.screen_ratio:
            dv[1] = -1
        return dv
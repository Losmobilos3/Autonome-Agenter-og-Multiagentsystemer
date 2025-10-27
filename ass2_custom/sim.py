import matplotlib.pyplot as plt

import numpy as np

S, A, C, turnFactor = 0.05, 0.05, 0.0005, 0.2

class Simulation:
    def __init__(self, no_agents, width, height, view_distance, protect_distance):
        self.size = np.vstack([width, height])
        self.agents: list[Agent] = []
        for n in range(no_agents):
            self.agents.append(Agent(view_distance, protect_distance, self))

        self.fig, self.ax = plt.subplots(figsize=(20, 15))


    def step(self):
        for agent in self.agents:
            agent.act()
            agent.update()

    def setup_plot(self):
        self.ax.set_xlim(0, self.size[0])
        self.ax.set_ylim(0, self.size[1])
        
        # init plot
        self.scatter = self.ax.scatter([], [], s=15, color='blue')
        
        return self.scatter

    def animate_frame(self, i):
        # Update the simulation state
        self.step() 
        
        # Extract agent positions
        x_coords = [a.pos[0][0] for a in self.agents]
        y_coords = [a.pos[1][0] for a in self.agents]
        
        # Update the data in the existing scatter plot object
        self.scatter.set_offsets(np.c_[x_coords, y_coords])
        
        # Return the updated plot object
        return (self.scatter,)

class Agent:
    def __init__(self, view_distance, protect_distance, sim_ref: Simulation):
        self.sim_ref = sim_ref
        self.view_distance = view_distance
        self.protect_distance = protect_distance
        self.pos = np.random.rand(2, 1) * self.sim_ref.size
        self.vel = np.random.rand(2, 1) * 3

    def update(self):
        # Limit velocity
        self.vel *= 3 / np.linalg.norm(self.vel)

        self.pos += self.vel

    def render(self, ax):
        return 0

    def act(self):
        neighbors = [a for a in self.sim_ref.agents if np.linalg.norm(a.pos - self.pos) < self.view_distance]

        near_neighbors = [a for a in neighbors if np.linalg.norm(a.pos - self.pos) < self.protect_distance]

        if (len(neighbors) == 0):
            return

        seperation = self.compute_separation(near_neighbors)

        alignment = self.compute_alignment(neighbors)

        cohesion = self.compute_cohesion(neighbors)

        self.vel += S * seperation + A * alignment + C * cohesion + turnFactor * self.stay_on_screen()


    def compute_separation(self, neighbors: list['Agent']):
        dv = 0
        for neighbor in neighbors:
            dv += self.pos - neighbor.pos
        return dv

    def compute_alignment(self, neighbors: list['Agent']):
        avg_v = np.mean([n.vel for n in neighbors])
        dv = avg_v
        return dv

    def compute_cohesion(self, neighbors: list['Agent']):
        avg_position = np.average([n.pos for n in neighbors], axis=0)
        dv = avg_position - self.pos
        return dv
        
    def stay_on_screen(self):
        dv = np.zeros((2,1))
        if self.pos[0] < 0:
            dv[0] = 1
        elif self.pos[0] > self.sim_ref.size[0]:
            dv[0] = -1
        if self.pos[1] < 0:
            dv[1] = 1
        elif self.pos[1] > self.sim_ref.size[1]:
            dv[1] = -1
        return dv
        
        
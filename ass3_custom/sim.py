import matplotlib.pyplot as plt

import numpy as np
from agent import Agent, Worker
from fruit import Fruit
from config import MAX_VEL, AGENT_PLOT_SIZE, FRUIT_PLOT_SIZE
from matplotlib.lines import Line2D 

class Simulation:
    def __init__(self, no_agents, no_fruits, width, height, screen_ratio=0.9):
        self.size = np.vstack([width, height])
        self.screen_ratio = screen_ratio
        self.agents: list[Agent] = []
        self.fruits: list[Fruit] = []
        self.avg_dist_leader = [] 
        self.avg_dist_followers = [] 

        # Create Workers
        for i in range(no_agents):
            self.agents.append(Worker(self, level=1))

        for i in range(no_fruits):
            self.fruits.append(Fruit(position=np.random.rand(2, 1) * self.size, level=1))

        self.fig, self.ax = plt.subplots(figsize=(20, 15))


    def step(self):
        for agent in self.agents:
            agent.act()
            agent.update()

    def setup_plot(self):
        self.ax.axis("equal")
        self.ax.set_xlim(0, self.size[0])
        self.ax.set_ylim(0, self.size[1])
        
        agent_pos = np.array([a.pos for a in self.agents]).squeeze().T
        fruit_pos = np.array([f.pos for f in self.fruits]).squeeze().T

        # init scatter plot
        self.agent_scatter = self.ax.scatter(*agent_pos, s=15, color="blue")
        self.fruit_scatter = self.ax.scatter(*fruit_pos, s=50, color="red")

        self.fruit_level_patches = []
        self.fruit_stems = [] # NEW: List to store stem Line2D objects
        for f in self.fruits:
            # Fruit level text (no offset, centered)
            text_patch = self.ax.text(
                f.pos[0], f.pos[1], 
                str(f.level),  
                ha='center', va='center', 
                color='black', fontsize=8,
                bbox=dict(facecolor='red', edgecolor='black', boxstyle='circle,pad=0.3', alpha=0.8)
            )
            self.fruit_level_patches.append(text_patch)

            # NEW: Add a stem for the fruit
            # Define the start and end points of the stem
            x_stem_start = f.pos[0] 
            y_stem_start = f.pos[1] + FRUIT_PLOT_SIZE/1000 # Adjust this offset as needed for visual appeal
            # A tiny horizontal shift for a slight curve, and then up
            x_stem_end = x_stem_start + FRUIT_PLOT_SIZE / 600 
            y_stem_end = y_stem_start + FRUIT_PLOT_SIZE / 1000 * 6

            stem_line = Line2D(
                [x_stem_start, x_stem_end], 
                [y_stem_start, y_stem_end],
                color='green', linewidth=2.5,
                solid_capstyle='round' # Makes the ends rounded
            )
            self.ax.add_line(stem_line)
            self.fruit_stems.append(stem_line)

        self.agent_level_patches = []
        for a in self.agents:
            # Agent level text
            text_patch = self.ax.text(
                a.pos[0], a.pos[1], 
                str(a.level), 
                ha='center', va='center', 
                color='white', fontsize=8,
                bbox=dict(facecolor='blue', edgecolor='black', boxstyle='circle,pad=0.2', alpha=0.8)
            )
            self.agent_level_patches.append(text_patch)
        
        # Return ALL artists that need to be redrawn
        return (self.agent_scatter, self.fruit_scatter, 
                *self.fruit_level_patches, *self.agent_level_patches,
                *self.fruit_stems) # NEW: Include the stems

    def save_metrics(self):
        pass
        
    def animate_frame(self, i):
        if i % 100 == 0:
            print(f"Animating frame {i}")

        # Update the simulation state
        self.step() 
        
        # Extract agent positions
        x_coords = [a.pos[0][0] for a in self.agents]
        y_coords = [a.pos[1][0] for a in self.agents]
        
        # Update the data in the existing scatter plot object
        self.agent_scatter.set_offsets(np.c_[x_coords, y_coords])

        # Save performance metrics
        self.save_metrics()
        
        # Return all updated plot objects (scatter and patches)
        return self.agent_scatter, 
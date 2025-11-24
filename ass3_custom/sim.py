import matplotlib.pyplot as plt

import numpy as np
from agent import Agent, Worker
from fruit import Fruit
from config import AGENT_PLOT_SIZE, FRUIT_PLOT_SIZE
from matplotlib.lines import Line2D 
import torch

class Simulation:
    def __init__(self, no_agents, no_fruits, width, height, screen_ratio=0.9):
        self.size = np.vstack([width, height])
        self.screen_ratio = screen_ratio
        self.agents: list[Agent] = []
        self.fruits: list[Fruit] = []
        self.avg_dist_leader = [] 
        self.avg_dist_followers = [] 

        # Used for learning
        self.prior_state = None

        # Create Workers
        for i in range(no_agents):
            self.agents.append(Worker(self, level=1, num_agents=no_agents, num_fruits=no_fruits))

        for i in range(no_fruits):
            self.fruits.append(Fruit(position=np.random.rand(2, 1) * self.size, level=1))

        self.fig, self.ax = plt.subplots(figsize=(20, 15))

    def init_env(self):
        for agent in self.agents:
            agent.pos = np.random.rand(2, 1) * self.size
            agent.vel = np.random.rand(2, 1) * 3

        for fruit in self.fruits:
            fruit.pos = np.random.rand(2, 1) * self.size
            fruit.picked = 0



    def run_episodes(self, no_episodes, max_steps_per_episode):
        for episode in range(no_episodes):
            print(f"Starting episode {episode+1}/{no_episodes}")
            self.init_env()
            for step in range(max_steps_per_episode):
                self.step()

    def step(self):
        self.prior_state = self.get_state_tensor()  # Store prior state for learning
        for agent in self.agents:
            agent.act()
            agent.update()

    def setup_plot(self):
        self.ax.axis("equal")
        self.ax.set_xlim(0, self.size[0])
        self.ax.set_ylim(0, self.size[1])
        
        # No need for scatter plots since text has circular backgrounds

        self.fruit_level_patches = []
        self.fruit_stems = [] # NEW: List to store stem Line2D objects
        for f in self.fruits:
            # Fruit level text (no offset, centered)
            text_patch = self.ax.text(
                f.pos[0], f.pos[1], 
                str(f.level),  
                ha='center', va='center', 
                color='black', fontsize=max(6, FRUIT_PLOT_SIZE/25),
                bbox=dict(facecolor='red', edgecolor='black', boxstyle=f'circle,pad={FRUIT_PLOT_SIZE/600}', alpha=1)
            )
            self.fruit_level_patches.append(text_patch)

            # NEW: Add a stem for the fruit
            # Define the start and end points of the stem
            x_stem_start = f.pos[0] 
            y_stem_start = f.pos[1] + FRUIT_PLOT_SIZE/1000 # Adjust this offset as needed for visual appeal
            # A tiny horizontal shift for a slight curve, and then up
            x_stem_end = x_stem_start + FRUIT_PLOT_SIZE / 800 
            y_stem_end = y_stem_start + FRUIT_PLOT_SIZE / 1000 * 4

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
                color='white', fontsize=max(6, AGENT_PLOT_SIZE/25),
                bbox=dict(facecolor='blue', edgecolor='black', boxstyle=f'circle,pad={AGENT_PLOT_SIZE/600}', alpha=1)
            )
            self.agent_level_patches.append(text_patch)
        
        # Return ALL artists that need to be redrawn
        return (*self.fruit_level_patches, *self.agent_level_patches,
                *self.fruit_stems) # Include the stems
        
    def animate_frame(self, i):
        if i % 100 == 0:
            print(f"Animating frame {i}")

        # Update the simulation state
        self.step() 
        
        # Update agent text positions
        for j, agent in enumerate(self.agents):
            self.agent_level_patches[j].set_position((agent.pos[0], agent.pos[1]))
            
        # Update fruit text positions and stems
        for j, fruit in enumerate(self.fruits):
            self.fruit_level_patches[j].set_position((fruit.pos[0], fruit.pos[1]))
            # Update stem positions
            x_stem_start = fruit.pos[0] 
            y_stem_start = fruit.pos[1] + FRUIT_PLOT_SIZE/1000
            x_stem_end = x_stem_start + FRUIT_PLOT_SIZE / 800 
            y_stem_end = y_stem_start + FRUIT_PLOT_SIZE / 1000 * 4
            self.fruit_stems[j].set_data([x_stem_start, x_stem_end], [y_stem_start, y_stem_end])

        # Save performance metrics
        self.save_metrics()
        
        # Return all updated plot objects (text patches and stems)
        return (*self.fruit_level_patches, *self.agent_level_patches, *self.fruit_stems) 
    

    def save_metrics(self):
        pass

    def get_state_tensor(self) -> torch.Tensor:
        # Extract agent positions
        agent_positions = []
        for agent in self.agents:
            agent_positions.append(agent.pos[0].item())
            agent_positions.append(agent.pos[1].item())

        # Extract fruit positions and picked status
        fruit_info = []
        for fruit in self.fruits:
            fruit_info.append(fruit.pos[0].item())
            fruit_info.append(fruit.pos[1].item())
            fruit_info.append(fruit.picked)

        # Concat and convert to tensor
        state_array = torch.tensor(agent_positions + fruit_info, dtype=torch.float32)
        return state_array
    
    def get_prior_state(self) -> torch.Tensor:
        return self.prior_state
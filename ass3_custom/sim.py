import matplotlib.pyplot as plt

import numpy as np
from agent import Agent, Worker
from fruit import Fruit
from config import AGENT_PLOT_SIZE, FRUIT_PLOT_SIZE, COLLECTION_DISTANCE, NEAREST_FRUITS_COUNT
from matplotlib.lines import Line2D 
import torch
from nn_Q import Model, supervised_train
from action_ds import ActionDataset
import copy

class Simulation:
    """Simulation class

    Attributes:
        size (np.NDArray[width, height]): Simulation size
        agents (List[Agent]): List of agents in the simulation
        fruits (List[Fruit]): List of fruits in the simulation
        abs_prior_state
        prior_state (torch.Tensor): State tensor for the prior state
        state_size (int): Size of the state sensor
        fig, ax (tuple[Figure, Axes]): Subplot figure for the simulation
    """

    def __init__(self, no_agents, no_fruits, width, height):
        """Initialize simulation

        Args:
            no_agent (int): Number of agents in the simulation 
            no_fruits (int): Number of fruits in the simulation
            width (float): Simulation width
            height (float): Simulation height
        """
        self.size = np.vstack([width, height])

        # Define size of state vector
        self.state_size = 3 * (no_agents - 1) + 4 * NEAREST_FRUITS_COUNT

        # Initalize Worker agents
        self.agents: list[Agent] = [Worker(self, i) for i in range(no_agents)]

        # Initialize fruits
        self.fruits: list[Fruit] = [
            Fruit(position=np.random.rand(2,1) * self.size, level=1) for _ in range(no_fruits)
        ]

        # Used for learning
        self.abs_prior_state = None

        # Shared policy approximations hat(pi)
        self.agent_policy_estimations = [Model(input_size=self.state_size, hidden_size=64) for _ in range(no_agents)]
        self.agent_action_buffers = [ActionDataset(state_size=self.state_size) for _ in range(no_agents)]  # To store past actions for each agent

        self.fig, self.ax = plt.subplots(figsize=(20, 15))

    def init_env(self):
        """Initialize the environment"""
        for agent in self.agents:
            agent.pos = np.random.rand(2, 1) * self.size
            agent.vel = np.random.rand(2, 1) * 3

        for fruit in self.fruits:
            fruit.pos = np.random.rand(2, 1) * self.size
            fruit.picked = 0

    def run_episodes(self, no_episodes, max_steps_per_episode):
        """Run _n_ episodes of the simulation

        Args:
            no_episodes (int): Number of episodes to run
            max_steps_per_episodes (int): Number of steps to run per episode
        """
        for episode in range(no_episodes):
            print(f"Starting episode {episode+1}/{no_episodes}")
            self.init_env()
            for step in range(max_steps_per_episode):
                self.step(step)

    def step(self, step_num=None):
        """Take a step in the simulation""" 
        self.save_prior_state()

        for i, agent in enumerate(self.agents):
            decision_id, vel = agent.act()
            agent.update()

            # Update agent action buffer with newly taken action
            self.agent_action_buffers[i].add_data(self.get_prior_state(i), decision_id, vel)

            # Fit policy approximations hat(pi) here
            # TODO: add some criteria to not train every step
            if step_num and step_num % 20 == 0:
                supervised_train(self.agent_policy_estimations[i], self.agent_action_buffers[i])

        # Check for fruit collection
        collecting_agents = [agent for agent in self.agents if agent.collecting]
        for fruit in self.fruits:
            # Skip if the fruit is already picked
            if fruit.picked:
                continue

            # Find all collecting agents that are close to the fruit
            close_agents = [
                agent for agent in collecting_agents
                if np.linalg.norm(fruit.pos - agent.pos) < COLLECTION_DISTANCE
            ]
            
            # Compute the total level of agents around the fruit
            collection_level = sum(agent.level for agent in close_agents)
                
            # Check if the fruit can be collected
            if fruit.level <= collection_level:
                # Pick the fruit
                fruit.picked = 1

                # Give all close agents a reward
                for agent in close_agents:
                    agent.give_reward(1)

                # Only pick up 1 fruit per frame
                break

        # Give rewards after all agents have moved
        for agent in self.agents:
            self.give_rewards(agent)

    def give_rewards(self, agent: Agent):
        """Distribute rewards to an agent

        Args:
            agent (Agent): Agent to be rewarded
        """
        # Give reward based on distance to nearest fruit
        dist_to_fruits = np.array([np.linalg.norm(fruit.pos - agent.pos) for fruit in self.fruits if not fruit.picked])
        min_dist_index = np.argmin(dist_to_fruits)
        
        min_dist = dist_to_fruits[min_dist_index] if min_dist_index is not None else None
        if min_dist is not None and min_dist < 5.0:
            agent.give_reward(1/min(dist_to_fruits))  # Small positive reward to encourage action
        else:
            agent.give_reward(-0.01)  # Small negative reward to encourage action
        # Punish for going away from fruits
        agentToFruitVec = self.fruits[min_dist_index].pos - agent.pos
        directionReward = (agentToFruitVec.T @ agent.vel).item()
        agent.give_reward(directionReward * 0.01)

    def setup_plot(self):
        """Initializes the plot"""
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
        
        # Add reward display for the first agent
        self.reward_text = self.ax.text(
            10, self.size[1] - 10,  # Top-left corner
            f"Agent 0 Reward: {self.agents[0].reward:.2f}",
            fontsize=14, color='black',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
        )
        
        # Return ALL artists that need to be redrawn
        return (*self.fruit_level_patches, *self.agent_level_patches,
                *self.fruit_stems, self.reward_text) # Include the stems and reward text
        
    def animate_frame(self, i):
        """Animates frame _i_

        Args:
            i (int): Frame index to animate
        """
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
            if fruit.picked:
                self.fruit_level_patches[j].set_alpha(0.2)  # Make picked fruits semi-transparent
                self.fruit_level_patches[j].set_bbox(dict(facecolor='red', edgecolor='black', boxstyle=f'circle,pad={FRUIT_PLOT_SIZE/600}', alpha=0.2))
                self.fruit_stems[j].set_alpha(0.2)  # Also make stem semi-transparent
            # Update stem positions
            x_stem_start = fruit.pos[0] 
            y_stem_start = fruit.pos[1] + FRUIT_PLOT_SIZE/1000
            x_stem_end = x_stem_start + FRUIT_PLOT_SIZE / 800 
            y_stem_end = y_stem_start + FRUIT_PLOT_SIZE / 1000 * 4
            self.fruit_stems[j].set_data([x_stem_start, x_stem_end], [y_stem_start, y_stem_end])

        # Update reward display for the first agent
        self.reward_text.set_text(f"Agent 0 Reward: {self.agents[0].reward:.2f}")

        # Save performance metrics
        self.save_metrics()
        
        # Return all updated plot objects (text patches and stems)
        return (*self.fruit_level_patches, *self.agent_level_patches, *self.fruit_stems, self.reward_text) 
    

    def save_metrics(self):
        # TODO: ?
        pass

    def save_prior_state(self):
        """Save the absolute prior state"""
        # Save prior absolute state as a tuple of lists of dicts
        self.abs_prior_state = (
            [{
                "x": agent.pos[0].item(),
                "y": agent.pos[1].item(),
                "level": agent.level,
                "agent_idx": agent.agent_idx
                } for agent in self.agents],
            [{
                "x": fruit.pos[0].item(), 
                "y": fruit.pos[1].item(),
                "picked": fruit.picked,
                "level": fruit.level
                } for fruit in self.fruits]
        )

    def get_state_tensor(self, agent_idx: int) -> torch.Tensor:
        """Get the state tensor
        
        Returns:
            torch.Tensor: Current state tensor
        """
        # Extract realtive agent positions, except for agent_idx
        subject = self.agents[agent_idx]
        
        agent_positions = []
        for agent in self.agents:
            if agent == self.agents[agent_idx]:
                continue
            agent_positions.append(agent.pos[0].item() - subject.pos[0].item())
            agent_positions.append(agent.pos[1].item() - subject.pos[1].item())
            agent_positions.append(agent.level)

        # Extract relative fruit positions and picked status
        fruit_info = []
        
        # Calculate distances and sort
        fruits_with_dist = []
        for fruit in self.fruits:
            dx = fruit.pos[0].item() - subject.pos[0].item()
            dy = fruit.pos[1].item() - subject.pos[1].item()
            dist = (dx**2 + dy**2)**0.5
            fruits_with_dist.append((dist, dx, dy, fruit.picked, fruit.level))
        
        fruits_with_dist.sort(key=lambda x: x[0])

        for i in range(NEAREST_FRUITS_COUNT):
            if i < len(fruits_with_dist):
                _, dx, dy, picked, level = fruits_with_dist[i]
                fruit_info.append(dx)
                fruit_info.append(dy)
                fruit_info.append(level)
                fruit_info.append(picked)
            else:
                # Padding with "picked" fruits (effectively invisible)
                fruit_info.extend([0, 0, 1])

        # Concat and convert to tensor
        state_array = torch.tensor(agent_positions + fruit_info, dtype=torch.float32)
        return state_array
    
    def get_prior_state(self, agent_idx) -> torch.Tensor:
        """Get the relative prior state tensor

        Returns:
            torch.Tensor: Relative prior state tensor
        """
        if self.abs_prior_state is None:
            return None

        abs_agents, abs_fruit_info = self.abs_prior_state
        subject = self.agents[agent_idx]
        
        # Extract realtive agent positions, except for agent_idx
        agent_positions = []
        for agent in abs_agents:
            if agent["agent_idx"] == agent_idx:
                continue
            agent_positions.append(agent["x"] - subject.pos[0].item())
            agent_positions.append(agent["y"] - subject.pos[1].item())
            agent_positions.append(agent["level"])

        # Extract relative fruit positions and picked status
        fruit_info = []
        
        # Calculate distances and sort
        fruits_with_dist = []
        for fruit in abs_fruit_info:
            dx = fruit["x"] - subject.pos[0].item()
            dy = fruit["y"] - subject.pos[1].item()
            dist = (dx**2 + dy**2)**0.5
            fruits_with_dist.append((dist, dx, dy, fruit["picked"], fruit["level"]))
            
        fruits_with_dist.sort(key=lambda x: x[0])

        for i in range(NEAREST_FRUITS_COUNT):
            if i < len(fruits_with_dist):
                _, dx, dy, picked, level = fruits_with_dist[i]
                fruit_info.append(dx)
                fruit_info.append(dy)
                fruit_info.append(level)
                fruit_info.append(picked)
            else:
                # Padding with "picked" fruits
                fruit_info.extend([0, 0, 1])

        # Concat and convert to tensor
        return torch.tensor(agent_positions + fruit_info, dtype=torch.float32)
       
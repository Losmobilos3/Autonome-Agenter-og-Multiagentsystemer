import matplotlib.pyplot as plt

import numpy as np
from agent import Agent
from fruit import Fruit
from config import AGENT_PLOT_SIZE, FRUIT_PLOT_SIZE, COLLECTION_DISTANCE, NEAREST_FRUITS_COUNT, FRAMES
from matplotlib.lines import Line2D 
import torch
from nn_Q import Model, supervised_train
from action_ds import ActionDataset

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
        self.size: np.ndarray = np.vstack([width, height])
        self.no_agents: int = no_agents

        # Define size of state vector
        self.state_size: int = 3 * (no_agents - 1) + 4 * NEAREST_FRUITS_COUNT

        # Initialize Worker agents
        self.agents: list[Agent] = [
            Agent(self, i) for i in range(no_agents)
        ]

        def random_level_gen(ratio_1_2: float = 0.7):
            return 1 if np.random.rand() < ratio_1_2 else 2

        # Initialize fruits
        self.fruits: list[Fruit] = [
            Fruit(position=np.random.randint(np.zeros((2, 1)), self.size, (2, 1)), level=random_level_gen()) for _ in range(no_fruits)
        ]

        # Used for learning
        self.abs_prior_state: dict = {}

        # Shared policy approximations hat(pi)
        self.agent_policy_estimations: list[Model] = [Model(input_size=self.state_size, hidden_size=64, include_softmax=True) for _ in range(no_agents)]
        self.agent_policy_optimizers: list[torch.optim.Optimizer] = [torch.optim.Adam(model.parameters(), lr=0.001) for model in self.agent_policy_estimations]
        self.agent_action_buffers: list[ActionDataset] = [ActionDataset(state_size=self.state_size) for _ in range(no_agents)]  # To store past actions for each agent

        self.fig, self.ax = plt.subplots(figsize=(17.5, 10))

        ### Variables used for performance metrics
        self.total_fruits_collected: np.ndarray = np.zeros(2)
        self.steps_used: int = 0

    def get_obstructed_cells(self) -> list[np.ndarray]:
        obstructed = [agent.pos for agent in self.agents] + [f.pos for f in self.fruits if not f.picked]
        return obstructed


    def init_env(self) -> None:
        """Initialize the environment"""
        for agent in self.agents:
            agent.pos = np.random.randint(np.zeros((2, 1)), self.size, (2,1)).astype(float)
            agent.vel = np.zeros((2, 1))
            agent._update_target_model()

        for fruit in self.fruits:
            fruit.pos = np.random.randint(np.zeros((2, 1)), self.size, (2, 1))
            fruit.picked = 0

        self.abs_prior_state = None

        ## Reset performance metrics
        self.total_fruits_collected = np.zeros(2) # Array to count fruits collected at levels (index = level - 1)
        self.steps_used = 0


    def run_episodes(self, no_episodes, max_steps_per_episode) -> None:
        """Run _n_ episodes of the simulation

        Args:
            no_episodes (int): Number of episodes to run
            max_steps_per_episodes (int): Number of steps to run per episode
        """
        for episode in range(no_episodes):
            print(f"Starting episode {episode+1}/{no_episodes}")
            self.init_env()
            for i in range(max_steps_per_episode):
                self.step(i)

    def step(self, step_num: int = None) -> None:
        """Take a step in the simulation
        
        Args:
            step_num (int): Step counter (if set, will run training every 20 steps)
        """ 
        for i, agent in enumerate(self.agents):
            decision_id = agent.act()

            # Update agent action buffer with newly taken action
            self.agent_action_buffers[i].add_data(self.get_state_tensor(i), decision_id)

            # Fit policy approximations hat(pi) here
            if step_num and step_num % 50 == 0:
                supervised_train(self.agent_policy_estimations[i], self.agent_action_buffers[i], optimizer=self.agent_policy_optimizers[i])
        
        self.save_prior_state()

        # Update the agents' states
        for agent in self.agents:
            agent.update()

        # Check for fruit collection
        collecting_agents = [agent for agent in self.agents if agent.collecting]
        for fruit in self.fruits:
            # Skip if the fruit is already picked
            if fruit.picked:
                continue

            # Find all collecting agents that are close to the fruit
            close_agents = [
                agent for agent in collecting_agents
                if np.linalg.norm(fruit.pos - agent.pos) <= COLLECTION_DISTANCE
            ]
            
            # Compute the total level of agents around the fruit
            collection_level = sum(agent.level for agent in close_agents)
                
            # Check if the fruit can be collected
            if fruit.level <= collection_level:
                # Pick the fruit
                fruit.picked = True

                # Count the collected fruit
                self.total_fruits_collected[fruit.level - 1] += 1 # Count the collected fruits

                # Give all close agents a reward
                for agent in close_agents:
                    agent.give_reward(5 * fruit.level) # Reward for collecting a fruit

                # Only pick up 1 fruit per frame
                break

        # Record the step index if all fruits have been picked
        if all([fruit.picked for fruit in self.fruits]):
            self.steps_used = step_num

        # Give rewards after all agents have moved
        for agent in self.agents:
            self.give_rewards(agent)



    def give_rewards(self, agent: Agent) -> None:
        """Distribute rewards to an agent

        Args:
            agent (Agent): Agent to be rewarded
        """
        remaining_fruit = [fruit for fruit in self.fruits if not fruit.picked]
        if not remaining_fruit:
            return

        # Give reward based on distance to nearest fruit
        dist_to_fruits = np.array([np.linalg.norm(fruit.pos - agent.pos) for fruit in remaining_fruit])
        direction_nearest_fruit = (remaining_fruit[np.argmin(dist_to_fruits)].pos - agent.pos)
        direction_reward = direction_nearest_fruit.T @ agent.vel / (np.linalg.norm(direction_nearest_fruit) * (np.linalg.norm(agent.vel) + 1e-5)) # cos(alpha)
        # Reward going to the nearest fruit
        if direction_reward > 0:
            agent.give_reward(direction_reward.item() * 0.1)

        # Reward for standing still near a fruit
        # if np.min(dist_to_fruits) <= COLLECTION_DISTANCE: #! Seems to make the agents wait and never pick up the fruit
        #     agent.give_reward(0.1)

        agent.give_reward(-0.01)  # Small time penalty to encourage efficiency



    def setup_plot(self):
        """Initializes the plot"""
        self.ax.axis("equal")
        self.ax.set_xlim(-0.5, self.size[0] + 0.5)
        self.ax.set_ylim(-0.5, self.size[1] + 0.5)

        # Add grid
        self.ax.set_xticks(np.arange(-0.5, self.size[0].item(), 1))
        self.ax.set_yticks(np.arange(-0.5, self.size[1].item(), 1))
        self.ax.grid(True)

        # Remove ticks
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
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
            y_stem_end = y_stem_start + FRUIT_PLOT_SIZE / 1000

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
        # self.reward_text = self.ax.text(
        #     10, self.size[1] - 10,  # Top-left corner
        #     f"Agent 0 Reward: {self.agents[0].reward:.2f}",
        #     fontsize=14, color='black',
        #     bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
        # )
        
        # Return ALL artists that need to be redrawn
        return (*self.fruit_level_patches, *self.agent_level_patches,
                *self.fruit_stems)#, self.reward_text) # Include the stems and reward text
        
    def animate_frame(self, i):
        """Animates frame _i_

        Args:
            i (int): Frame index to animate
        """
        if i % 100 == 0:
            print(f"Animating frame {i}")

        # Update the simulation state
        self.step(i)
        
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
            y_stem_end = y_stem_start + FRUIT_PLOT_SIZE / 1000
            self.fruit_stems[j].set_data([x_stem_start, x_stem_end], [y_stem_start, y_stem_end])

        # Update reward display for the first agent
        # self.reward_text.set_text(f"Agent 0 Reward: {self.agents[0].reward:.2f}")

        # Save performance metrics
        if i >= FRAMES - 1:
            self.save_metrics()
        
        # Return all updated plot objects (text patches and stems)
        return (*self.fruit_level_patches, *self.agent_level_patches, *self.fruit_stems)#, self.reward_text) 
    

    def save_metrics(self):
        with open("MARL_data.txt", "a") as f:
            f.write(f"{sum(self.total_fruits_collected)}, {self.total_fruits_collected[0]}, {self.total_fruits_collected[1]}\n")

    def save_prior_state(self) -> dict:
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
        # Extract relative agent positions, except for agent_idx
        subject = self.agents[agent_idx]
        
        agent_positions = []
        for agent in self.agents:

            # Skip if agent is itself
            if agent == subject:
                continue

            agent_positions.extend([
                (agent.pos[0] - subject.pos[0]).item(), # x-position
                (agent.pos[1] - subject.pos[1]).item(), # y-position
                agent.level # agent level
            ])

        # Extract relative fruit positions and picked status
        fruit_info = []

        # Calculate distances and sort
        fruits_with_dist = []
        for fruit in self.fruits:
            if fruit.picked:
                continue
            dx = fruit.pos[0].item() - subject.pos[0].item()
            dy = fruit.pos[1].item() - subject.pos[1].item()
            dist = np.sqrt(dx**2 + dy**2)

            fruits_with_dist.append((dist, dx, dy, fruit.picked, fruit.level))
        
        fruits_with_dist.sort(key=lambda x: x[0])

        for i in range(NEAREST_FRUITS_COUNT):
            if i < len(fruits_with_dist):
                _, dx, dy, picked, level = fruits_with_dist[i]
                fruit_info.append(dx)
                fruit_info.append(dy)
                fruit_info.append(level - subject.level)
                fruit_info.append(picked)
            else:
                # Padding with "picked" fruits (effectively invisible)
                fruit_info.extend([0, 0, 1, 1])

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
        
        # Find the subject's prior state
        subject_prior = next((a for a in abs_agents if a["agent_idx"] == agent_idx), None)
        if subject_prior is None:
            return None
            
        subject_prior_pos = np.array([subject_prior["x"], subject_prior["y"]])
        
        # Extract realtive agent positions, except for agent_idx
        agent_positions = []
        for agent in abs_agents:
            if agent["agent_idx"] == agent_idx:
                continue
            agent_positions.append(agent["x"] - subject_prior_pos[0])
            agent_positions.append(agent["y"] - subject_prior_pos[1])
            agent_positions.append(agent["level"])

        # Extract relative fruit positions and picked status
        fruit_info = []
        
        # Calculate distances and sort
        fruits_with_dist = []
        for fruit in abs_fruit_info:
            if fruit["picked"]:
                continue
            dx = fruit["x"] - subject_prior_pos[0]
            dy = fruit["y"] - subject_prior_pos[1]
            dist = (dx**2 + dy**2)**0.5
            fruits_with_dist.append((dist, dx, dy, fruit["picked"], fruit["level"] - subject_prior["level"]))
            
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
                fruit_info.extend([0, 0, 1, 1])

        # Concat and convert to tensor
        return torch.tensor(agent_positions + fruit_info, dtype=torch.float32)
       
import numpy as np
import torch
import torch.nn.functional as F
from nn_Q import Model

S, A, C, L = 0.05, 0.005, 0.005, 0.005

class Agent:
    """Agent class
    
    Attributes:
        sim_ref (Simulation): Reference to the simulation
        pos (np.NDArray[x, y]): Agent position
        vel (np.NDArray[lin, ang]): Agent velocity
        reward (float): Current agent reward
        collecting (bool): Flag determining whether the agent is currently collecting
        level (int): Agent level (determines what level fruit the agent can collect)
        Q_model (Model): Model used to determine the agents action
        discount_factor (float): Reward discount factor
        optim (torch.Optim.Adam): Agent optimization function
        prior_state (torch.Tensor): Prior state tensor
        prior_action (int): Decision ID for the previously executed action
    """
    def __init__(self, sim_ref: "Simulation", agent_idx: int, level: int = 1):
        """Initialize an Agent

        Args:
            sim_ref (Simulation): Reference the simulation
            agent_idx (int): Index of the agent
            level (int): Level of the agent (default: 1)
        """
        self.sim_ref = sim_ref
        self.agent_idx = agent_idx
        self.pos = np.random.randint(np.zeros((2, 1)), self.sim_ref.size, (2,1)).astype(float)
        self.vel = np.zeros((2, 1))
        self.reward = 0
        self.collecting = False
        self.level = level
        self.Q_model = Model(input_size=self.sim_ref.state_size, hidden_size=64)
        self.discount_factor = 0.9  # Discount factor for future rewards
        self.optim = torch.optim.Adam(self.Q_model.parameters(), lr=0.1)
        self.prior_action = None

    def give_reward(self, reward: float):
        """Give reward to the agent

        Args:
            reward (float): Value of the reward
        """
        self.reward += reward

    def update(self):
        """Update the agent state"""
        # Clamp velocity
        next_pos = self.pos + self.vel
        if np.any(next_pos < np.zeros((2, 1))) or np.any(next_pos > self.sim_ref.size):
            self.vel = np.zeros((2, 1))
        self.pos += self.vel

    def act(self):
        """Let the Agent take an action based on its model

        Returns:
            tuple[int, float]: Decision ID, and Agent velocity
        """
        self.collecting = False # Reset from prior frame
        decision_id = self.learn()

        # Execute action based on decision_id
        # print(f"Agent {self.agent_idx} decision: {decision_id}")
        match decision_id:
            case 0:  # Move up
                self.vel = np.array([0, 1]) [:, np.newaxis]
            case 1:  # Move down
                self.vel = np.array([0, -1]) [:, np.newaxis]
            case 2:  # Move left
                self.vel = np.array([-1, 0]) [:, np.newaxis]
            case 3:  # Move right
                self.vel = np.array([1, 0]) [:, np.newaxis]
            case 4:  # Collect
                self.collecting = True
        return decision_id

    def learn(self):
        """Update the Agent model, and return a decision and velocity

        Returns:
            tuple[int, float]: Decision ID, and Agent velocity
        """
        curr_state = self.sim_ref.get_state_tensor(self.agent_idx)
        Q_val_curr = self.Q_model(curr_state)

        # Select action based on Q-values (using argmax for stability)
        probs = torch.softmax(Q_val_curr, dim=0).detach().numpy()
        decision_id = torch.multinomial(torch.tensor(probs), num_samples=1).item()

        # if no prior decision, do not learn
        prior_state = self.sim_ref.get_prior_state(self.agent_idx)
        if prior_state is not None:
            # Get the reward, and reset the reward variable
            reward = self.reward
            self.reward = 0

            # Re-evaluate prior state to get gradient-attached Q-values
            Q_prev = self.Q_model(prior_state)
            predicted_q = Q_prev[self.prior_action]  # Chosen action Q-value from prior state

            # Target Q-value (detached) SE DEEP LEARNING SLIDES
            target = reward + self.discount_factor * torch.max(Q_val_curr)  # reward + best action Q-value from current state
            loss = F.mse_loss(predicted_q, target)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.prior_action = decision_id
        return decision_id

    def get_closest_fruit(self):
        """Returns the fruit closest to the Agent

        Returns:
            Fruit: Fruit object closest to the agent
        """
        closest_fruit = None
        min_dist = float('inf')
        for fruit in self.sim_ref.fruits:
            if fruit.picked:
                continue
            dist = np.linalg.norm(fruit.pos - self.pos)
            if dist < min_dist:
                min_dist = dist
                closest_fruit = fruit
        return closest_fruit

# NOT USED YET
class Memory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
    
    def push(self, state, action, reward, next_state):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states = zip(*batch)
        return states, actions, rewards, next_states
    
    def __len__(self):
        return len(self.memory)
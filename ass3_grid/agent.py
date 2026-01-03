import numpy as np
import torch
from config import COLLECTION_DISTANCE, MODE
from fruit import Fruit
import torch.nn.functional as F
from nn_Q import Model
import random
import copy
import itertools

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
        self.sim_ref: "Simulation" = sim_ref
        self.agent_idx: int = agent_idx
        self.pos: np.ndarray = np.random.randint(np.zeros((2, 1)), self.sim_ref.size, (2,1)).astype(float)
        self.vel: np.ndarray = np.zeros((2, 1))
        self.reward: int = 0
        self.collecting: bool = False
        self.level: int = level

        # Instanciate models
        self.Q_model: Model = Model(input_size=self.sim_ref.state_size, no_agents=self.sim_ref.no_agents, hidden_size=64)
        self.Q_target: Model = copy.deepcopy(self.Q_model)
        self.Q_target.eval()
        for param in self.Q_target.parameters():
            param.requires_grad = False

        self.discount_factor: float = 0.9  # Discount factor for future rewards
        self.optim: torch.optim.Optimizer = torch.optim.Adam(self.Q_model.parameters(), lr=0.001)
        self.prior_action: int = None
        self.epsilon: float = 1.0
        self.epsilon_decay: float = 0.99995
        self.epsilon_min: float = 0.01
        self.replay_memory: ReplayMemory = ReplayMemory(capacity=10000)

    def _update_target_model(self) -> None:
        self.Q_target = copy.deepcopy(self.Q_model)
        self.Q_target.eval()
        for param in self.Q_target.parameters():
            param.requires_grad = False

    def give_reward(self, reward: float) -> None:
        """Give reward to the agent

        Args:
            reward (float): Value of the reward
        """
        self.reward += reward

    def update(self) -> None:
        """Update the agent state"""
        # Clamp velocity
        next_pos = self.pos + self.vel
        obstructed_cells = self.sim_ref.get_obstructed_cells()
        cell_obstructed = any(np.array_equal(next_pos, cell) for cell in obstructed_cells)
        if (np.any(next_pos < np.zeros((2, 1))) or np.any(next_pos >= self.sim_ref.size)) or cell_obstructed:
            self.vel = np.zeros((2, 1))
        self.pos += self.vel

    def act(self) -> int:
        """Let the Agent take an action based on its model

        Returns:
            int: Decision ID
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.collecting = False # Reset from prior frame
        decision_id = self.learn()
        # decay learning

        # Execute action based on decision_id
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
                self.vel = np.zeros((2,1))
                self.collecting = True
        return decision_id
    
    def learn(self) -> int:
        """Update the Agent model, and return a decision and velocity

        Returns:
            int: Decision ID
        """
        curr_state = self.sim_ref.get_state_tensor(self.agent_idx)
        AV_curr = self.calc_AV(curr_state.unsqueeze(0))
        if np.random.rand() < self.epsilon:
            decision_id = torch.tensor(np.random.randint(0, 5))
        else:
            decision_id = torch.argmax(AV_curr)

        # if no prior decision, do not learn
        prior_state = self.sim_ref.get_prior_state(self.agent_idx)
        if prior_state is not None:
            # Get the reward, and reset the reward variable
            reward = self.reward
            self.reward = 0

            # Store experience in replay memory
            self.replay_memory.push(prior_state, self.prior_action, reward, curr_state)

            # Train on a batch from replay memory
            if len(self.replay_memory.memory) < 64:
                batch = self.replay_memory.memory  # Not enough data to train
            else:
                batch = random.sample(self.replay_memory.memory, 64)

            # Extract values
            prior_states, actions, rewards, curr_states = map(torch.stack, zip(*batch))

            AV_prev = self.calc_AV(prior_states)
            prev_AV = AV_prev[np.arange(0, len(actions)), actions]

            # Calc AV
            AV = self.calc_AV(curr_states)

            # Calculate target for model
            target = rewards + self.discount_factor * torch.max(AV, dim=1)[0].detach()

            # loss_b = F.mse_loss(prev_q.long(), target)
            loss_b = F.mse_loss(prev_AV, target.float())

            self.optim.zero_grad()
            loss_b.backward()
            self.optim.step()

        if isinstance(decision_id, torch.Tensor):
            self.prior_action = decision_id.item()
            return decision_id.item()
        
        self.prior_action = decision_id
        return decision_id

    def calc_AV(self, states) -> torch.Tensor:
        no_actions = 5
        batch_size = states.shape[0]

        # Get policy estimations hat(pi) for all agents, and filter away own
        piHatResults = [pi(states) for i, pi in enumerate(self.sim_ref.agent_policy_estimations) if i != self.agent_idx]

        # Create list of all possible joint actions
        joint_actions = list(itertools.product(range(no_actions), repeat=len(piHatResults)))
        
        AV = torch.zeros((batch_size, no_actions))
        for joint_action in joint_actions:
            joint_action = list(joint_action)
            # One-hot encode and concat joint action
            one_hot_joint_action = []
            for action in joint_action:
                action = torch.tensor(action).long()
                action = F.one_hot(action, num_classes=no_actions).float()
                one_hot_joint_action.append(action)
            
            # Stack and flatten, then expand to batch size
            one_hot_joint_action = torch.stack(one_hot_joint_action).flatten().unsqueeze(0).expand(batch_size, -1)

            Q_val = self.Q_model(states, one_hot_joint_action)

            # Calc product of all hat(pi) for each joint action
            joint_likelihood = torch.ones((batch_size, 1))
            with torch.no_grad():
                for i, piHatResult in enumerate(piHatResults):
                    # piHatResult is [batch_size, 5]
                    # We want the probability of action joint_action[i] for each batch element
                    probs = piHatResult[:, joint_action[i]].unsqueeze(1)
                    joint_likelihood *= probs
                
            # Multiply Q values with corresponding product of hat(pi) and add to AV
            AV += Q_val * joint_likelihood
            
        return AV

    def get_closest_fruit(self) -> Fruit:
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

# Used to do experience replay / mini-batching
class ReplayMemory(torch.utils.data.Dataset):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
    
    def push(self, prior_state, action, reward, curr_state):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((prior_state, torch.tensor(action), torch.tensor(reward), curr_state))
    
    def __getitem__(self, idx):
        return self.memory[idx]
    
    def __len__(self):
        return len(self.memory)
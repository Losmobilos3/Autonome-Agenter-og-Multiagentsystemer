import numpy as np
from config import MAX_VEL
import torch
import torch.nn.functional as F

S, A, C, L = 0.05, 0.005, 0.005, 0.005

class Agent:
    def __init__(self, sim_ref: "Simulation"):
        self.sim_ref = sim_ref
        self.pos = np.random.rand(2, 1) * self.sim_ref.size
        self.vel = np.random.rand(2, 1) * 3
        self.avg_dist_to_leader_data = []
        self.reward = 0

    def give_reward(self, reward: float):
        self.reward += reward

    def update(self):
        # Clamp velocity
        speed = np.linalg.norm(self.vel)
        if speed > MAX_VEL:
            self.vel = (self.vel / speed) * MAX_VEL

        self.pos += self.vel

    def act(self):
        pass  # To be implemented in subclasses


from nn_Q import Model

class Worker(Agent):
    def __init__(self, sim_ref: "Simulation", level: int, num_agents: int, num_fruits: int):
        super().__init__(sim_ref)
        self.level = level
        self.Q_model = Model(input_size=(2 * num_agents + 3 * num_fruits), hidden_size=64)
        self.discount_factor = 0.9  # Discount factor for future rewards
        self.optim = torch.optim.Adam(self.Q_model.parameters(), lr=0.1)
        self.prior_state = None
        self.prior_action = None


    def act(self):
        decision_id, vel = self.learn()
        # Execute action based on decision_id
        if decision_id == 0:  # Move
            self.vel = vel.detach().numpy().reshape(2, 1)
        elif decision_id == 1:  # Collect
            self.collect_fruit()
        elif decision_id == 2:  # Do nothing
            self.vel = np.zeros((2, 1))

    def learn(self):
        curr_state = self.sim_ref.get_state_tensor()
        Q_val_curr, vel = self.Q_model(curr_state)

        # Select action randomly based on Q-values (using softmax for probabilities)
        probs = torch.softmax(Q_val_curr, dim=0)
        decision_id = torch.multinomial(probs, num_samples=1).item()
        
        # if no prior decision, do not learn
        if self.prior_state is not None:
            reward = self.collect_reward()
            
            # Re-evaluate prior state to get gradient-attached Q-values
            Q_prev, mov_dir = self.Q_model(self.prior_state)
            predicted_q = Q_prev[self.prior_action] # Chosen action Q-value from prior state
            
            # Target Q-value (detached) SE DEEP LEARNING SLIDES
            target = reward + self.discount_factor * torch.max(Q_val_curr) # reward + best action Q-value from current state
            dir_to_fruit = torch.tensor(self.get_closest_fruit().pos - self.pos, dtype=torch.float32)
            loss = F.mse_loss(predicted_q, target) + F.mse_loss(mov_dir, dir_to_fruit)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.prior_state = curr_state.clone()
        self.prior_action = decision_id
        return decision_id, vel
        
    def collect_reward(self):
        reward = self.reward
        self.reward = 0  # Reset reward after using it
        return reward
    
    def collect_fruit(self):
        for fruit in self.sim_ref.fruits:
            if fruit.picked:
                continue
            if np.linalg.norm(fruit.pos - self.pos) < 3.0:
                fruit.picked = 1
                self.give_reward(1)  # Reward for collecting a fruit
                break # Only pick up 1 fruit per frame
        
    def get_closest_fruit(self):
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
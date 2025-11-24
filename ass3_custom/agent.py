import numpy as np
from config import MAX_VEL
import torch

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
        # Limit velocity
        self.vel *= MAX_VEL
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
        self.optim = torch.optim.Adam(self.Q_model.parameters(), lr=0.001)
        self.prior_action_q_value = None

    def update(self):
        # Limit velocity
        self.vel *= MAX_VEL / np.linalg.norm(self.vel)
        self.pos += self.vel

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
        decision_id = torch.argmax(Q_val_curr).item()
        
        # if no prior decision, do not learn
        if self.prior_action_q_value is not None:
            reward = self.collect_reward()
            self.Q_model.zero_grad()
            loss = (reward + self.discount_factor * torch.max(Q_val_curr) - self.prior_action_q_value)**2
            loss.backward()
            self.optim.step()

        self.prior_action_q_value = Q_val_curr[decision_id].detach()
        return decision_id, vel
        
    def collect_reward(self):
        reward = self.reward
        self.reward = 0  # Reset reward after using it
        return reward
    
    def collect_fruit(self):
        return 0
        
        
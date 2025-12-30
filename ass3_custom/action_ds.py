import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class ActionDataset(Dataset):
    def __init__(self, state_size):
        self.state_history = torch.zeros((0, state_size))
        self.action_history = torch.zeros((0, 3))  # Assuming 3 possible actions (Move, collect, nothing)
        self.vel_history = torch.zeros((0, 2)) # (x_vel, y_vel)

    def add_data(self, state, action, vel):
        self.state_history = torch.cat((self.state_history, state.unsqueeze(0)), dim=0)
        action_one_hot = F.one_hot(torch.tensor(action), num_classes=3).float()
        self.action_history = torch.cat((self.action_history, action_one_hot.unsqueeze(0)), dim=0)
        self.vel_history = torch.cat((self.vel_history, vel.unsqueeze(0)), dim=0)

    def __len__(self):
        return self.state_history.shape[0]

    def __getitem__(self, idx):
        return self.state_history[idx], self.action_history[idx], self.vel_history[idx] # X, action, velocity

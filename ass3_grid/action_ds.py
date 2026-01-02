import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class ActionDataset(Dataset):
    def __init__(self, state_size, max_size=300):
        self.max_size = max_size
        self.states = torch.zeros((max_size, state_size))
        self.actions = torch.zeros((max_size, 5))
        self.curr_idx = 0
        self.size = 0

    def add_data(self, state, action):
        if self.curr_idx >= self.max_size:
            self.curr_idx = 0  # Overwrite old data
        
        self.states[self.curr_idx] = state.detach()
        self.actions[self.curr_idx] = F.one_hot(torch.tensor(action), num_classes=5).float()
        
        self.curr_idx += 1
        self.size = max(self.size, self.curr_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
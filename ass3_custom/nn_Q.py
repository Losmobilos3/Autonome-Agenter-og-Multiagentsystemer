import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.reg_layer = torch.nn.Linear(hidden_size, 3) # (x_vel, y_vel, vel_magnitude)
        self.decision_layer = torch.nn.Linear(hidden_size, 3)  # Assuming 3 possible actions (Move, collect, nothing)


    def forward(self, x):
        x = self.fc1(x)
        features = torch.nn.functional.relu(x)
        vel = self.reg_layer(features)
        decision = self.decision_layer(features)
        decision = torch.nn.functional.softmax(decision, dim=0)
        return decision, vel
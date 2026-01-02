import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.decision_layer = torch.nn.Linear(hidden_size, 5)  # Assuming 5 possible actions (Up, Down, Left, Right, Collect)


    def forward(self, x):
        x = self.fc1(x)
        features = torch.nn.functional.relu(x)
        decision = self.decision_layer(features)
        return decision
    

def supervised_train(model, dataloader):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for X, y_action in dataloader:
        optim.zero_grad()

        Q_vals = model(X)

        loss_decision = torch.nn.functional.cross_entropy(Q_vals, y_action)
        loss = loss_decision
        loss.backward()

        optim.step()
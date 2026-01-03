import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, no_agents=1, hidden_size=64, include_softmax=False):
        super(Model, self).__init__()
        self.include_softmax = include_softmax
        self.fc1 = torch.nn.Linear(input_size + (no_agents-1)*5, hidden_size)
        self.decision_layer = torch.nn.Linear(hidden_size, 5)  # Assuming 5 possible actions (Up, Down, Left, Right, Collect)

    def forward(self, x, x_joint_action=None):
        if x_joint_action is not None: # Concat joint action if provided
            x = torch.cat((x, x_joint_action), dim=-1)

        x = self.fc1(x)
        features = torch.nn.functional.relu(x)
        decision = self.decision_layer(features)
        if self.include_softmax:
            decision = torch.nn.functional.softmax(decision, dim=-1)
        return decision
    

def supervised_train(model, dataloader):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    for X, y_action in dataloader:
        optim.zero_grad()

        Q_vals = model(X)

        loss_decision = torch.nn.functional.cross_entropy(Q_vals, y_action)
        loss = loss_decision
        loss.backward()

        optim.step()
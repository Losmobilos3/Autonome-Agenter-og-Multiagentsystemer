import torch

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.reg_layer = torch.nn.Linear(hidden_size, 2) # (x_vel, y_vel)
        self.decision_layer = torch.nn.Linear(hidden_size + 2, 3)  # Assuming 3 possible actions (Move, collect, nothing)


    def forward(self, x):
        x = self.fc1(x)
        features = torch.nn.functional.relu(x)
        vel = self.reg_layer(features)
        decision = self.decision_layer(torch.cat((features, vel), dim=0))
        # decision = torch.nn.functional.softmax(decision, dim=0)
        return decision, vel
    

def supervised_train(model, dataloader):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for X, y_action, y_vel in dataloader:
        optim.zero_grad()

        Q_vals, pred_vels = model(X)

        loss_decision = torch.nn.functional.cross_entropy(Q_vals, y_action)
        loss_velocity = torch.nn.functional.mse_loss(pred_vels, y_vel)
        loss = loss_decision + loss_velocity
        loss.backward()

        optim.step()
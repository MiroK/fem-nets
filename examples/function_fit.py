import fem_nets
import dolfin as df
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 2d spatial points
        self.lin1 = nn.Linear(1, 10)
        self.lin2 = nn.Linear(10, 1)
 
    def forward(self, x):
        y = self.lin1(x)
        y = torch.relu(y)
        y = self.lin2(y)
        return y


mesh = df.UnitIntervalMesh(7)
V = df.FunctionSpace(mesh, 'CG', 1)

model_fe = fem_nets.to_torch(V)
model_fe.double()

model_nn = Network()
model_nn.double()

params = list(model_fe.parameters()) + list(model_nn.parameters())

model = lambda x: model_fe(x).unsqueeze(2) + model_nn(x)

maxiter = 1

loss_fn = nn.MSELoss()

npts = 1000
# So we sample (0, 1)^2 to get points for the PDE residual evaluation
x = torch.rand(1, npts, 1, dtype=torch.float64)
y0 = x + torch.sin(4*np.pi*x)
x.requires_grad = True

nsteps = 0

def closure():
    global nsteps
    nsteps += 1
    optimizer.zero_grad()
    y = model(x)

    loss = torch.mean((y - y0)**2)
    loss.backward()
    print(f'\tStep {nsteps} => {float(loss)}')
    return loss

optimizer = optim.LBFGS(params, max_iter=maxiter,
                        history_size=1000, tolerance_change=1e-12,
                        line_search_fn="strong_wolfe")

for epoch in range(100):
    print(f'Epoch {epoch}')
    nsteps = 0
    optimizer.step(closure)
    print()
    
import matplotlib.pyplot as plt

x_ = x.detach().numpy().flatten()
y0_ = y0.detach().numpy().flatten()

y_fe = model_fe(x).detach().numpy().flatten()
y_nn = model_nn(x).detach().numpy().flatten()
y_ = model(x).detach().numpy().flatten()

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(x_, y0_, linestyle='none', marker='x', color='blue', label='data', markersize=10)

ax.plot(x_, y_, linestyle='none', marker='o', color='red', label='Combined')
ax.plot(x_, y_nn, linestyle='none', marker='o', color='orange', label='Network')
ax.plot(x_, y_fe, linestyle='none', marker='o', color='black', label='FE')

plt.legend()
plt.savefig('../docs/function_fit.png', bbox_inches='tight')
plt.show()

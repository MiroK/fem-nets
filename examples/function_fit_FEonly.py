import fem_nets
import dolfin as df
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


mesh = df.UnitIntervalMesh(5)
V1 = df.FunctionSpace(mesh, 'CG', 1)
model_lin = fem_nets.to_torch(V1)
model_lin.double()

mesh = df.UnitIntervalMesh(3)
V2 = df.FunctionSpace(mesh, 'CG', 2)
model_quad = fem_nets.to_torch(V2)
model_quad.double()

params = list(model_lin.parameters()) + list(model_quad.parameters())

model = lambda x: model_lin(x).unsqueeze(2) + model_quad(x).unsqueeze(2)

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

y_lin = model_lin(x).detach().numpy().flatten()
y_quad = model_quad(x).detach().numpy().flatten()
y_ = model(x).detach().numpy().flatten()

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(x_, y0_, linestyle='none', marker='x', color='blue', label='data', markersize=10)

ax.plot(x_, y_, linestyle='none', marker='o', color='red', label='Combined')
ax.plot(x_, y_quad, linestyle='none', marker='o', color='orange', label='FE 2')
ax.plot(x_, y_lin, linestyle='none', marker='o', color='black', label='FE 1')

plt.legend()
plt.savefig('../docs/function_fit2.png', bbox_inches='tight')
plt.show()

import torch
from fem_nets.networks import VectorLagrange1NN
import dolfin as df
import numpy as np
import gmshnics

mesh, _ = gmshnics.gDisk(center=(0, 0), inradius=0.2, outradius=1.0, size=0.8)

V = df.VectorFunctionSpace(mesh, 'CG', 1)
f = df.Expression(('x[0]-2*x[1]', '2*x[0]+3*x[1]'), degree=1)
fh = df.interpolate(f, V)

nn = VectorLagrange1NN(mesh)
nn.double()
nn.set_from_coefficients(fh.vector().get_local())

# We will do the visual comparison in piecewise constants
Q = df.VectorFunctionSpace(mesh, 'DG', 0)
fh_true = df.interpolate(f, Q)

Qi = Q.sub(0).collapse()
# NOTE: eval on subspace
x = torch.tensor(Qi.tabulate_dof_coordinates()).unsqueeze(0)
y = nn(x)  # first colum is x, second is y

# Want to build back a function
fh_x, fh_y = df.Function(Qi), df.Function(Qi)
fh_x.vector().set_local(y[..., 0].detach().numpy().flatten())
fh_y.vector().set_local(y[..., 1].detach().numpy().flatten())

fh_mine = df.Function(Q)
df.assign(fh_mine, [fh_x, fh_y])

error = df.sqrt(df.assemble(df.inner(fh_true - fh_mine, fh_true - fh_mine)*df.dx))
assert error < 1E-14
# Now you can just plot the two for example
df.File('nn.pvd') << fh_mine

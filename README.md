# Finite Element Neural Networks

Representation of some finite element function spaces (as defined in
FEniCS) in terms of neural networks. That is, we construct neural networks
whose weights are the coefficient vectors (as ordered in FEniCS)

## Dependencies
- `FEniCS` (2019.1.0 and higher) stack
- `pytorch`
- [`gmshnics`](https://github.com/MiroK/gmshnics) for some tests

## Usage
Basic idea (taken from [`examples/compate_plot.py`](https://github.com/MiroK/fem-nets/blob/master/examples/compare_plot.py))

```python
import torch
import fem_nets
import dolfin as df
import numpy as np
import gmshnics

mesh, _ = gmshnics.gDisk(center=(0, 0), inradius=0.2, outradius=1.0, size=0.8)

V = df.VectorFunctionSpace(mesh, 'CG', 1)
f = df.Expression(('x[0]-2*x[1]', '2*x[0]+3*x[1]'), degree=1)
fh = df.interpolate(f, V)

nn = fem_nets.to_torch(V)
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
```

And viola

  <p align="center">
    <img src="https://github.com/MiroK/fem-nets/blob/master/docs/nn.png">
  </p>

For more functionality see [tests](https://github.com/MiroK/fem-nets/blob/master/test/test_lagrange1.py#L36).

## TODO
- [x] Suport for 1, 2, 3 d
- [x] Support for Discontinuous Lagrange
- [ ] Support for higher order
- [ ] Tensor values spaces (symmetric and skew because why not)
- [x] Convenience functions `to_torch(V)` where `V` is a function space
- [ ] Example with training a hybrid model
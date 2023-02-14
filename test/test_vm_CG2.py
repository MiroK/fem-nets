from fem_nets.vandermonde import compute_vandermonde_CG2
from fem_nets.utils import cell_centers, dof_coordinates, random_inside_points, quadrature_points

import dolfin as df
import torch
import torch.nn as nn
import numpy as np

import pytest

    
@pytest.mark.parametrize('mesh', (df.UnitIntervalMesh(30), ))
@pytest.mark.parametrize('f', (df.Expression('2*x[0]*x[0]-1', degree=2), df.Constant(1))) # , ))
@pytest.mark.parametrize('get_points', (cell_centers, quadrature_points, dof_coordinates, random_inside_points))
def test_vm_1d(mesh, f, get_points):
    '''Use cell centers as points for comparison'''
    V = df.FunctionSpace(mesh, 'CG', 2)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    x_ = get_points(V)
    x = torch.tensor(x_).reshape((1, ) + x_.shape)
    vandermonde = compute_vandermonde_CG2(x, mesh)

    # Pushing this through linear layer with coefs behaves as eval of f at x
    lin = nn.Linear(V.dim(), 1, bias=False)
    lin.double()
    with torch.no_grad():
        lin.weight[0] = torch.tensor(coefs)
    
    mine = lin(vandermonde).squeeze(2)
    true = torch.tensor(np.array([fh(xi_) for xi_ in x_]))

    assert abs(torch.norm(true-mine, np.inf)) < 1E-12

# --------------------------------------------------------------------

if __name__ == '__main__':
    mesh = df.UnitIntervalMesh(1)
    f = df.Expression('2*x[0]*x[0]-1', degree=2)
    get_points = quadrature_points
    
    test_vm_1d(mesh, f, get_points)

    V = df.FunctionSpace(mesh, 'CG', 2)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    x_ = get_points(V)
    x = torch.tensor(x_).reshape((1, ) + x_.shape)
    vandermonde = compute_vandermonde_CG2(x, mesh)

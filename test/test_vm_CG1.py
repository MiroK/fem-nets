from fem_nets.vandermonde import compute_vandermonde_CG1
from fem_nets.utils import cell_centers, dof_coordinates, random_inside_points

import dolfin as df
import torch
import torch.nn as nn
import numpy as np

import pytest

meshes = (df.UnitSquareMesh(5, 5), )
try:
    import gmshnics

    square_unstructured, _ = gmshnics.gUnitSquare(0.8)
    disk, _ = gmshnics.gDisk(center=(0, 0), inradius=0.2, outradius=1.0, size=0.8)

    meshes = meshes + (square_unstructured, disk)
except ImportError:
    print('Gmshnics missing')


@pytest.mark.parametrize('mesh', meshes)
@pytest.mark.parametrize('f', (df.Constant(1), df.Expression('2*x[0]-3*x[1]', degree=1), ))
@pytest.mark.parametrize('get_points', (cell_centers, dof_coordinates, random_inside_points))
def test_cell_center(mesh, f, get_points):
    '''Use cell centers as points for comparison'''
    V = df.FunctionSpace(mesh, 'CG', 1)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    x_ = get_points(V)
    x = torch.tensor(x_).reshape((1, ) + x_.shape)
    vandermonde = compute_vandermonde_CG1(x, mesh)

    # Pushing this through linear layer with coefs behaves as eval of f at x
    lin = nn.Linear(V.dim(), 1, bias=False)
    lin.double()
    with torch.no_grad():
        lin.weight[0] = torch.tensor(coefs)
    
    mine = lin(vandermonde).squeeze(2)
    true = torch.tensor(np.array([fh(xi_) for xi_ in x_]))

    assert abs(torch.norm(true-mine, np.inf)) < 1E-12

from fem_nets.networks import VectorLagrange1NN
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
@pytest.mark.parametrize('f', (df.Constant((1, 2)), df.Expression(('x[0]-2*x[1]', '3*x[1]+4*x[0]'), degree=1)))
@pytest.mark.parametrize('get_points', (cell_centers, dof_coordinates, random_inside_points))
def test_value(mesh, f, get_points):
    '''Use cell centers as points for comparison'''
    V = df.VectorFunctionSpace(mesh, 'CG', 1)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    nn = VectorLagrange1NN(mesh)
    nn.double()
    nn.set_from_coefficients(coefs)

    x_ = get_points(V)
    x = torch.tensor(x_, dtype=torch.float64).reshape((1, ) + x_.shape)

    mine = nn(x)
    true = torch.tensor(np.array([fh(xi_) for xi_ in x_]))

    assert abs(torch.norm(true-mine, np.inf)) < 1E-12

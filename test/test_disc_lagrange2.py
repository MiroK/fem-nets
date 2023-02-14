from fem_nets.networks import DiscLagrangeNN
from fem_nets.utils import cell_centers, quadrature_points
from fem_nets.calculus import grad

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
@pytest.mark.parametrize('get_points', (cell_centers, quadrature_points))
def test_value_linear(mesh, f, get_points):
    '''Use cell centers as points for comparison'''
    V = df.FunctionSpace(mesh, 'DG', 2)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    nn = DiscLagrangeNN(V)
    nn.double()
    nn.set_from_coefficients(coefs)

    x_ = get_points(V)
    x = torch.tensor(x_, dtype=torch.float64).reshape((1, ) + x_.shape)

    mine = nn(x)
    true = torch.tensor(np.array([fh(xi_) for xi_ in x_]))

    assert abs(torch.norm(true-mine, np.inf)) < 1E-12



@pytest.mark.parametrize('mesh', meshes)
@pytest.mark.parametrize('get_points', (cell_centers, ))
def test_gradient_linear(mesh, get_points):
    '''Use cell centers as points for comparison'''
    V = df.FunctionSpace(mesh, 'DG', 2)

    f = df.Expression('2*x[0]-3*x[1]', degree=1)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    nn = DiscLagrangeNN(V)
    nn.double()
    nn.set_from_coefficients(coefs)

    x_ = get_points(V)
    x = torch.tensor(x_, dtype=torch.float64).reshape((1, ) + x_.shape)

    x.requires_grad = True
    mine = grad(nn(x), x)
    true = torch.tile(torch.tensor(np.array([2., -3.])), (len(x_), 1))

    assert abs(torch.norm(true-mine, np.inf)) < 1E-12

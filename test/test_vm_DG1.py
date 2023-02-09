from fem_nets.vandermonde import compute_vandermonde_DG1
from fem_nets.utils import cell_centers, dof_coordinates, random_inside_points, quadrature_points

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
@pytest.mark.parametrize('get_points', (cell_centers, ))
def test_vm_2d(mesh, f, get_points):
    '''Use cell centers as points for comparison'''
    V = df.FunctionSpace(mesh, 'DG', 1)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    x_ = get_points(V)
    x = torch.tensor(x_).reshape((1, ) + x_.shape)
    vandermonde = compute_vandermonde_DG1(x, mesh)

    # Pushing this through linear layer with coefs behaves as eval of f at x
    lin = nn.Linear(V.dim(), 1, bias=False)
    lin.double()
    with torch.no_grad():
        lin.weight[0] = torch.tensor(coefs)
    
    mine = lin(vandermonde).squeeze(2)
    true = torch.tensor(np.array([fh(xi_) for xi_ in x_]))

    assert abs(torch.norm(true-mine, np.inf)) < 1E-12

    
@pytest.mark.parametrize('mesh', (df.UnitCubeMesh(3, 3, 3), ))
@pytest.mark.parametrize('f', (df.Constant(1), df.Expression('2*x[0]-3*x[1]+x[2]', degree=1), ))
@pytest.mark.parametrize('get_points', (cell_centers, dof_coordinates, quadrature_points))
def test_vm_3d(mesh, f, get_points):
    '''Use cell centers as points for comparison'''
    V = df.FunctionSpace(mesh, 'DG', 1)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    x_ = get_points(V)
    x = torch.tensor(x_).reshape((1, ) + x_.shape)
    vandermonde = compute_vandermonde_DG1(x, mesh)

    # Pushing this through linear layer with coefs behaves as eval of f at x
    lin = nn.Linear(V.dim(), 1, bias=False)
    lin.double()
    with torch.no_grad():
        lin.weight[0] = torch.tensor(coefs)
    
    mine = lin(vandermonde).squeeze(2)
    true = torch.tensor(np.array([fh(xi_) for xi_ in x_]))

    assert abs(torch.norm(true-mine, np.inf)) < 1E-12

    
@pytest.mark.parametrize('mesh', (df.UnitIntervalMesh(30), ))
@pytest.mark.parametrize('f', (df.Constant(1), df.Expression('2*x[0]-1', degree=1), ))
@pytest.mark.parametrize('get_points', (cell_centers, dof_coordinates))
def test_vm_1d(mesh, f, get_points):
    '''Use cell centers as points for comparison'''
    V = df.FunctionSpace(mesh, 'DG', 1)
    fh = df.interpolate(f, V)

    coefs = fh.vector().get_local()

    x_ = get_points(V)
    x = torch.tensor(x_).reshape((1, ) + x_.shape)
    vandermonde = compute_vandermonde_DG1(x, mesh)

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

    test_vm_1d(mesh=df.UnitIntervalMesh(30),
               f=df.Constant(2),
               get_points=cell_centers)

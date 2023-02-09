# NOTE: Point eval here only makes sense if the points are inside triangles
import dolfin as df
import numpy as np
import torch

from . vandermonde_CG1 import _compute_vandermonde_Lagrange1


def compute_vandermonde_DG1(x, mesh, tol_=1E-13):
    '''Vandermonde matrix has in colums values of basis functions at x'''
    return _compute_vandermonde_Lagrange1(x, mesh, continuity='L2', tol_=1E-13)

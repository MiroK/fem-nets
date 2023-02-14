# NOTE: Point eval here only makes sense if the points are inside triangles
import dolfin as df
import numpy as np
import torch

import fem_nets.vandermonde.vandermonde_CG2 as CG2


def compute_vandermonde_DG2(x, mesh, tol_=1E-13):
    '''Vandermonde matrix has in colums values of basis functions at x'''
    gdim = mesh.geometry().dim()
    return {1: CG2._compute_vandermonde_p2g1,
            2: CG2._compute_vandermonde_p2g2,
            3: CG2._compute_vandermonde_p2g3}[gdim](x, mesh, continuity='L2', tol_=1E-13)


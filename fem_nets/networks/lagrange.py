# Continuous Lagrange networks
from fem_nets.vandermonde import compute_vandermonde_CG1
import dolfin as df
import numpy as np
import torch
import torch.nn as nn


class Lagrange1NN(nn.Module):
    '''Neural net that is 'CG1' function space on mesh'''
    def __init__(self, mesh, invalidate_cache=True):
        assert mesh.geometry().dim() == mesh.topology().dim()
        super().__init__()
        # The weights are coefficients that combine the basis functions
        self.lin = nn.Linear(mesh.num_vertices(), 1, bias=False)

        self.mesh = mesh
        self.vandermonde = None
        self.invalidate_cache = invalidate_cache
        
    def forward(self, x):
        # Build up "Vandermonde", then each column is what
        if self.vandermonde is None:
            # torch.zeros(bsize, npts, ndofs, dtype=x.dtype)
            self.vandermonde = compute_vandermonde_CG1(x, self.mesh)
        out = self.lin(self.vandermonde).squeeze(2)
        # NOTE: Vandermonde is the thing that depends on 'x'
        self.invalidate_cache and setattr(self, 'vandermonde', None)

        return out

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            self.lin.weight[0] = torch.tensor(coefs)


class VectorLagrange1NN(nn.Module):
    '''Neural net that is 'CG1' function space on mesh'''
    def __init__(self, mesh, dim=None, invalidate_cache=True):
        assert mesh.geometry().dim() == mesh.topology().dim()
        super().__init__()

        if dim is None:
            dim = mesh.geometry().dim()
        # The weights are coefficients that combine the basis functions
        # NOTE: We have a list so that it's easier to write the update logic ...
        self.lins = [nn.Linear(mesh.num_vertices(), 1, bias=False)
                     for sub in range(dim)]
        # ... however, for proper working up to(), double(), etc methods
        # pytorch needs to have them also individualy as attributes
        for i, lin in enumerate(self.lins):
            setattr(self, f'lin_{i}', lin)

        self.mesh = mesh
        self.vandermonde = None
        self.invalidate_cache = invalidate_cache

        V = df.VectorFunctionSpace(mesh, 'CG', 1, dim)
        self.dofs_i = [V.sub(sub).dofmap().dofs() for sub in range(dim)]
        
    def forward(self, x):
        # Build up "Vandermonde", then each column is what
        if self.vandermonde is None:
            # torch.zeros(bsize, npts, ndofs, dtype=x.dtype)
            self.vandermonde = compute_vandermonde_CG1(x, self.mesh)
        outs = [lin(self.vandermonde).squeeze(2) for lin in self.lins]
        # NOTE: Vandermonde is the thing that depends on 'x'
        self.invalidate_cache and setattr(self, 'vandermonde', None)
            
        out = torch.stack(outs, axis=2)

        return out

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            for i, lin in enumerate(self.lins):
                lin.weight[0] = torch.tensor(coefs[self.dofs_i[i]])

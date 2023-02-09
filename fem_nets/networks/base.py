import torch
import torch.nn as nn
import dolfin as df


class ScalarNN(nn.Module):
    '''Neural net representation of a scalar valued FE function'''    
    def __init__(self, V, invalidate_cache=True):
        mesh = V.mesh()
        assert mesh.geometry().dim() == mesh.topology().dim()
        assert V.ufl_element().value_shape() == ()

        super().__init__()
        # The weights are coefficients that combine the basis functions
        self.lin = nn.Linear(V.dim(), 1, bias=False)

        self.mesh = mesh
        self.V = V
        self.vandermonde = None
        self.invalidate_cache = invalidate_cache
        
    def forward(self, x):
        # Build up "Vandermonde", then each column is what
        pdegree = self.V.ufl_element().degree()
        if self.vandermonde is None:
            # torch.zeros(bsize, npts, ndofs, dtype=x.dtype)
            self.vandermonde = self._compute_vandermonde(pdegree)(x, self.mesh)
        out = self.lin(self.vandermonde).squeeze(2)
        # NOTE: Vandermonde is the thing that depends on 'x'
        self.invalidate_cache and setattr(self, 'vandermonde', None)

        return out

    def set_from_coefficients(self, coefs):
        '''Set the degrees of freedom'''
        with torch.no_grad():
            self.lin.weight[0] = torch.tensor(coefs)


class VectorNN(nn.Module):
    '''Neural net representation of vector valued function'''
    def __init__(self, V, invalidate_cache=True):
        mesh = V.mesh()
        assert mesh.geometry().dim() == mesh.topology().dim()
        assert len(V.ufl_element().value_shape()) == 1
        
        super().__init__()
        # How many components
        dim, = V.ufl_element().value_shape()
        # The weights are coefficients that combine the basis functions
        # NOTE: We have a list so that it's easier to write the update logic ...
        ndofs_per_comp = V.dim()//dim
        self.lins = [nn.Linear(ndofs_per_comp, 1, bias=False)
                     for sub in range(dim)]
        # ... however, for proper working up to(), double(), etc methods
        # pytorch needs to have them also individualy as attributes
        for i, lin in enumerate(self.lins):
            setattr(self, f'lin_{i}', lin)

        self.mesh = mesh
        self.V = V
        self.vandermonde = None
        self.invalidate_cache = invalidate_cache

        self.dofs_i = [V.sub(sub).dofmap().dofs() for sub in range(dim)]
        
    def forward(self, x):
        # Build up "Vandermonde", then each column is what
        pdegree = self.V.ufl_element().degree()        
        if self.vandermonde is None:
            # torch.zeros(bsize, npts, ndofs, dtype=x.dtype)
            self.vandermonde = self._compute_vandermonde(pdegree)(x, self.mesh)
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

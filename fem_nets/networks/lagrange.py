# Continuous Lagrange networks
from fem_nets.vandermonde import compute_vandermonde_CG1, compute_vandermonde_CG2
from fem_nets.networks.base import ScalarNN , VectorNN


class LagrangeBase():
    def _compute_vandermonde(self, pdegree):
        return {1: compute_vandermonde_CG1,
                2: compute_vandermonde_CG2}[pdegree]

    def _is_compatible(self, V):
        return V.ufl_element().family() == 'Lagrange'

    
class LagrangeNN(ScalarNN, LagrangeBase):
    '''Neural net that is 'CG1' function space on mesh'''    
    pass


class VectorLagrangeNN(VectorNN, LagrangeBase):
    '''Neural net representation of a vector valued FE function'''    
    pass

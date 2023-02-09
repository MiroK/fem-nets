# Continuous Lagrange networks
from fem_nets.vandermonde.vandermonde_DG import compute_vandermonde_DG1, compute_vandermonde_DG0
from fem_nets.networks.base import ScalarNN , VectorNN


class DiscLagrangeBase():
    def _compute_vandermonde(self, pdegree):
        return {1: compute_vandermonde_DG1,
                0: compute_vandermonde_DG0}[pdegree]

    def _is_compatible(self, V):
        return V.ufl_element().family() == 'Discontinuous Lagrange'
    

class DiscLagrangeNN(ScalarNN, DiscLagrangeBase):
    '''Neural net that is 'DG1' function space on mesh'''    
    pass


class DiscVectorLagrangeNN(VectorNN, DiscLagrangeBase):
    '''Neural net representation of a vector valued FE function in DG1'''    
    pass

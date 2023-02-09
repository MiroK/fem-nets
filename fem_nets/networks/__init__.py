from . lagrange import LagrangeNN, VectorLagrangeNN
from . disc_lagrange import DiscLagrangeNN, DiscVectorLagrangeNN


def to_torch(V):
    '''Try to represent V as a neural network'''
    Velm = V.ufl_element()
    family = Velm.family()
    rank = len(Velm.value_shape())

    if family == 'Lagrange':
        return {0: LagrangeNN, 1: VectorLagrangeNN}[rank](V)

    if family == 'Discontinuous Lagrange':
        return {0: DiscLagrangeNN, 1: DiscVectorLagrangeNN}[rank](V)
    
    raise NotImplementedError

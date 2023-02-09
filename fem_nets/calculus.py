import torch
import torch.autograd as autograd


def grad(u, x, retain_graph=True, create_graph=True):
    '''Gradient wrt to x where x is [x0, x1, ...]'''
    assert len(x.shape) >= 2
    
    # Scalar case
    if len(u.shape) == 2:
        adj_inp = torch.ones(u.shape, dtype=u.dtype)
        r, *_ = autograd.grad(u, x, adj_inp, retain_graph=retain_graph, create_graph=create_graph)
        return r
    # Handle vector row-wise
    assert len(u.shape) == 3
    grads = [grad(u[..., i], x) for i in range(u.shape[-1])]
    return torch.stack(grads, 2)
 
 
def div(u, x, retain_graph=True, create_graph=True):
    '''Divergence'''
    assert len(x.shape) >= 2
    # This is a 1d case for scalar
    if len(x.shape) == 2:
        assert len(u.shape) == 2
        return grad(u, x, retain_graph=retain_graph, create_graph=create_graph)

    # Vector:
    # u_i/x_i
    if len(u.shape) == 3:
        div_u = torch.zeros(u.shape[:-1], dtype=u.dtype)
        for i in range(u.shape[-1]):
            div_u = div_u + grad(u[..., i], x, retain_graph=retain_graph, create_graph=create_graph)[..., i]
        return div_u

    # Matrix
    assert len(u.shape) == 4
    n, m = u.shape[2:]
    assert n == m
    # div(u_i[:])
    return torch.stack([div(u[..., i, :], x, retain_graph=retain_graph, create_graph=create_graph)
                        for i in range(n)], axis=2)
    

def trace(tensor):
    '''Trace (batched version)'''
    # There is torch.trace but it was throwing errors
    assert len(tensor.shape) == 4
    m, n = tensor.shape[-2:]
    tr = torch.zeros(tensor.shape[:2])
    for i in range(n):
        tr = tr + tensor[..., i, i]
    return tr


def transpose(tensor):
    '''Transpose (batched version)'''
    assert len(tensor.shape) == 4
    return torch.transpose(tensor, 3, 2)


def dot(tensorA, tensorB):
    '''u.v or A.v or v.A'''
    # u.v
    if len(tensorA.shape) == 3 and len(tensorB.shape) == 3:
        return (tensorA*tensorB).sum(axis=2)

    if len(tensorA.shape) == 4 and len(tensorB.shape) == 3:
        nrows, ncols = tensorA.shape[-2:]
        nvec, = tensorB.shape[-1:]
        assert ncols == nvec

        return (tensorA*(tensorB.unsqueeze(2))).sum(axis=3)

    if len(tensorA.shape) == 3 and len(tensorB.shape) == 4:
        nrows, ncols = tensorB.shape[-2:]
        nvec, = tensorA.shape[-1:]
        assert nrows == nvec

        return (tensorA.unsqueeze(3)*(tensorB)).sum(axis=2)
    
    raise NotImplementedError


def outer(u, v):
    '''Outer product of two vector -> is out_{ij} u_i v_j'''
    assert u.shape == v.shape
    assert len(u.shape) == 3
    _, nsamples, ncomps = u.shape
    assert ncomps > 1

    return u.unsqueeze(3)*v.unsqueeze(2)


def cross(u, v):
    '''2 and 3 dim cross product'''
    try:
        return torch.cross(u, v)
    except RuntimeError:
        return u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]


def diff(u, axis):
    '''np.diff(u, axis=axis, n=1) - like'''
    shape = u.shape
    assert 0 <= axis < len(shape)

    plus = tuple(slice(1, s) if i == axis else slice(s) for i, s in enumerate(shape))
    minus = tuple(slice(0, s-1) if i == axis else slice(s) for i, s in enumerate(shape))

    return u[plus] - u[minus]

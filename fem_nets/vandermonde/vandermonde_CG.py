import dolfin as df
import numpy as np
import torch


def compute_vandermonde_CG1(x, mesh, tol_=1E-13):
    '''Vandermonde matrix has in colums values of basis functions at x'''
    batchdim, npts, gdim = x.shape
    assert batchdim == 1
    
    dtype = x.dtype
    
    assert gdim == mesh.geometry().dim()
    assert mesh.topology().dim() == gdim

    V = df.FunctionSpace(mesh, 'CG', 1)
    dm = V.dofmap()

    ndofs = V.dim()
    # The idea for evaluation is similar to how fenics does interpolation.
    # However, we do not find first the cell colliding with points, as we 
    # evaluate batches. Instead for everycell we decide based on barycentric
    # coordinates where which elements in `x` are inside. Based on the cooerds
    # we also compute the basis function value
    out = torch.zeros(npts, dtype=dtype)
    # The things with CG space and inside check (within tol) is that some points
    # migth end up found in several cells resulting in duplicate contributions.
    # We try to make exach xi contribute only once below
    x_has_contributed = torch.zeros(npts, dtype=bool)    
    
    vandermonde = torch.zeros(batchdim, npts, ndofs, dtype=dtype)
    for cell in df.cells(mesh):
        cell_dofs = dm.cell_dofs(cell.index())
        
        *A, C = np.array(cell.get_vertex_coordinates()).reshape((-1, gdim))
    
        T = (np.row_stack(A)-C).T
        
        Tinv = torch.tensor(np.linalg.inv(T).T, dtype=dtype)
        b = torch.tensor(-(C.dot(Tinv)), dtype=dtype)
        y = torch.add(torch.matmul(x, Tinv), b)
        # Here y has coordinates y0, y1 of all the points with
        # respect to the axis defined by columns of T. We need 0 <= yi <= 1
        # and 0 <= sum_i y_i <= 1 for the points to be inside
        sum_y = torch.sum(y, axis=2)

        is_inside = torch.logical_and(-tol_ < sum_y, sum_y < 1 + tol_)
        for i in range(gdim):
            yi = y[..., i]
            is_inside = torch.logical_and(is_inside, torch.logical_and(-tol_ < yi, yi < 1 + tol_))
        is_inside = is_inside.reshape(-1)
        
        is_inside = torch.logical_and(is_inside, ~x_has_contributed)
        for (ldof, dof) in enumerate(cell_dofs):
            if ldof < gdim:
                val = y[..., ldof]
            else:
                val = 1 - sum_y
            out[is_inside] = val.reshape(-1)[is_inside]
            vandermonde[..., dof] += 1*out
            out *= 0
        x_has_contributed[is_inside] = True
    return vandermonde

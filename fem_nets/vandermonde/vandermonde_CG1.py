import dolfin as df
import numpy as np
import torch


def compute_vandermonde_CG1(x, mesh, tol_=1E-13):

    batchdim, npts, gdim = x.shape
    assert gdim == 2
    assert batchdim == 1
    
    dtype = x.dtype
    
    assert gdim == mesh.geometry().dim()
    assert mesh.topology().dim() == gdim

    V = df.FunctionSpace(mesh, 'CG', 1)
    dm = V.dofmap()

    ndofs = V.dim()
    
    out = torch.zeros(npts, dtype=dtype)

    x_has_contributed = torch.zeros(npts, dtype=bool)    
    
    vandermonde = torch.zeros(batchdim, npts, ndofs, dtype=dtype)

    for cell in df.cells(mesh):
        cell_dofs = dm.cell_dofs(cell.index())
        
        A, B, C = np.array(cell.get_vertex_coordinates()).reshape((-1, gdim))
    
        T = np.column_stack([A-C, B-C])

        Tinv = torch.tensor(np.linalg.inv(T).T, dtype=dtype)
        b = torch.tensor(-(C.dot(Tinv)), dtype=dtype)
        y = torch.add(torch.matmul(x, Tinv), b)
        # Here y has coordinates y0, y1 of all the points with
        # respect to the axis defined by columns of T. We need 0 <= yi <= 1
        # and 0 <= sum_i y_i <= 1 for the points to be inside
        y0, y1 = y[..., 0], y[..., 1]
        # Flatten for setting
        y0, y1 = y0.reshape(-1), y1.reshape(-1)
        sum_y = y0 + y1
        
        is_inside = torch.logical_and(-tol_ < y0, y0 < 1 + tol_)
        is_inside = torch.logical_and(is_inside, torch.logical_and(-tol_ < y1, y1 < 1 + tol_))
        is_inside = torch.logical_and(is_inside, torch.logical_and(-tol_ < sum_y, sum_y < 1 + tol_))

        is_inside = torch.logical_and(is_inside, ~x_has_contributed)
        for (ldof, dof) in enumerate(cell_dofs):
            if ldof == 0:
                val = y0
            elif ldof == 1:
                val = y1
            else:
                val = 1 - sum_y
            out[is_inside] = val[is_inside]
            vandermonde[..., dof] += 1*out
            out *= 0
        x_has_contributed[is_inside] = True
    return vandermonde
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from utils import cell_centers
    
    mesh = df.UnitSquareMesh(1, 1)

    x_ = cell_centers(mesh)
    x = torch.tensor(x_, dtype=torch.float64).reshape((1, ) + x_.shape)

    V = compute_vandermonde(x, mesh)
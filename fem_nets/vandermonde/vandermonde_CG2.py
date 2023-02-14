import dolfin as df
import numpy as np
import torch


def compute_vandermonde_CG2(x, mesh, tol_=1E-13):
    '''Vandermonde matrix has in colums values of basis functions at x'''
    gdim = mesh.geometry().dim()
    return {1: _compute_vandermonde_p2g1,
            2: _compute_vandermonde_p2g2,
            3: _compute_vandermonde_p2g3}[gdim](x, mesh, continuity='H1', tol_=1E-13)


def _compute_vandermonde_p2g1(x, mesh, continuity, tol_=1E-13):    
    '''Vandermonde matrix has in colums values of basis functions at x'''
    assert continuity in ('H1', 'L2')
    batchdim, npts, gdim = x.shape
    assert batchdim == 1
    
    dtype = x.dtype
    
    assert gdim == 1
    assert mesh.topology().dim() == gdim

    if continuity == 'H1':
        V = df.FunctionSpace(mesh, 'CG', 2)
    else:
        V = df.FunctionSpace(mesh, 'DG', 2)        
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

    cell_vertices = mesh.coordinates()
    
    vandermonde = torch.zeros(batchdim, npts, ndofs, dtype=dtype)
    for cell in df.cells(mesh):
        cell_dofs = dm.cell_dofs(cell.index())

        C, A = cell_vertices[cell.entities(0)]
    
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

        y0 = y[..., 0]
        # We know how fenics orders the basis so ...
        for (ldof, dof) in enumerate(cell_dofs):
            if ldof == 0:
                val = 2*y0**2 - 3*y0 + 1
            elif ldof == 1:
                val = y0*(2*y0 - 1) 
            elif ldof == 2:
                val = 4*y0*(1 - y0)
            else:
                assert ValueError
            out[is_inside] = val.reshape(-1)[is_inside]
            vandermonde[..., dof] += 1*out
            out *= 0
        x_has_contributed[is_inside] = True
    return vandermonde


def _compute_vandermonde_p2g2(x, mesh, continuity, tol_=1E-13):    
    '''Vandermonde matrix has in colums values of basis functions at x'''
    assert continuity in ('H1', 'L2')
    batchdim, npts, gdim = x.shape
    assert batchdim == 1
    
    dtype = x.dtype
    
    assert gdim == mesh.geometry().dim()
    assert mesh.topology().dim() == gdim

    if continuity == 'H1':
        V = df.FunctionSpace(mesh, 'CG', 2)
    else:
        V = df.FunctionSpace(mesh, 'DG', 2)        
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
        
        A, B, C = np.array(cell.get_vertex_coordinates()).reshape((-1, gdim))
    
        T = (np.row_stack([B, C])-A).T
        
        Tinv = torch.tensor(np.linalg.inv(T).T, dtype=dtype)
        b = torch.tensor(-(A.dot(Tinv)), dtype=dtype)
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

        y0, y1 = y[..., 0], y[..., 1]
        
        is_inside = torch.logical_and(is_inside, ~x_has_contributed)
        for (ldof, dof) in enumerate(cell_dofs):
            if ldof == 0:
                val = 2*y0**2 + 4*y0*y1 - 3*y0 + 2*y1**2 - 3*y1 + 1
            elif ldof == 1:
                val = y0*(2*y0-1)
            elif ldof == 2:
                val = y1*(2*y1-1)
            elif ldof == 3:
                val = 4*y0*y1
            elif ldof == 4:
                val = 4*y1*(-y0-y1+1)
            elif ldof == 5:
                val = 4*y0*(-y0-y1+1)
            else:
                assert ValueError
            out[is_inside] = val.reshape(-1)[is_inside]
            vandermonde[..., dof] += 1*out
            out *= 0
        x_has_contributed[is_inside] = True
    return vandermonde


def _compute_vandermonde_p2g3(x, mesh, continuity, tol_=1E-13):    
    '''Vandermonde matrix has in colums values of basis functions at x'''
    assert continuity in ('H1', 'L2')
    batchdim, npts, gdim = x.shape
    assert batchdim == 1
    
    dtype = x.dtype
    
    assert gdim == mesh.geometry().dim()
    assert mesh.topology().dim() == gdim

    if continuity == 'H1':
        V = df.FunctionSpace(mesh, 'CG', 2)
    else:
        V = df.FunctionSpace(mesh, 'DG', 2)        
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
        
        A, B, C, D = np.array(cell.get_vertex_coordinates()).reshape((-1, gdim))
    
        T = (np.row_stack([B, C, D])-A).T
        
        Tinv = torch.tensor(np.linalg.inv(T).T, dtype=dtype)
        b = torch.tensor(-(A.dot(Tinv)), dtype=dtype)
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

        y0, y1, y2 = y[..., 0], y[..., 1], y[..., 2]
        
        is_inside = torch.logical_and(is_inside, ~x_has_contributed)
        for (ldof, dof) in enumerate(cell_dofs):
            if ldof == 0:
                val = 2*y0**2 + 4*y0*y1 + 4*y0*y2 - 3*y0 + 2*y1**2 + 4*y1*y2 - 3*y1 + 2*y2**2 -3*y2 + 1
            elif ldof == 1:
                val = y0*(2*y0-1)
            elif ldof == 2:
                val = y1*(2*y1-1)
            elif ldof == 3:
                val = y2*(2*y2-1)
            elif ldof == 4:
                val = 4*y1*y2
            elif ldof == 5:
                val = 4*y0*y2
            elif ldof == 6:
                val = 4*y0*y1
            elif ldof == 7:
                val = 4*y2*(-y0-y1-y2+1)
            elif ldof == 8:
                val = 4*y1*(-y0-y1-y2+1)
            elif ldof == 9:
                val = 4*y0*(-y0-y1-y2+1)
            else:
                assert ValueError
            out[is_inside] = val.reshape(-1)[is_inside]
            vandermonde[..., dof] += 1*out
            out *= 0
        x_has_contributed[is_inside] = True
    return vandermonde

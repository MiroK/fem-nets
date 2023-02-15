import dolfin as df
import numpy as np
import FIAT


def dof_coordinates(V):
    '''Spatial points associated with dofs'''
    gdim = V.mesh().geometry().dim()
    return V.tabulate_dof_coordinates()


def cell_centers(V):
    '''Arrays of cell center coordinates'''
    mesh = V.mesh()
    x = mesh.coordinates()
    c = np.mean(x[mesh.cells()], axis=1)

    return c


def mask_inside_points(mesh, points):
    '''True for points that are inside'''
    tree = mesh.bounding_box_tree()
    cs = np.fromiter((tree.compute_first_entity_collision(df.Point(point)) for point in points),
                     dtype=int)

    max_cell = mesh.num_entities_global(mesh.topology().dim())
    return cs < max_cell


def random_inside_points(V, npts=100):
    '''Random and mask'''
    mesh = V.mesh()

    X = mesh.coordinates()
    x0 = np.min(X, axis=0)
    x1 = np.max(X, axis=0)
    
    pts = np.random.rand(npts, 1)
    
    pts = x0*pts + (x1-x0)*pts

    mask = mask_inside_points(mesh, pts)
    return pts[mask]


def quadrature_points(V):
    '''Quadrature points to ingerate functions in V exactly'''
    deg = V.ufl_element().degree()
    mesh = V.mesh()
    
    Velm = df.FiniteElement('Quadrature', mesh.ufl_cell(), deg, quad_scheme='default')
    V = df.FunctionSpace(mesh, Velm)
    return V.tabulate_dof_coordinates()


def quad_points(V):
    '''Quadrature points'''
    mesh = V.mesh()
    gdim = mesh.geometry().dim()
    cell = FIAT.reference_element.ufc_simplex(gdim)

    deg = V.ufl_element().degree()
    quadrature = FIAT.quadrature_schemes.make_quadrature(cell, deg)

    ref_points = quadrature.get_points()
    
    x = mesh.coordinates()
    cells = mesh.cells()

    points = []
    for cell in x[cells]:
        *A, B = cell
        # FIXME: vectorize this
        V = (np.row_stack(A)-B).T
        for p in ref_points:
            q = V@p + B
            points.append(q)
    return np.array(points)


def quad_weights(V):
    ''''Quad weights for volume integration'''
    mesh = V.mesh()
    gdim = mesh.geometry().dim()
    cell = FIAT.reference_element.ufc_simplex(gdim)

    deg = V.ufl_element().degree()
    quadrature = FIAT.quadrature_schemes.make_quadrature(cell, deg)

    ref_weights = quadrature.get_weights()
    
    x = mesh.coordinates()
    cells = mesh.cells()

    weights = []
    for cell in df.cells(mesh):
        weights.extend(ref_weights*cell.volume())
    return 2*np.array(weights)

#  -------------------------------------------------------------------

if __name__ == '__main__':
    mesh = df.UnitSquareMesh(4, 4)
    V = df.FunctionSpace(mesh, 'CG', 2)

    Xq = quadrature_points(V)
    Yq = quad_points(V)
    
    Wq = quad_weights(V)

    f = df.Expression('3*x[0]*x[1]-2*x[1]', degree=V.ufl_element().degree())
    true = df.assemble(f*df.dx(domain=mesh))

    mine = np.inner(Wq, np.fromiter(map(f, Yq), dtype=float))
    print(abs(true-mine))

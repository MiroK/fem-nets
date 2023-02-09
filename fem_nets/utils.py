import dolfin as df
import numpy as np


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

#  -------------------------------------------------------------------

if __name__ == '__main__':
    mesh = df.UnitSquareMesh(3, 3)
    V = df.FunctionSpace(mesh, 'CG', 1)

    X = quadrature_points(V)

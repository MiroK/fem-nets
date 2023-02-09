import dolfin as df
import numpy as np


def dof_coordinates(V):
    '''Spatial points associated with dofs'''
    gdim = V.mesh().geometry().dim()
    return np.array(V.tabulate_dof_coordinates()).reshape(-1, gdim)


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

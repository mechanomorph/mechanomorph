"""Functions for resampling meshes."""

import numpy as np
import pyacvd
import pyvista as pv


def resample_mesh_voronoi(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_area: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a meshed surface using voronoi clustering.

    Parameters
    ----------
    vertices : np.ndarray
        (n x 3) array containing the coordinates of the vertices of the mesh
    faces : np.ndarray
        (n x 3) array containing the indices of the vertex for each face.
    target_area : float, optional
        The target area of each vertex in the mesh,
        by default 1.0

    Returns
    -------
    vertices : np.ndarray
        (n x 3) array containing the coordinates of the vertices of the mesh
    faces : np.ndarray
        (n x 3) array containing the indices of the vertex for each face.
    """
    # Prepend 3 for pyvista format
    faces = np.concatenate((np.ones((faces.shape[0], 1), dtype=int) * 3, faces), axis=1)

    # Create a mesh
    surf = pv.PolyData(vertices, faces)
    # surf = surf.smooth(n_iter=smoothing)

    # remesh to desired point size
    cluster_points = int(surf.area / target_area)
    # cluster_points = n_faces
    clus = pyacvd.Clustering(surf)
    # clus.subdivide(3)
    clus.cluster(cluster_points)
    remeshed = clus.create_mesh()

    verts = remeshed.points
    faces = remeshed.faces.reshape(-1, 4)[:, 1:]

    # switch face order to have outward normals
    faces = faces[:, [0, 2, 1]]

    return verts, faces

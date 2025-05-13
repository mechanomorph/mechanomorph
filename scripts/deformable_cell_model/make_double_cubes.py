"""Make a mesh of cube doublets contacting on a single face."""

import h5py
import napari
import numpy as np
import pyacvd
import pyvista as pv
from skimage.measure import marching_cubes


def resample_mesh(
    vertices,
    faces,
    target_area: float = 1.0,
):
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


def make_cube_meshes(
    edge_width: float = 30,
    target_area: float = 1.0,
):
    """Create a cube mesh with given edge width."""
    cube_mask = np.zeros((edge_width + 2, edge_width + 2, edge_width + 2), dtype=bool)
    cube_mask[1:-1, 1:-1, 1:-1] = True
    cube_vertices_0, cube_faces_0, _, _ = marching_cubes(cube_mask, 0)

    cube_vertices_0, cube_faces_0 = resample_mesh(
        cube_vertices_0, cube_faces_0, target_area=target_area
    )

    n_faces = cube_faces_0.shape[0]
    cell_indices_0 = np.arange(n_faces, dtype=int)

    # remove the offset added
    # so the lower corner is at (0, 0, 0)
    # cube_vertices_0 = cube_vertices_0 - 1

    # make the second cube by reflecting about the 1-2 plane
    cube_vertices_1 = cube_vertices_0 * np.array([-1, 1, 1])
    face_offset = cube_vertices_0.shape[0]
    cube_faces_1 = cube_faces_0 + face_offset
    cube_faces_1 = cube_faces_1[:, [0, 2, 1]]
    cell_indices_1 = np.arange(n_faces, dtype=int) + n_faces

    # make second cube
    # is shifted along the 0 axis
    # cube_offset = edge_width + cube_gap + 1
    # cube_vertices_1 = cube_vertices_0 + np.array([cube_offset, 0, 0])
    # face_offset = cube_vertices_0.shape[0]
    # cube_faces_1 = cube_faces_0 + face_offset
    # cell_indices_1 = np.arange(n_faces, dtype=int) + n_faces

    # combine the two cubes
    vertices = np.concatenate((cube_vertices_0, cube_vertices_1), axis=0)
    faces = np.concatenate((cube_faces_0, cube_faces_1), axis=0)
    face_cell_index = [cell_indices_0, cell_indices_1]

    print(len(vertices))

    # compute the bounding boxes
    # add a 1 unit margin to ensure cubes overlap
    bounding_box_0 = np.concat(
        (np.min(cube_vertices_0, axis=0) - 1, np.max(cube_vertices_0, axis=0) + 1),
        axis=0,
    )
    bounding_box_1 = np.concat(
        (np.min(cube_vertices_1, axis=0) - 1, np.max(cube_vertices_1, axis=0) + 1),
        axis=0,
    )
    bounding_boxes = np.stack((bounding_box_0, bounding_box_1), axis=0)

    return vertices, faces, face_cell_index, bounding_boxes


if __name__ == "__main__":
    edge_width = 30
    gap = 0.2
    contact_distance_threshold = 0.3

    vertices, faces, face_cell_index, bounding_boxes = make_cube_meshes(
        edge_width=edge_width,
        target_area=7,
    )
    # vertices.requires_grad = True

    # dummy_constant = torch.tensor(3.0, requires_grad=True)
    # vertices = vertices + dummy_constant

    with h5py.File("double_cube_mesh.h5", "w") as f:
        f.create_dataset("vertices", data=vertices)
        f.create_dataset("faces", data=faces)
        f.create_dataset("face_cell_index", data=face_cell_index)
        f.create_dataset("bounding_boxes", data=bounding_boxes)

    # make the clipping planes
    plane_parameters_neg = {
        "position": (-edge_width / 2, edge_width / 2, edge_width / 2),
        "normal": (1, 0, 0),
        "enabled": True,
    }

    plane_parameters_pos = {
        "position": (edge_width / 2, edge_width / 2, edge_width / 2),
        "normal": (-1, 0, 0),
        "enabled": True,
    }

    viewer = napari.Viewer()
    mesh_layer = viewer.add_surface(
        (vertices, faces),
        experimental_clipping_planes=[plane_parameters_neg, plane_parameters_pos],
    )
    mesh_layer.wireframe.visible = True
    mesh_layer.normals.face.visible = True

    viewer.dims.ndisplay = 3

    napari.run()

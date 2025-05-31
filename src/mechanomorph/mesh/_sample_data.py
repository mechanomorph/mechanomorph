"""Functions to generate example meshes."""

import numpy as np
from skimage.measure import marching_cubes

from mechanomorph.jax.dcm.utils import compute_face_normal_centroid_dot_product
from mechanomorph.mesh import resample_mesh_voronoi


def make_cube_doublet(
    edge_width: int = 30,
    target_area: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a double cube mesh with given edge width.

    The cubes symmetric about the plane defined by the normal (1, 0, 0) and
    the point (0, 0, 0). They are toughing and their vertices are aligned on
    that plane.

    Cube 0 goes from: (0, 0, 0) to (edge_width, edge_width, edge_width)
    Cube 1 goes from: (-edge_width, -edge_width, -edge_width) to (0, 0, 0)

    Both cubes are watertight objects and do not share any vertices or faces.
    Their normals point outward.

    Parameters
    ----------
    edge_width : int
        The width of the edge of the cubes. This must be an integer.
    target_area : float
        The target area of the triangles of the resulting mesh.

    Returns
    -------
    vertices : np.ndarray
        (n_vertices, 3) array containing the coordinates of each vertex.
    faces : np.ndarray
        (n_faces, 3) array containing the index of each vertex in each face.
    vertex_cell_mapping: np.ndarray
        (n_vertices,) array containing the index of the cell each vertex
        belongs to.
    face_cell_mapping : np.ndarray
        (n_faces,) array containing the index of the cell each face belongs to.
    bounding_boxes : np.ndarray
        (2, 6) array containing the bounding box for each cube. The bounding box
        has been extended by 1 unit in each direction to ensure they are overlapping.
        Each bounding box is: (min_0, min_1, min_2, max_0, max_1, max_2).
    """
    cube_mask = np.zeros((edge_width + 2, edge_width + 2, edge_width + 2), dtype=bool)
    cube_mask[1:-2, 1:-2, 1:-2] = True
    cube_vertices_0, cube_faces_0, _, _ = marching_cubes(cube_mask, 0)

    cube_vertices_0, cube_faces_0 = resample_mesh_voronoi(
        cube_vertices_0, cube_faces_0, target_area=target_area
    )

    n_vertices = cube_vertices_0.shape[0]
    n_faces = cube_faces_0.shape[0]
    vertex_cell_mapping_0 = np.zeros((n_vertices,), dtype=int)
    face_cell_mapping_0 = np.zeros((n_faces,), dtype=int)

    # make the second cube by reflecting about the 1-2 plane
    cube_vertices_1 = cube_vertices_0 * np.array([-1, 1, 1])
    face_offset = cube_vertices_0.shape[0]

    # make the second cube's faces
    # flip the faces that got flipped by the reflection
    cube_faces_1 = cube_faces_0.copy()
    normals_dot = compute_face_normal_centroid_dot_product(
        vertices=cube_vertices_1,
        faces=cube_faces_1,
    )
    flipped_faces_mask = normals_dot < 0
    cube_faces_1[flipped_faces_mask, :] = cube_faces_1[flipped_faces_mask][:, [2, 1, 0]]
    cube_faces_1 = cube_faces_1 + face_offset

    # make the vertex and face cell mappings
    vertex_cell_mapping_1 = np.ones((n_vertices,), dtype=int)
    face_cell_mapping_1 = np.ones((n_faces,), dtype=int)

    # combine the two cubes
    vertices = np.concatenate((cube_vertices_0, cube_vertices_1), axis=0)
    faces = np.concatenate((cube_faces_0, cube_faces_1), axis=0)
    vertex_cell_mapping = np.concatenate([vertex_cell_mapping_0, vertex_cell_mapping_1])
    face_cell_mapping = np.concatenate([face_cell_mapping_0, face_cell_mapping_1])

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

    return vertices, faces, vertex_cell_mapping, face_cell_mapping, bounding_boxes

"""Utilities for testing JAX-based mechanomorph models."""

import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray
from jax.experimental import sparse


def generate_two_cubes() -> (
    tuple[
        JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray
    ]
):
    """Make a mesh with two adjacent cubes sharing a face.

    Each cube has 8 vertices and 12 triangular faces.
    The edge length is 1.

    Returns
    -------
    all_vertices : JaxArray
        (n_vertices, 3) array of vertex coordinates
    all_faces : JaxArray
        (n_faces, 3) array of face indices
    vertex_cell_mapping : JaxArray
        (n_vertices,) array mapping vertices to cells
    face_cell_mapping : JaxArray
        (n_faces,) array mapping faces to cells
    vertices_cube1 : JaxArray
        (8, 3) array of cube 1 vertices
    vertices_cube2 : JaxArray
        (8, 3) array of cube 2 vertices
    faces_cube1 : JaxArray
        (12, 3) array of cube 1 faces
    faces_cube2 : JaxArray
        (12, 3) array of cube 2 faces
    """
    # Define 2 adjacent cubes sharing a face
    vertices_cube1 = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    vertices_cube2 = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    all_vertices = jnp.concatenate([vertices_cube1, vertices_cube2], axis=0)
    n_cube_vertices = vertices_cube1.shape[0]

    # Faces for each cube: define triangles for each face of a cube
    faces_cube1 = jnp.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 5, 6],
            [4, 6, 7],  # top
            [0, 1, 5],
            [0, 5, 4],  # front
            [2, 3, 7],
            [2, 7, 6],  # back
            [1, 2, 6],
            [1, 6, 5],  # right
            [3, 0, 4],
            [3, 4, 7],  # left
        ]
    )
    faces_cube2 = faces_cube1 + n_cube_vertices
    all_faces = jnp.concatenate([faces_cube1, faces_cube2], axis=0)
    n_cube_triangles = faces_cube1.shape[0]

    # Vertex and face cell mappings
    vertex_cell_mapping = jnp.concatenate(
        [
            jnp.zeros(n_cube_vertices, dtype=int),  # cell 0
            jnp.ones(n_cube_vertices, dtype=int),  # cell 1
        ]
    )
    face_cell_mapping = jnp.concatenate(
        [
            jnp.zeros(n_cube_triangles, dtype=int),
            jnp.ones(n_cube_triangles, dtype=int),
        ]
    )

    return (
        all_vertices,
        all_faces,
        vertex_cell_mapping,
        face_cell_mapping,
        vertices_cube1,
        vertices_cube2,
        faces_cube1,
        faces_cube2,
    )


def _get_cell_face_adjacency_matrix(nodes_ar, cell_faces_ar):
    """

    Compute the adjacency matrix of the faces of a given cell.

    Parameters
    ----------
    nodes_ar: np.ndarray (N_node_mesh, 3)
        Array of vertex positions.
    cell_faces_ar: np.ndarray (N_face_cell, 3)
        Contains the indices of the nodes of each face of the cell.

    Returns
    -------
    3x adjacency_matrix : jax.sparse.BCOO (N_face_cell, N_node)
        Returns 3 matrices. The first matrix contains the information
        of the first node of each face, the second matrix
        contains the information of the second node of each face, etc...

        These sparse matrices are organized as follows:

        adjacency_matrix_n1 =

        |    | n1 | n2 | n3 | ...
        |--- |----|----|----|----
        |c_f1|  1 |  0 |  0 | ...
        |c_f2|  1 |  0 |  0 | ...
        |c_f3|  0 |  0 |  1 | ...

        In this example, the first node of f1 is n1,
        the first node of f2 is n1 and the first node of f3 is n3.
    """
    N_node_mesh = nodes_ar.shape[0]
    N_face_cell = cell_faces_ar.shape[0]

    face_indices = np.arange(N_face_cell)
    data_n1 = np.ones((N_face_cell,), dtype=np.uint8)
    data_n2 = np.ones((N_face_cell,), dtype=np.uint8)
    data_n3 = np.ones((N_face_cell,), dtype=np.uint8)

    col_n1 = cell_faces_ar[:, 0]
    col_n2 = cell_faces_ar[:, 1]
    col_n3 = cell_faces_ar[:, 2]

    n1_indices = np.column_stack([face_indices, col_n1])
    n2_indices = np.column_stack([face_indices, col_n2])
    n3_indices = np.column_stack([face_indices, col_n3])

    sparse_cell_adjacency_matrix_n1 = sparse.BCOO(
        (data_n1, n1_indices), shape=(N_face_cell, N_node_mesh)
    )
    sparse_cell_adjacency_matrix_n2 = sparse.BCOO(
        (data_n2, n2_indices), shape=(N_face_cell, N_node_mesh)
    )
    sparse_cell_adjacency_matrix_n3 = sparse.BCOO(
        (data_n3, n3_indices), shape=(N_face_cell, N_node_mesh)
    )

    return (
        sparse_cell_adjacency_matrix_n1,
        sparse_cell_adjacency_matrix_n2,
        sparse_cell_adjacency_matrix_n3,
    )

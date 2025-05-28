"""Utilities for testing JAX-based mechanomorph models."""

import jax.numpy as jnp
from jax import Array as JaxArray


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
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    vertices_cube2 = jnp.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [1, 1, 1],
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

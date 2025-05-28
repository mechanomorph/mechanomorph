import jax.numpy as jnp
from jax import Array as JaxArray
from jax.experimental import sparse


def compute_cell_volume_packed(
    vertex_positions: JaxArray,
    faces: JaxArray,
    face_mask: JaxArray,
) -> JaxArray:
    """
    Computes the signed volume of a closed triangular mesh (cell).

    Parameters
    ----------
    vertex_positions : (max_vertices_per_cell, 3)
        Padded array of vertex coordinates for a single cell.
    faces : (max_faces_per_cell, 3)
        Indices into `vertex_positions`, defining each triangular face.
    face_mask : (max_faces_per_cell,)
        Boolean mask indicating valid faces.

    Returns
    -------
    volume : float
        Signed volume of the cell.
    """
    v1 = vertex_positions[faces[:, 0]]
    v2 = vertex_positions[faces[:, 1]]
    v3 = vertex_positions[faces[:, 2]]
    volume_contrib = jnp.einsum("ij,ij->i", v1, jnp.cross(v2, v3))
    volume_contrib = jnp.where(face_mask, volume_contrib, 0.0)
    return jnp.sum(volume_contrib) / 6.0


def compute_cell_volume(
    n1_coordinates: JaxArray, n2_coordinates: JaxArray, n3_coordinates: JaxArray
):
    """Compute the volume of a cell.

    Parameters
    ----------
    n1_coordinates : JaxArray
        (n_face_cell, 3) array containing the positions of
        the first vertex of each face of the cell.
    n2_coordinates : JaxArray
        (n_face_cell, 3)array containing the positions of
        the second vertex of each face of the cell.
    n3_coordinates : JaxArray
        (n_face_cell, 3) array containing the positions of
        the third vertex of each face of the cell.

    Returns
    -------
    cell_volume : float
        The volume of the cell
    """
    return (
        jnp.abs(jnp.vdot(n1_coordinates, jnp.cross(n2_coordinates, n3_coordinates)))
        / 6.0
    )


def gradient_cell_volume_wrt_node_positions(
    n1_coordinates: JaxArray,
    n2_coordinates: JaxArray,
    n3_coordinates: JaxArray,
    cell_adjacency_matrix_n1: sparse.BCOO,
    cell_adjacency_matrix_n2: sparse.BCOO,
    cell_adjacency_matrix_n3: sparse.BCOO,
):
    """
    Take the gradient of the cell volume with respect to the vertex positions.

    Parameters
    ----------
    n1_coordinates : JaxArray
        (n_face_cell, 3) array containing the positions of
        the first vertex of each face of the cell the gradient is being calculated for.
    n2_coordinates : JaxArray
        (n_face_cell, 3)array containing the positions of
        the second vertex of each face of the cell the gradient is being calculated for.
    n3_coordinates : JaxArray
        (n_face_cell, 3) array containing the positions of
        the third vertex of each face of the cell the gradient is being calculated for.
    cell_adjacency_matrix_n1: jax.experimental.sparse.BCOO
        (n_face, n_vertices_mesh) sparse array mapping the face index to
        the index of the face's first vertex. This should only be for
        faces of the cell the gradient is being calculated for.
    cell_adjacency_matrix_n2: jax.experimental.sparse.BCOO
        (n_face, n_vertices_mesh) sparse array mapping the face index to
        the index of the face's second vertex. This should only be for
        faces of the cell the gradient is being calculated for.
    cell_adjacency_matrix_n3: jax.experimental.sparse.BCOO
        (n_face, n_vertices_mesh) sparse array mapping the face index to
        the index of the face's third vertex. This should only be for
        faces of the cell the gradient is being calculated for.

    Returns
    -------
    grad_cell_volume: JaxArray
        (N_node, 3) array containing the gradient of the cell volume
        with respect to the node positions.
    """
    # These matrices have shape (N_face, 3)
    face_grad_cell_volume_ar_n1 = jnp.cross(n2_coordinates, n3_coordinates)
    face_grad_cell_volume_ar_n2 = jnp.cross(n3_coordinates, n1_coordinates)
    face_grad_cell_volume_ar_n3 = jnp.cross(n1_coordinates, n2_coordinates)

    # These matrices have shape (N_node, 3)
    node_grad_cell_volume_n1 = cell_adjacency_matrix_n1.T @ face_grad_cell_volume_ar_n1
    node_grad_cell_volume_n2 = cell_adjacency_matrix_n2.T @ face_grad_cell_volume_ar_n2
    node_grad_cell_volume_n3 = cell_adjacency_matrix_n3.T @ face_grad_cell_volume_ar_n3

    node_grad_cell_volume = (
        node_grad_cell_volume_n1 + node_grad_cell_volume_n2 + node_grad_cell_volume_n3
    ) / 6.0

    return node_grad_cell_volume

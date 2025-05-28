import jax.numpy as jnp
from jax import Array as JaxArray


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


def compute_cell_pressure_forces(
    vertex_positions: JaxArray,
    vertex_mask: JaxArray,
    faces: JaxArray,
    face_mask: JaxArray,
    target_volume: float,
    bulk_modulus: float,
) -> JaxArray:
    """
    Computes pressure forces for a single cell.

    This is intended to be used with padded vertex and face arrays.
    Generally, this will be used with vmap to apply it to all cells.

    Parameters
    ----------
    vertex_positions : (max_vertices_per_cell, 3)
        Padded vertex coordinates for a single cell.
    vertex_mask : (max_vertices_per_cell,)
        Boolean mask for valid vertices.
    faces : (max_faces_per_cell, 3)
        Face indices into the local vertex array.
    face_mask : (max_faces_per_cell,)
        Mask indicating which faces are valid.
    target_volume : float
        Desired (target) volume for the cell.
    bulk_modulus : float
        Elasticity coefficient.

    Returns
    -------
    pressure_forces : (max_vertices_per_cell, 3)
        The force on each vertex (0 if padded).
    """
    v1 = vertex_positions[faces[:, 0]]
    v2 = vertex_positions[faces[:, 1]]
    v3 = vertex_positions[faces[:, 2]]

    face_grad_1 = jnp.cross(v2, v3)
    face_grad_2 = jnp.cross(v3, v1)
    face_grad_3 = jnp.cross(v1, v2)

    face_grad_1 = jnp.where(face_mask[:, None], face_grad_1, 0.0)
    face_grad_2 = jnp.where(face_mask[:, None], face_grad_2, 0.0)
    face_grad_3 = jnp.where(face_mask[:, None], face_grad_3, 0.0)

    num_vertices = vertex_positions.shape[0]
    pressure_force = jnp.zeros((num_vertices, 3))

    pressure_force = pressure_force.at[faces[:, 0]].add(face_grad_1)
    pressure_force = pressure_force.at[faces[:, 1]].add(face_grad_2)
    pressure_force = pressure_force.at[faces[:, 2]].add(face_grad_3)

    pressure_force = pressure_force / 6.0

    cell_volume = compute_cell_volume_packed(vertex_positions, faces, face_mask)
    pressure_scale = -bulk_modulus * jnp.log(cell_volume / target_volume)
    pressure_force = pressure_force * pressure_scale

    pressure_force = jnp.where(vertex_mask[:, None], pressure_force, 0.0)

    return pressure_force

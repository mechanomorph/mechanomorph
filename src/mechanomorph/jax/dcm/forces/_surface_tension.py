import jax.numpy as jnp
from jax import Array as JaxArray


def compute_cell_surface_tension_forces(
    vertices: JaxArray,
    faces: JaxArray,
    valid_vertices: JaxArray,
    valid_faces: JaxArray,
    face_surface_tension: JaxArray,
    min_allowed_norm: float = 1e-10,
) -> JaxArray:
    """Compute surface tension forces for a single cell.

    Parameters
    ----------
    vertices : JaxArray
        (max_vertices_per_cell, 3) vertex coordinates.
    faces : JaxArray
        (max_faces_per_cell, 3) vertex indices for each face.
    valid_vertices : JaxArray
        (max_vertices_per_cell,) boolean mask for valid vertices.
    valid_faces : JaxArray
        (max_faces_per_cell,) boolean mask for valid faces.
    face_surface_tension : JaxArray
        (max_faces_per_cell,) surface tension value for each face.
    min_allowed_norm : float
        Minimum allowed norm for numerical stability.

    Returns
    -------
    vertex_forces : JaxArray
        (max_vertices_per_cell, 3) surface tension forces on each vertex.
    """
    # Gather vertex positions for each face
    # Shape: (max_faces_per_cell, 3, 3) -> (face, vertex_in_face, xyz)
    face_vertices = jnp.take(vertices, faces, axis=0)

    # Extract individual vertices
    # Shape: (max_faces_per_cell, 3)
    v0 = face_vertices[:, 0, :]
    v1 = face_vertices[:, 1, :]
    v2 = face_vertices[:, 2, :]

    # Compute face normals
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = jnp.cross(edge1, edge2)

    # Compute unit normals with numerical stability
    face_norms = jnp.linalg.norm(face_normals, axis=1, keepdims=True)
    face_norms = jnp.maximum(face_norms, min_allowed_norm)
    unit_normals = face_normals / face_norms

    # Mask invalid faces
    unit_normals = jnp.where(valid_faces[:, None], unit_normals, 0.0)

    # Compute forces for each vertex of each face
    # These are the gradients of face area with respect to vertex positions
    force_v0 = jnp.cross(unit_normals, v2 - v1) * face_surface_tension[:, None]
    force_v1 = jnp.cross(unit_normals, v0 - v2) * face_surface_tension[:, None]
    force_v2 = jnp.cross(unit_normals, v1 - v0) * face_surface_tension[:, None]

    # Mask invalid face contributions
    force_v0 = jnp.where(valid_faces[:, None], force_v0, 0.0)
    force_v1 = jnp.where(valid_faces[:, None], force_v1, 0.0)
    force_v2 = jnp.where(valid_faces[:, None], force_v2, 0.0)

    # Initialize vertex forces
    max_vertices = vertices.shape[0]
    vertex_forces = jnp.zeros((max_vertices, 3))

    # Scatter forces from faces to vertices
    # We need to accumulate forces from all faces that share each vertex
    vertex_forces = vertex_forces.at[faces[:, 0]].add(force_v0)
    vertex_forces = vertex_forces.at[faces[:, 1]].add(force_v1)
    vertex_forces = vertex_forces.at[faces[:, 2]].add(force_v2)

    # Apply the -0.5 factor and mask invalid vertices
    vertex_forces = -0.5 * vertex_forces
    vertex_forces = jnp.where(valid_vertices[:, None], vertex_forces, 0.0)

    return vertex_forces

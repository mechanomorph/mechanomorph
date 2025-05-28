import jax
import jax.numpy as jnp
from jax import Array as JaxArray


def pack_mesh_to_cells(
    vertices: JaxArray,
    faces: JaxArray,
    vertex_cell_mapping: JaxArray,
    face_cell_mapping: JaxArray,
    max_vertices_per_cell: int,
    max_faces_per_cell: int,
    max_cells: int,
):
    """
    Packs an unstructured mesh into dense per-cell arrays with padding.

    Parameters
    ----------
    vertices : JaxArray
        (n_vertices, 3) array containing coordinates of all mesh vertices.
    faces : JaxArray
        (n_faces, 3) array of indices into `vertices`, defining each triangular face.
    vertex_cell_mapping : JaxArray
        (n_vertices,) array mapping each vertex to its cell index.
    face_cell_mapping : JaxArray
        (n_faces,) array mapping each face to its cell index.
    max_vertices_per_cell : int
        Maximum number of vertices allowed per cell (used for padding).
    max_faces_per_cell : int
        Maximum number of faces allowed per cell (used for padding).
    max_cells : int
        Maximum number of total cells in the mesh.

    Returns
    -------
    vertices_packed : JaxArray
        (max_cells, max_vertices_per_cell, 3) padded vertex coordinates per cell.
    faces_packed : JaxArray
        (max_cells, max_faces_per_cell, 3) indices of the vertex for each face.
        The vertex indices must match the local indices in `vertices_packed`.
    valid_vertices_mask : JaxArray
        (max_cells, max_vertices_per_cell) boolean array marking valid vertices.
    valid_faces_mask : JaxArray
        (max_cells, max_faces_per_cell) boolean array marking valid faces.
    valid_cells_mask : JaxArray
        (max_cells,) boolean array indicating which cells are valid.
    vertex_overflow : bool
        True if any cell exceeds max_vertices_per_cell.
    face_overflow : bool
        True if any cell exceeds max_faces_per_cell.
    cell_overflow : bool
        True if any cell index exceeds max_cells.
    """
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]

    # preallocate the packed vertex data
    vertices_packed = jnp.zeros((max_cells, max_vertices_per_cell, 3))
    valid_vertices_mask = jnp.zeros((max_cells, max_vertices_per_cell), dtype=bool)
    vertex_counts = jnp.zeros((max_cells,), dtype=int)
    vertex_ids = jnp.arange(n_vertices)
    vertex_overflow = False

    vertex_to_local_idx = -jnp.ones((max_cells, n_vertices), dtype=int)

    def insert_vertex(carry, idx):
        """Insert a vertex into its assigned cell if space is available."""
        verts_packed, verts_mask, vert_counts, vert_overflow, vert_to_local_idx = carry
        cell_id = vertex_cell_mapping[idx]
        count = vert_counts[cell_id]
        is_valid = count < max_vertices_per_cell
        new_vp = verts_packed.at[cell_id, count].set(
            jnp.where(is_valid, vertices[idx], 0.0)
        )
        new_vm = verts_mask.at[cell_id, count].set(is_valid)
        new_vc = vert_counts.at[cell_id].add(jnp.where(is_valid, 1, 0))
        new_vo = vert_overflow | (~is_valid)
        new_v2l = vert_to_local_idx.at[cell_id, idx].set(jnp.where(is_valid, count, -1))
        return (new_vp, new_vm, new_vc, new_vo, new_v2l), None

    # insert the vertices into the packed arrays
    (
        (
            vertices_packed,
            valid_vertices_mask,
            vertex_counts,
            vertex_overflow,
            vertex_to_local_idx,
        ),
        _,
    ) = jax.lax.scan(
        insert_vertex,
        (
            vertices_packed,
            valid_vertices_mask,
            vertex_counts,
            vertex_overflow,
            vertex_to_local_idx,
        ),
        vertex_ids,
    )

    # preallocate the packed face data
    faces_packed = jnp.zeros((max_cells, max_faces_per_cell, 3), dtype=int)
    valid_faces_mask = jnp.zeros((max_cells, max_faces_per_cell), dtype=bool)
    face_counts = jnp.zeros((max_cells,), dtype=int)
    face_ids = jnp.arange(n_faces)
    face_overflow = False

    def insert_face(carry, idx):
        """Insert a face into its assigned cell using local vertex indices."""
        faces_packed, faces_mask, faces_count, faces_overflow = carry
        cell_id = face_cell_mapping[idx]
        count = faces_count[cell_id]
        is_valid = count < max_faces_per_cell
        global_face = faces[idx]
        local_face = jnp.array(
            [
                vertex_to_local_idx[cell_id, global_face[0]],
                vertex_to_local_idx[cell_id, global_face[1]],
                vertex_to_local_idx[cell_id, global_face[2]],
            ]
        )
        new_faces_packed = faces_packed.at[cell_id, count].set(
            jnp.where(is_valid, local_face, 0)
        )
        new_faces_mask = faces_mask.at[cell_id, count].set(is_valid)
        new_faces_count = faces_count.at[cell_id].add(jnp.where(is_valid, 1, 0))
        new_faces_overflow = faces_overflow | (~is_valid)
        return (
            new_faces_packed,
            new_faces_mask,
            new_faces_count,
            new_faces_overflow,
        ), None

    # insert the faces into the packed arrays
    (faces_packed, valid_faces_mask, face_counts, face_overflow), _ = jax.lax.scan(
        insert_face,
        (faces_packed, valid_faces_mask, face_counts, face_overflow),
        face_ids,
    )

    # check the cells data
    valid_cells_mask = (vertex_counts > 0) | (face_counts > 0)
    cell_overflow = jnp.any(vertex_cell_mapping >= max_cells) | jnp.any(
        face_cell_mapping >= max_cells
    )

    return (
        vertices_packed,
        faces_packed,
        valid_vertices_mask,
        valid_faces_mask,
        valid_cells_mask,
        vertex_overflow,
        face_overflow,
        cell_overflow,
    )

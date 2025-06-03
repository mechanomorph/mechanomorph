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


def compute_face_normal_centroid_dot_product(
    vertices: JaxArray, faces: JaxArray, epsilon: float = 1e-12
) -> JaxArray:
    """Check the orientation of mesh faces relative to the mesh centroid.

    This computes the dot product of the face normal
    with the vector from the mesh centroid to face centroid
    for all faces.

    Positive values indicate outward-pointing normals
    and negative values indicate inward-pointing normals.

    Parameters
    ----------
    vertices : JaxArray
        (n_vertices, 3) array of vertex coordinates.
    faces : JaxArray
        (n_faces, 3) array of vertex indices for each face.
    epsilon : float, optional
        Small value to avoid division by zero when normalizing vectors.
        Default is 1e-12.

    Returns
    -------
    dot_products : JaxArray
        (n_faces,) array containing the dot product for each face.
        Positive values indicate outward-pointing normals, negative values
        indicate inward-pointing normals.
    """
    # Get vertex coordinates for each face
    face_vertices = vertices[faces]  # (n_faces, 3, 3)

    # Extract individual vertex coordinates
    v0 = face_vertices[:, 0, :]  # (n_faces, 3)
    v1 = face_vertices[:, 1, :]  # (n_faces, 3)
    v2 = face_vertices[:, 2, :]  # (n_faces, 3)

    # Compute face normals using cross product
    edge1 = v1 - v0  # (n_faces, 3)
    edge2 = v2 - v0  # (n_faces, 3)
    face_normals = jnp.cross(edge1, edge2)  # (n_faces, 3)

    # Normalize to unit vectors
    face_normals_norm = jnp.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / jnp.maximum(face_normals_norm, epsilon)

    # Compute face centroids
    face_centroids = (v0 + v1 + v2) / 3.0  # (n_faces, 3)

    # Compute mesh centroid
    mesh_centroid = jnp.mean(vertices, axis=0)  # (3,)

    # Compute vectors from mesh centroid to face centroids
    centroid_to_face_vectors = face_centroids - mesh_centroid[None, :]  # (n_faces, 3)

    # Compute dot products
    dot_products = jnp.sum(
        face_normals * centroid_to_face_vectors, axis=1
    )  # (n_faces,)

    return dot_products


def detect_aabb_intersections(
    vertices_packed: JaxArray,
    valid_vertices_mask: JaxArray,
    valid_cells_mask: JaxArray,
    expansion: float,
    max_cells: int,
    max_cell_pairs: int,
) -> tuple[JaxArray, JaxArray, int, JaxArray]:
    """Detect cells with intersecting axis-aligned bounding boxes.

    Parameters
    ----------
    vertices_packed : JaxArray
        (max_cells, max_vertices_per_cell, 3) padded vertex coordinates
    valid_vertices_mask : JaxArray
        (max_cells, max_vertices_per_cell) boolean validity mask
    valid_cells_mask : JaxArray
        (max_cells,) boolean mask for valid cells
    expansion : float
        Scalar amount to expand AABBs (typically threshold * safety_factor)
    max_cells : int
        The maximum number of cells in the mesh.
        This parameter is used to keep array sizes static.
    max_cell_pairs : int
        The maximum number of cell pairs allowed for padding.

    Returns
    -------
    intersecting_pairs : JaxArray
        (max_cell_pairs, 2) array of intersecting cell indices.
    valid_pairs_mask : JaxArray
        (max_cell_pairs,) boolean mask indicating valid pairs.
        This is to account for the padding.
    n_intersecting : int
        The number of intersecting pairs found.
    bounding_boxes : JaxArray
        (max_cell_pairs, 6) array with the computed
        axis-aligned bounding boxes for each cell.
        Bounding box format: (min_0, min_1, min_2, max_0, max_1, max_2)


    """
    # Compute AABBs for each cell
    # Mask invalid vertices with extreme values for min/max operations
    masked_for_min = jnp.where(
        valid_vertices_mask[:, :, None], vertices_packed, jnp.inf
    )
    masked_for_max = jnp.where(
        valid_vertices_mask[:, :, None], vertices_packed, -jnp.inf
    )

    # Compute bounds
    aabb_mins = jnp.min(masked_for_min, axis=1) - expansion
    aabb_maxs = jnp.max(masked_for_max, axis=1) + expansion

    # Handle invalid cells by setting their AABBs to non-intersecting values
    aabb_mins = jnp.where(valid_cells_mask[:, None], aabb_mins, jnp.inf)
    aabb_maxs = jnp.where(valid_cells_mask[:, None], aabb_maxs, -jnp.inf)

    # Combine into single AABB array
    bounding_boxes = jnp.concatenate([aabb_mins, aabb_maxs], axis=1)

    # Check all pairs for intersection
    mins = bounding_boxes[:, :3]  # (max_cells, 3)
    maxs = bounding_boxes[:, 3:]  # (max_cells, 3)

    # Broadcast for pairwise comparison
    mins1 = mins[:, None, :]  # (max_cells, 1, 3)
    maxs1 = maxs[:, None, :]  # (max_cells, 1, 3)
    mins2 = mins[None, :, :]  # (1, max_cells, 3)
    maxs2 = maxs[None, :, :]  # (1, max_cells, 3)

    # Two AABBs intersect if max1 >= min2 AND max2 >= min1 in all dimensions
    intersects = jnp.all(maxs1 >= mins2, axis=2) & jnp.all(maxs2 >= mins1, axis=2)

    # Only consider valid cells
    valid_pairs = valid_cells_mask[:, None] & valid_cells_mask[None, :]
    intersects = intersects & valid_pairs

    # Extract upper triangle (no self-intersections, avoid duplicates)
    i_indices, j_indices = jnp.triu_indices(max_cells, k=1)
    upper_intersects = intersects[i_indices, j_indices]

    # Build output arrays
    intersecting_pairs = jnp.full((max_cell_pairs, 2), -1, dtype=jnp.int32)
    pair_valid_mask = jnp.zeros(max_cell_pairs, dtype=bool)

    def add_pair(carry, idx):
        """Add one intersecting pair to output arrays.

        Parameters
        ----------
        carry : tuple
            (pairs, mask, out_idx) current state
        idx : int
            Index into upper triangle arrays

        Returns
        -------
        updated_carry : tuple
            Updated state
        _ : None
            Unused (required by scan)
        """
        pairs, mask, out_idx = carry

        is_intersecting = upper_intersects[idx]
        has_space = out_idx < max_cell_pairs
        should_add = is_intersecting & has_space

        pairs = pairs.at[out_idx].set(
            jnp.where(
                should_add, jnp.array([i_indices[idx], j_indices[idx]]), pairs[out_idx]
            )
        )
        mask = mask.at[out_idx].set(jnp.where(should_add, True, mask[out_idx]))

        out_idx = jnp.where(should_add, out_idx + 1, out_idx)

        return (pairs, mask, out_idx), None

    # Process all possible pairs
    (intersecting_pairs, valid_pairs_mask, n_intersecting), _ = jax.lax.scan(
        add_pair,
        (intersecting_pairs, pair_valid_mask, 0),
        jnp.arange(len(upper_intersects)),
    )

    return (intersecting_pairs, valid_pairs_mask, n_intersecting, bounding_boxes)

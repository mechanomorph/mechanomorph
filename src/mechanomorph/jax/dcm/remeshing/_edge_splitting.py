import jax
import jax.numpy as jnp
from jax import Array as JaxArray


def canonicalize_edge(v1: int, v2: int) -> tuple[int, int]:
    """
    Canonicalize an edge by ordering vertices (smaller index first).

    Parameters
    ----------
    v1 : int
        First vertex index.
    v2 : int
        Second vertex index.

    Returns
    -------
    edge : tuple[int, int]
        Canonicalized edge with smaller index first.
    """
    return jnp.minimum(v1, v2), jnp.maximum(v1, v2)


def extract_unique_edges_from_faces(
    faces: JaxArray, face_mask: JaxArray, max_unique_edges: int
) -> tuple[JaxArray, JaxArray, JaxArray, int]:
    """
    Extract unique edges from faces and build face-to-edge mapping.

    Parameters
    ----------
    faces : JaxArray
        (max_faces, 3) face connectivity array.
    face_mask : JaxArray
        (max_faces,) boolean mask for valid faces.
    max_unique_edges : int
        Maximum number of unique edges to handle.

    Returns
    -------
    unique_edges : JaxArray
        (max_unique_edges, 2) array of unique edges with canonical ordering.
        Canonicalized as (min_vertex_index, max_vertex_index).
    unique_edge_mask : JaxArray
        (max_unique_edges,) boolean mask for valid unique edges.
    face_to_edge_idx : JaxArray
        (max_faces, 3) mapping from face edges to unique edge indices.
        Entry [i, j] gives the unique edge index for edge j of face i.
    n_unique_edges : int
        Number of unique edges found.
    """
    max_faces = faces.shape[0]

    # Extract all edges from faces (3 edges per face)
    all_edges = jnp.zeros((max_faces * 3, 2), dtype=jnp.int32)
    all_edges_mask = jnp.zeros(max_faces * 3, dtype=bool)

    def extract_face_edges(face_idx: int) -> tuple[JaxArray, JaxArray]:
        """
        Extract 3 edges from a single face.

        Parameters
        ----------
        face_idx : int
            Index of the face to process.

        Returns
        -------
        edges : JaxArray
            (3, 2) array of canonicalized edges.
        edge_mask : JaxArray
            (3,) boolean mask for valid edges.
        """
        is_valid = face_mask[face_idx]
        face_verts = faces[face_idx]

        # Extract edges with canonical ordering
        edges = jnp.array(
            [
                canonicalize_edge(face_verts[0], face_verts[1]),  # Edge 0
                canonicalize_edge(face_verts[1], face_verts[2]),  # Edge 1
                canonicalize_edge(face_verts[2], face_verts[0]),  # Edge 2
            ]
        )

        edge_mask = jnp.array([is_valid, is_valid, is_valid])

        return edges, edge_mask

    # Extract edges face by face
    def extract_single_face_edges(carry, face_idx):
        """
        Extract edges from a single face and add to arrays.

        Parameters
        ----------
        carry : tuple[JaxArray, JaxArray]
            Current (all_edges, all_edges_mask) arrays.
        face_idx : int
            Index of current face.

        Returns
        -------
        updated_carry : tuple[JaxArray, JaxArray]
            Updated arrays.
        _ : None
            Unused output for scan compatibility.
        """
        edges_array, mask_array = carry

        edges, mask = extract_face_edges(face_idx)

        # Write each edge separately to avoid dynamic slicing
        base_idx = face_idx * 3
        edges_array = edges_array.at[base_idx].set(edges[0])
        edges_array = edges_array.at[base_idx + 1].set(edges[1])
        edges_array = edges_array.at[base_idx + 2].set(edges[2])

        mask_array = mask_array.at[base_idx].set(mask[0])
        mask_array = mask_array.at[base_idx + 1].set(mask[1])
        mask_array = mask_array.at[base_idx + 2].set(mask[2])

        return (edges_array, mask_array), None

    (all_edges, all_edges_mask), _ = jax.lax.scan(
        extract_single_face_edges, (all_edges, all_edges_mask), jnp.arange(max_faces)
    )

    # Initialize unique edges storage
    unique_edges = jnp.zeros((max_unique_edges, 2), dtype=jnp.int32)
    unique_edge_mask = jnp.zeros(max_unique_edges, dtype=bool)
    edge_to_unique_idx = jnp.full(max_faces * 3, -1, dtype=jnp.int32)

    # Build unique edge list using scan
    def process_edge(carry, edge_idx):
        """
        Process a single edge, adding to unique list if new.

        Parameters
        ----------
        carry : tuple[JaxArray, JaxArray, JaxArray, int]
            Current (unique_edges, unique_mask, edge_map, n_unique) state.
        edge_idx : int
            Index of edge to process.

        Returns
        -------
        updated_carry : tuple[JaxArray, JaxArray, JaxArray, int]
            Updated state.
        _ : None
            Unused output for scan compatibility.
        """
        u_edges, u_mask, edge_map, n_unique = carry

        is_valid_edge = all_edges_mask[edge_idx]
        edge = all_edges[edge_idx]

        # Check if this edge already exists in unique edges
        matches = (u_edges[:, 0] == edge[0]) & (u_edges[:, 1] == edge[1]) & u_mask

        # Find first match (if any)
        match_indices = jnp.where(
            matches, jnp.arange(max_unique_edges), max_unique_edges
        )
        first_match_idx = jnp.min(match_indices)

        # Check if we found a match
        found_match = first_match_idx < max_unique_edges

        # If no match and edge is valid, add as new unique edge
        should_add_new = is_valid_edge & ~found_match & (n_unique < max_unique_edges)

        # Update unique edges
        u_edges = u_edges.at[n_unique].set(
            jnp.where(should_add_new, edge, u_edges[n_unique])
        )
        u_mask = u_mask.at[n_unique].set(
            jnp.where(should_add_new, True, u_mask[n_unique])
        )

        # Map this edge to unique index
        unique_idx = jnp.where(
            found_match, first_match_idx, jnp.where(should_add_new, n_unique, -1)
        )
        edge_map = edge_map.at[edge_idx].set(jnp.where(is_valid_edge, unique_idx, -1))

        # Increment unique count if we added a new edge
        n_unique = jnp.where(should_add_new, n_unique + 1, n_unique)

        return (u_edges, u_mask, edge_map, n_unique), None

    (unique_edges, unique_edge_mask, edge_to_unique_idx, n_unique_edges), _ = (
        jax.lax.scan(
            process_edge,
            (unique_edges, unique_edge_mask, edge_to_unique_idx, 0),
            jnp.arange(max_faces * 3),
        )
    )

    # Build face-to-edge mapping
    face_to_edge_idx = edge_to_unique_idx.reshape(max_faces, 3)

    return unique_edges, unique_edge_mask, face_to_edge_idx, n_unique_edges


def compute_unique_edge_lengths(
    unique_edges: JaxArray, unique_edge_mask: JaxArray, vertices: JaxArray
) -> JaxArray:
    """
    Compute lengths for unique edges.

    Parameters
    ----------
    unique_edges : JaxArray
        (max_unique_edges, 2) array of unique edge vertex pairs.
    unique_edge_mask : JaxArray
        (max_unique_edges,) boolean mask for valid edges.
    vertices : JaxArray
        (max_vertices, 3) vertex coordinates.

    Returns
    -------
    edge_lengths : JaxArray
        (max_unique_edges,) lengths of unique edges.
    """
    v1_coords = vertices[unique_edges[:, 0]]
    v2_coords = vertices[unique_edges[:, 1]]

    edge_vectors = v2_coords - v1_coords
    edge_lengths = jnp.linalg.norm(edge_vectors, axis=1)

    # Mask invalid edges
    edge_lengths = jnp.where(unique_edge_mask, edge_lengths, 0.0)

    return edge_lengths


def create_split_vertices_from_unique_edges(
    vertices: JaxArray,
    vertex_mask: JaxArray,
    unique_edges: JaxArray,
    unique_edge_mask: JaxArray,
    edge_lengths: JaxArray,
    edge_length_threshold: float,
) -> tuple[JaxArray, JaxArray, JaxArray, JaxArray, bool]:
    """
    Create new vertices at midpoints of edges that need splitting.

    Parameters
    ----------
    vertices : JaxArray
        (max_vertices_per_cell, 3) vertex coordinates.
    vertex_mask : JaxArray
        (max_vertices_per_cell,) boolean mask for active vertices.
    unique_edges : JaxArray
        (max_unique_edges, 2) array of unique edge vertex pairs.
    unique_edge_mask : JaxArray
        (max_unique_edges,) boolean mask for valid unique edges.
    edge_lengths : JaxArray
        (max_unique_edges,) lengths of unique edges.
    edge_length_threshold : float
        Maximum allowed edge length before splitting.

    Returns
    -------
    updated_vertices : JaxArray
        (max_vertices_per_cell, 3) vertex array with new vertices added.
    updated_vertex_mask : JaxArray
        (max_vertices_per_cell,) updated validity mask.
    edge_to_new_vertex : JaxArray
        (max_unique_edges,) mapping from unique edge index to new vertex index.
        -1 indicates edge not split.
    edges_to_split_mask : JaxArray
        (max_unique_edges,) boolean mask indicating which edges were split.
    vertex_overflow : bool
        True if there are more edges to split than available vertex slots.
    """
    max_vertices = vertices.shape[0]

    # Identify edges that need splitting
    edges_to_split_mask = unique_edge_mask & (edge_lengths > edge_length_threshold)

    # Count new vertices needed and check for overflow
    n_edges_to_split = jnp.sum(edges_to_split_mask)
    n_current_vertices = jnp.sum(vertex_mask)
    available_slots = max_vertices - n_current_vertices
    vertex_overflow = n_edges_to_split > available_slots

    # Create compact ordering for edges to split
    split_cumsum = jnp.cumsum(edges_to_split_mask.astype(jnp.int32))
    split_edge_order = jnp.where(edges_to_split_mask, split_cumsum - 1, -1)

    # Map edges to new vertex indices
    edge_to_new_vertex = jnp.where(
        edges_to_split_mask & (split_edge_order < available_slots),
        n_current_vertices + split_edge_order,
        -1,
    )

    # Create new vertices at edge midpoints
    updated_vertices = vertices.copy()
    updated_vertex_mask = vertex_mask.copy()

    def create_and_place_vertex(edge_idx: int) -> tuple[JaxArray, bool, int]:
        """
        Create a vertex at edge midpoint and determine placement.

        Parameters
        ----------
        edge_idx : int
            Index of the edge to potentially split.

        Returns
        -------
        midpoint : JaxArray
            (3,) midpoint coordinates.
        should_create : bool
            Whether this vertex should be created.
        target_idx : int
            Target index in vertex array.
        """
        should_split = edges_to_split_mask[edge_idx]
        new_vertex_idx = edge_to_new_vertex[edge_idx]
        has_valid_idx = new_vertex_idx >= 0

        # Compute midpoint
        edge = unique_edges[edge_idx]
        v1_pos = vertices[edge[0]]
        v2_pos = vertices[edge[1]]
        midpoint = 0.5 * (v1_pos + v2_pos)

        should_create = should_split & has_valid_idx
        target_idx = jnp.where(should_create, new_vertex_idx, 0)

        return midpoint, should_create, target_idx

    # Vectorize vertex creation
    midpoints, should_create_mask, target_indices = jax.vmap(create_and_place_vertex)(
        jnp.arange(unique_edges.shape[0])
    )

    # Place new vertices using scan to avoid dynamic indexing
    def place_single_vertex(carry, edge_idx):
        """
        Place a single new vertex if needed.

        Parameters
        ----------
        carry : tuple[JaxArray, JaxArray]
            Current (vertices, vertex_mask) state.
        edge_idx : int
            Edge index to process.

        Returns
        -------
        updated_carry : tuple[JaxArray, JaxArray]
            Updated state.
        _ : None
            Unused output for scan compatibility.
        """
        verts, v_mask = carry

        should_place = should_create_mask[edge_idx]
        target_idx = target_indices[edge_idx]
        midpoint = midpoints[edge_idx]

        # Update arrays conditionally
        verts = verts.at[target_idx].set(
            jnp.where(should_place, midpoint, verts[target_idx])
        )
        v_mask = v_mask.at[target_idx].set(
            jnp.where(should_place, True, v_mask[target_idx])
        )

        return (verts, v_mask), None

    (updated_vertices, updated_vertex_mask), _ = jax.lax.scan(
        place_single_vertex,
        (updated_vertices, updated_vertex_mask),
        jnp.arange(unique_edges.shape[0]),
    )

    return (
        updated_vertices,
        updated_vertex_mask,
        edge_to_new_vertex,
        edges_to_split_mask,
        vertex_overflow,
    )


def update_faces_from_edge_splits(
    faces: JaxArray,
    face_mask: JaxArray,
    face_to_edge_idx: JaxArray,
    edge_to_new_vertex: JaxArray,
    max_faces: int,
) -> tuple[JaxArray, JaxArray, bool]:
    """
    Update face connectivity based on edge splits.

    Parameters
    ----------
    faces : JaxArray
        (max_faces_per_cell, 3) original face connectivity.
    face_mask : JaxArray
        (max_faces_per_cell,) boolean mask for active faces.
    face_to_edge_idx : JaxArray
        (max_faces_per_cell, 3) mapping from face edges to unique edge indices.
    edge_to_new_vertex : JaxArray
        (max_unique_edges,) mapping from edge index to new vertex index.
        -1 indicates edge not split.
    max_faces : int
        Maximum number of faces allowed in output.

    Returns
    -------
    new_faces : JaxArray
        (max_faces, 3) updated face connectivity.
    new_face_mask : JaxArray
        (max_faces,) updated face validity mask.
    face_overflow : bool
        True if too many faces were created.
    """
    # Pre-define subdivision patterns for all possible edge split combinations
    subdivision_patterns = jnp.array(
        [
            # 0 splits: keep original triangle
            [[0, 1, 2], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            # 1 split on edge 0: v0->v1 split
            [[0, 3, 2], [3, 1, 2], [-1, -1, -1], [-1, -1, -1]],
            # 1 split on edge 1: v1->v2 split
            [[0, 1, 4], [0, 4, 2], [-1, -1, -1], [-1, -1, -1]],
            # 1 split on edge 2: v2->v0 split
            [[0, 1, 5], [5, 1, 2], [-1, -1, -1], [-1, -1, -1]],
            # 2 splits on edges 0,1
            [[0, 3, 5], [3, 1, 4], [3, 4, 5], [5, 4, 2]],
            # 2 splits on edges 0,2
            [[0, 3, 5], [3, 1, 2], [3, 2, 5], [-1, -1, -1]],
            # 2 splits on edges 1,2
            [[0, 1, 4], [0, 4, 5], [5, 4, 2], [-1, -1, -1]],
            # 3 splits: all edges
            [[0, 3, 5], [3, 1, 4], [4, 2, 5], [3, 4, 5]],
        ]
    )

    # Buffer for new faces
    new_faces_buffer = jnp.zeros((max_faces * 4, 3), dtype=jnp.int32)
    new_faces_valid = jnp.zeros(max_faces * 4, dtype=bool)

    def process_single_face(carry, face_idx):
        """
        Update connectivity for a single face based on edge splits.

        Parameters
        ----------
        carry : tuple[JaxArray, JaxArray, int, bool]
            Current (faces_buffer, validity_buffer, write_idx, overflow) state.
        face_idx : int
            Index of face to process.

        Returns
        -------
        updated_carry : tuple[JaxArray, JaxArray, int, bool]
            Updated state.
        _ : None
            Unused output for scan compatibility.
        """
        faces_buf, valid_buf, write_idx, overflow = carry

        # Get unique edge indices for this face's three edges
        edge_indices = face_to_edge_idx[face_idx]

        # Check which edges were split and get new vertex indices
        edge_0_new_v = edge_to_new_vertex[edge_indices[0]]
        edge_1_new_v = edge_to_new_vertex[edge_indices[1]]
        edge_2_new_v = edge_to_new_vertex[edge_indices[2]]

        # Determine which edges have valid new vertices
        has_split_0 = edge_0_new_v >= 0
        has_split_1 = edge_1_new_v >= 0
        has_split_2 = edge_2_new_v >= 0

        # Calculate pattern index (0-7) based on which edges are split
        pattern_idx = (
            has_split_0.astype(jnp.int32) * 1
            + has_split_1.astype(jnp.int32) * 2
            + has_split_2.astype(jnp.int32) * 4
        )

        # Map to our pattern array ordering
        pattern_mapping = jnp.array([0, 1, 2, 4, 3, 5, 6, 7])
        mapped_pattern_idx = pattern_mapping[pattern_idx]

        # Get subdivision pattern
        pattern = subdivision_patterns[mapped_pattern_idx]

        # Build vertex lookup array
        # Indices 0,1,2 are original vertices; 3,4,5 are new vertices on edges 0,1,2
        orig_verts = faces[face_idx]
        new_vert_indices = jnp.array([edge_0_new_v, edge_1_new_v, edge_2_new_v])
        all_verts = jnp.concatenate([orig_verts, new_vert_indices])

        # Generate subdivided faces
        def write_subdivided_face(write_carry, sub_idx):
            """
            Write a single subdivided face to the buffer.

            Parameters
            ----------
            write_carry : tuple[JaxArray, JaxArray, int, bool]
                Current write state.
            sub_idx : int
                Subdivision face index (0-3).

            Returns
            -------
            updated_carry : tuple[JaxArray, JaxArray, int, bool]
                Updated write state.
            _ : None
                Unused output for scan compatibility.
            """
            fb, vb, w_idx, ovf = write_carry

            pattern_face = pattern[sub_idx]
            is_valid_pattern = pattern_face[0] >= 0
            face_overflow = w_idx >= max_faces * 4

            should_write = face_mask[face_idx] & is_valid_pattern & ~face_overflow

            # Map pattern indices to actual vertex indices
            actual_face = jnp.array(
                [
                    all_verts[pattern_face[0]],
                    all_verts[pattern_face[1]],
                    all_verts[pattern_face[2]],
                ]
            )

            # Update buffers conditionally
            fb = fb.at[w_idx].set(jnp.where(should_write, actual_face, fb[w_idx]))
            vb = vb.at[w_idx].set(jnp.where(should_write, True, vb[w_idx]))

            new_w_idx = jnp.where(should_write, w_idx + 1, w_idx)
            new_ovf = ovf | face_overflow

            return (fb, vb, new_w_idx, new_ovf), None

        # Write all subdivided faces for this original face
        (faces_buf, valid_buf, write_idx, overflow), _ = jax.lax.scan(
            write_subdivided_face,
            (faces_buf, valid_buf, write_idx, overflow),
            jnp.arange(4),
        )

        return (faces_buf, valid_buf, write_idx, overflow), None

    # Process all faces
    (new_faces_buffer, new_faces_valid, final_write_idx, face_overflow), _ = (
        jax.lax.scan(
            process_single_face,
            (new_faces_buffer, new_faces_valid, 0, False),
            jnp.arange(faces.shape[0]),
        )
    )

    # Extract final faces (limited to max_faces)
    final_faces = new_faces_buffer[:max_faces]
    final_face_mask = new_faces_valid[:max_faces]

    # Check for overflow
    total_faces_created = jnp.sum(new_faces_valid)
    face_overflow = face_overflow | (total_faces_created > max_faces)

    # Apply stop_gradient to topology changes
    final_faces = jax.lax.stop_gradient(final_faces)
    final_face_mask = jax.lax.stop_gradient(final_face_mask)

    return final_faces, final_face_mask, face_overflow


def remesh_edge_split_single_cell(
    vertices: JaxArray,
    faces: JaxArray,
    vertex_mask: JaxArray,
    face_mask: JaxArray,
    edge_length_threshold: float,
) -> tuple[JaxArray, JaxArray, JaxArray, JaxArray, bool]:
    """
    Split long edges in a single cell mesh.

    This function:
    1. Identifies unique edges across all faces
    2. Splits edges longer than threshold by creating new vertices at midpoints
    3. Updates face topology to use the new vertices

    Parameters
    ----------
    vertices : JaxArray
        (max_vertices_per_cell, 3) vertex coordinates for this cell.
    faces : JaxArray
        (max_faces_per_cell, 3) face connectivity using local vertex indices.
    vertex_mask : JaxArray
        (max_vertices_per_cell,) boolean mask for active vertices.
    face_mask : JaxArray
        (max_faces_per_cell,) boolean mask for active faces.
    edge_length_threshold : float
        Maximum allowed edge length before splitting.

    Returns
    -------
    new_vertices : JaxArray
        (max_vertices_per_cell, 3) updated vertex coordinates.
    new_faces : JaxArray
        (max_faces_per_cell, 3) updated face connectivity.
    new_vertex_mask : JaxArray
        (max_vertices_per_cell,) updated vertex mask.
    new_face_mask : JaxArray
        (max_faces_per_cell,) updated face mask.
    overflow_occurred : bool
        True if vertex or face overflow occurred during edge splitting.
    """
    max_faces = faces.shape[0]
    max_unique_edges = max_faces * 3  # Upper bound on unique edges

    # Extract unique edges and build mappings
    unique_edges, unique_edge_mask, face_to_edge_idx, n_unique_edges = (
        extract_unique_edges_from_faces(faces, face_mask, max_unique_edges)
    )

    # Compute edge lengths for unique edges only
    edge_lengths = compute_unique_edge_lengths(unique_edges, unique_edge_mask, vertices)

    # Apply edge splits - create vertices and update connectivity
    (
        updated_vertices,
        updated_vertex_mask,
        edge_to_new_vertex,
        edges_to_split_mask,
        vertex_overflow,
    ) = create_split_vertices_from_unique_edges(
        vertices=vertices,
        vertex_mask=vertex_mask,
        unique_edges=unique_edges,
        unique_edge_mask=unique_edge_mask,
        edge_lengths=edge_lengths,
        edge_length_threshold=edge_length_threshold,
        # max_new_vertices=max_new_vertices
    )

    # Update face connectivity
    new_faces, new_face_mask, face_overflow = update_faces_from_edge_splits(
        faces=faces,
        face_mask=face_mask,
        face_to_edge_idx=face_to_edge_idx,
        edge_to_new_vertex=edge_to_new_vertex,
        max_faces=max_faces,
    )

    overflow_occurred = vertex_overflow | face_overflow

    return (
        updated_vertices,
        new_faces,
        updated_vertex_mask,
        new_face_mask,
        overflow_occurred,
    )

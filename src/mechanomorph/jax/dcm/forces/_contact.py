import jax
import jax.numpy as jnp


def find_contacting_vertices(
    vertices: jnp.ndarray,
    vertex_mask: jnp.ndarray,
    cell_contact_pairs: jnp.ndarray,
    cell_contact_mask: jnp.ndarray,
    distance_threshold: float,
    max_contacts: int = 10000,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find all vertex pairs within a distance threshold between contacting cells.

    Parameters
    ----------
    vertices : jnp.ndarray
        Array of shape (n_cells, max_vertices_per_cell, 3) containing
        vertex positions for each cell. Padded with zeros for invalid vertices.
    vertex_mask : jnp.ndarray
        Boolean array of shape (n_cells, max_vertices_per_cell) indicating
        which vertices are valid (True) or padding (False).
    cell_contact_pairs : jnp.ndarray
        Padded array of shape (max_contact_pairs, 2) containing pairs of cell
        indices that are potentially in contact. Padded with -1 for invalid entries.
    cell_contact_mask : jnp.ndarray
        Boolean array of shape (max_contact_pairs,) indicating which cell
        contact pairs are valid (True) or padding (False).
    distance_threshold : float
        Maximum distance between vertices to be considered in contact.
    max_contacts : int
        Maximum number of vertex contacts to detect. This is a compile-time
        constant for JIT compatibility.

    Returns
    -------
    contact_vertex_pairs : jnp.ndarray
        Array of shape (max_contacts, 2) containing pairs of global vertex
        indices that are in contact. Padded with -1 for invalid entries.
    contact_mask : jnp.ndarray
        Boolean array of shape (max_contacts,) indicating which contact
        pairs are valid (True) or padding (False).
    contact_distances : jnp.ndarray
        Array of shape (max_contacts,) containing distances between
        contacting vertices. Padded with infinity for invalid entries.
    """
    n_cells, max_vertices_per_cell, _ = vertices.shape

    # Pre-allocate output arrays
    contact_vertex_pairs = jnp.full((max_contacts, 2), -1, dtype=jnp.int32)
    contact_mask = jnp.zeros(max_contacts, dtype=bool)
    contact_distances = jnp.full(max_contacts, jnp.inf)

    def process_cell_pair(carry, pair_idx):
        """Process a single pair of potentially contacting cells.

        Parameters
        ----------
        carry : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
            Current state: (contact_pairs, contact_mask, distances, next_slot)
        pair_idx : int
            Index into cell_contact_pairs array.

        Returns
        -------
        carry : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
            Updated state after processing this cell pair.
        _ : None
            Unused output for scan compatibility.
        """
        pairs, mask, dists, next_slot = carry

        # Check if this is a valid cell pair
        is_valid_pair = cell_contact_mask[pair_idx]

        # Get the two cells to compare
        cell_i = cell_contact_pairs[pair_idx, 0]
        cell_j = cell_contact_pairs[pair_idx, 1]

        # Extract vertices and masks for both cells
        verts_i = vertices[cell_i]  # (max_vertices_per_cell, 3)
        verts_j = vertices[cell_j]  # (max_vertices_per_cell, 3)
        mask_i = vertex_mask[cell_i]  # (max_vertices_per_cell,)
        mask_j = vertex_mask[cell_j]  # (max_vertices_per_cell,)

        # Compute pairwise distances between all vertices in the two cells
        # Output shape: (max_vertices_per_cell, max_vertices_per_cell)
        pairwise_dists = jnp.linalg.norm(
            verts_i[:, None, :] - verts_j[None, :, :], axis=-1
        )

        # Create validity mask for vertex pairs
        # Only valid if both vertices are valid AND this is a valid cell pair
        valid_pairs = mask_i[:, None] & mask_j[None, :] & is_valid_pair

        # Find pairs within threshold
        within_threshold = (pairwise_dists < distance_threshold) & valid_pairs

        # Get indices of contacting vertices
        local_i, local_j = jnp.meshgrid(
            jnp.arange(max_vertices_per_cell),
            jnp.arange(max_vertices_per_cell),
            indexing="ij",
        )

        # Convert to global vertex indices
        global_i = cell_i * max_vertices_per_cell + local_i
        global_j = cell_j * max_vertices_per_cell + local_j

        # Flatten arrays for processing
        flat_within = within_threshold.ravel()
        flat_global_i = global_i.ravel()
        flat_global_j = global_j.ravel()
        flat_dists = pairwise_dists.ravel()

        # Process each potential contact in this cell pair
        def add_contact(inner_carry, flat_idx):
            """Add a single contact to the output arrays if valid.

            Parameters
            ----------
            inner_carry : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
                Current arrays and next available slot.
            flat_idx : int
                Flattened index into the pairwise arrays.

            Returns
            -------
            inner_carry : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
                Updated arrays and slot counter.
            _ : None
                Unused output for scan compatibility.
            """
            p, m, d, slot = inner_carry

            # Check if this pair is within threshold and we have space
            should_add = flat_within[flat_idx] & (slot < max_contacts)

            # Update arrays conditionally
            p = p.at[slot, 0].set(
                jnp.where(should_add, flat_global_i[flat_idx], p[slot, 0])
            )
            p = p.at[slot, 1].set(
                jnp.where(should_add, flat_global_j[flat_idx], p[slot, 1])
            )
            m = m.at[slot].set(jnp.where(should_add, True, m[slot]))
            d = d.at[slot].set(jnp.where(should_add, flat_dists[flat_idx], d[slot]))

            # Increment slot only if we added a contact
            new_slot = jnp.where(should_add, slot + 1, slot)

            return (p, m, d, new_slot), None

        # Process all potential contacts in this cell pair
        # n_potential = max_vertices_per_cell * max_vertices_per_cell
        n_potential = flat_within.shape[0]
        (pairs, mask, dists, next_slot), _ = jax.lax.scan(
            add_contact, (pairs, mask, dists, next_slot), jnp.arange(n_potential)
        )

        return (pairs, mask, dists, next_slot), None

    # Process all cell pairs
    n_pairs = cell_contact_pairs.shape[0]
    init_carry = (contact_vertex_pairs, contact_mask, contact_distances, 0)
    (final_pairs, final_mask, final_dists, _), _ = jax.lax.scan(
        process_cell_pair, init_carry, jnp.arange(n_pairs)
    )

    return final_pairs, final_mask, final_dists


def label_vertices(
    contact_vertex_pairs: jnp.ndarray,
    contact_mask: jnp.ndarray,
    n_total_vertices: int,
    max_iterations: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Assign connected component labels to contacting vertices.

    This uses the Union-Find algorithm.

    Parameters
    ----------
    contact_vertex_pairs : jnp.ndarray
        Array of shape (max_contacts, 2) containing pairs of vertex indices
        that are in contact. Invalid entries should be -1.
    contact_mask : jnp.ndarray
        Boolean array of shape (max_contacts,) indicating which contact
        pairs are valid.
    n_total_vertices : int
        Total number of vertices across all cells. This is a compile-time
        constant for JIT compatibility.
    max_iterations : int
        Maximum number of union-find iterations. Should be at least the
        diameter of the largest expected connected component.

    Returns
    -------
    vertex_labels : jnp.ndarray
        Array of shape (n_total_vertices,) containing the component label
        for each vertex. Non-contacting vertices have their own unique label
        equal to their vertex index.
    is_contacting : jnp.ndarray
        Boolean array of shape (n_total_vertices,) indicating which vertices
        are part of any contact.
    """
    # Initialize parent array - each vertex is its own parent initially
    parent = jnp.arange(n_total_vertices)

    # Mark which vertices are involved in any contact
    is_contacting = jnp.zeros(n_total_vertices, dtype=bool)

    def mark_contacting(carry, idx):
        """Mark vertices involved in contacts.

        Parameters
        ----------
        carry : jnp.ndarray
            Current is_contacting array.
        idx : int
            Index into contact arrays.

        Returns
        -------
        carry : jnp.ndarray
            Updated is_contacting array.
        _ : None
            Unused output for scan compatibility.
        """
        is_cont = carry
        valid = contact_mask[idx]
        v1 = contact_vertex_pairs[idx, 0]
        v2 = contact_vertex_pairs[idx, 1]

        # Only mark if this is a valid contact and indices are in bounds
        v1_valid = valid & (v1 >= 0) & (v1 < n_total_vertices)
        v2_valid = valid & (v2 >= 0) & (v2 < n_total_vertices)

        is_cont = is_cont.at[v1].set(jnp.where(v1_valid, True, is_cont[v1]))
        is_cont = is_cont.at[v2].set(jnp.where(v2_valid, True, is_cont[v2]))

        return is_cont, None

    is_contacting, _ = jax.lax.scan(
        mark_contacting, is_contacting, jnp.arange(contact_vertex_pairs.shape[0])
    )

    def find_root(parent_array, vertex):
        """Find root with path compression using fixed iterations.

        Parameters
        ----------
        parent_array : jnp.ndarray
            Current parent array.
        vertex : int
            Vertex index to find root for.

        Returns
        -------
        root : int
            Root vertex of the component.
        """

        # Fixed number of iterations for JIT compatibility
        def compress_step(v, _):
            """One step of path compression."""
            return parent_array[v], None

        root, _ = jax.lax.scan(compress_step, vertex, jnp.arange(max_iterations))
        return root

    def union_step(parent_array, _):
        """Perform one iteration of union operations.

        Parameters
        ----------
        parent_array : jnp.ndarray
            Current parent array.
        _ : Any
            Unused iteration index.

        Returns
        -------
        parent_array : jnp.ndarray
            Updated parent array after unions.
        _ : None
            Unused output for scan compatibility.
        """

        def process_edge(p, idx):
            """Process a single edge in the contact graph.

            Parameters
            ----------
            p : jnp.ndarray
                Current parent array.
            idx : int
                Index into contact arrays.

            Returns
            -------
            p : jnp.ndarray
                Updated parent array.
            _ : None
                Unused output for scan compatibility.
            """
            valid = contact_mask[idx]
            v1 = contact_vertex_pairs[idx, 0]
            v2 = contact_vertex_pairs[idx, 1]

            # Find roots of both vertices
            root1 = find_root(p, v1)
            root2 = find_root(p, v2)

            # Union by making root2's parent be root1
            # Only update if this is a valid edge
            should_update = valid & (root1 != root2)
            p = p.at[root2].set(jnp.where(should_update, root1, p[root2]))

            return p, None

        # Process all edges once
        new_parent, _ = jax.lax.scan(
            process_edge, parent_array, jnp.arange(contact_vertex_pairs.shape[0])
        )
        return new_parent, None

    # Run union-find for fixed number of iterations
    parent, _ = jax.lax.scan(union_step, parent, jnp.arange(max_iterations))

    # Final pass to ensure all vertices point to their root
    def finalize_labels(idx):
        """Get final root for each vertex.

        Parameters
        ----------
        idx : int
            Vertex index.

        Returns
        -------
        label : int
            Final component label for this vertex.
        """
        return find_root(parent, idx)

    vertex_labels = jax.vmap(finalize_labels)(jnp.arange(n_total_vertices))

    return vertex_labels, is_contacting


def average_vector_by_label(
    vertex_vectors: jnp.ndarray,
    vertex_mask: jnp.ndarray,
    vertex_labels: jnp.ndarray,
    max_components: int = 1000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Average vector values for vertices with the same label.

    Parameters
    ----------
    vertex_vectors : jnp.ndarray
        Array of shape (n_cells, max_vertices_per_cell, vector_dim) containing
        vector data to average (e.g., positions, velocities).
    vertex_mask : jnp.ndarray
        Boolean array of shape (n_cells, max_vertices_per_cell) indicating
        which vertices are valid.
    vertex_labels : jnp.ndarray
        Array of shape (n_total_vertices,) containing component labels
        from label_vertices function.
    max_components : int
        Maximum expected number of components. This is a compile-time
        constant for JIT compatibility.

    Returns
    -------
    averaged_vectors : jnp.ndarray
        Array of shape (n_cells, max_vertices_per_cell, vector_dim) with
        averaged vectors for vertices in the same component.
    was_averaged : jnp.ndarray
        Boolean array of shape (n_cells, max_vertices_per_cell) indicating
        which vertices had their vectors averaged (i.e., were part of a
        multi-vertex component).
    """
    n_cells, max_vertices_per_cell, vector_dim = vertex_vectors.shape
    n_total_vertices = n_cells * max_vertices_per_cell

    # Flatten vertex data for easier processing
    flat_vectors = vertex_vectors.reshape(n_total_vertices, vector_dim)
    flat_mask = vertex_mask.ravel()

    # Compute component sums and counts
    # We use segment_sum which requires sorted indices, but we'll work around this
    # by using a fixed-size accumulator array
    component_sums = jnp.zeros((max_components, vector_dim))
    component_counts = jnp.zeros(max_components)

    def accumulate_vertex(carry, vertex_idx):
        """Accumulate vector and count for one vertex.

        Parameters
        ----------
        carry : Tuple[jnp.ndarray, jnp.ndarray]
            Current (component_sums, component_counts).
        vertex_idx : int
            Global vertex index.

        Returns
        -------
        carry : Tuple[jnp.ndarray, jnp.ndarray]
            Updated sums and counts.
        _ : None
            Unused output for scan compatibility.
        """
        sums, counts = carry

        # Only process if vertex is valid
        is_valid = flat_mask[vertex_idx]
        label = vertex_labels[vertex_idx]
        vector = flat_vectors[vertex_idx]

        # Clip label to ensure it's within bounds
        safe_label = jnp.clip(label, 0, max_components - 1)

        # Update sums and counts conditionally
        sums = sums.at[safe_label].add(
            jnp.where(is_valid, vector, jnp.zeros(vector_dim))
        )
        counts = counts.at[safe_label].add(jnp.where(is_valid, 1.0, 0.0))

        return (sums, counts), None

    # Accumulate all vertices
    (component_sums, component_counts), _ = jax.lax.scan(
        accumulate_vertex,
        (component_sums, component_counts),
        jnp.arange(n_total_vertices),
    )

    # Compute averages, avoiding division by zero
    safe_counts = jnp.maximum(component_counts, 1.0)
    component_averages = component_sums / safe_counts[:, None]

    # Map averages back to vertices
    def get_averaged_vector(vertex_idx):
        """Get averaged vector for a vertex.

        Parameters
        ----------
        vertex_idx : int
            Global vertex index.

        Returns
        -------
        averaged : jnp.ndarray
            Averaged vector for this vertex's component.
        multi_vertex : bool
            Whether this vertex is part of a multi-vertex component.
        """
        label = vertex_labels[vertex_idx]
        safe_label = jnp.clip(label, 0, max_components - 1)
        count = component_counts[safe_label]

        # Check if this is part of a multi-vertex component
        is_multi = count > 1.5  # Use 1.5 to avoid floating point issues

        # Get averaged vector or keep original
        averaged = jnp.where(
            is_multi & flat_mask[vertex_idx],
            component_averages[safe_label],
            flat_vectors[vertex_idx],
        )

        return averaged, is_multi & flat_mask[vertex_idx]

    # Apply averaging to all vertices
    averaged_flat, was_averaged_flat = jax.vmap(get_averaged_vector)(
        jnp.arange(n_total_vertices)
    )

    # Reshape back to original shape
    averaged_vectors = averaged_flat.reshape(n_cells, max_vertices_per_cell, vector_dim)
    was_averaged = was_averaged_flat.reshape(n_cells, max_vertices_per_cell)

    return averaged_vectors, was_averaged

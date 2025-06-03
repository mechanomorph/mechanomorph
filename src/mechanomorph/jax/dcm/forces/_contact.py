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

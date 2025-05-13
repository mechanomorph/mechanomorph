"""Functions for detecting contacts and utilizing contacts between vertices."""

import torch


def find_contacting_vertices_from_cell_map(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_cell_index: torch.Tensor,
    cell_contact_map: torch.Tensor,
    distance_threshold: float,
):
    """Find vertices from different cells that are in contact.

    This iterates between pairs of contacting cells. For each pair, it computes
    the distance between all pairs of vertices from the two cells. If the distance is
    below the threshold, it marks the vertices as in contact.

    This will be slow. Some potential optimizations:
    - use spatial partitioning to reduce the number of pairs to check
    - make sure we aren't checking the same pair of cells twice
      (i.e., use upper triangle of cell contact map)

    Parameters
    ----------
    vertices : torch.Tensor
        Tensor of shape (n_vertices, 3) containing the
        3D coordinates of each vertex.
    faces : torch.Tensor
        Tensor of shape (num_faces, 3) containing vertex indices for each face.
    face_cell_index : list of torch.Tensor
        List where each element is a tensor of shape (num_faces_cell,)
        containing the face indices for each cell.
    cell_contact_map : torch.sparse_coo_tensor
        Sparse tensor of shape (n_cells, n_cells), with element
        set to 1 where cells are considered in contact.
    distance_threshold : float
        Distance threshold for marking vertices as in contact.

    Returns
    -------
    contact_map : torch.sparse_coo_tensor
        Sparse tensor of shape (n_vertices, n_vertices)
        with entries of 1 where vertices are in contact.
    """
    device = vertices.device
    n_vertices = vertices.shape[0]

    # Map from cell index -> unique vertex indices
    cell_vertices = []
    for face_indices in face_cell_index:
        vertex_indices = faces[face_indices].unique()
        cell_vertices.append(vertex_indices)

    contact_indices_list = []

    # Unpack the contacting cell pairs
    contacting_pairs = cell_contact_map.coalesce().indices().T  # (n_contacts, 2)

    for pair in contacting_pairs:
        cell_i = pair[0]
        cell_j = pair[1]

        verts_i = cell_vertices[cell_i]  # (n_i,)
        verts_j = cell_vertices[cell_j]  # (n_j,)

        pos_i = vertices[verts_i]  # (n_i, 3)
        pos_j = vertices[verts_j]  # (n_j, 3)

        # Compute pairwise distances
        diff = pos_i[:, None, :] - pos_j[None, :, :]  # (n_i, n_j, 3)
        dists = torch.norm(diff, dim=-1)  # (n_i, n_j)

        # Find where distance < threshold
        close = dists < distance_threshold

        # Get indices
        close_i, close_j = close.nonzero(as_tuple=True)

        verts_i_close = verts_i[close_i]
        verts_j_close = verts_j[close_j]

        # Stack and collect
        if verts_i_close.numel() > 0:
            contacts = torch.stack(
                [verts_i_close, verts_j_close], dim=0
            )  # (2, num_contacts)
            contact_indices_list.append(contacts)

    if contact_indices_list:
        all_contacts = torch.cat(contact_indices_list, dim=1)  # (2, total_contacts)

        contact_values = torch.ones(all_contacts.shape[1], device=device)
        contact_map = torch.sparse_coo_tensor(
            all_contacts, contact_values, (n_vertices, n_vertices)
        )

    else:
        # No contacts found
        contact_map = torch.sparse_coo_tensor(
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0,), device=device),
            (n_vertices, n_vertices),
        )

    return contact_map


def group_contacting_vertices_union_find(contact_matrix: torch.Tensor) -> torch.Tensor:
    """
    Group directly contacting vertices using Union-Find (disjoint set).

    This function will be slow for many vertices. We should find a faster way.

    Parameters
    ----------
    contact_matrix : (n_vertices, n_vertices) torch.Tensor
        Binary matrix (0 or 1) indicating contacts.

    Returns
    -------
    group_labels : (n_vertices,) torch.Tensor
        Group ID assigned to each vertex.
    """
    n_vertices = contact_matrix.size(0)
    device = contact_matrix.device

    parent = torch.arange(n_vertices, device=device)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # Path compression
            i = parent[i]
        return i

    def union(i, j):
        pi = find(i)
        pj = find(j)
        if pi != pj:
            parent[pj] = pi

    contacts = (contact_matrix.to_dense() > 0).nonzero(as_tuple=False)

    for i, j in contacts:
        union(i.item(), j.item())

    for i in range(n_vertices):
        find(i)

    return parent


def average_vector_by_group(vertices: torch.Tensor, labels: torch.Tensor):
    """
    Move vertices to the average position of their group.

    Parameters
    ----------
    vertices : torch.Tensor
        Tensor of shape (n_vertices, 3) containing
        the coordinates of each vertex.
    labels : torch.Tensor
        Tensor of shape (n_vertices,) where each entry is the
        group label of the corresponding vertex.

    Returns
    -------
    new_vertices : torch.Tensor
        Tensor of shape (n_vertices, 3) where vertices belonging
        to the same group have the same (average) position.
    """
    device = vertices.device
    n_vertices = vertices.shape[0]

    # Find unique labels
    unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)

    n_groups = unique_labels.shape[0]

    # Sum vertex positions per group
    group_sum = torch.zeros((n_groups, 3), device=device).scatter_add_(
        0, inverse_indices[:, None].expand(-1, 3), vertices
    )

    # Count how many vertices are in each group
    group_count = torch.zeros((n_groups,), device=device).scatter_add_(
        0, inverse_indices, torch.ones(n_vertices, device=device)
    )

    # Avoid division by zero
    group_count = group_count.clamp(min=1.0)

    # Average position per group
    group_mean = group_sum / group_count[:, None]  # (n_groups, 3)

    # Map each vertex to its group's mean
    new_vertices = group_mean[inverse_indices]

    return new_vertices

import torch

from mechanomorph.sim._deformable_cell_model._contact import (
    average_vector_by_group,
    find_contacting_vertices_from_cell_map,
    group_contacting_vertices_union_find,
)


def test_group_contacting_vertices():
    """Test grouping of vertices based on contact map."""
    # Create a sparse contact map
    # (0, 1, 4) and (2, 3) are in contact
    # 4, and 5 are alone
    contact_map = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2, 3, 0, 4, 1, 4], [1, 0, 3, 2, 4, 0, 4, 1]]),
        values=torch.ones(8),
        size=(6, 6),
    )

    labels = group_contacting_vertices_union_find(contact_map)

    expected_labels = torch.tensor([0, 0, 2, 2, 0, 5])

    torch.testing.assert_close(labels, expected_labels)


def test_average_vector_by_group():
    """Test averaging a vector that is grouped by labels."""
    vertices = torch.tensor(
        [
            [0.0, -1.0, 1.0],  # alone
            [0.0, -0.1, 1.0],  # 1, 4
            [0.0, -0.1, 0.0],  # 2, 5, 6
            [0.0, 1.0, 1.0],  # alone
            [0.0, 0.1, 1.0],  # 1, 4
            [0.0, 0.1, 0.0],  # 2, 5, 6
            [0.0, 0.0, 0.0],  # 2, 5, 6
        ]
    )
    labels = torch.tensor([0, 1, 2, 3, 1, 2, 2])

    averaged_vertices = average_vector_by_group(vertices, labels)

    expected_output = torch.tensor(
        [
            [0.0, -1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    torch.testing.assert_close(averaged_vertices, expected_output)


def test_find_contacting_vertices_from_cell_map():
    """Test finding contacting vertices."""

    vertices = torch.tensor(
        [
            [0.0, -1.0, 1.0],
            [0.0, -0.1, 1.0],
            [0.0, -0.1, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.0, 0.1, 0.0],
        ]
    )

    faces = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
        ]
    )

    face_cell_index = torch.tensor([[0], [1]])

    cell_contact_map = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.ones(2),
        size=(2, 2),
    )

    vertex_contact_map = find_contacting_vertices_from_cell_map(
        vertices=vertices,
        faces=faces,
        face_cell_index=face_cell_index,
        cell_contact_map=cell_contact_map,
        distance_threshold=0.3,
    )

    # check the result
    n_vertices = vertices.size(0)
    expected_contact_map = torch.sparse_coo_tensor(
        indices=torch.tensor([[1, 4, 2, 5], [4, 1, 5, 2]]),
        values=torch.ones(4),
        size=(n_vertices, n_vertices),
    )
    torch.testing.assert_close(
        vertex_contact_map.coalesce(),
        expected_contact_map.coalesce(),
    )

import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.dcm.forces import (
    average_vector_by_label,
    find_contacting_vertices,
    label_vertices,
)
from mechanomorph.jax.dcm.utils import pack_mesh_to_cells
from mechanomorph.jax.utils.testing import (
    generate_two_cubes,
)


def test_find_contacting_vertices():
    """Test that the correct contacting vertices are found."""

    # make the mesh
    (
        all_vertices,
        all_faces,
        vertex_cell_mapping,
        face_cell_mapping,
        vertices_cube1,
        vertices_cube2,
        faces_cube1,
        faces_cube2,
    ) = generate_two_cubes()

    # pack the mesh to cells
    (
        vertices_packed,
        faces_packed,
        valid_vertices_mask,
        valid_faces_mask,
        valid_cells_mask,
        vertex_overflow,
        face_overflow,
        cell_overflow,
    ) = pack_mesh_to_cells(
        vertices=all_vertices,
        faces=all_faces,
        vertex_cell_mapping=vertex_cell_mapping,
        face_cell_mapping=face_cell_mapping,
        max_vertices_per_cell=vertices_cube1.shape[0],
        max_faces_per_cell=faces_cube1.shape[0],
        max_cells=3,
    )

    # make cell contact data
    cell_contact_pairs = jnp.array(
        [
            [0, 1],  # Contact between cube 1 and cube 2
            [-1, -1],  # Padding for invalid contact pair
        ]
    )
    cell_contact_mask = jnp.array([True, False])

    # find the contacts
    distance_threshold = 0.5
    max_contacts = 10000  # large enough to not truncate results
    contact_pairs, contact_mask, distances = find_contacting_vertices(
        vertices_packed,
        valid_vertices_mask,
        cell_contact_pairs,
        cell_contact_mask,
        distance_threshold,
        max_contacts,
    )

    # convert the contact pairs to a set of tuples for easy comparison
    valid_contact_pairs = contact_pairs[contact_mask]
    valid_contact_pairs_set = {
        (int(pair[0]), int(pair[1])) for pair in valid_contact_pairs
    }

    # expected results
    expected_pairs = {(1, 8), (2, 11), (5, 12), (6, 15)}

    # check the results
    assert expected_pairs == valid_contact_pairs_set

    np.testing.assert_allclose(
        np.zeros(len(valid_contact_pairs)), distances[contact_mask]
    )


def test_label_vertices():
    """Test the connected components labeling of contacts."""

    n_total_vertices = 10
    max_iterations = 5
    contact_pairs = jnp.array(
        [
            [0, 5],
            [1, 6],
            [2, 6],
            [-1, -1],  # Padding for invalid contact pair
        ]
    )
    contact_mask = jnp.array([True, True, True, False])

    vertex_labels, is_contacting = label_vertices(
        contact_pairs, contact_mask, n_total_vertices, max_iterations
    )

    # check the results
    assert vertex_labels.shape == (n_total_vertices,)
    assert (
        vertex_labels[0] == vertex_labels[5]
    )  # 0 and 5 should be in the same component
    assert (
        vertex_labels[1] == vertex_labels[6]
    )  # 1 and 6 should be in the same component
    assert (
        vertex_labels[2] == vertex_labels[6]
    )  # 2 and 6 should be in the same component

    n_expected_unique_labels = 7
    assert len(jnp.unique(vertex_labels)) == n_expected_unique_labels

    expected_is_contacting = jnp.array(
        [True, True, True, False, False, True, True, False, False, False]
    )
    np.testing.assert_array_equal(is_contacting, expected_is_contacting)


def test_average_3d_vector_by_label():
    """Test averaging 3D vectors by label."""
    vertex_vectors = jnp.array(
        [
            [
                [0, 0, 0],
                [1, 1, 1],
                [-1, -1, -1],  # padding
            ],
            [
                [10, 10, 10],
                [0.5, 0.5, 0.5],
                [-1, -1, -1],  # padding
            ],
        ]
    )
    vertex_mask = jnp.array([[True, True, False], [True, True, False]])
    vertex_labels = jnp.array([1, 0, -1, 10, 1, -1])
    max_components = vertex_vectors.shape[0] * vertex_vectors.shape[1]

    averaged_vectors, was_averaged = average_vector_by_label(
        vertex_vectors, vertex_mask, vertex_labels, max_components
    )

    # check the result
    expected_vectors = jnp.array(
        [
            [
                [0.25, 0.25, 0.25],
                [1, 1, 1],
                [-1, -1, -1],  # padding
            ],
            [
                [10, 10, 10],
                [0.25, 0.25, 0.25],
                [-1, -1, -1],  # padding
            ],
        ]
    )
    np.testing.assert_allclose(averaged_vectors, expected_vectors)

    expected_was_averaged = jnp.array([[True, False, False], [False, True, False]])
    np.testing.assert_array_equal(was_averaged, expected_was_averaged)


def test_average_4d_vector_by_label():
    """Test averaging 3D vectors by label."""
    vertex_vectors = jnp.array(
        [
            [
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [-1, -1, -1, -1],  # padding
            ],
            [
                [10, 10, 10, 0],
                [0.5, 0.5, 0.5, 0],
                [-1, -1, -1, -1],  # padding
            ],
        ]
    )
    vertex_mask = jnp.array([[True, True, False], [True, True, False]])
    vertex_labels = jnp.array([1, 0, -1, 10, 1, -1])
    max_components = vertex_vectors.shape[0] * vertex_vectors.shape[1]

    averaged_vectors, was_averaged = average_vector_by_label(
        vertex_vectors, vertex_mask, vertex_labels, max_components
    )

    # check the result
    expected_vectors = jnp.array(
        [
            [
                [0.25, 0.25, 0.25, 0],
                [1, 1, 1, 0],
                [-1, -1, -1, -1],  # padding
            ],
            [
                [10, 10, 10, 0],
                [0.25, 0.25, 0.25, 0],
                [-1, -1, -1, -1],  # padding
            ],
        ]
    )
    np.testing.assert_allclose(averaged_vectors, expected_vectors)

    expected_was_averaged = jnp.array([[True, False, False], [False, True, False]])
    np.testing.assert_array_equal(was_averaged, expected_was_averaged)

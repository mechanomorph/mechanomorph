import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.dcm.forces import find_contacting_vertices
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

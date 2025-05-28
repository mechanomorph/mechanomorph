import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.dcm.utils import pack_mesh_to_cells
from mechanomorph.jax.utils.testing import generate_two_cubes


def test_pack_mesh_to_cells_two_cubes():
    """
    Test mesh packing on two adjacent cubes with shared face.

    Creates two adjacent cubes and checks:
    - Vertices are correctly packed into cells
    - Faces are remapped to local vertex indices
    - Valid masks are set properly
    - No overflow occurs
    """
    (
        all_vertices,
        all_faces,
        vertex_cell_mapping,
        face_cell_mapping,
        vertices_cube1,
        vertices_cube2,
        faces_cube1,
        _,
    ) = generate_two_cubes()

    max_vertices_per_cell = 8
    max_faces_per_cell = 12
    max_cells = 3

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
        max_vertices_per_cell=max_vertices_per_cell,
        max_faces_per_cell=max_faces_per_cell,
        max_cells=max_cells,
    )

    # Validate the vertices (there are only 2 cells)
    np.testing.assert_array_equal(
        vertices_packed[0:2, :], jnp.stack((vertices_cube1, vertices_cube2), axis=0)
    )
    expected_vertices_mask = jnp.concatenate(
        (
            jnp.ones((2, max_vertices_per_cell), dtype=bool),
            jnp.zeros((1, max_vertices_per_cell), dtype=bool),
        )
    )
    np.testing.assert_array_equal(valid_vertices_mask, expected_vertices_mask)

    # validate the faces (there are only 2 cells)
    # note that the faces are remapped to indices in the packed format
    np.testing.assert_array_equal(
        faces_packed[0:2, :], jnp.stack((faces_cube1, faces_cube1), axis=0)
    )
    expected_faces_mask = jnp.concatenate(
        (
            jnp.ones((2, max_faces_per_cell), dtype=bool),
            jnp.zeros((1, max_faces_per_cell), dtype=bool),
        )
    )
    np.testing.assert_array_equal(valid_faces_mask, expected_faces_mask)

    # validate the cells
    np.testing.assert_array_equal(valid_cells_mask, jnp.array([True, True, False]))

    # check that overflow flags
    assert not vertex_overflow
    assert not face_overflow
    assert not cell_overflow


def test_pack_mesh_to_cells_vertex_overflow():
    """Test that the overflow flag is set for vertex overflow."""
    (
        all_vertices,
        all_faces,
        vertex_cell_mapping,
        face_cell_mapping,
        vertices_cube1,
        vertices_cube2,
        faces_cube1,
        _,
    ) = generate_two_cubes()

    max_vertices_per_cell = 7
    max_faces_per_cell = 12
    max_cells = 3

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
        max_vertices_per_cell=max_vertices_per_cell,
        max_faces_per_cell=max_faces_per_cell,
        max_cells=max_cells,
    )

    # Validate the vertices (there are only 2 cells)
    np.testing.assert_array_equal(
        vertices_packed[0:2, :],
        jnp.stack((vertices_cube1[:7, :], vertices_cube2[:7, :]), axis=0),
    )
    expected_vertices_mask = jnp.concatenate(
        (
            jnp.ones((2, max_vertices_per_cell), dtype=bool),
            jnp.zeros((1, max_vertices_per_cell), dtype=bool),
        )
    )
    np.testing.assert_array_equal(valid_vertices_mask, expected_vertices_mask)

    # overflow flag should be set
    assert vertex_overflow
    assert not face_overflow
    assert not cell_overflow


def test_pack_mesh_to_cells_face_overflow():
    """Test that the overflow flag is set for face overflow."""
    (
        all_vertices,
        all_faces,
        vertex_cell_mapping,
        face_cell_mapping,
        vertices_cube1,
        vertices_cube2,
        faces_cube1,
        _,
    ) = generate_two_cubes()

    max_vertices_per_cell = 8
    max_faces_per_cell = 11
    max_cells = 3

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
        max_vertices_per_cell=max_vertices_per_cell,
        max_faces_per_cell=max_faces_per_cell,
        max_cells=max_cells,
    )

    # Validate the vertices (there are only 2 cells)
    np.testing.assert_array_equal(
        vertices_packed[0:2, :], jnp.stack((vertices_cube1, vertices_cube2), axis=0)
    )
    expected_vertices_mask = jnp.concatenate(
        (
            jnp.ones((2, max_vertices_per_cell), dtype=bool),
            jnp.zeros((1, max_vertices_per_cell), dtype=bool),
        )
    )
    np.testing.assert_array_equal(valid_vertices_mask, expected_vertices_mask)

    # validate the faces (there are only 2 cells)
    # note that the faces are remapped to indices in the packed format
    np.testing.assert_array_equal(
        faces_packed[0:2, :],
        jnp.stack(
            (faces_cube1[:max_faces_per_cell, :], faces_cube1[:max_faces_per_cell, :]),
            axis=0,
        ),
    )
    expected_faces_mask = jnp.concatenate(
        (
            jnp.ones((2, max_faces_per_cell), dtype=bool),
            jnp.zeros((1, max_faces_per_cell), dtype=bool),
        )
    )
    np.testing.assert_array_equal(valid_faces_mask, expected_faces_mask)

    # validate the cells
    np.testing.assert_array_equal(valid_cells_mask, jnp.array([True, True, False]))

    # overflow flag should be set
    assert not vertex_overflow
    assert face_overflow
    assert not cell_overflow


def test_pack_mesh_to_cells_cell_overflow():
    """Test that the overflow flag is set for cell overflow."""
    (
        all_vertices,
        all_faces,
        vertex_cell_mapping,
        face_cell_mapping,
        vertices_cube1,
        vertices_cube2,
        faces_cube1,
        _,
    ) = generate_two_cubes()

    max_vertices_per_cell = 8
    max_faces_per_cell = 12
    max_cells = 1

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
        max_vertices_per_cell=max_vertices_per_cell,
        max_faces_per_cell=max_faces_per_cell,
        max_cells=max_cells,
    )

    # Validate the vertices (there is only 1 cell)
    np.testing.assert_array_equal(
        vertices_packed, jnp.expand_dims(vertices_cube1, axis=0)
    )
    expected_vertices_mask = jnp.ones((1, max_vertices_per_cell), dtype=bool)
    np.testing.assert_array_equal(valid_vertices_mask, expected_vertices_mask)

    # validate the faces (there is only 1 cell)
    # note that the faces are remapped to indices in the packed format
    np.testing.assert_array_equal(faces_packed, jnp.expand_dims(faces_cube1, axis=0))
    expected_faces_mask = jnp.ones((1, max_faces_per_cell), dtype=bool)
    np.testing.assert_array_equal(valid_faces_mask, expected_faces_mask)

    # validate the cells
    np.testing.assert_array_equal(
        valid_cells_mask,
        jnp.array(
            [
                True,
            ]
        ),
    )

    # overflow flag should be set
    # since we have only one cell, the vertices and faces overflow too
    assert vertex_overflow
    assert face_overflow
    assert cell_overflow

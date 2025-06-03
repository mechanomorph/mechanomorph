import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.dcm.utils import (
    compute_face_normal_centroid_dot_product,
    detect_aabb_intersections,
    pack_mesh_to_cells,
)
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


def test_compute_face_normal_centroid_dot_product_simple_tetrahedron():
    """Test with a simple tetrahedron with mixed inward and outward facing normals."""
    # Simple tetrahedron with one vertex at origin and three at unit distances
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Define faces with mixed orientations
    faces = jnp.array(
        [
            [2, 1, 0],  # outward facing (counter-clockwise from outside)
            [3, 2, 0],  # outward facing (counter-clockwise from outside)
            [1, 0, 3],  # inward facing (clockwise from outside)
            [2, 1, 3],  # inward facing (clockwise from outside)
        ]
    )

    dot_products = compute_face_normal_centroid_dot_product(vertices, faces)

    # Check basic properties
    assert dot_products.shape == (
        4,
    ), f"Expected 4 dot products, got {dot_products.shape}"

    # Check that first two faces (outward) have positive dot products
    assert jnp.all(dot_products[:2] > 0)

    # Check that last two faces (inward) have negative dot products
    assert jnp.all(dot_products[2:] < 0)


def test_detect_aabb_intersections_two_cubes():
    """Test AABB intersection detection on two adjacent cubes."""
    # make a mesh with three objects.
    # 0, 1 are in contact, 2 is by itself
    vertices_packed = jnp.array(
        [
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 0, 0],  # padding
            ],
            [
                [0, 0, 1.1],
                [0, 0, 2],
                [0, 1, 2],
                [0, 0, 0],  # padding
            ],
            [
                [10, 0, 0],
                [10, 0, 1],
                [10, 1, 1],
                [0, 0, 0],  # padding
            ],
            [
                [0, 0, 0],  # padding
                [0, 0, 0],  # padding
                [0, 0, 0],  # padding
                [0, 0, 0],  # padding
            ],
        ]
    )
    valid_vertices_mask = jnp.array(
        [
            [True, True, True, False],  # first cell
            [True, True, True, False],  # second cell
            [True, True, True, False],  # third cell
            [False, False, False, False],  # padding cell
        ]
    )

    # Make the valid cells mask
    # The last cell is padding.
    valid_cells_mask = jnp.array([True, True, True, False])

    # Check intersection
    (intersecting_pairs, valid_pairs_mask, n_intersecting, bounding_boxes) = (
        detect_aabb_intersections(
            vertices_packed=vertices_packed,
            valid_vertices_mask=valid_vertices_mask,
            valid_cells_mask=valid_cells_mask,
            expansion=0.5,
            max_cells=vertices_packed.shape[0],
            max_cell_pairs=10,
        )
    )

    valid_intersecting_pairs = intersecting_pairs[valid_pairs_mask]
    np.testing.assert_array_equal(valid_intersecting_pairs, jnp.array([[0, 1]]))
    assert n_intersecting == 1

    assert bounding_boxes.shape == (4, 6)  # 3 cells, each with 6 bounding box values
    np.testing.assert_allclose(
        bounding_boxes[0],
        [-0.5, -0.5, -0.5, 0.5, 1.5, 1.5],  # Bounding box for first cube
    )
    np.testing.assert_allclose(
        bounding_boxes[1],
        [-0.5, -0.5, 0.6, 0.5, 1.5, 2.5],  # Bounding box for second cube
    )
    np.testing.assert_allclose(
        bounding_boxes[2],
        [9.5, -0.5, -0.5, 10.5, 1.5, 1.5],  # Bounding box for third cube
    )

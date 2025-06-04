import jax.numpy as jnp
import numpy as np
import pytest

from mechanomorph.jax.dcm.remeshing import remesh_edge_split_single_cell


def _compute_face_normal(vertices, face):
    """Compute normal vector of a face."""
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    return jnp.cross(edge1, edge2)


@pytest.mark.parametrize("face_order", [[0, 1, 2], [2, 1, 0], [1, 0, 2]])
def test_single_edge_splitting_single_cell(face_order):
    """Test splitting a single edge in a single cell mesh."""
    edge_length_threshold = 1.1
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
        ]
    )
    faces = jnp.array(
        [
            face_order,
            [-1, -1, -1],  # padding
        ]
    )

    vertex_mask = jnp.array([True, True, True, False, False, False])
    faces_mask = jnp.array([True, False])
    new_vertices, new_faces, new_vertex_mask, new_face_mask, overflow = (
        remesh_edge_split_single_cell(
            vertices=vertices,
            faces=faces,
            vertex_mask=vertex_mask,
            face_mask=faces_mask,
            edge_length_threshold=edge_length_threshold,
        )
    )

    # no overflow should have occurred
    assert not overflow

    # check that the new vertices are correct
    expected_vertices = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
    )
    np.testing.assert_allclose(new_vertices[new_vertex_mask], expected_vertices)

    # check that a new face was added
    valid_new_faces = new_faces[new_face_mask]
    assert valid_new_faces.shape == (2, 3)

    # verify that the new faces have the same normal
    # as the original face
    original_face_normal = _compute_face_normal(vertices, faces[0])
    new_normals = jnp.array(
        [
            _compute_face_normal(new_vertices, new_faces[0]),
            _compute_face_normal(new_vertices, new_faces[1]),
        ]
    )
    normal_dot_products = jnp.dot(new_normals, original_face_normal)
    assert np.all(normal_dot_products > 0.0)


@pytest.mark.parametrize(
    "face_order,expected_vertex_order",
    [([0, 1, 2], [0, 1]), ([2, 1, 0], [0, 1]), ([1, 0, 2], [1, 0])],
)
def test_two_edge_splitting_single_cell(face_order, expected_vertex_order):
    """Test splitting two edges in a face in a single cell mesh."""
    edge_length_threshold = 1.1
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 5.0, 0.0],
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
        ]
    )
    faces = jnp.array(
        [
            face_order,
            [-1, -1, -1],  # padding
            [-1, -1, -1],  # padding
        ]
    )
    vertex_mask = jnp.array([True, True, True, False, False])
    faces_mask = jnp.array([True, False, False])

    # split the edges
    new_vertices, new_faces, new_vertex_mask, new_face_mask, overflow = (
        remesh_edge_split_single_cell(
            vertices=vertices,
            faces=faces,
            vertex_mask=vertex_mask,
            face_mask=faces_mask,
            edge_length_threshold=edge_length_threshold,
        )
    )

    # no overflow should have occurred
    assert not overflow

    # check the vertices
    expected_new_vertices = [[0.75, 2.5, 0.0], [0.25, 2.5, 0.0]]
    expected_vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 5.0, 0.0],
            expected_new_vertices[expected_vertex_order[0]],
            expected_new_vertices[expected_vertex_order[1]],
        ]
    )
    np.testing.assert_allclose(new_vertices[new_vertex_mask], expected_vertices)

    # check that two faces were added
    valid_new_faces = new_faces[new_face_mask]
    assert valid_new_faces.shape == (3, 3)

    # verify that the new faces have the same normal
    original_face_normal = _compute_face_normal(vertices, faces[0])
    new_normals = jnp.array(
        [
            _compute_face_normal(new_vertices, new_faces[0]),
            _compute_face_normal(new_vertices, new_faces[1]),
            _compute_face_normal(new_vertices, new_faces[2]),
        ]
    )
    normal_dot_products = jnp.dot(new_normals, original_face_normal)
    assert np.all(normal_dot_products > 0.0)


@pytest.mark.parametrize(
    "face_order,expected_vertex_order",
    [([0, 1, 2], [0, 1, 2]), ([2, 1, 0], [1, 0, 2]), ([1, 0, 2], [0, 2, 1])],
)
def test_three_edge_splitting_single_cell(face_order, expected_vertex_order):
    """Test splitting three edges in a face in a single cell mesh."""
    edge_length_threshold = 0.9
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 5.0, 0.0],
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
        ]
    )
    faces = jnp.array(
        [
            face_order,
            [-1, -1, -1],  # padding
            [-1, -1, -1],  # padding
            [-1, -1, -1],  # padding
        ]
    )
    vertex_mask = jnp.array([True, True, True, False, False, False])
    faces_mask = jnp.array([True, False, False, False])

    # split the edges
    new_vertices, new_faces, new_vertex_mask, new_face_mask, overflow = (
        remesh_edge_split_single_cell(
            vertices=vertices,
            faces=faces,
            vertex_mask=vertex_mask,
            face_mask=faces_mask,
            edge_length_threshold=edge_length_threshold,
        )
    )

    # no overflow should have occurred
    assert not overflow

    # check the vertices
    expected_new_vertices = [[0.5, 0.0, 0.0], [0.75, 2.5, 0.0], [0.25, 2.5, 0.0]]
    expected_vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 5.0, 0.0],
            expected_new_vertices[expected_vertex_order[0]],
            expected_new_vertices[expected_vertex_order[1]],
            expected_new_vertices[expected_vertex_order[2]],
        ]
    )
    np.testing.assert_allclose(new_vertices[new_vertex_mask], expected_vertices)

    # check that two faces were added
    valid_new_faces = new_faces[new_face_mask]
    assert valid_new_faces.shape == (4, 3)

    # verify that the new faces have the same normal
    original_face_normal = _compute_face_normal(vertices, faces[0])
    new_normals = jnp.array(
        [
            _compute_face_normal(new_vertices, new_faces[0]),
            _compute_face_normal(new_vertices, new_faces[1]),
            _compute_face_normal(new_vertices, new_faces[2]),
        ]
    )
    normal_dot_products = jnp.dot(new_normals, original_face_normal)
    assert np.all(normal_dot_products > 0.0)


def test_edge_splitting_single_cell_vertex_overflow():
    """Test the overflow flag is set when too many vertices are created."""
    edge_length_threshold = 0.9
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 5.0, 0.0],
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
        ]
    )
    faces = jnp.array(
        [
            [0, 1, 2],
            [-1, -1, -1],  # padding
            [-1, -1, -1],  # padding
            [-1, -1, -1],  # padding
        ]
    )
    vertex_mask = jnp.array([True, True, True, False, False])
    faces_mask = jnp.array([True, False, False, False])

    # split the edges
    new_vertices, new_faces, new_vertex_mask, new_face_mask, overflow = (
        remesh_edge_split_single_cell(
            vertices=vertices,
            faces=faces,
            vertex_mask=vertex_mask,
            face_mask=faces_mask,
            edge_length_threshold=edge_length_threshold,
        )
    )

    # overflow should have occurred
    # (5 vertex slots, 6 needed)
    assert overflow


def test_edge_splitting_single_cell_face_overflow():
    """Test the overflow flag is set when too many faces are created."""
    edge_length_threshold = 0.1
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 5.0, 0.0],
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
        ]
    )
    faces = jnp.array(
        [
            [0, 1, 2],
            [-1, -1, -1],  # padding
            [-1, -1, -1],  # padding
        ]
    )
    vertex_mask = jnp.array([True, True, True, False, False, False])
    faces_mask = jnp.array([True, False, False])

    # split the edges
    new_vertices, new_faces, new_vertex_mask, new_face_mask, overflow = (
        remesh_edge_split_single_cell(
            vertices=vertices,
            faces=faces,
            vertex_mask=vertex_mask,
            face_mask=faces_mask,
            edge_length_threshold=edge_length_threshold,
        )
    )

    # overflow should have occurred
    # (3 face slots, 4 needed)
    assert overflow


def test_edge_splitting_single_cell_shared_edge():
    """Test edge splitting with a shared edge."""
    """Test the overflow flag is set when too many faces are created."""
    edge_length_threshold = 0.9
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.2, 0.0],
            [0.5, -0.2, 0.0],
            [-1.0, -1.0, -1.0],  # padding
            [-1.0, -1.0, -1.0],  # padding
        ]
    )
    faces = jnp.array(
        [
            [0, 1, 2],
            [0, 3, 1],  # shared edge (0, 1)
            [-1, -1, -1],  # padding
            [-1, -1, -1],  # padding
        ]
    )
    vertex_mask = jnp.array([True, True, True, True, False, False])
    faces_mask = jnp.array([True, True, False, False])

    # split the edges
    new_vertices, new_faces, new_vertex_mask, new_face_mask, overflow = (
        remesh_edge_split_single_cell(
            vertices=vertices,
            faces=faces,
            vertex_mask=vertex_mask,
            face_mask=faces_mask,
            edge_length_threshold=edge_length_threshold,
        )
    )

    # check the vertices
    expected_vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.2, 0.0],
            [0.5, -0.2, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(new_vertices[new_vertex_mask], expected_vertices)

    # check that two faces were added
    valid_new_faces = new_faces[new_face_mask]
    assert valid_new_faces.shape == (4, 3)

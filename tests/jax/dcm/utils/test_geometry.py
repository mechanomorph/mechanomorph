import numpy as np

from mechanomorph.jax.dcm.utils import (
    compute_cell_volume,
    compute_cell_volume_packed,
    gradient_cell_volume_wrt_node_positions,
    pack_mesh_to_cells,
)
from mechanomorph.jax.utils.testing import (
    _get_cell_face_adjacency_matrix,
    generate_two_cubes,
)


def test_compute_cell_volume():
    """Test the computation of cell volumes for two cubes."""
    (
        all_vertices,
        _,
        _,
        _,
        _,
        _,
        faces_cube1,
        faces_cube2,
    ) = generate_two_cubes()

    computed_c1_volume = compute_cell_volume(
        all_vertices[faces_cube1[:, 0]],
        all_vertices[faces_cube1[:, 1]],
        all_vertices[faces_cube1[:, 2]],
    )

    computed_c2_volume = compute_cell_volume(
        all_vertices[faces_cube2[:, 0]],
        all_vertices[faces_cube2[:, 1]],
        all_vertices[faces_cube2[:, 2]],
    )

    np.testing.assert_almost_equal(computed_c1_volume, 1.0)
    np.testing.assert_almost_equal(computed_c2_volume, 1.0)


def test_compute_cell_volume_packed():
    """Validate the packed cell volume computation."""
    (
        all_vertices,
        all_faces,
        vertex_cell_mapping,
        face_cell_mapping,
        vertices_cube1,
        _,
        faces_cube1,
        _,
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
        max_cells=2,
    )

    volume_c1 = compute_cell_volume_packed(
        vertex_positions=vertices_packed[0, ...],
        faces=faces_packed[0, ...],
        face_mask=valid_faces_mask[0, ...],
    )
    volume_c2 = compute_cell_volume_packed(
        vertex_positions=vertices_packed[1, ...],
        faces=faces_packed[1, ...],
        face_mask=valid_faces_mask[1, ...],
    )

    np.testing.assert_almost_equal(volume_c1, 1.0)
    np.testing.assert_almost_equal(volume_c2, 1.0)


def test_gradient_cell_volume_wrt_node_positions():
    """
    Check that the computation of the gradient is correct.
    """

    (
        all_vertices,
        _,
        _,
        _,
        _,
        _,
        faces_cube1,
        faces_cube2,
    ) = generate_two_cubes()

    # Get the cell face adjacency matrix
    # get the vertex-face adjacency matrices for cube 1
    (
        c1_face_n1_adjacency_matrices,
        c1_face_n2_adjacency_matrices,
        c1_face_n3_adjacency_matrices,
    ) = _get_cell_face_adjacency_matrix(all_vertices, faces_cube1)

    # get the vertex-face adjacency matrices for cube 2
    (
        c2_face_n1_adjacency_matrices,
        c2_face_n2_adjacency_matrices,
        c2_face_n3_adjacency_matrices,
    ) = _get_cell_face_adjacency_matrix(all_vertices, faces_cube2)

    # Extract the node positions of the faces
    c1_n1_pos_ar = all_vertices[faces_cube1[:, 0]]
    c1_n2_pos_ar = all_vertices[faces_cube1[:, 1]]
    c1_n3_pos_ar = all_vertices[faces_cube1[:, 2]]

    c2_n1_pos_ar = all_vertices[faces_cube2[:, 0]]
    c2_n2_pos_ar = all_vertices[faces_cube2[:, 1]]
    c2_n3_pos_ar = all_vertices[faces_cube2[:, 2]]

    # Compute the gradient of the cell volume wrt the node positions
    computed_c1_grad_vol = gradient_cell_volume_wrt_node_positions(
        c1_n1_pos_ar,
        c1_n2_pos_ar,
        c1_n3_pos_ar,
        c1_face_n1_adjacency_matrices,
        c1_face_n2_adjacency_matrices,
        c1_face_n3_adjacency_matrices,
    )

    computed_c2_grad_vol = gradient_cell_volume_wrt_node_positions(
        c2_n1_pos_ar,
        c2_n2_pos_ar,
        c2_n3_pos_ar,
        c2_face_n1_adjacency_matrices,
        c2_face_n2_adjacency_matrices,
        c2_face_n3_adjacency_matrices,
    )

    # calculate the correct solution manually
    # todo: use an example with a simple analytical solution
    correct_c1_grad_vol = np.zeros_like(all_vertices)
    correct_c2_grad_vol = np.zeros_like(all_vertices)

    for face in faces_cube1:
        n1_id, n2_id, n3_id = face

        n1_pos = all_vertices[n1_id]
        n2_pos = all_vertices[n2_id]
        n3_pos = all_vertices[n3_id]

        n1_grad_vol = np.cross(n2_pos, n3_pos) / 6.0
        n2_grad_vol = np.cross(n3_pos, n1_pos) / 6.0
        n3_grad_vol = np.cross(n1_pos, n2_pos) / 6.0

        correct_c1_grad_vol[n1_id] += n1_grad_vol
        correct_c1_grad_vol[n2_id] += n2_grad_vol
        correct_c1_grad_vol[n3_id] += n3_grad_vol

    for face in faces_cube2:
        n1_id, n2_id, n3_id = face

        n1_pos = all_vertices[n1_id]
        n2_pos = all_vertices[n2_id]
        n3_pos = all_vertices[n3_id]

        n1_grad_vol = np.cross(n2_pos, n3_pos) / 6.0
        n2_grad_vol = np.cross(n3_pos, n1_pos) / 6.0
        n3_grad_vol = np.cross(n1_pos, n2_pos) / 6.0

        correct_c2_grad_vol[n1_id] += n1_grad_vol
        correct_c2_grad_vol[n2_id] += n2_grad_vol
        correct_c2_grad_vol[n3_id] += n3_grad_vol

    # Check that the computed gradients match the expected solution
    np.testing.assert_almost_equal(computed_c1_grad_vol, correct_c1_grad_vol)
    np.testing.assert_almost_equal(computed_c2_grad_vol, correct_c2_grad_vol)

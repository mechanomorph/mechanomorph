import numpy as np

from mechanomorph.jax.dcm.utils import (
    compute_cell_volume,
    gradient_cell_volume_wrt_node_positions,
)
from mechanomorph.jax.utils.testing import TissueMesh, generate_dummy_mesh


def test_compute_cell_volume():
    dummy_vertices, dummy_faces, dummy_face_cells = generate_dummy_mesh()

    # Create a tissue mesh object
    tissue_mesh = TissueMesh(dummy_vertices, dummy_faces, dummy_face_cells)

    # Get the cell face adjacency matrix
    [
        face_n1_adjacency_matrices,
        face_n2_adjacency_matrices,
        face_n3_adjacency_matrices,
    ] = tissue_mesh.cell_adjacency_matrix

    # Extract the face adjacency matrices of the two cells
    c1_face_n1_adjacency_matrices, c2_face_n1_adjacency_matrices = (
        face_n1_adjacency_matrices
    )
    c1_face_n2_adjacency_matrices, c2_face_n2_adjacency_matrices = (
        face_n2_adjacency_matrices
    )
    c1_face_n3_adjacency_matrices, c2_face_n3_adjacency_matrices = (
        face_n3_adjacency_matrices
    )

    # Extract the node positions of the faces
    c1_n1_pos_ar = c1_face_n1_adjacency_matrices @ dummy_vertices
    c1_n2_pos_ar = c1_face_n2_adjacency_matrices @ dummy_vertices
    c1_n3_pos_ar = c1_face_n3_adjacency_matrices @ dummy_vertices

    c2_n1_pos_ar = c2_face_n1_adjacency_matrices @ dummy_vertices
    c2_n2_pos_ar = c2_face_n2_adjacency_matrices @ dummy_vertices
    c2_n3_pos_ar = c2_face_n3_adjacency_matrices @ dummy_vertices

    computed_c1_volume = compute_cell_volume(c1_n1_pos_ar, c1_n2_pos_ar, c1_n3_pos_ar)

    computed_c2_volume = compute_cell_volume(c2_n1_pos_ar, c2_n2_pos_ar, c2_n3_pos_ar)

    np.testing.assert_almost_equal(computed_c1_volume, 1.0)
    np.testing.assert_almost_equal(computed_c2_volume, 1.0)


def test_gradient_cell_volume_wrt_node_positions():
    """
    Check that the computation of the gradient is correct.
    """

    dummy_vertices, dummy_faces, dummy_face_cells = generate_dummy_mesh()

    # Create a tissue mesh object
    tissue_mesh = TissueMesh(dummy_vertices, dummy_faces, dummy_face_cells)

    # Get the cell face adjacency matrix
    [
        face_n1_adjacency_matrices,
        face_n2_adjacency_matrices,
        face_n3_adjacency_matrices,
    ] = tissue_mesh.cell_adjacency_matrix

    # Extract the face adjacency matrices of the two cells
    c1_face_n1_adjacency_matrices, c2_face_n1_adjacency_matrices = (
        face_n1_adjacency_matrices
    )
    c1_face_n2_adjacency_matrices, c2_face_n2_adjacency_matrices = (
        face_n2_adjacency_matrices
    )
    c1_face_n3_adjacency_matrices, c2_face_n3_adjacency_matrices = (
        face_n3_adjacency_matrices
    )

    # Extract the node positions of the faces
    c1_n1_pos_ar = c1_face_n1_adjacency_matrices @ dummy_vertices
    c1_n2_pos_ar = c1_face_n2_adjacency_matrices @ dummy_vertices
    c1_n3_pos_ar = c1_face_n3_adjacency_matrices @ dummy_vertices

    c2_n1_pos_ar = c2_face_n1_adjacency_matrices @ dummy_vertices
    c2_n2_pos_ar = c2_face_n2_adjacency_matrices @ dummy_vertices
    c2_n3_pos_ar = c2_face_n3_adjacency_matrices @ dummy_vertices

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

    correct_c1_grad_vol = np.zeros_like(dummy_vertices)
    correct_c2_grad_vol = np.zeros_like(dummy_vertices)

    # Get all the faces that belong to cell 1 and cell 2.
    # The order of the nodes have to be flipped in certain faces
    # to make sure that the normals always point outwards.
    c1_faces = np.vstack(
        [
            dummy_faces[dummy_face_cells[:, 0] == 1],
            dummy_faces[dummy_face_cells[:, 1] == 1][:, [0, 2, 1]],
        ]
    )
    c2_faces = np.vstack(
        [
            dummy_faces[dummy_face_cells[:, 0] == 2],
            dummy_faces[dummy_face_cells[:, 1] == 2][:, [0, 2, 1]],
        ]
    )

    for face in c1_faces:
        n1_id, n2_id, n3_id = face

        n1_pos = dummy_vertices[n1_id]
        n2_pos = dummy_vertices[n2_id]
        n3_pos = dummy_vertices[n3_id]

        n1_grad_vol = np.cross(n2_pos, n3_pos) / 6.0
        n2_grad_vol = np.cross(n3_pos, n1_pos) / 6.0
        n3_grad_vol = np.cross(n1_pos, n2_pos) / 6.0

        correct_c1_grad_vol[n1_id] += n1_grad_vol
        correct_c1_grad_vol[n2_id] += n2_grad_vol
        correct_c1_grad_vol[n3_id] += n3_grad_vol

    for face in c2_faces:
        n1_id, n2_id, n3_id = face

        n1_pos = dummy_vertices[n1_id]
        n2_pos = dummy_vertices[n2_id]
        n3_pos = dummy_vertices[n3_id]

        n1_grad_vol = np.cross(n2_pos, n3_pos) / 6.0
        n2_grad_vol = np.cross(n3_pos, n1_pos) / 6.0
        n3_grad_vol = np.cross(n1_pos, n2_pos) / 6.0

        correct_c2_grad_vol[n1_id] += n1_grad_vol
        correct_c2_grad_vol[n2_id] += n2_grad_vol
        correct_c2_grad_vol[n3_id] += n3_grad_vol

    # Check that the computed gradients match the expected solution
    np.testing.assert_almost_equal(computed_c1_grad_vol, correct_c1_grad_vol)
    np.testing.assert_almost_equal(computed_c2_grad_vol, correct_c2_grad_vol)

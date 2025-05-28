import jax
import numpy as np

from mechanomorph.jax.dcm.forces import (
    compute_cell_pressure_forces,
)
from mechanomorph.jax.dcm.utils import (
    gradient_cell_volume_wrt_node_positions,
    pack_mesh_to_cells,
)
from mechanomorph.jax.utils.testing import (
    _get_cell_face_adjacency_matrix,
    generate_two_cubes,
)

jax.config.update("jax_enable_x64", True)


def test_compute_pressure_forces_vmap():
    """
    Check that the computation of the pressure forces is correct.
    """
    # Define the cell pressures
    cell_pressure_ar = np.array([100.0, 200.0])  # Pa
    bulk_modulus = 2500.0  # Pa
    initial_cell_volume_ar = np.array([1.5, 1.0])  # m^3
    target_cell_volumes = initial_cell_volume_ar * np.exp(
        cell_pressure_ar / bulk_modulus
    )

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
        max_cells=2,
    )

    # verify the wasn't overflow during packing
    assert not vertex_overflow
    assert not face_overflow
    assert not cell_overflow

    # JIT-compiled single-cell pressure force computation
    jit_compute_cell_pressure_forces_dense = jax.jit(compute_cell_pressure_forces)

    # Vectorized computation across all cells
    vmap_compute_cell_pressure_forces_dense = jax.vmap(
        jit_compute_cell_pressure_forces_dense,
        in_axes=(0, 0, 0, 0, 0, None),
        out_axes=0,
    )

    # compute the forces
    per_cell_forces = vmap_compute_cell_pressure_forces_dense(
        vertices_packed,
        valid_vertices_mask,
        faces_packed,
        valid_faces_mask,
        target_cell_volumes,
        bulk_modulus,
    )

    # now calculate the expected forces.
    # todo: make expected values from the analytical solution

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

    # combine the adjacency matrices
    face_n1_adjacency_matrices = [
        c1_face_n1_adjacency_matrices,
        c2_face_n1_adjacency_matrices,
    ]
    face_n2_adjacency_matrices = [
        c1_face_n2_adjacency_matrices,
        c2_face_n2_adjacency_matrices,
    ]
    face_n3_adjacency_matrices = [
        c1_face_n3_adjacency_matrices,
        c2_face_n3_adjacency_matrices,
    ]

    # Compute the correct pressure forces
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
    c1_n1_pos_ar = c1_face_n1_adjacency_matrices @ all_vertices
    c1_n2_pos_ar = c1_face_n2_adjacency_matrices @ all_vertices
    c1_n3_pos_ar = c1_face_n3_adjacency_matrices @ all_vertices

    c2_n1_pos_ar = c2_face_n1_adjacency_matrices @ all_vertices
    c2_n2_pos_ar = c2_face_n2_adjacency_matrices @ all_vertices
    c2_n3_pos_ar = c2_face_n3_adjacency_matrices @ all_vertices

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

    c1_pressure_force = (
        -bulk_modulus * np.log(1.0 / target_cell_volumes[0]) * computed_c1_grad_vol
    )
    c2_pressure_force = (
        -bulk_modulus * np.log(1.0 / target_cell_volumes[1]) * computed_c2_grad_vol
    )
    correct_pressure_forces = c1_pressure_force + c2_pressure_force

    # Make sure that the arrays are equal
    np.testing.assert_array_almost_equal(
        np.concatenate(per_cell_forces), correct_pressure_forces
    )

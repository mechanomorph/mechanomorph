import jax
import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.dcm.forces import compute_cell_surface_tension_forces
from mechanomorph.jax.dcm.utils import pack_mesh_to_cells
from mechanomorph.jax.utils.testing import generate_two_cubes


def test_compute_surface_tension():
    """Test computing surface tension batched over cells."""
    # Generate a mesh with two cubes
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

    # parameters
    min_norm = 1e-10
    face_surface_tension_packed = jnp.ones(
        (faces_packed.shape[0], faces_packed.shape[1])
    )

    # make the batched and JIT-ed function
    batched_surface_tension_forces = jax.vmap(
        compute_cell_surface_tension_forces,
        in_axes=(0, 0, 0, 0, 0, None),
        out_axes=0,
    )
    jit_batched_surface_tension_forces = jax.jit(batched_surface_tension_forces)

    # Compute the surface tension forces for each cell
    surface_tensions = jit_batched_surface_tension_forces(
        vertices_packed,
        faces_packed,
        valid_vertices_mask,
        valid_faces_mask,
        face_surface_tension_packed,
        min_norm,
    )

    # Compute the correct surface tension forces
    face_surface_tension_ar = np.concatenate(face_surface_tension_packed)
    correct_surface_tension_ar = np.zeros_like(all_vertices)

    for face_id, face in enumerate(all_faces):
        n1_id, n2_id, n3_id = face

        n1_pos = all_vertices[n1_id]
        n2_pos = all_vertices[n2_id]
        n3_pos = all_vertices[n3_id]

        # Compute the unit normal vector of the face
        normal_vector = np.cross(n2_pos - n1_pos, n3_pos - n1_pos)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # Compute the surface tension force
        n1_grad_face_area = np.cross(normal_vector, n3_pos - n2_pos) / 2.0
        n2_grad_face_area = np.cross(normal_vector, n1_pos - n3_pos) / 2.0
        n3_grad_face_area = np.cross(normal_vector, n2_pos - n1_pos) / 2.0

        correct_surface_tension_ar[n1_id] -= (
            face_surface_tension_ar[face_id] * n1_grad_face_area
        )
        correct_surface_tension_ar[n2_id] -= (
            face_surface_tension_ar[face_id] * n2_grad_face_area
        )
        correct_surface_tension_ar[n3_id] -= (
            face_surface_tension_ar[face_id] * n3_grad_face_area
        )

    # Check that the computed surface tension forces match the analytical solution
    np.testing.assert_almost_equal(
        np.concatenate(surface_tensions), correct_surface_tension_ar
    )

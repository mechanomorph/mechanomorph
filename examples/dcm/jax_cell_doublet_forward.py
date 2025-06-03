"""Forward simulation of cell doublet.

Each cell has a pressure and surface tension. The simulation
starts out with two cubes then they relax to equilibrium.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import napari
import numpy as np
from jax import Array as JaxArray

from mechanomorph.jax.dcm.forces import (
    average_vector_by_label,
    compute_cell_pressure_forces,
    compute_cell_surface_tension_forces,
    find_contacting_vertices,
    label_vertices,
)
from mechanomorph.jax.dcm.utils import detect_aabb_intersections, pack_mesh_to_cells
from mechanomorph.mesh import make_cube_doublet


class SimulationState(NamedTuple):
    """Container for all simulation state variables.

    Attributes
    ----------
    vertices_packed : jax.Array
        (max_cells, max_vertices_per_cell, 3) vertex coordinates.
    faces_packed : jax.Array
        (max_cells, max_faces_per_cell, 3) vertex indices per face.
    valid_vertices_mask : jax.Array
        (max_cells, max_vertices_per_cell) boolean validity mask.
    valid_faces_mask : jax.Array
        (max_cells, max_faces_per_cell) boolean validity mask.
    valid_cells_mask : jax.Array
        (max_cells,) boolean validity mask.
    face_properties : jax.Array
        (max_cells, max_faces_per_cell) per-face properties.
    cell_properties : jax.Array
        (max_cells,) per-cell properties.
    """

    vertices_packed: jax.Array
    faces_packed: jax.Array
    valid_vertices_mask: jax.Array
    valid_faces_mask: jax.Array
    valid_cells_mask: jax.Array
    surface_tensions: jax.Array
    pressures: jax.Array


# make versions that are batched over the cells
batched_compute_cell_pressure_forces = jax.vmap(
    compute_cell_pressure_forces,
    in_axes=(0, 0, 0, 0, 0, None),
    out_axes=0,
)
batched_surface_tension_forces = jax.vmap(
    compute_cell_surface_tension_forces,
    in_axes=(0, 0, 0, 0, 0, None),
    out_axes=0,
)


def calculate_forces(
    simulation_state: SimulationState,
    target_cell_volumes: JaxArray,
    bulk_modulus: float,
    min_norm: float = 1e-10,
) -> JaxArray:
    """Calculate forces acting on vertices based on current simulation state.

    Parameters
    ----------
    simulation_state : SimulationState
        Current state of the simulation containing vertices, faces, and properties.
    target_cell_volumes : JaxArray
        (max_cells,) target volumes for each cell used to compute pressure forces.
    bulk_modulus : float
        The bulk modulus used to compute pressure forces.
    min_norm : float
        Minimum norm for surface tension forces to avoid division by zero.

    Returns
    -------
    forces : jax.Array
        (max_cells, max_vertices_per_cell, 3) forces acting on each vertex.
    """
    pressure_forces = batched_compute_cell_pressure_forces(
        simulation_state.vertices_packed,
        simulation_state.valid_vertices_mask,
        simulation_state.faces_packed,
        simulation_state.valid_faces_mask,
        target_cell_volumes,
        bulk_modulus,
    )

    surface_tension_forces = batched_surface_tension_forces(
        simulation_state.vertices_packed,
        simulation_state.faces_packed,
        simulation_state.valid_vertices_mask,
        simulation_state.valid_faces_mask,
        simulation_state.surface_tensions,
        min_norm,
    )

    return pressure_forces + surface_tension_forces


def process_contacts(
    vertices: jnp.ndarray,
    vertex_mask: jnp.ndarray,
    cell_contact_pairs: jnp.ndarray,
    cell_contact_mask: jnp.ndarray,
    distance_threshold: float,
    max_contacts: int = 10000,
    max_components: int = 1000,
    max_iterations: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process contacts between cells: detect, label, and average.

    Parameters
    ----------
    vertices : jnp.ndarray
        Array of shape (n_cells, max_vertices_per_cell, 3) containing
        vertex positions.
    vertex_mask : jnp.ndarray
        Boolean array of shape (n_cells, max_vertices_per_cell) for
        valid vertices.
    cell_contact_pairs : jnp.ndarray
        Padded array of shape (max_contact_pairs, 2) with potentially
        contacting cells.
    cell_contact_mask : jnp.ndarray
        Boolean array of shape (max_contact_pairs,) indicating which
        cell contact pairs are valid.
    distance_threshold : float
        Maximum distance for contact detection.
    max_contacts : int
        Maximum number of contacts (compile-time constant).
    max_components : int
        Maximum number of components (compile-time constant).
    max_iterations : int
        Maximum union-find iterations (compile-time constant).

    Returns
    -------
    contact_pairs : jnp.ndarray
        Array of shape (max_contacts, 2) with contacting vertex pairs.
    contact_mask : jnp.ndarray
        Boolean array of shape (max_contacts,) for valid contacts.
    vertex_labels : jnp.ndarray
        Array of shape (n_total_vertices,) with component labels.
    averaged_vectors : jnp.ndarray
        Averaged vectors of shape (n_cells, max_vertices_per_cell, vector_dim).
    was_averaged : jnp.ndarray
        Boolean array of shape (n_cells, max_vertices_per_cell).
    """
    # find contacting vertices
    contact_pairs, contact_mask, distances = find_contacting_vertices(
        vertices,
        vertex_mask,
        cell_contact_pairs,
        cell_contact_mask,
        distance_threshold,
        max_contacts,
    )

    # label connected components
    n_total_vertices = vertices.shape[0] * vertices.shape[1]
    vertex_labels, is_contacting = label_vertices(
        contact_pairs, contact_mask, n_total_vertices, max_iterations
    )

    # average vectors
    averaged_vectors, was_averaged = average_vector_by_label(
        vertices, vertex_mask, vertex_labels, max_components
    )

    return contact_pairs, contact_mask, vertex_labels, averaged_vectors, was_averaged


def make_forward_simulation(time_step: float, n_iterations: int):
    """Create a JIT-compiled forward simulation.

    Parameters
    ----------
    time_step : float
        Time step for integration (compile-time constant).
    n_iterations : int
        Number of iterations to run (compile-time constant).

    Returns
    -------
    callable
        JIT-compiled simulation function that returns final state.
    """

    @jax.jit
    def forward_simulation(
        vertices_packed: jax.Array,
        faces_packed: jax.Array,
        valid_vertices_mask: jax.Array,
        valid_faces_mask: jax.Array,
        valid_cells_mask: jax.Array,
        surface_tensions: jax.Array,
        pressures: jax.Array,
    ) -> SimulationState:
        """Run the forward simulation.

        Parameters
        ----------
        vertices_packed : jax.Array
            (max_cells, max_vertices_per_cell, 3) initial vertex coordinates.
        faces_packed : jax.Array
            (max_cells, max_faces_per_cell, 3) face connectivity.
        valid_vertices_mask : jax.Array
            (max_cells, max_vertices_per_cell) vertex validity.
        valid_faces_mask : jax.Array
            (max_cells, max_faces_per_cell) face validity.
        valid_cells_mask : jax.Array
            (max_cells,) cell validity.
        surface_tensions : jax.Array
            (max_cells, max_faces_per_cell) per-face properties.
        pressures : jax.Array
            (max_cells,) per-cell properties.

        Returns
        -------
        final_state : SimulationState
            Final simulation state after all iterations.
        """
        # Bundle initial state
        initial_state = SimulationState(
            vertices_packed=vertices_packed,
            faces_packed=faces_packed,
            valid_vertices_mask=valid_vertices_mask,
            valid_faces_mask=valid_faces_mask,
            valid_cells_mask=valid_cells_mask,
            surface_tensions=surface_tensions,
            pressures=pressures,
        )

        def scan_body(
            carry: SimulationState, step_idx: int
        ) -> tuple[SimulationState, None]:
            """Single iteration of the simulation loop.

            Parameters
            ----------
            carry : SimulationState
                Current simulation state.
            step_idx : int
                Current iteration index (unused but required by scan).

            Returns
            -------
            updated_state : SimulationState
                Updated state after one iteration.
            output : None
                No per-step output.
            """
            state = carry

            # compute the contacts
            (intersecting_pairs, valid_pairs_mask, n_intersecting, bounding_boxes) = (
                detect_aabb_intersections(
                    vertices_packed=vertices_packed,
                    valid_vertices_mask=valid_vertices_mask,
                    valid_cells_mask=valid_cells_mask,
                    expansion=2e-6,
                    max_cells=2,
                    max_cell_pairs=2,
                )
            )

            (
                contact_pairs,
                contact_mask,
                vertex_labels,
                averaged_vertices,
                was_averaged,
            ) = process_contacts(
                vertices_packed,
                valid_vertices_mask,
                intersecting_pairs,
                valid_pairs_mask,
                1e-6,
                max_contacts=1000,
                max_components=10,
                max_iterations=5,
            )

            # update the state with averaged vertices
            state._replace(
                vertices_packed=jnp.where(
                    state.valid_vertices_mask[..., None],
                    averaged_vertices,
                    state.vertices_packed,
                )
            )

            # Calculate forces
            forces = calculate_forces(
                state,
                target_cell_volumes=jnp.array([(30e-6) ** 3, (30e-6) ** 3]),
                bulk_modulus=2500.0,
                min_norm=1e-10,
            )

            # average the forces of the contacting vertices
            forces_with_contacts, _ = average_vector_by_label(
                vertex_vectors=forces,
                vertex_mask=state.valid_vertices_mask,
                vertex_labels=vertex_labels,
                max_components=10000,
            )

            # Integrate vertex positions
            new_vertices = state.vertices_packed + time_step * forces_with_contacts
            state = state._replace(
                vertices_packed=jnp.where(
                    state.valid_vertices_mask[..., None],
                    new_vertices,
                    state.vertices_packed,
                )
            )

            return state, None

        # Run simulation loop
        final_state, _ = jax.lax.scan(
            scan_body, initial_state, jnp.arange(n_iterations)
        )

        return final_state

    return forward_simulation


if __name__ == "__main__":
    edge_width = 30
    target_area = 7
    grid_size = 1e-6
    time_step = 30.0
    n_forward_iterations = 2000

    # mechanical properties
    surface_tension_value = 0.009
    pressures_value = 100

    # make the mesh
    vertices, faces, vertex_cell_mapping, face_cell_mapping, _ = make_cube_doublet(
        edge_width=edge_width,
        target_area=target_area,
    )

    # convert the vertices to the physical units size
    vertices = vertices * grid_size

    # convert to the packed format.
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
        vertices=jnp.array(vertices),
        faces=jnp.array(faces),
        vertex_cell_mapping=jnp.array(vertex_cell_mapping),
        face_cell_mapping=jnp.array(face_cell_mapping),
        max_vertices_per_cell=int(vertices.shape[0] / 2),
        max_faces_per_cell=int(faces.shape[0] / 2),
        max_cells=2,
    )

    # verify the wasn't overflow during packing
    assert not vertex_overflow
    assert not face_overflow
    assert not cell_overflow

    # make the mechanical properties arrays
    surface_tensions = surface_tension_value * jnp.ones(
        (faces_packed.shape[0], faces_packed.shape[1]), dtype=float
    )
    pressures = pressures_value * jnp.ones((2,))

    # make the simulation function
    simulate = make_forward_simulation(
        time_step=time_step, n_iterations=n_forward_iterations
    )

    # make the simulation
    print("\nRunning simulation...")
    start_time = time.time()
    final_state = simulate(
        vertices_packed,
        faces_packed,
        valid_vertices_mask,
        valid_faces_mask,
        valid_cells_mask,
        surface_tensions,
        pressures,
    )
    # ensure all computations are complete
    final_state.vertices_packed.block_until_ready()

    # compute the total simulation time
    simulation_time = time.time() - start_time
    iteration_time = simulation_time / n_forward_iterations
    print(f"iteration time: {iteration_time} s")

    # make the clipping planes
    plane_parameters_neg = {
        "position": (-edge_width / 2, edge_width / 2, edge_width / 2),
        "normal": (1, 0, 0),
        "enabled": True,
    }

    plane_parameters_pos = {
        "position": (edge_width / 2, edge_width / 2, edge_width / 2),
        "normal": (-1, 0, 0),
        "enabled": True,
    }

    viewer = napari.Viewer()
    initial_mesh = viewer.add_surface(
        (vertices / grid_size, faces),
        experimental_clipping_planes=[plane_parameters_neg, plane_parameters_pos],
    )
    initial_mesh.wireframe.visible = True
    initial_mesh.normals.face.visible = True

    # visualize the final layer
    final_vertices = np.concatenate(final_state.vertices_packed)
    final_mesh = viewer.add_surface(
        (final_vertices / grid_size, faces),
        experimental_clipping_planes=[plane_parameters_neg, plane_parameters_pos],
    )
    final_mesh.wireframe.visible = True
    final_mesh.normals.face.visible = True

    viewer.dims.ndisplay = 3

    napari.run()

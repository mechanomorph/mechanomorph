"""Forward simulation of a single cell.

The cell has an internal pressure and surface tension.
The cell starts out shaped like a cube and then relaxes
to a sphere due to the surface tension and pressure forces.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import napari
import numpy as np
from jax import Array as JaxArray

from mechanomorph.jax.dcm.forces import (
    compute_cell_pressure_forces,
    compute_cell_surface_tension_forces,
)
from mechanomorph.jax.dcm.utils import pack_mesh_to_cells
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
        target_cell_volumes: jax.Array,
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
            (max_cells, max_faces_per_cell) surface tension value for each face.
        pressures : jax.Array
            (max_cells,) pressure for each cell.
        target_cell_volumes : jax.Array
            (max_cells,) target volume for each cell.

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
            """Single time step of the simulation loop.

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

            # Calculate forces
            forces = calculate_forces(
                state,
                target_cell_volumes=target_cell_volumes,
                bulk_modulus=2500.0,
                min_norm=1e-10,
            )

            # Integrate vertex positions
            new_vertices = state.vertices_packed + time_step * forces
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

    # take the vertices from the first cell only
    vertices = vertices[vertex_cell_mapping == 0]
    faces = faces[face_cell_mapping == 0]

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
        vertex_cell_mapping=jnp.zeros(vertices.shape[0], dtype=int),
        face_cell_mapping=jnp.zeros(faces.shape[0], dtype=int),
        max_vertices_per_cell=int(vertices.shape[0]),
        max_faces_per_cell=int(faces.shape[0]),
        max_cells=1,
    )

    # verify the wasn't overflow during packing
    assert not vertex_overflow
    assert not face_overflow
    assert not cell_overflow

    # make the mechanical properties arrays
    surface_tensions = surface_tension_value * jnp.ones(
        (faces_packed.shape[0], faces_packed.shape[1]), dtype=float
    )
    pressures = pressures_value * jnp.ones((1,))
    target_cell_volumes = jnp.array([(edge_width * grid_size) ** 3])

    # make the simulation function
    simulate = make_forward_simulation(
        time_step=time_step, n_iterations=n_forward_iterations
    )

    # make the simulation
    print("Running simulation...")
    start_time = time.time()
    final_state = simulate(
        vertices_packed,
        faces_packed,
        valid_vertices_mask,
        valid_faces_mask,
        valid_cells_mask,
        surface_tensions,
        pressures,
        target_cell_volumes=target_cell_volumes,
    )

    # visualize the result with napari
    print("Launching napari...")
    viewer = napari.Viewer()
    initial_mesh = viewer.add_surface(
        (vertices / grid_size, faces),
        name="Initial Mesh",
    )
    initial_mesh.wireframe.visible = True

    # visualize the final layer
    final_vertices = np.concatenate(final_state.vertices_packed)
    final_mesh = viewer.add_surface(
        (final_vertices / grid_size, faces),
        name="Final Mesh",
    )
    final_mesh.wireframe.visible = True

    viewer.dims.ndisplay = 3

    napari.run()

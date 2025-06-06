"""Demonstration of agents moving randomly with adhesion."""

from functools import partial

import jax
import jax.numpy as jnp
import napari
import numpy as np
from jax import Array as JaxArray
from scipy.ndimage import distance_transform_edt
from tqdm import trange

from mechanomorph.jax.agent.forces import (
    biased_random_locomotion_force,
    cell_boundary_adhesion_potential,
    cell_boundary_repulsion_potential,
    cell_cell_adhesion_potential,
    cell_cell_repulsion_potential,
)
from mechanomorph.jax.data import (
    sample_scalar_field_linear,
    sample_scalar_field_nearest,
    sample_vector_field_nearest,
)


def random_3d_unit_vectors(n_vectors: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Generate an array of random 3D unit vectors in JAX.

    Parameters
    ----------
    n_vectors : int
        The number of random unit vectors to generate.
    key : jax.random.KeyArray
        A PRNG key for random number generation.

    Returns
    -------
    vectors : jnp.ndarray
        An array of shape (n_vectors, 3) containing random unit vectors.
    """
    vectors = jax.random.uniform(key, shape=(n_vectors, 3), minval=-1.0, maxval=1.0)
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def position_random_uniform(
    n_positions: int,
    lower_left_coordinate: tuple[float, float, float],
    upper_right_coordinate: tuple[float, float, float],
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Generate random positions in 3D space using JAX.

    This draws from a uniform distribution in a bounding box.

    Parameters
    ----------
    n_positions : int
        The number of random positions to generate.
    lower_left_coordinate : tuple[float, float, float]
        The lower left coordinate of the bounding box.
    upper_right_coordinate : tuple[float, float, float]
        The upper right coordinate of the bounding box.
    key : jax.random.KeyArray
        A PRNG key for random number generation.

    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_positions, 3) containing the random positions.
    """
    ll = jnp.array(lower_left_coordinate)
    ur = jnp.array(upper_right_coordinate)
    key, subkey = jax.random.split(key)
    rand = jax.random.uniform(subkey, shape=(n_positions, 3))
    positions = rand * (ur - ll) + ll
    return positions


def bounding_box_lines(
    lower_left_coordinate: tuple[float, float, float],
    upper_right_coordinate: tuple[float, float, float],
) -> np.ndarray:
    """Get the edges of a bounding box.

    Parameters
    ----------
    lower_left_coordinate : tuple[float, float, float]
        The lower left coordinate of the bounding box.
    upper_right_coordinate : tuple[float, float, float]
        The upper right coordinate of the bounding box.

    Returns
    -------
    lines : np.ndarray
        An array of shape (12, 2, 3) containing the start and end points of each line.
    """
    # Extract coordinates
    x_min, y_min, z_min = lower_left_coordinate
    x_max, y_max, z_max = upper_right_coordinate

    # Define all 8 vertices of the box
    vertices = np.array(
        [
            [x_min, y_min, z_min],  # 0: lower left front
            [x_max, y_min, z_min],  # 1: lower right front
            [x_max, y_max, z_min],  # 2: upper right front
            [x_min, y_max, z_min],  # 3: upper left front
            [x_min, y_min, z_max],  # 4: lower left back
            [x_max, y_min, z_max],  # 5: lower right back
            [x_max, y_max, z_max],  # 6: upper right back
            [x_min, y_max, z_max],  # 7: upper left back
        ]
    )

    # Define the 12 edges by vertex indices (start, end)
    edge_indices = [
        # Front face
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # Back face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # Connecting edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Create the lines array with shape (12, 2, 3)
    lines = np.zeros((12, 2, 3))

    for i, (start_idx, end_idx) in enumerate(edge_indices):
        lines[i, 0] = vertices[start_idx]  # Start point
        lines[i, 1] = vertices[end_idx]  # End point

    return lines


def field_gradient(field: jnp.ndarray) -> jnp.ndarray:
    """Compute the unit vector gradient field using JAX.

    Parameters
    ----------
    field : jnp.ndarray
        (h, w, d) array containing the scalar field to compute the gradient of.

    Returns
    -------
    gradient_vectors : jnp.ndarray
        (3, h, w, d) array containing the gradient vector field. All vectors
        are unit vectors.
    """
    # Compute gradients along each axis
    axis_gradients = jnp.gradient(field, axis=(0, 1, 2))  # tuple of 3 arrays

    # Stack into (3, h, w, d)
    gradients = jnp.stack(axis_gradients, axis=0)

    # Normalize along the vector axis (axis=0)
    norm = (
        jnp.linalg.norm(gradients, axis=0, keepdims=True) + 1e-8
    )  # prevent div-by-zero
    unit_gradients = gradients / norm

    return unit_gradients


def make_domain(
    lower_left_coordinate: tuple[int, int, int],
    upper_right_coordinate: tuple[int, int, int],
    shape: tuple[int, int, int] = (50, 50, 50),
) -> tuple[jax.Array, jax.Array]:
    """Make a domain for the simulation.

    Parameters
    ----------
    lower_left_coordinate : tuple[int, int, int]
        The lower left coordinate of the domain (voxel indices).
    upper_right_coordinate : tuple[int, int, int]
        The upper right coordinate of the domain (voxel indices).
    shape : tuple[int, int, int]
        The shape of the array the domain is embedded in.

    Returns
    -------
    distance_field : jax.Array
        The field of distances to the boundary.
    normals_field : VectorField
        (3, h, w, d) vector field of normals to the boundary.
    """
    domain_segmentation = np.zeros(shape, dtype=bool)
    domain_segmentation[
        lower_left_coordinate[0] : upper_right_coordinate[0],
        lower_left_coordinate[1] : upper_right_coordinate[1],
        lower_left_coordinate[2] : upper_right_coordinate[2],
    ] = True

    distances = distance_transform_edt(domain_segmentation)
    distance_field = jnp.array(distances, dtype=jnp.float32)

    normals_field = field_gradient(distance_field)

    return distance_field, normals_field


@jax.jit
def compute_forces(
    positions: JaxArray,
    cell_adhesion_radii: JaxArray,
    cell_adhesion_strength: JaxArray,
    cell_repulsion_radii: JaxArray,
    cell_repulsion_strength: JaxArray,
    boundary_adhesion_radii: JaxArray,
    boundary_adhesion_strength: JaxArray,
    boundary_repulsion_radii: JaxArray,
    boundary_repulsion_strength: JaxArray,
    boundary_distances: JaxArray,
    boundary_normal_vectors: JaxArray,
    previous_locomotion_direction: JaxArray,
    locomotion_speed: JaxArray,
    rng_key: jax.random.PRNGKey,
) -> tuple[JaxArray, JaxArray]:
    """Compute the forces at a given time step.

    This includes random locomotion, cell-cell repulsion,
    and cell-boundary repulsion forces.

    Parameters
    ----------
    positions : JaxArray
        The current position of each agent.
    cell_adhesion_radii : JaxArray
        (n_agents,) array of the cell-cell interaction radius of each agent.
    cell_adhesion_strength : JaxArray
        (n_agents,) array of the cell-cell adhesion strength of each agent.
    cell_repulsion_radii : JaxArray
        (n_agents,) array of the cell-cell interaction radius of each agent.
    cell_repulsion_strength : JaxArray
        (n_agents,) array of the cell-cell repulsion strength of each agent.
    boundary_adhesion_radii : JaxArray
        (n_agents,) array of the cell-boundary interaction radius of each agent.
    boundary_adhesion_strength : JaxArray
        (n_agents,) array of the cell-boundary adhesion strength of each agent.
    boundary_repulsion_radii : JaxArray
        (n_agents,) array of the boundary interaction radius of each agent.
    boundary_repulsion_strength : JaxArray
        (n_agents,) array of the repulsion strength of each agent.
    boundary_distances : JaxArray
        (n_agents,) array of the distance to the boundary for each agent.
    boundary_normal_vectors : JaxArray
        (n_agents, 3) array of the boundary normal vector for each agent.
    previous_locomotion_direction : JaxArray
        (n_agents, 3) array of the locomotion direction of each agent.
    locomotion_speed : JaxArray
        (n_agents,) array of the locomotion speed of each agent.
    rng_key : jax.random.PRNGKey
        The random number generator key for stochastic processes.

    Returns
    -------
    forces : torch.Tensor
        (n_agents, 3) array of the force vector for each agent.
    locomotion_direction : torch.Tensor
        (n_agents, 3) array of the locomotion direction of each agent.
    """
    n_agents = positions.shape[0]
    valid_agents_mask = jnp.ones((n_agents,), dtype=bool)

    # compute the locomotion force
    locomotion_direction_change_probability = 0.01 * jnp.ones((n_agents,))
    locomotion_forces, updated_locomotion_direction = biased_random_locomotion_force(
        previous_direction=previous_locomotion_direction,
        direction_change_probability=locomotion_direction_change_probability,
        bias_direction=jnp.zeros((n_agents, 3)),
        bias_constant=jnp.zeros((n_agents,)),
        locomotion_speed=locomotion_speed,
        valid_agents_mask=valid_agents_mask,
        key=rng_key,
    )

    # compute the cell-cell adhesion forces
    cell_adhesion_forces = cell_cell_adhesion_potential(
        positions=positions,
        interaction_radii=cell_adhesion_radii,
        adhesion_strength=cell_adhesion_strength,
        power=0.0,
        valid_agents_mask=valid_agents_mask,
    )

    # compute the cell-cell repulsion forces
    cell_repulsion_forces = cell_cell_repulsion_potential(
        positions=positions,
        interaction_radii=cell_repulsion_radii,
        repulsion_strength=cell_repulsion_strength,
        power=0.0,
        valid_agents_mask=valid_agents_mask,
    )

    # compute the cell-boundary adhesion forces
    boundary_adhesion_forces = cell_boundary_adhesion_potential(
        distances=boundary_distances,
        normal_vectors=boundary_normal_vectors,
        interaction_radii=boundary_adhesion_radii,
        adhesion_strength=boundary_adhesion_strength,
        power=0,
        valid_agents_mask=valid_agents_mask,
    )

    # compute the boundary repulsion force
    boundary_repulsion_forces = cell_boundary_repulsion_potential(
        distances=boundary_distances,
        normal_vectors=boundary_normal_vectors,
        interaction_radii=boundary_repulsion_radii,
        repulsion_strength=boundary_repulsion_strength,
        power=0,
        valid_agents_mask=valid_agents_mask,
    )

    forces = (
        locomotion_forces
        + cell_repulsion_forces
        + cell_adhesion_forces
        + boundary_adhesion_forces
        + boundary_repulsion_forces
    )

    return forces, updated_locomotion_direction


if __name__ == "__main__":
    # domain parameters
    domain_lower_left = jnp.array([10, 10, 10])
    domain_upper_right = jnp.array([210, 110, 50])

    # cell parameters
    n_cells = 5000
    cell_radius = 5.0
    min_cell_adhesion = 0.04
    max_cell_adhesion = 0.4
    cell_adhesion_radius = 1.25
    min_boundary_adhesion = 0.04
    max_boundary_adhesion = 0.6
    boundary_repulsion_constant = 10.0
    locomotion_speed_constant = 0.3

    # simulation parameters
    device = "cpu"
    time_step = 0.1
    damping_coefficient = 1.0
    n_time_steps = 500
    log_every_n_steps = 10

    # shape of the full space
    # this includes padding around the domain
    total_shape = (220, 120, 60)

    random_key = jax.random.PRNGKey(0)

    # make the positions
    positions = position_random_uniform(
        n_positions=n_cells,
        lower_left_coordinate=domain_lower_left + cell_radius,
        upper_right_coordinate=domain_upper_right - cell_radius,
        key=random_key,
    )

    # make the domain
    distance_field, normals_field = make_domain(
        lower_left_coordinate=domain_lower_left,
        upper_right_coordinate=domain_upper_right,
        shape=total_shape,
    )

    # locomotion parameters
    locomotion_direction = random_3d_unit_vectors(n_cells, key=random_key)
    locomotion_speed = locomotion_speed_constant * jnp.ones((n_cells,))  # µm/s

    # cell-cell adhesion force parameters
    cell_adhesion_radii = cell_adhesion_radius * cell_radius * jnp.ones((n_cells,))
    cell_adhesion_strength_field = min_cell_adhesion * np.ones(total_shape)
    cell_adhesion_slope = (max_cell_adhesion - min_cell_adhesion) / (
        domain_upper_right[0] - domain_lower_left[0]
    )
    for x_index in range(domain_lower_left[0], domain_upper_right[0]):
        cell_adhesion_strength_field[x_index:, :, :] = (
            min_cell_adhesion + cell_adhesion_slope * (x_index - domain_lower_left[0])
        )
    cell_adhesion_strength_field = jnp.array(cell_adhesion_strength_field)

    # cell-cell repulsion force parameters
    cell_repulsion_radii = cell_radius * jnp.ones((n_cells,))
    cell_repulsion_strength = 1.0 * jnp.ones((n_cells,))

    # cell-boundary adhesion force parameters
    boundary_adhesion_radii = cell_radius * jnp.ones((n_cells,))
    boundary_adhesion_strength_field = min_boundary_adhesion * np.ones(total_shape)
    boundary_adhesion_slope = (max_boundary_adhesion - min_boundary_adhesion) / (
        domain_upper_right[0] - domain_lower_left[0]
    )
    for x_index in range(domain_lower_left[0], domain_upper_right[0]):
        boundary_adhesion_strength_field[x_index:, :, :] = (
            min_boundary_adhesion
            + boundary_adhesion_slope * (x_index - domain_lower_left[0])
        )
    boundary_adhesion_strength_field = jnp.array(boundary_adhesion_strength_field)

    # cell-boundary repulsion force parameters
    boundary_repulsion_radii = cell_radius * jnp.ones((n_cells,))
    boundary_repulsion_strength = boundary_repulsion_constant * jnp.ones((n_cells,))

    # run the simulation
    logged_times = []
    logged_positions = []
    n_logged_time_points = 0

    sample_cell_adhesion_strength = jax.jit(
        partial(
            sample_scalar_field_linear,
            field=cell_adhesion_strength_field,
            origin=jnp.array([0, 0, 0]),
            scale=jnp.array([1, 1, 1]),
            cval=0.0,
        )
    )
    sample_boundary_adhesion_strength = jax.jit(
        partial(
            sample_scalar_field_linear,
            field=boundary_adhesion_strength_field,
            origin=jnp.array([0, 0, 0]),
            scale=jnp.array([1, 1, 1]),
            cval=0.0,
        )
    )
    sample_distance_field = jax.jit(
        partial(
            sample_scalar_field_nearest,
            field=distance_field,
            origin=jnp.array([0, 0, 0]),
            scale=jnp.array([1, 1, 1]),
            cval=0.0,
        )
    )
    sample_normal_vectors = jax.jit(
        partial(
            sample_vector_field_nearest,
            field=normals_field,
            origin=jnp.array([0, 0, 0]),
            scale=jnp.array([1, 1, 1]),
            cval=-1.0,
        )
    )

    rng_keys = jax.random.split(jax.random.key(42), num=n_time_steps)
    for time_step_index in trange(int(n_time_steps)):
        # get the adhesion strengths for the current locations
        cell_adhesion_strength = sample_cell_adhesion_strength(positions)
        boundary_adhesion_strength = sample_boundary_adhesion_strength(
            positions,
        )

        # get the boundary distances and normals
        boundary_distances = sample_distance_field(
            positions,
        )
        boundary_normal_vectors = sample_normal_vectors(positions)

        forces, locomotion_direction = compute_forces(
            positions=positions,
            cell_adhesion_radii=cell_adhesion_radii,
            cell_adhesion_strength=cell_adhesion_strength,
            cell_repulsion_radii=cell_repulsion_radii,
            cell_repulsion_strength=cell_repulsion_strength,
            boundary_adhesion_radii=boundary_adhesion_radii,
            boundary_adhesion_strength=boundary_adhesion_strength,
            boundary_repulsion_radii=boundary_repulsion_radii,
            boundary_repulsion_strength=boundary_repulsion_strength,
            boundary_distances=boundary_distances,
            boundary_normal_vectors=boundary_normal_vectors,
            previous_locomotion_direction=locomotion_direction,
            locomotion_speed=locomotion_speed,
            rng_key=rng_keys[time_step_index],
        )
        positions = positions + forces * (time_step / damping_coefficient)

        # log data
        if time_step_index % log_every_n_steps == 0:
            logged_times.append(time_step_index * time_step)
            time_indices = n_logged_time_points * np.ones((n_cells,))
            logged_positions.append(np.column_stack((time_indices, positions)))
            n_logged_time_points += 1

    logged_positions = np.concatenate(logged_positions)
    print(logged_positions.shape)
    np.save(f"n_{n_cells}.npy", logged_positions)

    # view the results
    viewer = napari.Viewer()
    viewer.add_points(
        positions,
        size=2 * cell_radius,
        shading="spherical",
        name="initial_positions",
        visible=False,
    )
    viewer.add_points(
        logged_positions,
        size=2 * cell_radius,
        shading="spherical",
        name="time series",
    )
    viewer.add_shapes(
        bounding_box_lines(
            lower_left_coordinate=domain_lower_left,
            upper_right_coordinate=domain_upper_right,
        ),
        shape_type="line",
    )
    viewer.add_image(
        np.asarray(distance_field),
        name="distance field",
        colormap="gray",
        visible=False,
    )
    viewer.add_image(
        np.asarray(normals_field), name="normals field", colormap="green", visible=False
    )

    viewer.dims.ndisplay = 3
    napari.run()

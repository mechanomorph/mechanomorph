"""Demonstration of agents moving randomly without adhesion."""

import napari
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from tqdm import trange

from mechanomorph.data import ScalarField, VectorField
from mechanomorph.sim.agent.forces import (
    biased_random_locomotion_force,
    cell_boundary_repulsion_potential,
)
from mechanomorph.sim.agent.utils.field import field_gradient


def random_3d_unit_vectors(n_vectors: int) -> torch.Tensor:
    """Generate an array of random 3D unit vectors.

    Parameters
    ----------
    n_vectors : int
        The number of random unit vectors to generate.

    Returns
    -------
    vectors : torch.Tensor
        A tensor of shape (n_vectors, 3) containing random unit vectors.
    """
    vectors = torch.randn((n_vectors, 3))
    return torch.nn.functional.normalize(vectors, dim=1)


def position_grid_3d(
    lower_left_coordinate: tuple[float, float, float],
    grid_spacing: tuple[float, float, float],
    grid_shape: tuple[int, int, int],
    device: str = "cpu",
) -> torch.Tensor:
    """Make a grid of positions in 3D space.

    Parameters
    ----------
    lower_left_coordinate : tuple[float, float, float]
        The lower left coordinate of the grid.
    grid_spacing : tuple[float, float, float]
        The spacing between the grid points.
    grid_shape : tuple[int, int, int]
        The shape of the grid (number of points in each dimension).
    device : str
        The device to store the tensor on. Default is "cpu".

    Returns
    -------
    grid : torch.Tensor
        A tensor of shape (n_points, 3) containing the positions of the grid points.
    """
    x = torch.arange(
        lower_left_coordinate[0],
        lower_left_coordinate[0] + grid_spacing[0] * grid_shape[0],
        grid_spacing[0],
        device=device,
    )
    y = torch.arange(
        lower_left_coordinate[1],
        lower_left_coordinate[1] + grid_spacing[1] * grid_shape[1],
        grid_spacing[1],
        device=device,
    )
    z = torch.arange(
        lower_left_coordinate[2],
        lower_left_coordinate[2] + grid_spacing[2] * grid_shape[2],
        grid_spacing[2],
        device=device,
    )
    return torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).reshape(-1, 3)


def position_random_uniform(
    n_positions: int,
    lower_left_coordinate: tuple[float, float, float],
    upper_right_coordinate: tuple[float, float, float],
    device: str = "cpu",
) -> torch.Tensor:
    """Generate random positions in 3D space.

    This draws from a uniform distribution in a bounding box.

    Parameters
    ----------
    n_positions : int
        The number of random positions to generate.
    lower_left_coordinate : tuple[float, float, float]
        The lower left coordinate of the bounding box.
    upper_right_coordinate : tuple[float, float, float]
        The upper right coordinate of the bounding box.
    device : str
        The device to store the tensor on. Default is "cpu".

    Returns
    -------
    positions : torch.Tensor
        A tensor of shape (n_positions, 3) containing the random positions.
    """
    positions = torch.empty((n_positions, 3), device=device)
    positions[:, 0] = (
        torch.rand(n_positions) * (upper_right_coordinate[0] - lower_left_coordinate[0])
        + lower_left_coordinate[0]
    )
    positions[:, 1] = (
        torch.rand(n_positions) * (upper_right_coordinate[1] - lower_left_coordinate[1])
        + lower_left_coordinate[1]
    )
    positions[:, 2] = (
        torch.rand(n_positions) * (upper_right_coordinate[2] - lower_left_coordinate[2])
        + lower_left_coordinate[2]
    )
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


def make_domain(
    lower_left_coordinate: tuple[float, float, float],
    upper_right_coordinate: tuple[float, float, float],
    shape: tuple[int, int, int] = (50, 50, 50),
    device: str = "cpu",
) -> tuple[ScalarField, VectorField]:
    """Make a domain for the simulation.

    Parameters
    ----------
    lower_left_coordinate : tuple[float, float, float]
        The lower left coordinate of the domain.
    upper_right_coordinate : tuple[float, float, float]
        The upper right coordinate of the domain.
    shape : tuple[int, int, int]
        The shape of the array the domain is embedded in.
    device : str
        The device to store the arrays on.

    Returns
    -------
    distance_field : ScalarField
        The field of distances to the boundary.
    normals_field : VectorField
        The field of normals to the boundary.
        These are calculated as the gradient of the distance field.
    """
    domain_segmentation = np.zeros(shape, dtype=bool)
    domain_segmentation[
        lower_left_coordinate[0] : upper_right_coordinate[0],
        lower_left_coordinate[1] : upper_right_coordinate[1],
        lower_left_coordinate[2] : upper_right_coordinate[2],
    ] = True
    distances = distance_transform_edt(domain_segmentation)
    distance_field = ScalarField(torch.from_numpy(distances).float().to(device=device))

    normals_field = VectorField(
        field_gradient(distance_field.field),
    )

    return distance_field, normals_field


def compute_forces(
    positions: torch.Tensor,
    boundary_repulsion_radii: torch.Tensor,
    boundary_repulsion_strength: torch.Tensor,
    boundary_distance_field: ScalarField,
    boundary_normals_field: VectorField,
    previous_locomotion_direction: torch.Tensor,
    locomotion_speed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the forces at a given time step.

    This includes random locomotion, cell-cell repulsion,
    and cell-boundary repulsion forces.

    Parameters
    ----------
    positions : torch.Tensor
        The current position of each agent.
    boundary_repulsion_radii : torch.Tensor
        (n_agents,) array of the interaction radius of each agent.
    boundary_repulsion_strength : torch.Tensor
        (n_agents,) array of the repulsion strength of each agent.
    boundary_distance_field : ScalarField
        The distance scalar field of the boundary.
    boundary_normals_field : VectorField
        (3, h, w, d) vector field of the normals of the boundary.
        This should be computed as the gradient of the distance field.
        All vectors must be unit vectors and point away
        from the boundary.
    previous_locomotion_direction : torch.Tensor
        (n_agents, 3) array of the locomotion direction of each agent.
    locomotion_speed : torch.Tensor
        (n_agents,) array of the locomotion speed of each agent.

    Returns
    -------
    forces : torch.Tensor
        (n_agents, 3) array of the force vector for each agent.
    locomotion_direction : torch.Tensor
        (n_agents, 3) array of the locomotion direction of each agent.
    """
    n_agents = positions.size(dim=0)

    # compute the locomotion force
    locomotion_direction_change_probability = 0.1 * torch.ones((n_agents,))
    locomotion_forces, updated_locomotion_direction = biased_random_locomotion_force(
        previous_direction=previous_locomotion_direction,
        direction_change_probability=locomotion_direction_change_probability,
        bias_direction=torch.zeros((n_agents, 3)),
        bias_constant=torch.zeros((n_agents,)),
        locomotion_speed=locomotion_speed,
    )

    # compute the boundary repulsion force
    boundary_repulsion_forces = cell_boundary_repulsion_potential(
        positions=positions,
        interaction_radii=boundary_repulsion_radii,
        repulsion_strength=boundary_repulsion_strength,
        distance_field=boundary_distance_field,
        normals_field=boundary_normals_field,
        power=0,
    )

    forces = locomotion_forces + boundary_repulsion_forces
    # forces = locomotion_forces
    return forces, updated_locomotion_direction


if __name__ == "__main__":
    # domain parameters
    domain_lower_left = np.array([10, 10, 10])
    domain_upper_right = np.array([210, 110, 50])

    # cell parameters
    n_cells = 500
    cell_radius = 5.0

    # simulation parameters
    device = "cpu"
    time_step = 0.1
    damping_coefficient = 1.0
    n_time_steps = 2 * 3600 / time_step
    log_every_n_steps = 10

    # make the positions
    positions = position_random_uniform(
        n_positions=n_cells,
        lower_left_coordinate=domain_lower_left + cell_radius,
        upper_right_coordinate=domain_upper_right - cell_radius,
        device=device,
    )

    # make the domain
    distance_field, normals_field = make_domain(
        lower_left_coordinate=domain_lower_left,
        upper_right_coordinate=domain_upper_right,
        shape=(220, 120, 60),
        device=device,
    )

    # locomotion parameters
    locomotion_direction = random_3d_unit_vectors(n_cells)
    locomotion_speed = 0.3 * torch.ones((n_cells,), device=device)  # µm/s

    # boundary forces
    boundary_repulsion_radii = cell_radius * torch.ones((n_cells,), device=device)
    boundary_repulsion_strength = 1.0 * torch.ones((n_cells,), device=device)

    # run the simulation
    logged_times = []
    logged_positions = []
    n_logged_time_points = 0
    for time_step_index in trange(int(n_time_steps)):
        forces, locomotion_direction = compute_forces(
            positions=positions,
            boundary_repulsion_radii=boundary_repulsion_radii,
            boundary_repulsion_strength=boundary_repulsion_strength,
            boundary_distance_field=distance_field,
            boundary_normals_field=normals_field,
            previous_locomotion_direction=locomotion_direction,
            locomotion_speed=locomotion_speed,
        )
        positions = positions + forces * (time_step / damping_coefficient)

        # log data
        if time_step_index % log_every_n_steps == 0:
            logged_times.append(time_step_index * time_step)
            time_indices = n_logged_time_points * torch.ones((n_cells,), device=device)
            logged_positions.append(torch.column_stack((time_indices, positions)))
            n_logged_time_points += 1

    logged_positions = torch.cat(logged_positions).numpy(force=True)
    print(logged_positions.shape)

    # view the results
    viewer = napari.Viewer()
    viewer.add_points(
        positions, size=10, shading="spherical", name="initial_positions", visible=False
    )
    viewer.add_points(
        logged_positions,
        size=10,
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
    viewer.dims.ndisplay = 3
    napari.run()

"""Script to validate the boundary adhesion force.

There is an agent that starts at the equilibrium point
between the locomotion and adhesion forces.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

from mechanomorph.data import ScalarField, VectorField
from mechanomorph.sim.agent.forces import (
    biased_random_locomotion_force,
    cell_boundary_adhesion_potential,
)
from mechanomorph.sim.agent.utils.field import field_gradient


def compute_forces(
    positions: torch.Tensor,
    boundary_interaction_radii: torch.Tensor,
    boundary_adhesion_strength: torch.Tensor,
    boundary_distance_field: ScalarField,
    boundary_normals_field: VectorField,
    locomotion_direction: torch.Tensor,
    locomotion_speed: torch.Tensor,
) -> torch.Tensor:
    """Compute the forces at a given time step.

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the current position of each agent.
    boundary_interaction_radii : torch.Tensor
        (n_agents,) array of the interaction radius of each agent.
    boundary_adhesion_strength : torch.Tensor
        (n_agents,) array of the adhesion strength of each agent.
    boundary_distance_field : ScalarField
        The distance scalar field of the boundary.
    boundary_normals_field : VectorField
        (3, h, w, d) vector field of the normals of the boundary.
        This should be computed as the gradient of the distance field.
        All vectors must be unit vectors and point away
        from the boundary.
    locomotion_direction : torch.Tensor
        (n_agents, 3) array of the locomotion direction of each agent.
    locomotion_speed : torch.Tensor
        (n_agents,) array of the locomotion speed of each agent.

    Returns
    -------
    forces : torch.Tensor
        The force vector for each agent.
    """
    n_agents = positions.size(dim=0)

    # compute the locomotion force
    locomotion_direction_change_probability = torch.zeros((n_agents,))
    locomotion_forces, _ = biased_random_locomotion_force(
        previous_direction=locomotion_direction,
        direction_change_probability=locomotion_direction_change_probability,
        bias_direction=locomotion_direction,
        bias_constant=torch.ones((n_agents,)),
        locomotion_speed=locomotion_speed,
    )

    # compute the boundary repulsion force
    boundary_repulsion_forces = cell_boundary_adhesion_potential(
        positions=positions,
        interaction_radii=boundary_interaction_radii,
        adhesion_strength=boundary_adhesion_strength,
        distance_field=boundary_distance_field,
        normals_field=boundary_normals_field,
        power=0,
    )

    return locomotion_forces + boundary_repulsion_forces


if __name__ == "__main__":
    # simulation parameters
    time_step = 0.25
    damping_coefficient = 1.0
    n_time_steps = 50

    # run parameters
    device = "cpu"
    plot_path = "boundary_adhesion_positions.png"

    # set up the domain
    domain_segmentation = np.zeros((50, 50, 50), dtype=bool)
    domain_segmentation[5:45, 5:45, 5:45] = True
    distances = distance_transform_edt(domain_segmentation)
    distance_field = ScalarField(torch.from_numpy(distances).float().to(device=device))

    normals_field = VectorField(
        field_gradient(distance_field.field),
    )

    # initialize agent position
    positions = torch.tensor(
        [
            [7.0, 25.0, 25.0],
        ],
        device=device,
    )

    # boundary repulsion parameters
    interaction_radii = torch.tensor([6])
    adhesion_strength = torch.tensor([2])

    # locomotion is towards the boundary
    locomotion_direction = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    locomotion_speed = torch.tensor([1])

    logged_positions = []
    logged_times = []
    for time_step_index in range(n_time_steps):
        forces = compute_forces(
            positions=positions,
            boundary_interaction_radii=interaction_radii,
            boundary_adhesion_strength=adhesion_strength,
            boundary_distance_field=distance_field,
            boundary_normals_field=normals_field,
            locomotion_direction=locomotion_direction,
            locomotion_speed=locomotion_speed,
        )
        positions = positions + forces * (time_step / damping_coefficient)

        # log data
        logged_times.append(time_step_index * time_step)
        logged_positions.append(positions)

    # convert logs to numpy arrays
    logged_times = np.asarray(logged_times)
    logged_positions = torch.cat(logged_positions).numpy(force=True)

    print(f"final position: {logged_positions[-1, :]}")

    # plot
    f, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].axhline(7.0, c="gray", label="expected")
    axs[0].plot(logged_times, logged_positions[:, 0], label="actual")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("position x")
    axs[0].legend()

    axs[1].axhline(25, c="gray")
    axs[1].plot(logged_times, logged_positions[:, 1])
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("position y")

    axs[2].axhline(25, c="gray")
    axs[2].plot(logged_times, logged_positions[:, 2])
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("position z")

    plt.tight_layout()
    f.savefig(plot_path, dpi=300)

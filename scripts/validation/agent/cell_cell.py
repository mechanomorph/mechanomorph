"""Script to validate the cell-cell interaction force.

Two cells are placed close to each other. They should
equillibrate where the adhesion and repulsion forces are equal
and opposite.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt

from mechanomorph.sim.agent.forces import (
    cell_cell_adhesion_potential,
    cell_cell_repulsion_potential,
)


def compute_forces(
    positions: torch.Tensor,
    adhesion_interaction_radii: torch.Tensor,
    adhesion_strength: torch.Tensor,
    repulsion_interaction_radii: torch.Tensor,
    repulsion_strength: torch.Tensor,
) -> torch.Tensor:
    """Compute the forces at a given time step.

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the current position of each agent.
    adhesion_interaction_radii : torch.Tensor
        (n_agents,) array of the interaction radius of each agent.
    adhesion_strength : torch.Tensor
        (n_agents,) array of the adhesion strength of each agent.
    repulsion_interaction_radii : torch.Tensor
        (n_agents,) array of the interaction radius of each agent.
    repulsion_strength : torch.Tensor
        (n_agents,) array of the repulsion strength of each agent.

    Returns
    -------
    forces : torch.Tensor
        The force vector for each agent.
    """
    adhesion_forces = cell_cell_adhesion_potential(
        positions=positions,
        interaction_radii=adhesion_interaction_radii,
        adhesion_strength=adhesion_strength,
        power=0.0,
    )

    repulsion_forces = cell_cell_repulsion_potential(
        positions=positions,
        interaction_radii=repulsion_interaction_radii,
        repulsion_strength=repulsion_strength,
        power=0.0,
    )

    return adhesion_forces + repulsion_forces


if __name__ == "__main__":
    # simulation parameters
    time_step = 0.1
    damping_coefficient = 1.0
    n_time_steps = 1000

    # run parameters
    device = "cpu"
    plot_path = "cell_cell_positions.png"

    # initialize agent position
    positions = torch.tensor(
        [
            [7.0, 0.0, 0.0],
            [-7.0, 0.0, 0.0],
        ],
        device=device,
    )

    # adhesion parameters
    adhesion_interaction_radii = torch.tensor([12.5, 12.5], device=device)
    adhesion_strength = torch.tensor([0.04, 0.04], device=device)

    # repulsion parameters
    repulsion_interaction_radii = torch.tensor([10.0, 10.0], device=device)
    repulsion_strength = torch.tensor([1.0, 1.0], device=device)

    # run the simulation
    logged_positions = []
    logged_times = []
    for time_step_index in range(n_time_steps):
        forces = compute_forces(
            positions=positions,
            adhesion_interaction_radii=adhesion_interaction_radii,
            adhesion_strength=adhesion_strength,
            repulsion_interaction_radii=repulsion_interaction_radii,
            repulsion_strength=repulsion_strength,
        )
        positions = positions + forces * (time_step / damping_coefficient)

        # log data
        logged_times.append(time_step_index * time_step)
        logged_positions.append(positions)

    # get the results as numpy arrays
    logged_times = np.asarray(logged_times)
    logged_positions_0 = np.stack(
        [time_point[0, :].numpy(force=True) for time_point in logged_positions]
    )
    logged_positions_1 = np.stack(
        [time_point[1, :].numpy(force=True) for time_point in logged_positions]
    )

    print(f"final position 0: {logged_positions_0[-1, :]}")
    print(f"final position 1: {logged_positions_1[-1, :]}")

    # plot
    f, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].axhline(10, c="gray", label="expected 0")
    axs[0].axhline(-10, c="gray", label="expected 1")
    axs[0].scatter(logged_times, logged_positions_0[:, 0], s=0.1, label="agent 0")
    axs[0].scatter(logged_times, logged_positions_1[:, 0], s=0.1, label="agent 1")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("position x")
    axs[0].legend()

    axs[1].axhline(0, c="gray", label="expected 0")
    axs[1].axhline(0, c="gray", label="expected 1")
    axs[1].scatter(logged_times, logged_positions_0[:, 1], s=0.5, label="agent 0")
    axs[1].scatter(logged_times, logged_positions_1[:, 1], s=0.5, label="agent 1")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("position y")

    axs[2].axhline(0, c="gray", label="expected 0")
    axs[2].axhline(0, c="gray", label="expected 1")
    axs[2].scatter(logged_times, logged_positions_0[:, 2], s=0.5, label="agent 0")
    axs[2].scatter(logged_times, logged_positions_1[:, 2], s=0.5, label="agent 1")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("position z")

    plt.tight_layout()
    f.savefig(plot_path, dpi=300)

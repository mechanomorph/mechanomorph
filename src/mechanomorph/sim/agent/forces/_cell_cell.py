import torch

from mechanomorph.sim.agent.forces._contact_utils import (
    vectors_distances_between_agents,
)


def cell_cell_adhesion_potential(
    positions: torch.Tensor,
    interaction_radii: torch.Tensor,
    adhesion_strength: torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    """Compute an adhesion force with a potential function.

    This is expensive and computes for all potential interactions.
    There are a few improvements we should consider:
        - use spatial partitioning to reduce the number of interactions
        - only compute the potential for the interacting agents

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the positions of the agents.
    interaction_radii : torch.Tensor
        (n_agents,) array of the radius of each agent.
    adhesion_strength : torch.Tensor
        The strength of the adhesion. Must be broadcastable
        to (n_agents, n_agents).
    power : float
        The power of the potential function. Default is 1.0.

    Returns
    -------
    forces : torch.Tensor
        (n_agents, 3) array of the cell-cell adhesion
        force vector for each agent.
    """
    distances, vectors = vectors_distances_between_agents(positions)

    maximum_interaction_distance = interaction_radii.unsqueeze(
        1
    ) + interaction_radii.unsqueeze(0)

    force_magnitude = adhesion_strength * torch.pow(
        1 - (distances / maximum_interaction_distance), exponent=power + 1
    ).fill_diagonal_(0)

    # set forces to zero if the distance is greater
    # than the maximum interaction distance
    force_magnitude[distances > maximum_interaction_distance] = 0

    forces = force_magnitude.unsqueeze(dim=-1) * vectors
    return forces.sum(dim=1)


def cell_cell_repulsion_potential(
    positions: torch.Tensor,
    interaction_radii: torch.Tensor,
    repulsion_strength: torch.Tensor,
    power: float = 1.0,
) -> torch.Tensor:
    """Compute an adhesion force with a potential function.

    This is expensive and computes for all potential interactions.
    There are a few improvements we should consider:
        - use spatial partitioning to reduce the number of interactions
        - only compute the potential for the interacting agents

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the positions of the agents.
    interaction_radii : torch.Tensor
        (n_agents,) array of the radius of each agent.
    repulsion_strength : torch.Tensor
        The strength of the adhesion. Must be broadcastable
        to (n_agents, n_agents).
    power : float
        The power of the potential function. Default is 1.0.

    Returns
    -------
    forces : torch.Tensor
        (n_agents, 3) array of the cell-cell adhesion
        force vector for each agent.
    """
    distances, vectors = vectors_distances_between_agents(positions)

    maximum_interaction_distance = interaction_radii.unsqueeze(
        1
    ) + interaction_radii.unsqueeze(0)

    force_magnitude = -repulsion_strength * torch.pow(
        1 - (distances / maximum_interaction_distance), exponent=power + 1
    ).fill_diagonal_(0)

    # set forces to zero if the distance is greater
    # than the maximum interaction distance
    force_magnitude[distances > maximum_interaction_distance] = 0

    forces = force_magnitude.unsqueeze(dim=-1) * vectors
    return forces.sum(dim=1)

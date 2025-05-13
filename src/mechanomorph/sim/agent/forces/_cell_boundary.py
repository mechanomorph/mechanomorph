import torch


def cell_boundary_adhesion_potential(
    positions: torch.Tensor,
    interaction_radii: torch.Tensor,
    distance_field: torch.Tensor,
) -> torch.Tensor:
    """Compute an adhesion force between a cell and boundary with a potential function.

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the positions of the agents.
    interaction_radii : torch.Tensor
        (n_agents,) array of the radius of each agent.
    distance_field
        The distance field of the boundary.

    Returns
    -------
    forces : torch.Tensor
        (n_agents, 3) array of the cell-boundary adhesion
        force vector for each agent.
    """

import torch


def vectors_distances_between_agents(
    positions: torch.Tensor, epsilon: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the vector between all agents.

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the positions of the agents.
    epsilon : float
        Small value to prevent divide by zero when computing unit vectors.

    Returns
    -------
    distances : torch.Tensor
        (n_agents, n_agents) array of distances between all agents.
    unit_vectors : torch.Tensor
        (n_agents, n_agents, 3) array of unit vectors between all agents.
        unit_vectors[i, j, :] is the unit vector from agent i to agent j.
    """
    vectors = positions.unsqueeze(0) - positions.unsqueeze(1)

    # compute the distances
    distances = torch.linalg.norm(vectors, dim=2).fill_diagonal_(0)

    # create the unit vectors
    unit_vectors = vectors / torch.max(
        distances, torch.tensor(epsilon, device=positions.device)
    ).unsqueeze(dim=2)

    return distances, unit_vectors

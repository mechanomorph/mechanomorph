import torch


def _random_unit_vectors(shape: tuple[int, int], device: str = "cpu") -> torch.Tensor:
    """Generate an array of random unit vectors.

    This uses a uniform distribution.

    Parameters
    ----------
    shape: tuple[int, int]
        The shape of the array to generate.
        First element is the number of vectors.
        Second element is the dimensionality of each vector.
    device : str
        The device to use for the resulting tensor.

    Returns
    -------
    torch.Tensor
        An array of random unit vectors.
    """
    # generate random vectors (range -1 to 1)
    random_vectors = (torch.rand(shape, device=device) * 2) - 1

    # normalize the vectors
    return torch.nn.functional.normalize(random_vectors, dim=1)


def biased_random_locomotion_force(
    previous_direction: torch.Tensor,
    direction_change_probability: torch.Tensor,
    bias_direction: torch.Tensor,
    bias_constant: torch.Tensor,
    locomotion_speed: torch.Tensor,
) -> torch.Tensor:
    """Compute the locomotion forces for an array of agents.

    Parameters
    ----------
    previous_direction : torch.Tensor
        (n_agents, 3) array containing unit vectors giving
        the previous direction of the agent.
    direction_change_probability : torch.Tensor
        (n_agents,) array containing the probability of changing direction.
        This is generally calculated as time_step / persistence_time.
    bias_direction : torch.Tensor
        (n_agents, 3) array containing unit vectors giving
        the bias direction of the agent.
    bias_constant : torch.Tensor
        (n_agents,) array containing the bias constant for each agent. This should be
        in range 0-1. 0 is fully random and 1 is fully biased.
    locomotion_speed : torch.Tensor
        (n_agents,) array containing the magnitude of
        the locomotion speed for each agent.
    """
    # get the number of agents
    n_agents = previous_direction.shape[0]

    # determine which agents will change direction
    change_direction_mask = (
        torch.rand(n_agents, device=previous_direction.device)
        <= direction_change_probability
    )

    # compute the new directions
    new_biased_direction = (
        bias_constant.unsqueeze(dim=1) * bias_direction
    ) * previous_direction
    new_random_direction = (1 - bias_constant.unsqueeze(dim=1)) * _random_unit_vectors(
        (n_agents, 3), device=previous_direction.device
    )
    new_direction = torch.nn.functional.normalize(
        new_biased_direction + new_random_direction, dim=1
    )
    # insert the new velocities
    previous_direction[change_direction_mask] = new_direction[change_direction_mask]

    return locomotion_speed.unsqueeze(dim=1) * previous_direction, previous_direction

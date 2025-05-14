import torch

from mechanomorph.data import ScalarField, VectorField


def cell_boundary_adhesion_potential(
    positions: torch.Tensor,
    interaction_radii: torch.Tensor,
    adhesion_strength: torch.Tensor,
    distance_field: ScalarField,
    normals_field: VectorField,
    power: float = 1.0,
) -> torch.Tensor:
    """Compute an adhesion force between a cell and boundary with a potential function.

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the positions of the agents.
    interaction_radii : torch.Tensor
        (n_agents,) array of the interaction radius of each agent.
    adhesion_strength : torch.Tensor
        (n_agents,) array of the adhesion strength of each agent.
    distance_field : ScalarField
        The distance scalar field of the boundary.
    normals_field : VectorField
        (3, h, w, d) vector field of the normals of the boundary.
        This should be computed as the gradient of the distance field.
        All vectors must be unit vectors.
    power : float
        The exponent of the adhesion potential.
        Note the computed exponent is power + 1.
        Default value is 1.0.

    Returns
    -------
    forces : torch.Tensor
        (n_agents, 3) array of the cell-boundary adhesion
        force vector for each agent.
    """
    distances = distance_field.sample(positions)
    normalized_distances = distances / interaction_radii
    potential_magnitude = torch.pow(1 - normalized_distances, exponent=power + 1)

    # set forces to zero if the distance is greater
    # the interaction radius
    potential_magnitude[distances > interaction_radii] = 0

    normal_vectors = normals_field.sample(positions)

    return adhesion_strength * potential_magnitude * normal_vectors


def cell_boundary_repulsion_potential(
    positions: torch.Tensor,
    interaction_radii: torch.Tensor,
    repulsion_strength: torch.Tensor,
    distance_field: ScalarField,
    normals_field: VectorField,
    power: float = 1.0,
) -> torch.Tensor:
    """Compute a repulsion force between a cell and boundary with a potential function.

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the positions of the agents.
    interaction_radii : torch.Tensor
        (n_agents,) array of the interaction radius of each agent.
    repulsion_strength : torch.Tensor
        (n_agents,) array of the repulsion strength of each agent.
    distance_field : ScalarField
        The distance scalar field of the boundary.
    normals_field : VectorField
        (3, h, w, d) vector field of the normals of the boundary.
        This should be computed as the gradient of the distance field.
        All vectors must be unit vectors.
    power : float
        The exponent of the adhesion potential.
        Note the computed exponent is power + 1.
        Default value is 1.0.

    Returns
    -------
    forces : torch.Tensor
        (n_agents, 3) array of the cell-boundary adhesion
        force vector for each agent.
    """
    distances = distance_field.sample(positions)
    normalized_distances = distances / interaction_radii
    potential_magnitude = torch.pow(1 - normalized_distances, exponent=power + 1)

    # set forces to zero if the distance is greater
    # the interaction radius
    potential_magnitude[distances > interaction_radii] = 0

    normal_vectors = normals_field.sample(positions)

    return -repulsion_strength * potential_magnitude * normal_vectors

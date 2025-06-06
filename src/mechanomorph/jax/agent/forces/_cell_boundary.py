"""Functions for computing cell-boundary forces in JAX."""

import jax.numpy as jnp
from jax import Array as JaxArray


def cell_boundary_adhesion_potential(
    distances: JaxArray,
    normal_vectors: JaxArray,
    valid_agents_mask: JaxArray,
    interaction_radii: JaxArray,
    adhesion_strength: JaxArray,
    power: float = 1.0,
) -> JaxArray:
    """Compute an adhesion force between cells and boundary with a potential function.

    Parameters
    ----------
    distances : JaxArray
        (n_agents,) array containing distance to the boundary for each agent.
    normal_vectors : JaxArray
        (n_agents, 3) array of the boundary normal vector for each agent.
    valid_agents_mask : JaxArray
        Array of shape (n_agents,) containing boolean values indicating
        which positions are valid (True) vs padding (False).
    interaction_radii : JaxArray
        Array of shape (n_agents,) containing the interaction radius of each agent.
    adhesion_strength : JaxArray
        Array of shape (n_agents,) containing the adhesion strength of each agent.
    power : float, optional
        The exponent of the adhesion potential. Note the computed exponent is
        power + 1. Default value is 1.0.

    Returns
    -------
    forces : JaxArray
        Array of shape (n_agents, 3) containing the cell-boundary adhesion
        force vector for each agent. Invalid agents will have zero forces.
    """
    # Compute the potential magnitude
    potential_magnitude = _compute_potential_magnitude(
        distances=distances,
        interaction_radii=interaction_radii,
        valid_positions=valid_agents_mask,
        power=power,
    )

    # Compute adhesion forces (negative sign for attractive force)
    forces = _assemble_boundary_forces(
        strength=adhesion_strength,
        potential_magnitude=potential_magnitude,
        normal_vectors=normal_vectors,
        valid_positions=valid_agents_mask,
        force_sign=-1.0,
    )

    return forces


def cell_boundary_repulsion_potential(
    distances: JaxArray,
    normal_vectors: JaxArray,
    valid_agents_mask: JaxArray,
    interaction_radii: JaxArray,
    repulsion_strength: JaxArray,
    power: float = 1.0,
) -> JaxArray:
    """Compute a repulsion force between cells and boundary with a potential function.

    Parameters
    ----------
    distances : JaxArray
        (n_agents,) array containing distance to the boundary for each agent
    normal_vectors : JaxArray
        (n_agents, 3) array of the boundary normal vector for each agent.
    valid_agents_mask : JaxArray
        Array of shape (n_agents,) containing boolean values indicating
        which positions are valid (True) vs padding (False).
    interaction_radii : JaxArray
        Array of shape (n_agents,) containing the interaction radius of each agent.
    repulsion_strength : JaxArray
        Array of shape (n_agents,) containing the repulsion strength of each agent.

    origin : JaxArray
        Array of shape (3,) specifying the origin of the field coordinate system.
    scale : JaxArray
        Array of shape (3,) specifying the scale (voxel size) of the field.
    power : float, optional
        The exponent of the repulsion potential. Note the computed exponent is
        power + 1. Default value is 1.0.
    cval : float, optional
        Constant value to use for coordinates outside field bounds.
        Default is 0.0.

    Returns
    -------
    forces : JaxArray
        Array of shape (n_agents, 3) containing the cell-boundary repulsion
        force vector for each agent. Invalid agents will have zero forces.
    """
    # Compute the potential magnitude
    potential_magnitude = _compute_potential_magnitude(
        distances=distances,
        interaction_radii=interaction_radii,
        valid_positions=valid_agents_mask,
        power=power,
    )

    # Compute repulsion forces (positive sign for repulsive force)
    forces = _assemble_boundary_forces(
        strength=repulsion_strength,
        potential_magnitude=potential_magnitude,
        normal_vectors=normal_vectors,
        valid_positions=valid_agents_mask,
        force_sign=1.0,
    )

    return forces


def _compute_potential_magnitude(
    distances: JaxArray,
    interaction_radii: JaxArray,
    valid_positions: JaxArray,
    power: float,
) -> JaxArray:
    """Compute the magnitude of the potential force.

    Parameters
    ----------
    distances : JaxArray
        Array of shape (n_agents,) containing distances from each agent
        to the boundary.
    interaction_radii : JaxArray
        Array of shape (n_agents,) containing the interaction radius of each agent.
    valid_positions : JaxArray
        Array of shape (n_agents,) containing boolean values indicating
        which positions are valid.
    power : float
        The exponent of the potential function.

    Returns
    -------
    potential_magnitude : JaxArray
        Array of shape (n_agents,) containing the potential magnitude
        for each agent.
    """
    # Normalize distances by interaction radii
    normalized_distances = distances / jnp.maximum(interaction_radii, 1e-12)

    # Compute potential magnitude using power law
    raw_magnitude = jnp.power(1.0 - normalized_distances, power + 1)

    # Set forces to zero if distance is greater than interaction radius
    within_range = distances <= interaction_radii

    # Apply masking for valid positions and interaction range
    potential_magnitude = jnp.where(
        valid_positions & within_range,
        raw_magnitude,
        0.0,
    )

    return potential_magnitude


def _assemble_boundary_forces(
    strength: JaxArray,
    potential_magnitude: JaxArray,
    normal_vectors: JaxArray,
    valid_positions: JaxArray,
    force_sign: float,
) -> JaxArray:
    """Assemble the final boundary forces from components.

    Parameters
    ----------
    strength : JaxArray
        Array of shape (n_agents,) containing the force strength for each agent.
    potential_magnitude : JaxArray
        Array of shape (n_agents,) containing the potential magnitude for each agent.
    normal_vectors : JaxArray
        Array of shape (n_agents, 3) containing the normal vectors at each
        agent position.
    valid_positions : JaxArray
        Array of shape (n_agents,) containing boolean values indicating
        which positions are valid.
    force_sign : float
        Sign of the force: -1.0 for attractive (adhesion), 1.0 for repulsive.

    Returns
    -------
    forces : JaxArray
        Array of shape (n_agents, 3) containing the assembled force vectors.
    """
    # Compute force magnitude for each agent
    force_magnitude = force_sign * strength * potential_magnitude

    # Expand dimensions for broadcasting with normal vectors
    force_magnitude_expanded = jnp.expand_dims(force_magnitude, axis=1)

    # Compute force vectors
    forces = force_magnitude_expanded * normal_vectors

    # Apply validity mask to final forces
    valid_mask_expanded = jnp.expand_dims(valid_positions, axis=1)
    forces = jnp.where(valid_mask_expanded, forces, 0.0)

    return forces

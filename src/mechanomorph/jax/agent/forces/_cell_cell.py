import jax
import jax.numpy as jnp
from jax import Array as JaxArray


def vectors_distances_between_agents(
    positions: JaxArray, valid_positions_mask: JaxArray, epsilon: float = 1e-12
) -> tuple[JaxArray, JaxArray]:
    """Find the vector between all agents.

    Parameters
    ----------
    positions : JaxArray
        (max_agents, 3) array of the padded positions of the agents.
    valid_positions_mask : JaxArray
        (max_agents,) boolean array indicating which positions are valid.
    epsilon : float
        Small value to prevent divide by zero when computing unit vectors.
        Default is 1e-12.

    Returns
    -------
    distances : JaxArray
        (max_agents, max_agents) array of distances between all agents.
        Invalid agent pairs have distance 0.
    unit_vectors : JaxArray
        (max_agents, max_agents, 3) array of unit vectors between all agents.
        unit_vectors[i, j, :] is the unit vector from agent i to agent j.
        Invalid agent pairs have zero vectors.
    """
    # Create validity mask for pairs
    valid_pairs = valid_positions_mask[:, None] & valid_positions_mask[None, :]

    # Compute vectors between all positions
    vectors = positions[None, :, :] - positions[:, None, :]

    # Compute distances
    distances = jnp.linalg.norm(vectors, axis=2)

    # Mask out invalid pairs and diagonal
    distances = jnp.where(valid_pairs, distances, 0.0)
    distances = distances.at[jnp.diag_indices_from(distances)].set(0.0)

    # Create unit vectors with epsilon to avoid division by zero
    safe_distances = jnp.maximum(distances, epsilon)
    unit_vectors = vectors / safe_distances[:, :, None]

    # Mask out invalid pairs
    unit_vectors = jnp.where(valid_pairs[:, :, None], unit_vectors, 0.0)

    return distances, unit_vectors


def cell_cell_adhesion_potential(
    positions: jnp.ndarray,
    interaction_radii: jnp.ndarray,
    adhesion_strength: jnp.ndarray,
    valid_positions_mask: jnp.ndarray,
    power: float = 1.0,
) -> jnp.ndarray:
    """Compute an adhesion force with a potential function.

    This function computes adhesion forces between all pairs of valid agents
    within their interaction radii using a potential function.

    Parameters
    ----------
    positions : jnp.ndarray
        (max_agents, 3) array of the padded positions of the agents.
    interaction_radii : jnp.ndarray
        (max_agents,) array of the padded radius of each agent.
    adhesion_strength : jnp.ndarray
        The padded adhesion strength. Must be broadcastable
        to (max_agents, max_agents).
    valid_positions_mask : jnp.ndarray
        (max_agents,) boolean array indicating which positions are valid.
    power : float
        The power of the potential function.
        Note the computed exponent is power + 1.
        Default is 1.0.

    Returns
    -------
    forces : JaxArray
        (max_agents, 3) array of the cell-cell adhesion
        force vector for each agent. Invalid agents have zero force.
    """
    distances, vectors = vectors_distances_between_agents(
        positions, valid_positions_mask
    )

    jax.debug.print("Vector 0,1 {vectors}", vectors=vectors[0, 1, :])

    # Compute maximum interaction distance for each pair
    maximum_interaction_distance = (
        interaction_radii[:, None] + interaction_radii[None, :]
    )

    # Compute normalized distances (safe division by using where)
    normalized_distances = jnp.where(
        maximum_interaction_distance > 0,
        distances / maximum_interaction_distance,
        1.0,  # Set to 1.0 to make force 0 when max_interaction_distance is 0
    )

    # Compute force magnitudes using potential function
    force_magnitude = adhesion_strength * jnp.power(1 - normalized_distances, power + 1)

    # Mask out forces beyond interaction distance and diagonal
    within_range = distances <= maximum_interaction_distance
    force_magnitude = jnp.where(within_range, force_magnitude, 0.0)
    force_magnitude = force_magnitude.at[jnp.diag_indices_from(force_magnitude)].set(
        0.0
    )

    # Apply validity mask
    valid_pairs = valid_positions_mask[:, None] & valid_positions_mask[None, :]
    force_magnitude = jnp.where(valid_pairs, force_magnitude, 0.0)

    # Compute force vectors
    forces = force_magnitude[:, :, None] * vectors

    # Sum forces for each agent
    total_forces = jnp.sum(forces, axis=1)

    # Mask out invalid agents
    total_forces = jnp.where(valid_positions_mask[:, None], total_forces, 0.0)

    return total_forces


def cell_cell_repulsion_potential(
    positions: JaxArray,
    interaction_radii: JaxArray,
    repulsion_strength: JaxArray,
    valid_positions_mask: JaxArray,
    power: float = 1.0,
) -> JaxArray:
    """Compute a repulsion force with a potential function.

    This function computes repulsion forces between all pairs of valid agents
    within their interaction radii using a potential function.

    Parameters
    ----------
    positions : JaxArray
        (max_agents, 3) array of the padded positions of the agents.
    interaction_radii : JaxArray
        (max_agents,) array of the padded radius of each agent.
    repulsion_strength : JaxArray
        The padded repulsion strength. Must be broadcastable
        to (max_agents, max_agents).
    valid_positions_mask : JaxArray
        (max_agents,) boolean array indicating which positions are valid.
    power : float
        The power of the potential function.
        Note the computed exponent is power + 1.
        Default is 1.0.

    Returns
    -------
    forces : JaxArray
        (max_agents, 3) array of the cell-cell repulsion
        force vector for each agent. Invalid agents have zero force.
    """
    distances, vectors = vectors_distances_between_agents(
        positions, valid_positions_mask
    )

    # Compute maximum interaction distance for each pair
    maximum_interaction_distance = (
        interaction_radii[:, None] + interaction_radii[None, :]
    )

    # Compute normalized distances (safe division by using where)
    normalized_distances = jnp.where(
        maximum_interaction_distance > 0,
        distances / maximum_interaction_distance,
        1.0,  # Set to 1.0 to make force 0 when max_interaction_distance is 0
    )

    # Compute force magnitudes using potential function (negative for repulsion)
    force_magnitude = -repulsion_strength * jnp.power(
        1 - normalized_distances, power + 1
    )

    # Mask out forces beyond interaction distance and diagonal
    within_range = distances <= maximum_interaction_distance
    force_magnitude = jnp.where(within_range, force_magnitude, 0.0)
    force_magnitude = force_magnitude.at[jnp.diag_indices_from(force_magnitude)].set(
        0.0
    )

    # Apply validity mask
    valid_pairs = valid_positions_mask[:, None] & valid_positions_mask[None, :]
    force_magnitude = jnp.where(valid_pairs, force_magnitude, 0.0)

    # Compute force vectors
    forces = force_magnitude[:, :, None] * vectors

    # Sum forces for each agent
    total_forces = jnp.sum(forces, axis=1)

    # Mask out invalid agents
    total_forces = jnp.where(valid_positions_mask[:, None], total_forces, 0.0)

    return total_forces

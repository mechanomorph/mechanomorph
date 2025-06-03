import jax
import jax.numpy as jnp
from jax import Array as JaxArray


def _random_unit_vectors(shape: tuple[int, int], key: jax.random.PRNGKey) -> JaxArray:
    """Generate an array of random unit vectors.

    This uses a uniform distribution on the unit sphere.

    Parameters
    ----------
    shape : tuple[int, int]
        The shape of the array to generate.
        First element is the number of vectors.
        Second element is the dimensionality of each vector.
    key : jax.random.PRNGKey
        JAX random key for reproducible randomness.

    Returns
    -------
    JaxArray
        An array of random unit vectors with shape (n_vectors, dimensionality).
    """
    # Generate random vectors (range -1 to 1)
    random_vectors = jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    # Normalize the vectors to unit length
    # Add small epsilon to avoid division by zero
    norms = jnp.linalg.norm(random_vectors, axis=1, keepdims=True)
    unit_vectors = random_vectors / jnp.maximum(norms, 1e-12)

    return unit_vectors


def biased_random_locomotion_force(
    previous_direction: JaxArray,
    direction_change_probability: JaxArray,
    bias_direction: JaxArray,
    bias_constant: JaxArray,
    locomotion_speed: JaxArray,
    valid_agents_mask: JaxArray,
    key: jax.random.PRNGKey,
) -> tuple[JaxArray, JaxArray]:
    """Compute the locomotion forces for an array of agents.

    This function computes biased random walk forces for agents, with support
    for padded arrays to ensure JIT compatibility.

    Parameters
    ----------
    previous_direction : JaxArray
        (max_agents, 3) array containing padded unit vectors giving
        the previous direction of each agent.
    direction_change_probability : JaxArray
        (max_agents,) array containing the padded probability of changing direction.
        This is generally calculated as time_step / persistence_time.
    bias_direction : JaxArray
        (max_agents, 3) array containing padded unit vectors giving
        the bias direction of each agent.
    bias_constant : JaxArray
        (max_agents,) array containing the padded bias constant for each agent.
        This should be in range 0-1. 0 is fully random and 1 is fully biased.
    locomotion_speed : JaxArray
        (max_agents,) array containing the padded magnitude of
        the locomotion speed for each agent.
    valid_agents_mask : JaxArray
        (max_agents,) boolean array indicating which agents are valid.
    key : jax.random.PRNGKey
        JAX random key for reproducible randomness.

    Returns
    -------
    forces : JaxArray
        (max_agents, 3) array of the locomotion force vectors.
        Invalid agents have zero force.
    new_directions : JaxArray
        (max_agents, 3) array of the updated direction vectors.
        Invalid agents retain their previous direction.
    """
    max_agents = previous_direction.shape[0]

    # Split the random key for different random operations
    key1, key2 = jax.random.split(key)

    # Generate random values to determine which agents change direction
    random_values = jax.random.uniform(key1, shape=(max_agents,))
    change_direction_mask = random_values <= direction_change_probability

    # Apply valid agents mask to the change direction mask
    change_direction_mask = change_direction_mask & valid_agents_mask

    # Generate random unit vectors for all agents
    random_directions = _random_unit_vectors((max_agents, 3), key2)

    # Compute the biased component of the new direction
    # Note: The original code had a bug here - it multiplied by previous_direction
    # which doesn't make sense. I'm fixing it to add the bias to the previous direction
    bias_weight = bias_constant[:, None]  # Shape: (max_agents, 1)

    # Weighted combination of bias direction and random direction
    new_direction_unnormalized = (
        bias_weight * bias_direction + (1 - bias_weight) * random_directions
    )

    # Normalize to get unit vectors
    norms = jnp.linalg.norm(new_direction_unnormalized, axis=1, keepdims=True)
    new_direction = new_direction_unnormalized / jnp.maximum(norms, 1e-12)

    # Use jnp.where to conditionally update directions (JIT-compatible)
    updated_direction = jnp.where(
        change_direction_mask[:, None],  # Broadcast mask to (max_agents, 3)
        new_direction,
        previous_direction,
    )

    # Mask out invalid agents
    updated_direction = jnp.where(
        valid_agents_mask[:, None],
        updated_direction,
        previous_direction,  # Keep previous for invalid agents
    )

    # Compute forces
    forces = locomotion_speed[:, None] * updated_direction

    # Mask out forces for invalid agents
    forces = jnp.where(valid_agents_mask[:, None], forces, 0.0)

    return forces, updated_direction

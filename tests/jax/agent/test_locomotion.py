import jax
import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.agent.forces import biased_random_locomotion_force


def test_biased_random_locomotion():
    """Test the biased random locomotion force function."""
    # set the parameters
    previous_direction = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    direction_change_probability = jnp.array([0.5, 0.5])
    bias_direction = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    bias_direction = bias_direction / jnp.linalg.norm(
        bias_direction, axis=1, keepdims=True
    )
    bias_constant = jnp.array([0.5, 0.5])
    locomotion_speed = jnp.array([1.0, 1.0])
    valid_agents_mask = jnp.array([True, True])

    # call the function
    new_velocity, new_direction = biased_random_locomotion_force(
        previous_direction,
        direction_change_probability,
        bias_direction,
        bias_constant,
        locomotion_speed,
        valid_agents_mask,
        jax.random.key(42),
    )

    # check the shape of the output
    assert new_velocity.shape == (2, 3)
    assert new_direction.shape == (2, 3)


def test_biased_random_locomotion_broadcasting():
    """Test that parameters can be broadcasted correctly."""
    # set the parameters
    # this is fully biased motion
    previous_direction = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    direction_change_probability = jnp.array([0.0])
    bias_direction = jnp.array([1.0, 0.0, 0.0])
    bias_constant = jnp.array([1.0])
    locomotion_speed = jnp.array([1.0])
    valid_agents_mask = jnp.array([True, True])

    new_velocity, new_direction = biased_random_locomotion_force(
        previous_direction,
        direction_change_probability,
        bias_direction,
        bias_constant,
        locomotion_speed,
        valid_agents_mask,
        jax.random.key(42),
    )

    np.testing.assert_allclose(
        new_velocity, jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    np.testing.assert_allclose(new_direction, previous_direction)

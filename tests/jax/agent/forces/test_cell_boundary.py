import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.agent.forces import (
    cell_boundary_adhesion_potential,
    cell_boundary_repulsion_potential,
)


def test_boundary_adhesion_potential():
    """Test cell-boundary adhesion potential forces."""
    distances = jnp.array([0.0, 0.5, 10.0])
    normal_vectors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    valid_agents_mask = jnp.array([True, True, True])
    interaction_radii = jnp.array([1.0, 1.0, 1.0])
    adhesion_strength = jnp.array([2.0, 2.0, 2.0])
    power = 0.0

    # Compute the forces
    forces = cell_boundary_adhesion_potential(
        distances=distances,
        normal_vectors=normal_vectors,
        valid_agents_mask=valid_agents_mask,
        interaction_radii=interaction_radii,
        adhesion_strength=adhesion_strength,
        power=power,
    )

    # Check the forces
    expected_forces = jnp.array(
        [
            [-2.0, 0.0, 0.0],  # Force for the first agent
            [0.0, -1.0, 0.0],  # Force for the second agent
            [0.0, 0.0, 0.0],  # (beyond interaction radius)
        ]
    )
    np.testing.assert_allclose(forces, expected_forces)


def test_boundary_repulsion_potential():
    """Test cell-boundary repulsion potential forces."""
    distances = jnp.array([0.0, 0.5, 10.0])
    normal_vectors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    valid_agents_mask = jnp.array([True, True, True])
    interaction_radii = jnp.array([1.0, 1.0, 1.0])
    repulsion_strength = jnp.array([2.0, 2.0, 2.0])
    power = 0.0

    # Compute the forces
    forces = cell_boundary_repulsion_potential(
        distances=distances,
        normal_vectors=normal_vectors,
        valid_agents_mask=valid_agents_mask,
        interaction_radii=interaction_radii,
        repulsion_strength=repulsion_strength,
        power=power,
    )

    # Check the forces
    expected_forces = jnp.array(
        [
            [2.0, 0.0, 0.0],  # Force for the first agent
            [0.0, 1.0, 0.0],  # Force for the second agent
            [0.0, 0.0, 0.0],  # (beyond interaction radius)
        ]
    )
    np.testing.assert_allclose(forces, expected_forces)

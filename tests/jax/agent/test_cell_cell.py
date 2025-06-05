from math import sqrt

import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.agent.forces import (
    cell_cell_adhesion_potential,
    cell_cell_repulsion_potential,
)


def test_cell_cell_adhesion_potential():
    """Test computing cell-cell adhesion forces."""

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0],  # padding
        ]
    )
    valid_positions_mask = jnp.array([True, True, True, False])
    interaction_radii = jnp.array([0.5, 2.0, 0.1, 0.0])
    adhesion_strength = jnp.array(1.0)

    forces = cell_cell_adhesion_potential(
        positions=positions,
        interaction_radii=interaction_radii,
        adhesion_strength=adhesion_strength,
        valid_positions_mask=valid_positions_mask,
        power=0.0,
    )

    # Create force magnitudes (zeros initially)
    expected_force_magnitudes = jnp.zeros((3, 3))
    expected_force_magnitudes = expected_force_magnitudes.at[0, 1].set(1 - (1 / 2.5))
    expected_force_magnitudes = expected_force_magnitudes.at[1, 0].set(1 - (1 / 2.5))
    expected_force_magnitudes = expected_force_magnitudes.at[1, 2].set(
        1 - (sqrt(2) / 2.1)
    )
    expected_force_magnitudes = expected_force_magnitudes.at[2, 1].set(
        1 - (sqrt(2) / 2.1)
    )

    # Define expected vectors
    expected_vectors = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1 / sqrt(2), 1 / sqrt(2), 0.0]],
            [[0.0, -1.0, 0.0], [1 / sqrt(2), -1 / sqrt(2), 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    # Calculate expected forces
    expected_forces = jnp.sum(
        jnp.expand_dims(expected_force_magnitudes, axis=-1) * expected_vectors, axis=1
    )

    # Check the forces
    np.testing.assert_allclose(forces[valid_positions_mask], expected_forces)


def test_cell_cell_repulsion_potential():
    """Test computing cell-cell repulsion forces."""

    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0],  # padding
        ]
    )
    valid_positions_mask = jnp.array([True, True, True, False])
    interaction_radii = jnp.array([0.5, 2.0, 0.1, 0.0])
    repulsion_strength = jnp.array(1.0)

    forces = cell_cell_repulsion_potential(
        positions=positions,
        interaction_radii=interaction_radii,
        repulsion_strength=repulsion_strength,
        valid_positions_mask=valid_positions_mask,
        power=0.0,
    )

    # Create force magnitudes (zeros initially)
    expected_force_magnitudes = jnp.zeros((3, 3))
    expected_force_magnitudes = expected_force_magnitudes.at[0, 1].set(1 - (1 / 2.5))
    expected_force_magnitudes = expected_force_magnitudes.at[1, 0].set(1 - (1 / 2.5))
    expected_force_magnitudes = expected_force_magnitudes.at[1, 2].set(
        1 - (sqrt(2) / 2.1)
    )
    expected_force_magnitudes = expected_force_magnitudes.at[2, 1].set(
        1 - (sqrt(2) / 2.1)
    )

    # Define expected vectors
    expected_vectors = jnp.array(
        [
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1 / sqrt(2), -1 / sqrt(2), 0.0]],
            [[0.0, 1.0, 0.0], [-1 / sqrt(2), 1 / sqrt(2), 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    # Calculate expected forces
    expected_forces = jnp.sum(
        jnp.expand_dims(expected_force_magnitudes, axis=-1) * expected_vectors, axis=1
    )

    # Check the forces
    np.testing.assert_allclose(forces[valid_positions_mask], expected_forces)

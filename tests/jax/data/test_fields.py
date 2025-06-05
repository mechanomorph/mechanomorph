import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.data import (
    sample_scalar_field_linear,
    sample_scalar_field_nearest,
)


def test_sample_scalar_field_nearest():
    # Define a small 3x3x3 field for easy verification
    field = jnp.arange(27).reshape((3, 3, 3)).astype(jnp.float32)

    # Choose origin and scale
    origin = jnp.array([1.0, 1.0, 1.0])
    scale = jnp.array([2.0, 2.0, 2.0])

    # Define coordinates in world space
    coordinates = jnp.array(
        [
            [1.0, 1.0, 1.0],  # Maps to voxel (0,0,0) -> exact match
            [3.9, 3.9, 3.9],  # Maps to (1.45, 1.45, 1.45) -> round to (1,1,1)
            [
                6.1,
                6.1,
                6.1,
            ],  # Maps to (2.55, 2.55, 2.55) -> round to (3,3,3) -> out of bounds
            [-0.2, -0.2, -0.2],  # Maps to (-0.6, -0.6, -0.6) -> out of bounds
            [10.0, 10.0, 10.0],  # Maps to (4.5, 4.5, 4.5) -> out of bounds
        ]
    )

    valid_coordinates = jnp.array([True, True, True, True, True])

    cval = -1.0
    result = sample_scalar_field_nearest(
        coordinates, valid_coordinates, field, origin, scale, cval
    )

    # Expected sampled values:
    # field[0,0,0] = 0
    # field[1,1,1] = 13
    # out of bounds = cval
    # invalid = cval
    expected = jnp.array([0.0, 13.0, cval, cval, cval])

    np.testing.assert_allclose(result, expected)


def test_sample_scalar_field_linear():
    # Simple 3x3x3 field: values range from 0 to 26
    field = jnp.arange(27).reshape((3, 3, 3)).astype(jnp.float32)

    origin = jnp.array([1.0, 1.0, 1.0])
    scale = jnp.array([2.0, 2.0, 2.0])
    cval = -1.0

    # Coordinates in world space
    coordinates = jnp.array(
        [
            [3.0, 3.0, 3.0],  # Maps to (1.0, 1.0, 1.0) — interior, center of cube
            [1.0, 1.0, 1.0],  # Maps to (0.0, 0.0, 0.0) — exact voxel, no interp
            [20.0, 20.0, 20.0],  # out of bounds
            [-20.0, -20.0, -20.0],  # out of bounds
        ]
    )

    valid_coordinates = jnp.array([True, True, True, True])

    result = sample_scalar_field_linear(
        coordinates, valid_coordinates, field, origin, scale, cval
    )

    # Expected:
    # (1.0, 1.0, 1.0) -> voxel center: no interpolation, value = field[1,1,1] = 13
    # (0.0, 0.0, 0.0) -> no interpolation, value = field[0,0,0] = 0
    # out of bounds -> cval
    # invalid -> cval
    expected = jnp.array([13.0, 0.0, cval, cval])
    np.testing.assert_allclose(result, expected)

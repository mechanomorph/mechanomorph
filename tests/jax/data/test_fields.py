import jax.numpy as jnp
import numpy as np

from mechanomorph.jax.data import (
    sample_scalar_field_linear,
    sample_scalar_field_nearest,
    sample_vector_field_linear,
    sample_vector_field_nearest,
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
    """Test sampling a scalar field with linear interpolation."""
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


def test_sample_vector_field_nearest():
    """Test sampling a vector field with nearest neighbor interpolation."""
    # Define a (2,2,2) vector field with 3D vectors at each voxel
    field_last = jnp.array(
        [
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
            [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]]],
        ]
    )  # shape (2, 2, 2, 3)

    field_first = jnp.transpose(field_last, (3, 0, 1, 2))  # shape (3, 2, 2, 2)

    # Origin and scale
    origin = jnp.array([1.0, 1.0, 1.0])
    scale = jnp.array([2.0, 2.0, 2.0])
    cval = -1.0

    # Coordinates in world space
    coordinates = jnp.array(
        [
            [1.0, 1.0, 1.0],  # maps to (0.0, 0.0, 0.0) -> field[0,0,0]
            [2.9, 2.9, 2.9],  # maps to (0.95, 0.95, 0.95) -> round to (1,1,1)
            [5.0, 5.0, 5.0],  # out of bounds
            [-0.1, -0.1, -0.1],  # out of bounds
        ]
    )

    valid_coordinates = jnp.array([True, True, True, False])

    # perform the sampling
    result_first = sample_vector_field_nearest(
        coordinates, valid_coordinates, field_first, origin, scale, cval
    )

    # Calculate the expected results
    expected = jnp.array(
        [
            [1.0, 0.0, 0.0],  # field[0,0,0]
            [0.1, 0.2, 0.3],  # field[1,1,1]
            [cval, cval, cval],  # out-of-bounds
            [cval, cval, cval],  # invalid
        ]
    )
    np.testing.assert_allclose(result_first, expected)


def test_sample_vector_field_linear():
    """Test sampling a vector field with linear interpolation."""
    # Define field in (x, y, z, 3) shape
    field_last = jnp.array(
        [
            [[[0, 10, 20], [1, 11, 21]], [[2, 12, 22], [3, 13, 23]]],
            [[[4, 14, 24], [5, 15, 25]], [[6, 16, 26], [7, 17, 27]]],
        ],
        dtype=jnp.float32,
    )  # shape: (2, 2, 2, 3)

    # Transpose to (3, x, y, z) as required
    field = jnp.transpose(field_last, (3, 0, 1, 2))  # shape: (3, 2, 2, 2)

    # Transformation params
    origin = jnp.array([1.0, 1.0, 1.0])
    scale = jnp.array([2.0, 2.0, 2.0])
    cval = -1.0

    # World-space coordinates to sample
    coordinates = jnp.array(
        [
            [3.0, 3.0, 3.0],  # maps to (1.0, 1.0, 1.0) — corner
            [2.0, 2.0, 2.0],  # maps to (0.5, 0.5, 0.5) — center
            [-10.0, -10.0, -10.0],  # out-of-bounds
            [3.0, 3.0, 3.0],  # valid location but marked invalid
        ]
    )
    valid_coordinates = jnp.array([True, True, True, False])

    # Perform the sampling
    result = sample_vector_field_linear(
        coordinates, valid_coordinates, field, origin, scale, cval
    )

    # Compute expected result for center interpolation
    corner_vectors = jnp.array(
        [
            [0, 10, 20],
            [1, 11, 21],
            [2, 12, 22],
            [3, 13, 23],
            [4, 14, 24],
            [5, 15, 25],
            [6, 16, 26],
            [7, 17, 27],
        ]
    )
    expected_center = jnp.mean(corner_vectors, axis=0)

    expected = jnp.array(
        [
            [7.0, 17.0, 27.0],  # field[1,1,1]
            expected_center,  # interpolated center
            [cval, cval, cval],  # out-of-bounds
            [cval, cval, cval],  # invalid
        ]
    )

    # Compare the results
    np.testing.assert_allclose(result, expected)

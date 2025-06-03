"""Functions to sample fields at specified positions."""

import jax
from jax import Array as JaxArray
from jax import numpy as jnp


def sample_scalar_field_nearest(
    coordinates: JaxArray,
    valid_coordinates: JaxArray,
    field: JaxArray,
    origin: JaxArray,
    scale: JaxArray,
    cval: float = 0.0,
) -> JaxArray:
    """Sample values from a 3D scalar field using nearest neighbor interpolation.

    Parameters
    ----------
    coordinates : JaxArray
        Array of shape (n_coordinates, 3) containing the 3D coordinates
        at which to sample the field. May be padded with invalid entries.
    valid_coordinates : JaxArray
        Array of shape (n_coordinates,) containing boolean values indicating
        which coordinates are valid (True) vs padding (False).
    field : JaxArray
        Array of shape (shape_x, shape_y, shape_z) containing the scalar field values.
    origin : JaxArray, optional
        Array of shape (3,) specifying the origin of the field coordinate system.
        Default is [0.0, 0.0, 0.0].
    scale : JaxArray, optional
        Array of shape (3,) specifying the scale (voxel size) of the field.
        Default is [1.0, 1.0, 1.0].
    cval : float, optional
        Constant value to use for invalid coordinates or coordinates outside
        field bounds. Default is 0.0.

    Returns
    -------
    JaxArray
        Array of shape (n_coordinates,) containing the sampled field values.
        Invalid coordinates will have value `cval`.
    """
    # Transform coordinates to field coordinate system
    field_coords = (coordinates - origin) / scale

    # Get field shape as array for JIT compatibility
    field_shape = jnp.array(field.shape)

    # Vectorize over all coordinates
    def _sample_single_coord(coord: JaxArray, is_valid: JaxArray) -> JaxArray:
        """Sample a single coordinate with validity check.

        Parameters
        ----------
        coord : JaxArray
            Single coordinate of shape (3,).
        is_valid : JaxArray
            Boolean indicating if this coordinate is valid.

        Returns
        -------
        JaxArray
            Sampled value or cval if invalid.
        """
        sampled_value = _sample_scalar_nearest_neighbor(coord, field, field_shape, cval)
        return jax.lax.cond(is_valid, lambda: sampled_value, lambda: cval)

    vectorized_sample = jax.vmap(_sample_single_coord)
    return vectorized_sample(field_coords, valid_coordinates)


def sample_scalar_field_linear(
    coordinates: JaxArray,
    valid_coordinates: JaxArray,
    field: JaxArray,
    origin: JaxArray,
    scale: JaxArray,
    cval: float = 0.0,
) -> JaxArray:
    """Sample values from a 3D scalar field using trilinear interpolation.

    Parameters
    ----------
    coordinates : JaxArray
        Array of shape (n_coordinates, 3) containing the 3D coordinates
        at which to sample the field. May be padded with invalid entries.
    valid_coordinates : JaxArray
        Array of shape (n_coordinates,) containing boolean values indicating
        which coordinates are valid (True) vs padding (False).
    field : JaxArray
        Array of shape (shape_x, shape_y, shape_z) containing the scalar field values.
    origin : JaxArray, optional
        Array of shape (3,) specifying the origin of the field coordinate system.
        Default is [0.0, 0.0, 0.0].
    scale : JaxArray, optional
        Array of shape (3,) specifying the scale (voxel size) of the field.
        Default is [1.0, 1.0, 1.0].
    cval : float, optional
        Constant value to use for invalid coordinates or coordinates outside
        field bounds. Default is 0.0.

    Returns
    -------
    JaxArray
        Array of shape (n_coordinates,) containing the sampled field values.
        Invalid coordinates will have value `cval`.
    """
    # Transform coordinates to field coordinate system
    field_coords = (coordinates - origin) / scale

    # Get field shape as array for JIT compatibility
    field_shape = jnp.array(field.shape)

    # Vectorize over all coordinates
    def _sample_single_coord(coord: JaxArray, is_valid: JaxArray) -> JaxArray:
        """Sample a single coordinate with validity check.

        Parameters
        ----------
        coord : JaxArray
            Single coordinate of shape (3,).
        is_valid : JaxArray
            Boolean indicating if this coordinate is valid.

        Returns
        -------
        JaxArray
            Sampled value or cval if invalid.
        """
        sampled_value = _sample_scalar_linear(coord, field, field_shape, cval)
        return jax.lax.cond(is_valid, lambda: sampled_value, lambda: cval)

    vectorized_sample = jax.vmap(_sample_single_coord)
    return vectorized_sample(field_coords, valid_coordinates)


def sample_vector_field_nearest(
    coordinates: JaxArray,
    valid_coordinates: JaxArray,
    field: JaxArray,
    origin: JaxArray,
    scale: JaxArray,
    cval: float = 0.0,
    vectors_last: bool = True,
) -> JaxArray:
    """Sample vectors from a 3D vector field using nearest neighbor interpolation.

    Parameters
    ----------
    coordinates : JaxArray
        Array of shape (n_coordinates, 3) containing the 3D coordinates
        at which to sample the field. May be padded with invalid entries.
    valid_coordinates : JaxArray
        Array of shape (n_coordinates,) containing boolean values indicating
        which coordinates are valid (True) vs padding (False).
    field : JaxArray
        Vector field array. Shape depends on vectors_last parameter:
        - If vectors_last=True: (shape_x, shape_y, shape_z, 3)
        - If vectors_last=False: (3, shape_x, shape_y, shape_z)
    origin : JaxArray, optional
        Array of shape (3,) specifying the origin of the field coordinate system.
    scale : JaxArray, optional
        Array of shape (3,) specifying the scale (voxel size) of the field.
    cval : float, optional
        Constant value to use for invalid coordinates or coordinates outside
        field bounds. Default is 0.0.
    vectors_last : bool, optional
        Whether vectors are stored in the last dimension. If True, field shape
        is (shape_x, shape_y, shape_z, 3). If False, field shape is
        (3, shape_x, shape_y, shape_z). Default is True.

    Returns
    -------
    JaxArray
        Array of shape (n_coordinates, 3) containing the sampled vector values.
        Invalid coordinates will have vectors filled with `cval`.
    """
    # Transform coordinates to field coordinate system
    field_coords = (coordinates - origin) / scale

    # Handle different vector storage formats using lax.cond for JIT compatibility
    processed_field = jax.lax.cond(
        vectors_last,
        lambda f: f,  # Field shape: (shape_x, shape_y, shape_z, 3)
        lambda f: jnp.transpose(
            f, (1, 2, 3, 0)
        ),  # Transpose (3, x, y, z) -> (x, y, z, 3)
        field,
    )

    spatial_shape = jnp.array(processed_field.shape[:3])

    # Vectorize over all coordinates
    def _sample_single_coord(coord: JaxArray, is_valid: JaxArray) -> JaxArray:
        """Sample a single coordinate with validity check.

        Parameters
        ----------
        coord : JaxArray
            Single coordinate of shape (3,).
        is_valid : JaxArray
            Boolean indicating if this coordinate is valid.

        Returns
        -------
        JaxArray
            Sampled vector or cval-filled vector if invalid.
        """
        sampled_vector = _sample_vector_nearest_neighbor(
            coord, processed_field, spatial_shape, cval
        )
        invalid_vector = jnp.full(processed_field.shape[-1], cval)
        return jax.lax.cond(is_valid, lambda: sampled_vector, lambda: invalid_vector)

    vectorized_sample = jax.vmap(_sample_single_coord)
    return vectorized_sample(field_coords, valid_coordinates)


def sample_vector_field_linear(
    coordinates: JaxArray,
    valid_coordinates: JaxArray,
    field: JaxArray,
    origin: JaxArray,
    scale: JaxArray,
    cval: float = 0.0,
    vectors_last: bool = True,
) -> JaxArray:
    """Sample vectors from a 3D vector field using trilinear interpolation.

    Parameters
    ----------
    coordinates : JaxArray
        Array of shape (n_coordinates, 3) containing the 3D coordinates
        at which to sample the field. May be padded with invalid entries.
    valid_coordinates : JaxArray
        Array of shape (n_coordinates,) containing boolean values indicating
        which coordinates are valid (True) vs padding (False).
    field : JaxArray
        Vector field array. Shape depends on vectors_last parameter:
        - If vectors_last=True: (shape_x, shape_y, shape_z, 3)
        - If vectors_last=False: (3, shape_x, shape_y, shape_z)
    origin : JaxArray, optional
        Array of shape (3,) specifying the origin of the field coordinate system.
    scale : JaxArray, optional
        Array of shape (3,) specifying the scale (voxel size) of the field.
    cval : float, optional
        Constant value to use for invalid coordinates or coordinates outside
        field bounds. Default is 0.0.
    vectors_last : bool, optional
        Whether vectors are stored in the last dimension. If True, field shape
        is (shape_x, shape_y, shape_z, 3). If False, field shape is
        (3, shape_x, shape_y, shape_z). Default is True.

    Returns
    -------
    JaxArray
        Array of shape (n_coordinates, 3) containing the sampled vector values.
        Invalid coordinates will have vectors filled with `cval`.
    """
    # Transform coordinates to field coordinate system
    field_coords = (coordinates - origin) / scale

    # Handle different vector storage formats using lax.cond for JIT compatibility
    processed_field = jax.lax.cond(
        vectors_last,
        lambda f: f,  # Field shape: (shape_x, shape_y, shape_z, 3)
        lambda f: jnp.transpose(
            f, (1, 2, 3, 0)
        ),  # Transpose (3, x, y, z) -> (x, y, z, 3)
        field,
    )

    spatial_shape = jnp.array(processed_field.shape[:3])

    # Vectorize over all coordinates
    def _sample_single_coord(coord: JaxArray, is_valid: JaxArray) -> JaxArray:
        """Sample a single coordinate with validity check.

        Parameters
        ----------
        coord : JaxArray
            Single coordinate of shape (3,).
        is_valid : JaxArray
            Boolean indicating if this coordinate is valid.

        Returns
        -------
        JaxArray
            Sampled vector or cval-filled vector if invalid.
        """
        sampled_vector = _sample_vector_linear(
            coord, processed_field, spatial_shape, cval
        )
        invalid_vector = jnp.full(processed_field.shape[-1], cval)
        return jax.lax.cond(is_valid, lambda: sampled_vector, lambda: invalid_vector)

    vectorized_sample = jax.vmap(_sample_single_coord)
    return vectorized_sample(field_coords, valid_coordinates)


def _sample_scalar_nearest_neighbor(
    coord: JaxArray,
    field: JaxArray,
    field_shape: JaxArray,
    cval: float,
) -> JaxArray:
    """Sample a single point from a scalar field using nearest neighbor interpolation.

    Parameters
    ----------
    coord : JaxArray
        Array of shape (3,) containing the coordinate to sample.
    field : JaxArray
        Array of shape (shape_x, shape_y, shape_z) containing the scalar field.
    field_shape : JaxArray
        Array of shape (3,) containing the field dimensions.
    cval : float
        Constant value for out-of-bounds coordinates.

    Returns
    -------
    JaxArray
        Scalar value sampled from the field.
    """
    # Round to nearest integer indices
    indices = jnp.round(coord).astype(jnp.int32)

    # Check if coordinates are within bounds
    in_bounds = jnp.all((indices >= 0) & (indices < field_shape))

    # Get value with constant boundary handling
    return jax.lax.cond(
        in_bounds, lambda: field[indices[0], indices[1], indices[2]], lambda: cval
    )


def _sample_scalar_linear(
    coord: JaxArray,
    field: JaxArray,
    field_shape: JaxArray,
    cval: float,
) -> JaxArray:
    """Sample a single point from a scalar field using trilinear interpolation.

    Parameters
    ----------
    coord : JaxArray
        Array of shape (3,) containing the coordinate to sample.
    field : JaxArray
        Array of shape (shape_x, shape_y, shape_z) containing the scalar field.
    field_shape : JaxArray
        Array of shape (3,) containing the field dimensions.
    cval : float
        Constant value for out-of-bounds coordinates.

    Returns
    -------
    JaxArray
        Scalar value interpolated from the field.
    """
    # Get integer part (lower corner) and fractional part
    lower_indices = jnp.floor(coord).astype(jnp.int32)
    fractions = coord - lower_indices

    # Define the 8 corner offsets for trilinear interpolation
    offsets = jnp.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    def _get_corner_value(offset: JaxArray) -> JaxArray:
        """Get value at a corner of the interpolation cube.

        Parameters
        ----------
        offset : JaxArray
            Array of shape (3,) containing the offset from the lower corner.

        Returns
        -------
        JaxArray
            The field value at the corner with boundary handling.
        """
        corner_indices = lower_indices + offset
        in_bounds = jnp.all((corner_indices >= 0) & (corner_indices < field_shape))

        return jax.lax.cond(
            in_bounds,
            lambda: field[corner_indices[0], corner_indices[1], corner_indices[2]],
            lambda: cval,
        )

    # Get values at all 8 corners
    corner_values = jax.vmap(_get_corner_value)(offsets)

    # Compute trilinear interpolation weights
    weights = jnp.array(
        [
            (1 - fractions[0]) * (1 - fractions[1]) * (1 - fractions[2]),  # [0,0,0]
            fractions[0] * (1 - fractions[1]) * (1 - fractions[2]),  # [1,0,0]
            (1 - fractions[0]) * fractions[1] * (1 - fractions[2]),  # [0,1,0]
            fractions[0] * fractions[1] * (1 - fractions[2]),  # [1,1,0]
            (1 - fractions[0]) * (1 - fractions[1]) * fractions[2],  # [0,0,1]
            fractions[0] * (1 - fractions[1]) * fractions[2],  # [1,0,1]
            (1 - fractions[0]) * fractions[1] * fractions[2],  # [0,1,1]
            fractions[0] * fractions[1] * fractions[2],  # [1,1,1]
        ]
    )

    # Weighted sum of corner values
    return jnp.sum(weights * corner_values)


def _sample_vector_nearest_neighbor(
    coord: JaxArray,
    field: JaxArray,
    spatial_shape: JaxArray,
    cval: float,
) -> JaxArray:
    """Sample a single point from a vector field using nearest neighbor interpolation.

    Parameters
    ----------
    coord : JaxArray
        Array of shape (3,) containing the coordinate to sample.
    field : JaxArray
        Array of shape (shape_x, shape_y, shape_z, vector_dim)
        containing the vector field.
    spatial_shape : JaxArray
        Array of shape (3,) containing the spatial dimensions of the field.
    cval : float
        Constant value for out-of-bounds coordinates.

    Returns
    -------
    JaxArray
        Vector value sampled from the field.
    """
    # Round to nearest integer indices
    indices = jnp.round(coord).astype(jnp.int32)

    # Check if coordinates are within bounds
    in_bounds = jnp.all((indices >= 0) & (indices < spatial_shape))

    # Get vector with constant boundary handling
    return jax.lax.cond(
        in_bounds,
        lambda: field[indices[0], indices[1], indices[2]],
        lambda: jnp.full(field.shape[-1], cval),
    )


def _sample_vector_linear(
    coord: JaxArray,
    field: JaxArray,
    spatial_shape: JaxArray,
    cval: float,
) -> JaxArray:
    """Sample a single point from a vector field using trilinear interpolation.

    Parameters
    ----------
    coord : JaxArray
        Array of shape (3,) containing the coordinate to sample.
    field : JaxArray
        Array of shape (shape_x, shape_y, shape_z, vector_dim)
        containing the vector field.
    spatial_shape : JaxArray
        Array of shape (3,) containing the spatial dimensions of the field.
    cval : float
        Constant value for out-of-bounds coordinates.

    Returns
    -------
    JaxArray
        Vector value interpolated from the field.
    """
    # Get integer part (lower corner) and fractional part
    lower_indices = jnp.floor(coord).astype(jnp.int32)
    fractions = coord - lower_indices

    # Define the 8 corner offsets for trilinear interpolation
    offsets = jnp.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    def _get_corner_vector(offset: JaxArray) -> JaxArray:
        """Get vector value at a corner of the interpolation cube.

        Parameters
        ----------
        offset : JaxArray
            Array of shape (3,) containing the offset from the lower corner.

        Returns
        -------
        JaxArray
            The field vector at the corner with boundary handling.
        """
        corner_indices = lower_indices + offset
        in_bounds = jnp.all((corner_indices >= 0) & (corner_indices < spatial_shape))

        return jax.lax.cond(
            in_bounds,
            lambda: field[corner_indices[0], corner_indices[1], corner_indices[2]],
            lambda: jnp.full(field.shape[-1], cval),
        )

    # Get vectors at all 8 corners
    corner_vectors = jax.vmap(_get_corner_vector)(offsets)

    # Compute trilinear interpolation weights
    weights = jnp.array(
        [
            (1 - fractions[0]) * (1 - fractions[1]) * (1 - fractions[2]),  # [0,0,0]
            fractions[0] * (1 - fractions[1]) * (1 - fractions[2]),  # [1,0,0]
            (1 - fractions[0]) * fractions[1] * (1 - fractions[2]),  # [0,1,0]
            fractions[0] * fractions[1] * (1 - fractions[2]),  # [1,1,0]
            (1 - fractions[0]) * (1 - fractions[1]) * fractions[2],  # [0,0,1]
            fractions[0] * (1 - fractions[1]) * fractions[2],  # [1,0,1]
            (1 - fractions[0]) * fractions[1] * fractions[2],  # [0,1,1]
            fractions[0] * fractions[1] * fractions[2],  # [1,1,1]
        ]
    )

    # Weighted sum of corner vectors
    return jnp.sum(weights[:, None] * corner_vectors, axis=0)

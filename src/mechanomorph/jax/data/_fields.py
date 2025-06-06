import jax.numpy as jnp
from jax import Array as JaxArray


def sample_scalar_field_linear(
    coordinates: JaxArray, field: JaxArray, origin: JaxArray, scale: JaxArray, cval=0.0
) -> JaxArray:
    """
    Sample a 3D scalar field using trilinear interpolation.

    The coordinate system follows the convention:
    - coordinates[:, 0] corresponds to x-axis (field width dimension)
    - coordinates[:, 1] corresponds to y-axis (field height dimension)
    - coordinates[:, 2] corresponds to z-axis (field depth dimension)
    - field shape is (W, H, D) corresponding to (width, height, depth)

    Parameters
    ----------
    coordinates : JaxArray
        (N, 3) array of coordinates to sample at in world space.
        Each row is [x, y, z] corresponding to [width, height, depth].
    field : JaxArray
        (W, H, D) array representing the 3D scalar field.
        Indexed as field[x_idx, y_idx, z_idx].
    origin : JaxArray
        (3,) array representing the origin of the field coordinate system [x, y, z].
    scale : JaxArray
        (3,) array representing the voxel size in each dimension [x, y, z].
    cval : float
        Constant value for coordinates outside the field bounds.
        Default is 0.0.

    Returns
    -------
    values : JaxArray
        (N,) array of interpolated scalar values.
    """
    W, H, D = field.shape

    # Transform world coordinates to field index space
    coordinates = (coordinates - origin) / scale

    # Extract coordinates: x=width, y=height, z=depth
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Floor and ceil indices for interpolation
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    z0 = jnp.floor(z).astype(jnp.int32)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Interpolation weights (fractional parts)
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Weights for each corner (complement weights)
    wx0, wy0, wz0 = 1.0 - dx, 1.0 - dy, 1.0 - dz
    wx1, wy1, wz1 = dx, dy, dz

    def get_vals(x_idx, y_idx, z_idx):
        """Get field values with bounds checking."""
        # Check bounds
        in_bounds = (
            (x_idx >= 0)
            & (x_idx < W)
            & (y_idx >= 0)
            & (y_idx < H)
            & (z_idx >= 0)
            & (z_idx < D)
        )
        # Clamp indices to valid range for safe indexing
        x_idx = jnp.clip(x_idx, 0, W - 1)
        y_idx = jnp.clip(y_idx, 0, H - 1)
        z_idx = jnp.clip(z_idx, 0, D - 1)
        # Get values and apply cval for out-of-bounds
        vals = field[x_idx, y_idx, z_idx]
        return jnp.where(in_bounds, vals, cval)

    # Get values at all 8 corners of the interpolation cube
    c000 = get_vals(x0, y0, z0)  # (x0, y0, z0)
    c001 = get_vals(x0, y0, z1)  # (x0, y0, z1)
    c010 = get_vals(x0, y1, z0)  # (x0, y1, z0)
    c011 = get_vals(x0, y1, z1)  # (x0, y1, z1)
    c100 = get_vals(x1, y0, z0)  # (x1, y0, z0)
    c101 = get_vals(x1, y0, z1)  # (x1, y0, z1)
    c110 = get_vals(x1, y1, z0)  # (x1, y1, z0)
    c111 = get_vals(x1, y1, z1)  # (x1, y1, z1)

    # Trilinear interpolation
    interp = (
        c000 * wx0 * wy0 * wz0
        + c001 * wx0 * wy0 * wz1
        + c010 * wx0 * wy1 * wz0
        + c011 * wx0 * wy1 * wz1
        + c100 * wx1 * wy0 * wz0
        + c101 * wx1 * wy0 * wz1
        + c110 * wx1 * wy1 * wz0
        + c111 * wx1 * wy1 * wz1
    )

    return interp


def sample_scalar_field_nearest(
    coordinates: JaxArray, field: JaxArray, origin: JaxArray, scale: JaxArray, cval=0.0
) -> JaxArray:
    """
    Sample a 3D scalar field using nearest-neighbor interpolation.

    The coordinate system follows the convention:
    - coordinates[:, 0] corresponds to x-axis (field width dimension)
    - coordinates[:, 1] corresponds to y-axis (field height dimension)
    - coordinates[:, 2] corresponds to z-axis (field depth dimension)
    - field shape is (W, H, D) corresponding to (width, height, depth)

    Parameters
    ----------
    coordinates : JaxArray
        (N, 3) array of coordinates to sample at in world space.
        Each row is [x, y, z] corresponding to [width, height, depth].
    field : JaxArray
        (W, H, D) array representing the 3D scalar field.
        Indexed as field[x_idx, y_idx, z_idx].
    origin : JaxArray
        (3,) array representing the origin of the field coordinate system [x, y, z].
    scale : JaxArray
        (3,) array representing the voxel size in each dimension [x, y, z].
    cval : float
        Constant value for coordinates outside the field bounds.
        Default is 0.0.

    Returns
    -------
    values : JaxArray
        (N,) array of sampled scalar values.
    """
    W, H, D = field.shape

    # Transform world coordinates to field index space
    coordinates = (coordinates - origin) / scale

    # Extract coordinates: x=width, y=height, z=depth
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Round to nearest integer index
    x_idx = jnp.round(x).astype(jnp.int32)
    y_idx = jnp.round(y).astype(jnp.int32)
    z_idx = jnp.round(z).astype(jnp.int32)

    # Check bounds
    in_bounds = (
        (x_idx >= 0)
        & (x_idx < W)
        & (y_idx >= 0)
        & (y_idx < H)
        & (z_idx >= 0)
        & (z_idx < D)
    )

    # Clamp indices to valid range for safe indexing
    x_idx = jnp.clip(x_idx, 0, W - 1)
    y_idx = jnp.clip(y_idx, 0, H - 1)
    z_idx = jnp.clip(z_idx, 0, D - 1)

    # Fetch values: field is indexed as [x, y, z]
    vals = field[x_idx, y_idx, z_idx]

    # Replace out-of-bounds with cval
    return jnp.where(in_bounds, vals, cval)


def sample_vector_field_nearest(
    coordinates: JaxArray,
    field: JaxArray,
    origin: JaxArray,
    scale: JaxArray,
    cval: float = 0.0,
) -> JaxArray:
    """
    Sample a 3D vector field using nearest-neighbor interpolation.

    The coordinate system follows the convention:
    - coordinates[:, 0] corresponds to x-axis (field width dimension)
    - coordinates[:, 1] corresponds to y-axis (field height dimension)
    - coordinates[:, 2] corresponds to z-axis (field depth dimension)
    - field shape is (C, W, H, D) corresponding to (channels, width, height, depth)

    Parameters
    ----------
    coordinates : JaxArray
        (N, 3) array of coordinates to sample at in world space.
        Each row is [x, y, z] corresponding to [width, height, depth].
    field : JaxArray
        (C, W, H, D) array representing the vector field, where C is the
        number of vector components. Indexed as field[c, x_idx, y_idx, z_idx].
    origin : JaxArray
        (3,) array representing the origin of the field coordinate system [x, y, z].
    scale : JaxArray
        (3,) array representing the voxel size in each dimension [x, y, z].
    cval : float
        Constant value to use for out-of-bounds samples.
        Default is 0.0.

    Returns
    -------
    values : JaxArray
        (N, C) array containing the vector field values sampled at each input point.
    """
    C, W, H, D = field.shape

    # Transform world coordinates to field index space
    coordinates = (coordinates - origin) / scale

    # Extract coordinates: x=width, y=height, z=depth
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Round to nearest integer index
    x_idx = jnp.round(x).astype(jnp.int32)
    y_idx = jnp.round(y).astype(jnp.int32)
    z_idx = jnp.round(z).astype(jnp.int32)

    # Check bounds
    in_bounds = (
        (x_idx >= 0)
        & (x_idx < W)
        & (y_idx >= 0)
        & (y_idx < H)
        & (z_idx >= 0)
        & (z_idx < D)
    )

    # Clamp to valid range for indexing
    x_idx_safe = jnp.clip(x_idx, 0, W - 1)
    y_idx_safe = jnp.clip(y_idx, 0, H - 1)
    z_idx_safe = jnp.clip(z_idx, 0, D - 1)

    # Sample vector field: field is indexed as [c, x, y, z]
    samples_all = field[:, x_idx_safe, y_idx_safe, z_idx_safe].T  # (N, C)

    # Apply cval to out-of-bounds points
    return jnp.where(in_bounds[:, None], samples_all, cval)


def sample_vector_field_linear(
    coordinates: JaxArray,
    field: JaxArray,
    origin: JaxArray,
    scale: JaxArray,
    cval: float = 0.0,
) -> JaxArray:
    """
    Sample a 3D vector field using trilinear interpolation.

    The coordinate system follows the convention:
    - coordinates[:, 0] corresponds to x-axis (field width dimension)
    - coordinates[:, 1] corresponds to y-axis (field height dimension)
    - coordinates[:, 2] corresponds to z-axis (field depth dimension)
    - field shape is (C, W, H, D) corresponding to (channels, width, height, depth)

    Parameters
    ----------
    coordinates : JaxArray
        (N, 3) array of coordinates to sample at in world space.
        Each row is [x, y, z] corresponding to [width, height, depth].
    field : JaxArray
        (C, W, H, D) array representing the vector field, where C is the
        number of vector components. Indexed as field[c, x_idx, y_idx, z_idx].
    origin : JaxArray
        (3,) array representing the origin of the field coordinate system [x, y, z].
    scale : JaxArray
        (3,) array representing the voxel size in each dimension [x, y, z].
    cval : float
        Constant value to use for out-of-bounds samples.
        Default is 0.0.

    Returns
    -------
    values : JaxArray
        (N, C) array containing the vector field values sampled at each input point.
    """
    C, W, H, D = field.shape

    # Transform world coordinates to field index space
    coordinates = (coordinates - origin) / scale

    # Extract coordinates: x=width, y=height, z=depth
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Floor and ceil indices for interpolation
    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = y0 + 1
    z0 = jnp.floor(z).astype(jnp.int32)
    z1 = z0 + 1

    # Interpolation weights (fractional parts)
    dx = x - x0
    dy = y - y0
    dz = z - z0

    def get_sample(x_idx, y_idx, z_idx):
        """Get vector field samples with bounds checking."""
        # Check bounds for this corner
        in_bounds = (
            (x_idx >= 0)
            & (x_idx < W)
            & (y_idx >= 0)
            & (y_idx < H)
            & (z_idx >= 0)
            & (z_idx < D)
        )

        # Clamp indices for safe access
        x_idx_safe = jnp.clip(x_idx, 0, W - 1)
        y_idx_safe = jnp.clip(y_idx, 0, H - 1)
        z_idx_safe = jnp.clip(z_idx, 0, D - 1)

        # Get samples: field is indexed as [c, x, y, z]
        samples = field[:, x_idx_safe, y_idx_safe, z_idx_safe].T  # (N, C)

        # Apply cval where out of bounds
        return jnp.where(in_bounds[:, None], samples, cval)

    # Get samples at all 8 corners of the interpolation cube
    c000 = get_sample(x0, y0, z0)  # (x0, y0, z0)
    c001 = get_sample(x0, y0, z1)  # (x0, y0, z1)
    c010 = get_sample(x0, y1, z0)  # (x0, y1, z0)
    c011 = get_sample(x0, y1, z1)  # (x0, y1, z1)
    c100 = get_sample(x1, y0, z0)  # (x1, y0, z0)
    c101 = get_sample(x1, y0, z1)  # (x1, y0, z1)
    c110 = get_sample(x1, y1, z0)  # (x1, y1, z0)
    c111 = get_sample(x1, y1, z1)  # (x1, y1, z1)

    # Trilinear interpolation weights for each corner
    w000 = (1 - dx) * (1 - dy) * (1 - dz)  # (x0, y0, z0)
    w001 = (1 - dx) * (1 - dy) * dz  # (x0, y0, z1)
    w010 = (1 - dx) * dy * (1 - dz)  # (x0, y1, z0)
    w011 = (1 - dx) * dy * dz  # (x0, y1, z1)
    w100 = dx * (1 - dy) * (1 - dz)  # (x1, y0, z0)
    w101 = dx * (1 - dy) * dz  # (x1, y0, z1)
    w110 = dx * dy * (1 - dz)  # (x1, y1, z0)
    w111 = dx * dy * dz  # (x1, y1, z1)

    # Weighted sum of all corners
    values = (
        w000[:, None] * c000
        + w001[:, None] * c001
        + w010[:, None] * c010
        + w011[:, None] * c011
        + w100[:, None] * c100
        + w101[:, None] * c101
        + w110[:, None] * c110
        + w111[:, None] * c111
    )

    return values

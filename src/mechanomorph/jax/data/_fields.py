import jax.numpy as jnp
from jax import Array as JaxArray


def sample_scalar_field_linear(
    coordinates: JaxArray, field: JaxArray, origin: JaxArray, scale: JaxArray, cval=0.0
) -> JaxArray:
    """
    Performs sampling with linear interpolation on a 3D scalar field.

    Parameters
    ----------
    coordinates : JaxArray
        (N, 3) array of float coordinates to sample at.
    field : JaxArray
        (D, H, W) array representing the 3D scalar field.
    origin : JaxArray
        (3,) array representing the origin of the field.
    scale : JaxArray
        (3,) array representing the scale of the field.
    cval : float
        Constant value for coordinates outside the field.

    Returns
    -------
    values : JaxArray
        (n_coords,) array of interpolated scalar values.
    """
    D, H, W = field.shape

    coordinates = (coordinates - origin) / scale

    # Split coordinates
    z, y, x = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Floor and ceil indices
    z0 = jnp.floor(z).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)

    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1

    # Weights for interpolation
    dz = z - z0
    dy = y - y0
    dx = x - x0

    wz0, wy0, wx0 = 1.0 - dz, 1.0 - dy, 1.0 - dx
    wz1, wy1, wx1 = dz, dy, dx

    # Prepare all 8 corners
    def get_vals(z_idx, y_idx, x_idx):
        # Check bounds
        in_bounds = (
            (z_idx >= 0)
            & (z_idx < D)
            & (y_idx >= 0)
            & (y_idx < H)
            & (x_idx >= 0)
            & (x_idx < W)
        )
        z_idx = jnp.clip(z_idx, 0, D - 1)
        y_idx = jnp.clip(y_idx, 0, H - 1)
        x_idx = jnp.clip(x_idx, 0, W - 1)
        vals = field[z_idx, y_idx, x_idx]
        return jnp.where(in_bounds, vals, cval)

    c000 = get_vals(z0, y0, x0)
    c001 = get_vals(z0, y0, x1)
    c010 = get_vals(z0, y1, x0)
    c011 = get_vals(z0, y1, x1)
    c100 = get_vals(z1, y0, x0)
    c101 = get_vals(z1, y0, x1)
    c110 = get_vals(z1, y1, x0)
    c111 = get_vals(z1, y1, x1)

    # Combine with weights
    interp = (
        c000 * wz0 * wy0 * wx0
        + c001 * wz0 * wy0 * wx1
        + c010 * wz0 * wy1 * wx0
        + c011 * wz0 * wy1 * wx1
        + c100 * wz1 * wy0 * wx0
        + c101 * wz1 * wy0 * wx1
        + c110 * wz1 * wy1 * wx0
        + c111 * wz1 * wy1 * wx1
    )

    return interp


def sample_scalar_field_nearest(
    coordinates: JaxArray, field: JaxArray, origin: JaxArray, scale: JaxArray, cval=0.0
) -> JaxArray:
    """
    Performs fused nearest-neighbor sampling on a 3D scalar field.

    Args:
        field: (D, H, W) array representing the 3D scalar field.
        coordinates: (N, 3) array of float coordinates to sample at.
        cval: Constant value for coordinates outside the field.

    Returns
    -------
        (N,) array of sampled scalar values.
    """
    D, H, W = field.shape

    # apply the transform to the coordinates
    coordinates = (coordinates - origin) / scale

    z, y, x = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Round to nearest integer index
    z_idx = jnp.round(z).astype(jnp.int32)
    y_idx = jnp.round(y).astype(jnp.int32)
    x_idx = jnp.round(x).astype(jnp.int32)

    # Check bounds
    in_bounds = (
        (z_idx >= 0)
        & (z_idx < D)
        & (y_idx >= 0)
        & (y_idx < H)
        & (x_idx >= 0)
        & (x_idx < W)
    )

    # Clamp indices to valid range
    z_idx = jnp.clip(z_idx, 0, D - 1)
    y_idx = jnp.clip(y_idx, 0, H - 1)
    x_idx = jnp.clip(x_idx, 0, W - 1)

    # Fetch values
    vals = field[z_idx, y_idx, x_idx]

    # Replace out-of-bounds with cval
    return jnp.where(in_bounds, vals, cval)


def sample_vector_field_nearest(
    coordinates: JaxArray,
    field: JaxArray,
    origin: JaxArray,
    scale: JaxArray,
    cval: float,
) -> JaxArray:
    """
    Sample a 3D vector field using nearest-neighbor sampling.

    Out-of-bounds coordinates return a constant value.

    Parameters
    ----------
    field : JaxArray
        Array of shape (C, D, H, W) representing the vector field,
        where C is the number of dimensions of the vectors.
    coordinates : JaxArray
        Array of shape (N, 3) containing N (x, y, z) coordinates in voxel index space.
    origin : JaxArray
        Origin of the field coordinate system, used to transform coordinates.
    scale : JaxArray
        Scale of the field coordinate system, used to transform coordinates.
    cval : float
        Constant value to use for out-of-bounds samples.

    Returns
    -------
    values : JaxArray
        Array of shape (N, C) containing the vector field values
        sampled at each input point.
    """
    _, D, H, W = field.shape

    coordinates = (coordinates - origin) / scale
    coordinates = jnp.round(coordinates).astype(jnp.int32)
    xi, yi, zi = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Check bounds
    in_bounds = (xi >= 0) & (xi < D) & (yi >= 0) & (yi < H) & (zi >= 0) & (zi < W)

    # Clamp to valid range for indexing (only used where valid == True)
    xi_safe = jnp.clip(xi, 0, D - 1)
    yi_safe = jnp.clip(yi, 0, H - 1)
    zi_safe = jnp.clip(zi, 0, W - 1)

    # Sample vector field
    samples_all = field[:, xi_safe, yi_safe, zi_safe].T  # (N, C)

    # Apply cval to invalid points
    return jnp.where(in_bounds[:, None], samples_all, cval)


def sample_vector_field_linear(
    coordinates: JaxArray,
    field: JaxArray,
    origin: JaxArray,
    scale: JaxArray,
    cval: float,
) -> JaxArray:
    """
    Sample a 3D vector field using trilinear interpolation.

    Out-of-bounds coordinates return a constant value.

    Parameters
    ----------
    field : JaxArray
        Array of shape (C, D, H, W) representing the vector field,
        where C is the number of dimensions in the vectors.
    coordinates : JaxArray
        Array of shape (N, 3) containing N (x, y, z) coordinates in voxel index space.
    origin : JaxArray
        The origin of the field coordinate system.
    scale : JaxArray
        The scale (voxel size) of the field.
    cval : float
        Constant value to use for out-of-bounds samples.

    Returns
    -------
    values : JaxArray
        Array of shape (N, C) containing the vector field values
        sampled at each input point.
    """
    _, D, H, W = field.shape

    coordinates = (coordinates - origin) / scale
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

    # Floor and ceil indices
    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = y0 + 1
    z0 = jnp.floor(z).astype(jnp.int32)
    z1 = z0 + 1

    # Clip indices for safe access
    x0c = jnp.clip(x0, 0, W - 1)
    x1c = jnp.clip(x1, 0, W - 1)
    y0c = jnp.clip(y0, 0, H - 1)
    y1c = jnp.clip(y1, 0, H - 1)
    z0c = jnp.clip(z0, 0, D - 1)
    z1c = jnp.clip(z1, 0, D - 1)

    # Fractional parts
    xd = x - x0
    yd = y - y0
    zd = z - z0

    def get_sample(x_idx, y_idx, z_idx):
        return field[:, x_idx, y_idx, z_idx].T  # shape (N, C)

    # Gather values at 8 neighboring corners
    c000 = get_sample(x0c, y0c, z0c)
    c001 = get_sample(x0c, y0c, z1c)
    c010 = get_sample(x0c, y1c, z0c)
    c011 = get_sample(x0c, y1c, z1c)
    c100 = get_sample(x1c, y0c, z0c)
    c101 = get_sample(x1c, y0c, z1c)
    c110 = get_sample(x1c, y1c, z0c)
    c111 = get_sample(x1c, y1c, z1c)

    # Interpolation weights
    w000 = (1 - xd) * (1 - yd) * (1 - zd)
    w001 = (xd) * (1 - yd) * (1 - zd)
    w010 = (1 - xd) * (yd) * (1 - zd)
    w011 = (xd) * (yd) * (1 - zd)
    w100 = (1 - xd) * (1 - yd) * (zd)
    w101 = (xd) * (1 - yd) * (zd)
    w110 = (1 - xd) * (yd) * (zd)
    w111 = (xd) * (yd) * (zd)

    # Weighted sum of neighbors
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

    # Determine if any of the 8 neighbors are out-of-bounds
    in_bounds = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H) & (z0 >= 0) & (z1 < D)

    values = jnp.where(in_bounds[:, None], values, cval)

    return values

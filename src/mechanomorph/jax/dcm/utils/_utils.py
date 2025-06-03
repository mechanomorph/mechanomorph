"""Miscellaneous utility functions for DCM."""

import jax.numpy as jnp


def reshape_flat_array_to_padded(
    flat_array: jnp.ndarray,
    n_cells: int,
    max_vertices_per_cell: int,
) -> jnp.ndarray:
    """Convert a flat array to padded cell format.

    Parameters
    ----------
    flat_array : jnp.ndarray
        Array of shape (n_total_vertices,) containing component labels
        for each vertex in flattened format.
    n_cells : int
        Number of cells.
    max_vertices_per_cell : int
        Maximum number of vertices per cell (padding size).

    Returns
    -------
    padded_labels : jnp.ndarray
        Array of shape (n_cells, max_vertices_per_cell) containing
        the component labels in padded format matching the input vertices.
    """
    # Reshape the flat array back to the padded format
    padded_labels = flat_array.reshape(n_cells, max_vertices_per_cell)
    return padded_labels


def convert_flat_indices_to_padded(
    flat_indices: jnp.ndarray,
    n_cells: int,
    max_vertices_per_cell: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert flat vertex indices to cell and local vertex indices.

    Parameters
    ----------
    flat_indices : jnp.ndarray
        Array of shape (...,) containing global vertex indices
        in flattened format.
    n_cells : int
        Number of cells.
    max_vertices_per_cell : int
        Maximum number of vertices per cell.

    Returns
    -------
    cell_indices : jnp.ndarray
        Array of shape (...,) containing cell indices for each vertex.
    local_indices : jnp.ndarray
        Array of shape (...,) containing local vertex indices within
        each cell (0 to max_vertices_per_cell-1).
    """
    # Integer division to get cell index
    cell_indices = flat_indices // max_vertices_per_cell

    # Modulo to get local vertex index within cell
    local_indices = flat_indices % max_vertices_per_cell

    # Clip to valid ranges to handle any invalid indices (-1)
    cell_indices = jnp.clip(cell_indices, 0, n_cells - 1)
    local_indices = jnp.clip(local_indices, 0, max_vertices_per_cell - 1)

    # Mask invalid indices
    valid_mask = flat_indices >= 0
    cell_indices = jnp.where(valid_mask, cell_indices, -1)
    local_indices = jnp.where(valid_mask, local_indices, -1)

    return cell_indices, local_indices

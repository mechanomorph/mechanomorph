import numpy as np
from jax import numpy as jnp

from mechanomorph.jax.dcm.utils import (
    convert_flat_indices_to_padded,
    reshape_flat_array_to_padded,
)


def test_convert_flat_indices_to_padded():
    """Test converting indices from flat to padded format."""
    indices = jnp.array([0, 1, 2, 3, 4, 5])
    n_cells = 2
    max_vertices_per_cell = 3

    cell_indices, local_indices = convert_flat_indices_to_padded(
        indices,
        n_cells=n_cells,
        max_vertices_per_cell=max_vertices_per_cell,
    )

    # check the results
    np.testing.assert_allclose(cell_indices, [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(local_indices, [0, 1, 2, 0, 1, 2])


def test_reshape_flat_array_to_padded():
    """Test reshaping a flat array to padded format."""
    flat_array = jnp.array([1, 2, 3, 4, 5, 6])
    n_cells = 2
    max_vertices_per_cell = 3

    padded_array = reshape_flat_array_to_padded(
        flat_array,
        n_cells=n_cells,
        max_vertices_per_cell=max_vertices_per_cell,
    )

    # check the results
    expected_padded_array = jnp.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(padded_array, expected_padded_array)

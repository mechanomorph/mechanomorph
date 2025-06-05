"""Functions for operating on data."""

from mechanomorph.jax.data._fields import (
    sample_scalar_field_linear,
    sample_scalar_field_nearest,
    sample_vector_field_linear,
    sample_vector_field_nearest,
)

__all__ = [
    "sample_scalar_field_linear",
    "sample_scalar_field_nearest",
    "sample_vector_field_linear",
    "sample_vector_field_nearest",
]

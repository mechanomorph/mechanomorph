"""Utility functions for the DCM model."""

from mechanomorph.jax.dcm.utils._geometry import (
    compute_cell_volume,
    compute_cell_volume_packed,
    gradient_cell_volume_wrt_node_positions,
)
from mechanomorph.jax.dcm.utils._mesh import (
    compute_face_normal_centroid_dot_product,
    detect_aabb_intersections,
    pack_mesh_to_cells,
)

__all__ = [
    "compute_cell_volume",
    "compute_cell_volume_packed",
    "detect_aabb_intersections",
    "gradient_cell_volume_wrt_node_positions",
    "pack_mesh_to_cells",
    "compute_face_normal_centroid_dot_product",
]

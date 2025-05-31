"""Tooling for working with 3D triangular meshes."""

from mechanomorph.mesh._resample import resample_mesh_voronoi
from mechanomorph.mesh._sample_data import make_cube_doublet

__all__ = ["resample_mesh_voronoi", "make_cube_doublet"]

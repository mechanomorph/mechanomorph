"""Utilities for reading and writing images."""

import os

import dask.array as da
import tifffile


def read_lazy_tiff(path: os.PathLike) -> da.Array:
    """Load a tif as a lazy dask array.

    Parameters
    ----------
    path : os.PathLike
        The path to the io to load.

    Returns
    -------
    dask.array.Array
        The io as a lazy dask array.
    """
    zarr_store = tifffile.imread(path, aszarr=True)
    return da.from_zarr(zarr_store)

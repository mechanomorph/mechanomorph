"""Example script to convert tracks to trackastra data for training."""

import glob
import pickle
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import networkx as nx
import numpy as np
import tifffile
from skimage.filters import gaussian
from skimage.io import imread
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from mechanomorph.track.convert import graph_to_ctc


def load_graph_pickle(file_path: PathLike) -> nx.DiGraph:
    """Load a networkx tracking graph that was pickled."""
    with open(file_path, "rb") as file_handle:
        graph = pickle.load(file_handle)
    return graph


def load_segmentations_from_paths(file_paths: list[PathLike]) -> np.ndarray:
    """Load segmentations from an ordered lists of paths.

    Parameters
    ----------
    file_paths : list[PathLike]
        The ordered list of paths to load from.
        Must be readable my skimage imread.

    Returns
    -------
    np.ndarray
        The (t, z, y, x) array of segmantations.
    """
    return np.stack([imread(path) for path in file_paths])


def make_image_from_mask(mask: np.ndarray, gaussian_sigma: float = 1) -> np.ndarray:
    """Convert a segmentation mask into an image.

    Applies a Gaussian blur to the segment edges.

    Parameters
    ----------
    mask : np.ndarray
        The label image for the segmentation.
    gaussian_sigma : float
        The standard deviation of the gaussian kernel.
        Default value is 1.

    Returns
    -------
    np.ndarray
        The synthetic edge image.
    """
    # make the boundary image
    boundary_mask = find_boundaries(mask) > 0
    boundary_image = boundary_mask.astype(float)

    return gaussian(boundary_image, sigma=gaussian_sigma)


def convert_to_trackastra_training_data(
    graph_path: PathLike,
    segmentation_paths: list[PathLike],
    raw_output_directory_path: PathLike,
    track_output_directory_path: PathLike,
    frame_attribute: str,
    label_attribute: str,
):
    """Convert to the trackastra CTCC formatted data."""
    graph = load_graph_pickle(graph_path)
    segmentations = load_segmentations_from_paths(segmentation_paths)

    # convert and write out the data
    tracking_table, masks = graph_to_ctc(
        graph=graph,
        masks_original=segmentations,
        check=True,
        frame_attribute=frame_attribute,
        label_attribute=label_attribute,
        outdir=None,
    )

    # make the output folders
    track_output_directory = Path(track_output_directory_path)
    track_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    raw_output_directory = Path(raw_output_directory_path)
    raw_output_directory.mkdir(parents=True, exist_ok=True)

    # save the table
    tracking_table.to_csv(
        track_output_directory / "man_track.txt", index=False, header=False, sep=" "
    )

    # save the images
    for i, mask in tqdm(enumerate(masks), total=len(masks), desc="Saving masks"):
        file_name = f"man_track{i:04d}.tif"

        # write the mask
        tifffile.imwrite(
            track_output_directory / file_name,
            mask,
            compression="zstd",
        )

        # write the raw image
        # todo use real images
        tifffile.imwrite(
            raw_output_directory / file_name,
            make_image_from_mask(mask, gaussian_sigma=2),
        )


@dataclass
class DataToConvert:
    """Information for a single dataset to convert to trackastra."""

    graph_path: PathLike
    raw_directory_path: PathLike
    segmentation_directory_path: PathLike
    segmentation_file_template: str
    base_name: str
    frame_attribute: str
    label_attribute: str


if __name__ == "__main__":
    # base directory in which to save all ground truth
    base_directory = Path("./track_data")

    # node data key on the tracking graph for the time index
    frame_attribute = "t"

    # node data key on the tracking graph for the segmentation label
    label_attribute = "opticell_label"

    # paths to the data to convert
    all_datasets = [
        DataToConvert(
            graph_path=f"/nas/groups/iber/Projects/Embryo_parameter_estimation/old/process_all_opticell3d_20240812/track_old/embryo_{embryo_index}/curated_graph.pkl",
            raw_directory_path=f"/nas/groups/iber/Projects/Embryo_parameter_estimation/old/process_all_opticell3d_20240812/pre_processed/embryo_{embryo_index}",
            segmentation_directory_path=f"/nas/groups/iber/Projects/Embryo_parameter_estimation/old/process_all_opticell3d_20240812/track_old/embryo_{embryo_index}/relabeled_segmentation",
            segmentation_file_template=f"embryo{embryo_index}" + "_t_{}_tracked.tif",
            base_name=f"embryo{embryo_index}",
            frame_attribute=frame_attribute,
            label_attribute=label_attribute,
        )
        for embryo_index in [3, 4, 5, 6]
    ]

    # process all files
    for dataset in all_datasets:
        segmentation_directory_path = Path(dataset.segmentation_directory_path)
        n_images = len(glob.glob((segmentation_directory_path / "*.tif").as_posix()))
        segmentation_paths = [
            segmentation_directory_path
            / dataset.segmentation_file_template.format(time_index)
            for time_index in range(n_images)
        ]

        # directory to store all of re-formmated data from this dataset
        ground_truth_directory = base_directory / f"{dataset.base_name}_GT"

        # directory to save the reformmated tracks to
        tracks_directory = ground_truth_directory / "TRA"

        # directory to save the raw image to
        raw_directory = base_directory / dataset.base_name

        convert_to_trackastra_training_data(
            graph_path=dataset.graph_path,
            segmentation_paths=segmentation_paths,
            track_output_directory_path=tracks_directory,
            raw_output_directory_path=raw_directory,
            frame_attribute=dataset.frame_attribute,
            label_attribute=dataset.label_attribute,
        )

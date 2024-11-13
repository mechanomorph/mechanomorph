"""Example script to convert tracks to trackastra data for training."""

import glob
import pickle
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import networkx as nx
import numpy as np
from skimage.io import imread

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


def convert_to_trackastra_training_data(
    graph_path: PathLike,
    segmentation_paths: list[PathLike],
    output_directory_path: PathLike,
    frame_attribute: str,
    label_attribute: str,
):
    """Convert to the trackastra CTCC formatted data."""
    graph = load_graph_pickle(graph_path)
    segmentations = load_segmentations_from_paths(segmentation_paths)

    # convert and write out the data
    graph_to_ctc(
        graph=graph,
        masks_original=segmentations,
        check=True,
        frame_attribute=frame_attribute,
        label_attribute=label_attribute,
        outdir=output_directory_path,
    )


@dataclass
class DataToConvert:
    """Information for a single dataset to convert to trackastra."""

    graph_path: PathLike
    segmentation_directory_path: PathLike
    segmentation_file_template: str
    base_name: str
    frame_attribute: str
    label_attribute: str


if __name__ == "__main__":
    # base directory in which to save all ground truth
    base_directory = Path(".")

    # node data key on the tracking graph for the time index
    frame_attribute = "t"

    # node data key on the tracking graph for the segmentation label
    label_attribute = "opticell_label"

    # paths to the data to convert
    all_datasets = [
        DataToConvert(
            graph_path="/Users/kyamauch/Documents/early_embryo/mzg_20240424/curated_graph.pkl",
            segmentation_directory_path="/Users/kyamauch/Documents/early_embryo/mzg_20240424/track_preprocessed/relabeled_segmentation",
            segmentation_file_template="embryo3_t_{}_tracked.tif",
            base_name="embryo3",
            frame_attribute=frame_attribute,
            label_attribute=label_attribute,
        ),
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

        convert_to_trackastra_training_data(
            graph_path=dataset.graph_path,
            segmentation_paths=segmentation_paths,
            output_directory_path=tracks_directory,
            frame_attribute=dataset.frame_attribute,
            label_attribute=dataset.label_attribute,
        )

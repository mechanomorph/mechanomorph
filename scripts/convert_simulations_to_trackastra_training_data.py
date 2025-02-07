"""Example script to convert tracks to trackastra data for training."""

import glob
import pickle
from dataclasses import dataclass
from functools import partial
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

    # convert the graph to a directed graph
    # note that this creates a directed edge in both directions
    # directed_graph = graph.to_directed()
    #
    # # get the edges that point backward in time
    # edges_to_remove = []
    # for edge in directed_graph.edges():
    #     # edges are ((time_start, label_start), (time_end, label_end))
    #     start_node_time = edge[0][0]
    #     end_node_time = edge[1][0]
    #     if start_node_time > end_node_time:
    #         edges_to_remove.append(edge)
    #
    # # remove the edges that point backwards in time
    # for edge in edges_to_remove:
    #     directed_graph.remove_edge(edge[0], edge[1])

    # return directed_graph

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

    print(
        f"{track_output_directory_path}: {segmentations.shape[0]} segmentations loaded"
    )

    # add the node data
    node_data = {}
    for time_index, label_value in graph.nodes():
        node_data[(time_index, label_value)] = {
            frame_attribute: time_index - 1,
            label_attribute: label_value,
        }
    nx.set_node_attributes(graph, node_data)

    # remove the node with negative time
    nodes_to_remove = []
    for node_key, node_data in graph.nodes(data=True):
        time_index = node_data[frame_attribute]
        if time_index < 0:
            nodes_to_remove.append(node_key)

    for node_key in nodes_to_remove:
        print(node_key)
        graph.remove_node(node_key)

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
            compression="zstd",
        )


def convert_directory(
    directory_path: Path,
    output_base_directory: Path,
    graph_name: str = "graph.gpickle",
    segmentation_file_template: str = "result_{}.tif",
    frame_attribute: str = "t",
    label_attribute: str = "opticell_label",
) -> None:
    """Iterate over files in a directory to convert to trackastra training data."""
    graph_path = directory_path / graph_name

    for subdirectory in ["voxel_size_05_um", "voxel_size_075_um"]:
        # get the segmentation paths
        segmentation_directory_path = directory_path / "seg_label_images" / subdirectory
        n_images = len(glob.glob((segmentation_directory_path / "*.tif").as_posix()))
        segmentation_paths = [
            segmentation_directory_path / segmentation_file_template.format(time_index)
            for time_index in range(n_images)
        ]

        # make the output directory path
        output_base_name = f"{directory_path.stem}_{subdirectory}"

        # directory to store all of re-formmated data from this dataset
        ground_truth_directory = output_base_directory / f"{output_base_name}_GT"

        # directory to save the reformmated tracks to
        track_output_directory_path = ground_truth_directory / "TRA"

        # directory to save the raw image to
        raw_output_directory = output_base_directory / output_base_name

        convert_to_trackastra_training_data(
            graph_path=graph_path,
            segmentation_paths=segmentation_paths,
            track_output_directory_path=track_output_directory_path,
            raw_output_directory_path=raw_output_directory,
            frame_attribute=frame_attribute,
            label_attribute=label_attribute,
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
    # a list of the simulation output directories
    base_simulation_directory = Path(
        "/nas/groups/iber/Projects/Embryo_parameter_estimation/ground_truth_movies_tracking/img_tracking/simulation_outputs"
    )

    directory_pattern = str(base_simulation_directory / "simulation_*")
    simulation_directories = [Path(path) for path in glob.glob(directory_pattern)]

    # base directory in which to save all ground truth
    output_base_directory = Path("./track_data")

    # node data key on the tracking graph for the time index
    frame_attribute = "t"

    # node data key on the tracking graph for the segmentation label
    label_attribute = "opticell_label"

    # format for making the segmentation file name
    # the time frame index is inserted in the {}
    segmentation_file_template = "result_{}.tif"

    # the name of the ground truth tracking graph
    graph_name = "graph.gpickle"

    conversion_function = partial(
        convert_directory,
        output_base_directory=output_base_directory,
        graph_name=graph_name,
        segmentation_file_template=segmentation_file_template,
        frame_attribute=frame_attribute,
        label_attribute=label_attribute,
    )

    # with mp.get_context('spawn').Pool() as pool:
    #     pool.map(conversion_function, simulation_directories)

    for directory in simulation_directories:
        conversion_function(directory)

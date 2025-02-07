"""Functions to convert between tracking formats.

Adapted from https://github.com/weigertlab/trackastra

BSD 3-Clause License

Copyright (c) 2024, Benjamin Gallusser, Martin Weigert

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import logging
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CtcTracklet:
    """Model for a single tracklet in the CTC format."""

    def __init__(self, parent: int, nodes: list[int], start_frame: int) -> None:
        self.parent = parent
        self.nodes = nodes
        self.start_frame = start_frame

    def __lt__(self, other):
        """Return True if this tracklet comes before another tracklet."""
        if self.start_frame < other.start_frame:
            return True
        if self.start_frame > other.start_frame:
            return False
        if self.start_frame == other.start_frame:
            return self.parent < other.parent

    def __str__(self) -> str:
        """Return a string representation of this tracklet."""
        return f"Tracklet(parent={self.parent}, nodes={self.nodes})"

    def __repr__(self) -> str:
        """Return a string representation of this tracklet."""
        return str(self)


def ctc_tracklets(G: nx.DiGraph, frame_attribute: str = "time") -> list[CtcTracklet]:
    """Return all CTC tracklets in a graph, i.e.

    - first node after
        - a division (out_degree of parent = 2)
        - an appearance (in_degree=0)
        - a gap closing event (delta_t to parent node > 1)
    - inner nodes have in_degree=1 and out_degree=1, delta_t=1
    - last node:
        - before a division (out_degree = 2)
        - before a disappearance (out_degree = 0)
        - before a gap closing event (delta_t to next node > 1)
    """
    tracklets = []
    # get all nodes with out_degree == 2 (i.e. parent of a tracklet)

    # Queue of tuples(parent id, start node id)
    starts = deque()
    starts.extend(
        [(p, d) for p in G.nodes for d in G.successors(p) if G.out_degree[p] == 2]
    )
    # set parent = -1 since there is no parent
    starts.extend([(-1, n) for n in G.nodes if G.in_degree[n] == 0])
    while starts:
        _p, _s = starts.popleft()
        nodes = [_s]
        # build a tracklet
        c = _s
        while True:
            if G.out_degree[c] > 2:
                raise ValueError("More than two daughters!")
            if G.out_degree[c] == 2:
                break
            if G.out_degree[c] == 0:
                break
            t_c = G.nodes[c][frame_attribute]
            suc = next(iter(G.successors(c)))
            t_suc = G.nodes[suc][frame_attribute]
            if t_suc - t_c > 1:
                logger.debug(
                    f"Gap closing edge from `{c} (t={t_c})` to `{suc} (t={t_suc})`"
                )
                starts.append((c, suc))
                break
            # Add node to tracklet
            c = next(iter(G.successors(c)))
            nodes.append(c)

        tracklets.append(
            CtcTracklet(
                parent=_p, nodes=nodes, start_frame=G.nodes[_s][frame_attribute]
            )
        )

    return tracklets


def _check_ctc_df(df: pd.DataFrame, masks: np.ndarray):
    """Sanity check of all labels in a CTC dataframe are present in the masks."""
    # Check for empty df
    if len(df) == 0 and np.all(masks == 0):
        return True

    for t in range(df.t1.min(), df.t1.max()):
        sub = df[(df.t1 <= t) & (df.t2 >= t)]
        sub_lab = set(sub.label)
        # Since we have non-negative integer labels,
        # we can np.bincount instead of np.unique for speedup
        masks_lab = set(np.where(np.bincount(masks[t].ravel()))[0]) - {0}
        if not sub_lab.issubset(masks_lab):
            print(f"Missing labels in masks at t={t}: {sub_lab - masks_lab}")
            return False
    return True


def graph_to_ctc(
    graph: nx.DiGraph,
    masks_original: np.ndarray,
    check: bool = True,
    frame_attribute: str = "time",
    label_attribute: str = "label",
    outdir: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Convert graph to ctc track Dataframe and relabeled masks.

    Args:
        graph: with node attributes `frame_attribute` and "label"
        masks_original: list of masks with unique labels
        check: Check CTC format
        frame_attribute: Name of the frame attribute in the graph nodes.
        outdir: path to save results in CTC format.

    Returns
    -------
        pd.DataFrame: track dataframe with columns
            ['track_id', 't_start', 't_end', 'parent_id']
        np.ndarray: masks with unique color for each track
    """
    # each tracklet is a linear chain in the graph
    tracklets = ctc_tracklets(graph, frame_attribute=frame_attribute)

    regions = tuple(
        {reg.label: reg.slice for reg in regionprops(m)}
        for _, m in enumerate(masks_original)
    )

    masks = np.stack([np.zeros_like(m) for m in masks_original])
    rows = []
    # To map parent references to tracklet ids.
    # -1 means no parent, which is mapped to 0 in CTC format.
    node_to_tracklets = {-1: 0}

    # Sort tracklets by parent id
    for i, _tracklet in tqdm(
        enumerate(sorted(tracklets)),
        total=len(tracklets),
        desc="Converting graph to CTC results",
    ):
        _parent = _tracklet.parent
        _nodes = _tracklet.nodes
        label = i + 1

        _start, end = _nodes[0], _nodes[-1]

        t1 = _tracklet.start_frame
        # t1 = graph.nodes[start][frame_attribute]
        t2 = graph.nodes[end][frame_attribute]

        node_to_tracklets[end] = label

        # relabel masks
        for _n in _nodes:
            node = graph.nodes[_n]
            t = int(node[frame_attribute])
            lab = int(node[label_attribute])
            ss = regions[t][lab]
            m = masks_original[t][ss] == lab
            if masks[t][ss][m].max() > 0:
                raise RuntimeError(f"Overlapping masks at t={t}, label={lab}")
            if np.count_nonzero(m) == 0:
                raise RuntimeError(f"Empty mask at t={t}, label={lab}")
            masks[t][ss][m] = label

        rows.append([label, t1, t2, node_to_tracklets[_parent]])

    df = pd.DataFrame(rows, columns=["label", "t1", "t2", "parent"], dtype=int)

    masks = np.stack(masks)

    if check:
        _check_ctc_df(df, masks)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(
            # mode=775,
            parents=True,
            exist_ok=True,
        )
        df.to_csv(outdir / "man_track.txt", index=False, header=False, sep=" ")
        for i, m in tqdm(enumerate(masks), total=len(masks), desc="Saving masks"):
            tifffile.imwrite(
                outdir / f"man_track{i:04d}.tif",
                m,
                compression="zstd",
            )

    return df, masks

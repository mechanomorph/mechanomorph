"""Utility functions for 3D geometry operations."""

import torch


def find_intersecting_bounding_boxes(boxes: torch.Tensor):
    """Find all intersecting 3D axis-aligned bounding boxes.

    Parameters
    ----------
    boxes : torch.Tensor
        A tensor of shape (N, 6) containing the bounding boxes.
        Each box is represented by its min and max coordinates
        in the format (x_min, y_min, z_min, x_max, y_max, z_max).

    Returns
    -------
    intersecting_pairs : torch.Tensor
        A tensor of shape (M, 2) containing the indices of the intersecting boxes.
        Each row represents a pair of intersecting boxes.
    """
    n_boxes = boxes.size(0)

    # Expand boxes to compare every pair
    boxes1 = boxes.unsqueeze(1)  # (N, 1, 6)
    boxes2 = boxes.unsqueeze(0)  # (1, N, 6)

    # Check if boxes overlap on each axis
    overlap_0 = (boxes1[:, :, 3] >= boxes2[:, :, 0]) & (
        boxes2[:, :, 3] >= boxes1[:, :, 0]
    )
    overlap_1 = (boxes1[:, :, 4] >= boxes2[:, :, 1]) & (
        boxes2[:, :, 4] >= boxes1[:, :, 1]
    )
    overlap_2 = (boxes1[:, :, 5] >= boxes2[:, :, 2]) & (
        boxes2[:, :, 5] >= boxes1[:, :, 2]
    )

    intersection = overlap_0 & overlap_1 & overlap_2  # (N, N)

    # Remove self-intersections
    intersection.fill_diagonal_(False)

    # Only get (i, j) where i < j to avoid duplicate pairs
    i, j = torch.triu_indices(n_boxes, n_boxes, offset=1)
    mask = intersection[i, j]

    intersecting_pairs = torch.stack([i[mask], j[mask]], dim=1)  # (M, 2)

    return intersecting_pairs

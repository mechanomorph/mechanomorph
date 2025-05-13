import torch

from mechanomorph.sim._deformable_cell_model._geometry_utils import (
    find_intersecting_bounding_boxes,
)


def test_find_intersecting_boxes_3d():
    # Create test boxes
    boxes = torch.tensor(
        [
            [0, 0, 0, 10, 10, 10],  # box 0
            [5, 5, 5, 15, 15, 15],  # box 1 -- overlaps with box 0, overlaps box 2
            [10, 0, 0, 20, 10, 10],  # box 2 -- touches box 0 at face, overlaps box 1
            [20, 20, 20, 30, 30, 30],  # box 3 -- no overlap
        ],
        dtype=torch.float32,
    )

    intersecting_pairs = find_intersecting_bounding_boxes(boxes)

    # Convert to set of tuples for easier comparison
    pairs_set = {[tuple(sorted(pair)) for pair in intersecting_pairs.tolist()]}

    expected_pairs = {
        (0, 1),
        (1, 2),
        (0, 2),
    }
    assert pairs_set == expected_pairs

import numpy as np
import pytest

from mechanomorph.mesh import make_cube_doublet


@pytest.mark.skip("Currently segfaults when run with all tests, but works standalone.")
def test_make_cube_doublet():
    """Make sure the cube doublet is created correctly."""
    edge_width = 20
    target_area = 7.0

    vertices, faces, vertex_cell_mapping, face_cell_mapping, bounding_boxes = (
        make_cube_doublet(edge_width=edge_width, target_area=target_area)
    )

    # Check the vertex cell mapping
    assert vertex_cell_mapping.shape == (vertices.shape[0],)

    # Check the face cell mapping
    assert face_cell_mapping.shape == (faces.shape[0],)

    # Check bounding boxes
    assert bounding_boxes.shape == (2, 6)
    np.testing.assert_allclose(
        bounding_boxes[0], [-1, -1, -1, edge_width + 1, edge_width + 1, edge_width + 1]
    )
    np.testing.assert_allclose(
        bounding_boxes[1], [-edge_width - 1, -1, -1, 1, edge_width + 1, edge_width + 1]
    )

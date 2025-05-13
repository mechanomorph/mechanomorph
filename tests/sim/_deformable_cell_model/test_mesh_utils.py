import pytest
import torch

from mechanomorph.sim._deformable_cell_model._mesh_utils import (
    compute_face_unit_normals,
    get_face_vertex_mapping,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_get_face_vertex_mapping(dtype):
    """Test the face-vertex mapping function."""
    vertices = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=dtype,
    )

    faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
    )

    face_vertex_n1, face_vertex_n2, face_vertex_n3 = get_face_vertex_mapping(
        vertices=vertices, faces=faces, dtype=dtype
    )

    # test node 1
    expected_face_vertex_n1 = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(face_vertex_n1.to_dense(), expected_face_vertex_n1)

    n1_coordinates = face_vertex_n1 @ vertices
    expected_n1_coordinates = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(n1_coordinates, expected_n1_coordinates)

    # test node 2
    expected_face_vertex_n2 = torch.tensor(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(face_vertex_n2.to_dense(), expected_face_vertex_n2)

    n2_coordinates = face_vertex_n2 @ vertices
    expected_n2_coordinates = torch.tensor(
        [
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(n2_coordinates, expected_n2_coordinates)

    # test node 3
    expected_face_vertex_n3 = torch.tensor(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(face_vertex_n3.to_dense(), expected_face_vertex_n3)

    n3_coordinates = face_vertex_n3 @ vertices
    expected_n3_coordinates = torch.tensor(
        [
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(n3_coordinates, expected_n3_coordinates)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compute_unit_normals(dtype):
    vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype)

    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
        ]
    )

    unit_normals = compute_face_unit_normals(vertices=vertices, faces=faces)
    expected_unit_normals = torch.tensor(
        [
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(unit_normals, expected_unit_normals)

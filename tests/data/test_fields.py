import pytest
import torch

from mechanomorph.data import ScalarField, VectorField


def test_scalar_field():
    """Test sampling from a scalar field."""
    field = torch.zeros((3, 3, 3))
    field[1, :, :] = 1
    field[2, :, :] = 2

    # create the ScalarField with an identity transform
    scalar_field = ScalarField(
        field=field, origin=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)
    )

    # test the sampling
    sample_points = torch.tensor(
        [
            [0.4, 0.4, 0.4],  # should be 0
            [1.5, 1, 1],  # should be 2
            [1.2, 2, 1],  # should be 1
        ]
    )
    values = scalar_field.sample(positions=sample_points, order=0)

    expected_values = torch.tensor([0.0, 2.0, 1.0], dtype=field.dtype)
    torch.testing.assert_close(values, expected_values)

    with pytest.raises(NotImplementedError):
        _ = scalar_field.sample(positions=sample_points, order=1)


def test_scalar_field_transform():
    """Test sampling a scalar field with transform."""
    """Test sampling from a scalar field."""
    field = torch.zeros((3, 3, 3))
    field[1, :, :] = 1
    field[2, :, :] = 2

    # create the ScalarField with a transform
    scalar_field = ScalarField(
        field=field, origin=(5.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0)
    )

    # test the sampling
    sample_points = torch.tensor(
        [
            [5.8, 0.4, 0.4],  # should be 0
            [8.0, 1, 1],  # should be 2
            [7.4, 2, 1],  # should be 1
        ]
    )
    values = scalar_field.sample(positions=sample_points, order=0)

    expected_values = torch.tensor([0.0, 2.0, 1.0], dtype=field.dtype)
    torch.testing.assert_close(values, expected_values)


def test_vector_field():
    """Test sampling a vector field."""
    field = torch.zeros((3, 2, 4, 5))
    field[:, 1, 1, 1] = torch.tensor([1.0, 2.0, 3.0])

    # create the VectorField with an identity transform
    vector_field = VectorField(
        field=field, origin=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)
    )

    # test the sampling
    sample_points = torch.tensor(
        [
            [0.4, 0.4, 0.4],  # should be [0, 0, 0]
            [1.3, 1, 1],  # should be [1, 2, 3]
        ]
    )

    values = vector_field.sample(positions=sample_points, order=0)

    expected_values = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=field.dtype
    )
    torch.testing.assert_close(values, expected_values)

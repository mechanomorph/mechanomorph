from math import sqrt

import torch

from mechanomorph.sim.agent.utils.field import field_gradient


def test_field_gradient():
    """Test computing the gradient of a scalar field."""

    field = torch.zeros((2, 4, 5))
    field[0, 0, 0] = 1.0

    gradient_vectors = field_gradient(field)

    expected_gradient_vectors = torch.zeros(3, 2, 4, 5)
    expected_gradient_vectors[:, 0, 0, 0] = torch.tensor(
        [-1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)]
    )
    expected_gradient_vectors[:, 1, 0, 0] = torch.tensor([-1.0, 0.0, 0.0])
    expected_gradient_vectors[:, 0, 1, 0] = torch.tensor([0.0, -1.0, 0.0])
    expected_gradient_vectors[:, 0, 0, 1] = torch.tensor([0, 0, -1.0])
    torch.testing.assert_close(gradient_vectors, expected_gradient_vectors)

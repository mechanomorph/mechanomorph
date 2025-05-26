import torch

from mechanomorph.sim.agent.forces._contact_utils import (
    vectors_distances_between_agents,
)


def test_vectors_distances_between_agents():
    """Test finding the vectors and distances between agents."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    distances, unit_vectors = vectors_distances_between_agents(positions)

    # check the distances
    expected_distances = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(distances, expected_distances)

    # check the unit vectors
    expected_unit_vectors = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    torch.testing.assert_close(unit_vectors, expected_unit_vectors)

from math import sqrt

import torch

from mechanomorph.sim.agent.forces import cell_cell_adhesion_potential


def test_cell_cell_adhesion_potential():
    """Test computing adhesion forces."""
    # make the parameters
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    interaction_radii = torch.tensor([0.5, 2.0, 0.1])
    adhesion_strength = torch.tensor(1.0)

    forces = cell_cell_adhesion_potential(
        positions=positions,
        interaction_radii=interaction_radii,
        adhesion_strength=adhesion_strength,
        power=0.0,
    )

    # compute the expected forces
    expected_force_magnitudes = torch.zeros((3, 3))
    expected_force_magnitudes[0, 1] = 1 - (1 / 2.5)
    expected_force_magnitudes[1, 0] = 1 - (1 / 2.5)
    expected_force_magnitudes[1, 2] = 1 - (sqrt(2) / 2.1)
    expected_force_magnitudes[2, 1] = 1 - (sqrt(2) / 2.1)

    expected_vectors = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1 / sqrt(2), 1 / sqrt(2), 0.0]],
            [[0.0, -1.0, 0.0], [1 / sqrt(2), -1 / sqrt(2), 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    expected_forces = (
        expected_force_magnitudes.unsqueeze(dim=-1) * expected_vectors
    ).sum(dim=1)

    # check the forces
    torch.testing.assert_close(forces, expected_forces)

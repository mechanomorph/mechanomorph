import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from mechanomorph.data import ScalarField, VectorField
from mechanomorph.sim.agent.forces import (
    cell_boundary_adhesion_potential,
    cell_boundary_repulsion_potential,
)
from mechanomorph.sim.agent.utils.field import field_gradient


def test_boundary_adhesion_potential():
    """Test cell-boundary adhesion potential forces."""
    domain_segmentation = np.zeros((20, 20, 20), dtype=bool)
    domain_segmentation[5:15, 5:15, 5:15] = True
    distances = distance_transform_edt(domain_segmentation)
    distance_field = ScalarField(torch.from_numpy(distances).float())

    normals_field = VectorField(
        field_gradient(distance_field.field),
    )

    positions = torch.tensor(
        [
            [5, 10, 10],
            [10, 10, 10],
        ]
    )
    interaction_radii = torch.tensor([2, 2])
    adhesion_strength = torch.tensor([1, 1])

    forces = cell_boundary_adhesion_potential(
        positions=positions,
        interaction_radii=interaction_radii,
        adhesion_strength=adhesion_strength,
        distance_field=distance_field,
        normals_field=normals_field,
        power=0,
    )

    # check the values
    expected_forces = torch.tensor(
        [
            [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(forces, expected_forces)


def test_boundary_repulsion_potential():
    """Test cell-boundary repulsion potential forces."""
    domain_segmentation = np.zeros((20, 20, 20), dtype=bool)
    domain_segmentation[5:15, 5:15, 5:15] = True
    distances = distance_transform_edt(domain_segmentation)
    distance_field = ScalarField(torch.from_numpy(distances).float())

    normals_field = VectorField(
        field_gradient(distance_field.field),
    )

    positions = torch.tensor(
        [
            [5, 10, 10],
            [10, 10, 10],
        ]
    )
    interaction_radii = torch.tensor([2, 2])
    repulsion_strength = torch.tensor([1, 1])

    forces = cell_boundary_repulsion_potential(
        positions=positions,
        interaction_radii=interaction_radii,
        repulsion_strength=repulsion_strength,
        distance_field=distance_field,
        normals_field=normals_field,
        power=0,
    )

    # check the values
    expected_forces = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(forces, expected_forces)

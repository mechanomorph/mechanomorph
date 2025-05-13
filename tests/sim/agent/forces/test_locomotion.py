import torch

from mechanomorph.sim.agent.forces import biased_random_locomotion_force


def test_biased_random_locomotion():
    """Test the biased random locomotion force function."""

    # set the parameters
    previous_direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    direction_change_probability = torch.tensor([0.5, 0.5])
    bias_direction = torch.nn.functional.normalize(
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]), dim=1
    )
    bias_constant = torch.tensor([0.5, 0.5])
    locomotion_speed = torch.tensor([1.0, 1.0])

    # call the function
    new_velocity, new_direction = biased_random_locomotion_force(
        previous_direction,
        direction_change_probability,
        bias_direction,
        bias_constant,
        locomotion_speed,
    )

    # check the shape of the output
    assert new_velocity.shape == (2, 3)
    assert new_direction.shape == (2, 3)

    # check that the output is a tensor
    assert isinstance(new_velocity, torch.Tensor)
    assert isinstance(new_direction, torch.Tensor)


def test_biased_random_locomotion_broadcasting():
    """Test that parameters can be broadcasted correctly."""
    # set the parameters
    # this is fully biased motion
    previous_direction = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    direction_change_probability = torch.tensor([0.0])
    bias_direction = torch.tensor([1.0, 0.0, 0.0])
    bias_constant = torch.tensor([1.0])
    locomotion_speed = torch.tensor([1.0])

    new_velocity, new_direction = biased_random_locomotion_force(
        previous_direction,
        direction_change_probability,
        bias_direction,
        bias_constant,
        locomotion_speed,
    )

    torch.testing.assert_close(
        new_velocity, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(new_direction, previous_direction)

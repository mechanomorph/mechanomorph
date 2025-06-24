from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray

from mechanomorph.jax.agent.integration import make_simulation_loop


def test_euler_simulation_loop():
    """Test a simple Euler time integration loop."""

    class ParticleState(NamedTuple):
        """State of a particle system."""

        positions: JaxArray
        iteration: int

    def condition_function(state: ParticleState) -> bool:
        """Check if simulation should continue.

        Parameters
        ----------
        state : ParticleState
            Current simulation state.

        Returns
        -------
        bool
            True if iteration < 100, False otherwise.
        """
        return state.iteration < 100

    def compute_forces(state: ParticleState, params: JaxArray) -> JaxArray:
        """Compute the forces for a single iteration.

        Parameters----------
        state : ParticleState
            Current simulation state with positions and iteration.
        params : JaxArray
            Array of shape (2,) containing [param_a, param_b].

        Returns
        -------
        updated_state : ParticleState
            Unchanged state (no mutations in this simple example).
        forces : JaxArray
            Forces array of shape (n_particles, 3).
            Each particle experiences force: [1, 0, 0] * param_a
        """

        # Unit vector in x-direction
        unit_vector = jnp.array([1.0, 0.0, 0.0])

        # Compute force: [1, 0, 0] * param_a
        # This creates force [param_a, 0, 0]
        force_vector = unit_vector * params

        # Apply same force to all particles
        n_particles = state.positions.shape[0]
        forces = jnp.tile(force_vector, (n_particles, 1))

        # Return unchanged state and forces
        return state, forces

    # Set up the initial conditions
    n_particles = 3
    initial_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    initial_state = ParticleState(positions=initial_positions, iteration=0)

    # Create simulation function
    max_iterations = 100
    checkpoint_every_n = 10
    time_step = 0.01
    sim_fn = make_simulation_loop(
        condition_function=condition_function,
        compute_forces_function=compute_forces,
        time_step=time_step,
        checkpoint_every_n=checkpoint_every_n,
        max_iterations=max_iterations,
        checkpoint_properties={"positions": (n_particles, 3)},
    )

    # Define a loss function
    def loss_function(params: JaxArray) -> float:
        """Compute loss as mean displacement in the x-direction.

        Parameters
        ----------
        params : JaxArray
            Force parameters [param_a, param_b].

        Returns
        -------
        float
            Sum of final x-positions.
        """
        final_state, checkpoints = sim_fn(initial_state, params)
        final_positions = final_state.positions

        # get the displacements in the x-direction
        displacements = final_positions[:, 0] - initial_positions[:, 0]

        # Calculate the mean displacement
        return jnp.mean(displacements)

    # Run the simulation
    parameters = jnp.array(5.0)
    final_state, checkpoints = sim_fn(initial_state, parameters)

    # check the final state
    assert final_state.iteration == max_iterations
    expected_final_positions = (
        initial_positions
        + parameters * jnp.array([1.0, 0.0, 0.0]) * max_iterations * time_step
    )
    np.testing.assert_allclose(
        final_state.positions, expected_final_positions, rtol=1e-5
    )

    # check the checkpoint data
    assert "positions" in checkpoints
    n_expected_checkpoints = max_iterations // checkpoint_every_n + 1
    checkpoint_period = max_iterations // checkpoint_every_n
    assert checkpoints["positions"].shape == (n_expected_checkpoints, n_particles, 3)
    expected_checkpoint_positions = np.zeros((n_expected_checkpoints, n_particles, 3))
    for checkpoint_index in range(n_expected_checkpoints):
        checkpoint_iteration = checkpoint_period * checkpoint_index
        checkpoint_time = checkpoint_iteration * time_step
        expected_checkpoint_positions[checkpoint_index, ...] = (
            np.array(initial_positions)
            + np.array(parameters) * np.array([1.0, 0.0, 0.0]) * checkpoint_time
        )
    np.testing.assert_allclose(
        checkpoints["positions"], expected_checkpoint_positions, rtol=1e-5
    )

    assert "times" in checkpoints
    np.testing.assert_allclose(
        checkpoints["times"], jnp.linspace(0, max_iterations * time_step, 11)
    )

    assert "iterations" in checkpoints
    np.testing.assert_allclose(
        checkpoints["iterations"], jnp.arange(0, max_iterations + 1, checkpoint_every_n)
    )

    # check the gradient
    grad_fn = jax.grad(loss_function)
    gradients = grad_fn(parameters)
    # loss is the mean displacement in the x-direction.
    # All positions have the same displacement in the x-direction,
    # so we consider the gradient for one particle
    # loss = x_f - x_i
    # loss = (parameters * time_step * max_iterations + x_i) - x_i
    # loss = parameters * time_step * max_iterations
    # d(loss)/d(parameters) = 1
    expected_gradients = jnp.array(1.0)
    np.testing.assert_allclose(gradients, expected_gradients, rtol=1e-6)

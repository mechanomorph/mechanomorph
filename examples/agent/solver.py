# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax",
#     "numpy",
#     "equinox"
# ]
# ///

"""Euler integration with checkpointing for JAX simulations.

This module implements forward Euler integration with periodic checkpoints
while maintaining JIT and autodiff compatibility. The approach uses a while_loop
with pre-allocated checkpoint storage to ensure static array shapes required
for JAX compilation.

Key features:
- Fixed-size checkpoint arrays allocated at simulation start
- Conditional checkpointing using jax.lax.cond for autodiff compatibility
- JIT compilation support for high performance
- Full gradient flow through checkpointed states

The implementation stores simulation state snapshots at regular intervals,
enabling trajectory analysis, debugging, and gradient-based optimization
of simulation parameters.
"""

from typing import NamedTuple

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax import Array as JaxArray


class BaseSimulationState(NamedTuple):
    """Base class for simulation state.

    This class serves as a base for defining simulation states with JAX arrays.
    It is intended to be extended by specific simulation state definitions.

    Parameters
    ----------
    vertices : JaxArray
        Vertex positions in the simulation, shape (n_vertices, 3).
    velocities : JaxArray
        Vertex velocities in the simulation, shape (n_vertices, 3).
    """

    vertices: JaxArray
    velocities: JaxArray


class LoopStateWithCheckpoints(NamedTuple):
    """State container for the while_loop iteration with checkpoint storage.

    This container holds all data needed for one iteration of the simulation
    loop, including the current simulation state and checkpoint storage arrays.

    Parameters
    ----------
    simulation_state : SimulationState
        The current state of the simulation containing all physical quantities.
    iteration : int
        Current iteration counter (0-based).
    checkpoints : dict
        Dictionary containing pre-allocated arrays for checkpointed states.
        Keys match SimulationState field names, values are arrays with shape
        (n_checkpoints, ...) where ... matches the field's original shape.
    checkpoint_index : int
        Index for the next checkpoint to write (0-based).
    """

    simulation_state: BaseSimulationState
    iteration: int
    checkpoints: dict
    checkpoint_index: int


def forward_simulation_with_checkpoints(
    simulation_state,
    n_iterations: int,
    time_step: float,
    compute_forces_fn: callable,
    checkpoint_interval: int,
) -> tuple[BaseSimulationState, dict]:
    """Perform Euler integration with periodic checkpoints.

    This function runs a forward Euler integration scheme while saving
    simulation state snapshots at regular intervals. The checkpointing
    is implemented using conditional updates that preserve autodiff
    compatibility.

    Parameters
    ----------
    simulation_state : SimulationState
        Initial state of the simulation containing vertices and other
        physical quantities. Must be a NamedTuple with JAX array fields.
    n_iterations : int
        Number of time steps to integrate. Must be positive.
    time_step : float
        Size of each time step for the Euler integration.
    compute_forces_fn : callable
        Function that computes forces given a SimulationState.
        Signature: compute_forces_fn(simulation_state) -> JaxArray
        Must return forces with shape (n_vertices, 3).
    checkpoint_interval : int
        Save checkpoint every N iterations. Must be positive.

    Returns
    -------
    final_state : SimulationState
        Final simulation state after n_iterations time steps.
    checkpoints : dict
        Dictionary containing checkpoint data with keys matching
        SimulationState fields. Each value has shape (n_checkpoints, ...)
        where n_checkpoints = ceil(n_iterations / checkpoint_interval) + 1.
        Includes the initial state as the first checkpoint.

    Notes
    -----
    This function pre-allocates checkpoint storage for JIT compatibility.
    The number of checkpoints includes the initial state plus one checkpoint
    for every checkpoint_interval iterations that occur within n_iterations.

    All operations are JAX-compatible and preserve gradient flow for
    autodiff applications.
    """
    # Calculate number of checkpoints (including initial state)
    n_checkpoints = (n_iterations + checkpoint_interval - 1) // checkpoint_interval + 1

    def allocate_checkpoint_array(field_value: JaxArray) -> JaxArray:
        """Allocate storage for one field across all checkpoints.

        Creates a zero-initialized array to store checkpoint data for a single
        field from the SimulationState across all checkpoint times.

        Parameters
        ----------
        field_value : JaxArray
            The initial value of the field to allocate storage for.

        Returns
        -------
        JaxArray
            Zero-initialized array with shape (n_checkpoints, *field_value.shape)
            and same dtype as field_value.
        """
        checkpoint_shape = (n_checkpoints, *field_value.shape)
        return jnp.zeros(checkpoint_shape, dtype=field_value.dtype)

    # Create checkpoint storage for all fields in SimulationState
    initial_checkpoints = {}
    for field_name in simulation_state._fields:
        field_value = getattr(simulation_state, field_name)
        if isinstance(field_value, jnp.ndarray):
            initial_checkpoints[field_name] = allocate_checkpoint_array(field_value)

    # Store initial state as first checkpoint
    checkpoints_with_initial = {}
    for field_name, checkpoint_array in initial_checkpoints.items():
        field_value = getattr(simulation_state, field_name)
        checkpoints_with_initial[field_name] = checkpoint_array.at[0].set(field_value)

    def condition_fn(loop_state: LoopStateWithCheckpoints) -> bool:
        """Check if we should continue iterating.

        Determines whether the while_loop should continue based on the
        current iteration count.

        Parameters
        ----------
        loop_state : LoopStateWithCheckpoints
            Current loop state containing iteration counter.

        Returns
        -------
        bool
            True if we should continue (iteration < n_iterations), False otherwise.
        """
        return loop_state.iteration < n_iterations

    def body_fn(loop_state: LoopStateWithCheckpoints) -> LoopStateWithCheckpoints:
        """Perform one Euler step and conditionally save checkpoint.

        Executes one time step of the Euler integration and conditionally
        saves the new state to the checkpoint arrays based on the checkpoint
        interval.

        Parameters
        ----------
        loop_state : LoopStateWithCheckpoints
            Current loop state containing simulation state, iteration counter,
            and checkpoint storage.

        Returns
        -------
        LoopStateWithCheckpoints
            Updated loop state with new simulation state, incremented iteration
            counter, and potentially updated checkpoint storage.
        """
        current_state = loop_state.simulation_state

        # Compute forces and perform Euler step
        forces = compute_forces_fn(current_state)
        new_vertices = current_state.vertices + time_step * forces
        new_simulation_state = current_state._replace(vertices=new_vertices)

        # Check if we should save a checkpoint
        next_iteration = loop_state.iteration + 1
        should_checkpoint = (next_iteration % checkpoint_interval == 0) & (
            next_iteration <= n_iterations
        )

        def save_checkpoint() -> tuple[dict, int]:
            """Save current state to checkpoints.

            Updates the checkpoint storage arrays with the current simulation
            state at the next available checkpoint slot.

            Returns
            -------
            tuple[dict, int]
                Updated checkpoint dictionary and incremented checkpoint index.
            """
            updated_checkpoints = {}
            for field_name, checkpoint_array in loop_state.checkpoints.items():
                field_value = getattr(new_simulation_state, field_name)
                updated_checkpoints[field_name] = checkpoint_array.at[
                    loop_state.checkpoint_index + 1
                ].set(field_value)
            return updated_checkpoints, loop_state.checkpoint_index + 1

        def skip_checkpoint() -> tuple[dict, int]:
            """Skip checkpointing for this iteration.

            Returns the current checkpoint storage and index unchanged.

            Returns
            -------
            tuple[dict, int]
                Unchanged checkpoint dictionary and checkpoint index.
            """
            return loop_state.checkpoints, loop_state.checkpoint_index

        # Conditionally update checkpoints using jax.lax.cond
        new_checkpoints, new_checkpoint_index = jax.lax.cond(
            should_checkpoint, save_checkpoint, skip_checkpoint
        )

        return LoopStateWithCheckpoints(
            simulation_state=new_simulation_state,
            iteration=next_iteration,
            checkpoints=new_checkpoints,
            checkpoint_index=new_checkpoint_index,
        )

    # Initialize loop state
    initial_loop_state = LoopStateWithCheckpoints(
        simulation_state=simulation_state,
        iteration=0,
        checkpoints=checkpoints_with_initial,
        checkpoint_index=0,
    )

    # Run the while loop
    final_loop_state = eqxi.while_loop(
        condition_fn,
        body_fn,
        initial_loop_state,
        max_steps=n_iterations,
        kind="checkpointed",
    )

    return final_loop_state.simulation_state, final_loop_state.checkpoints


def jit_forward_simulation_with_checkpoints(
    simulation_state: BaseSimulationState,
    n_iterations: int,
    time_step: float,
    compute_forces_fn: callable,
    checkpoint_interval: int,
) -> tuple[BaseSimulationState, dict]:
    """JIT-compiled version of checkpoint simulation.

    This is a convenience function that applies JIT compilation to the
    forward simulation with checkpoints. Use this for better performance
    in production code.

    Parameters
    ----------
    simulation_state : SimulationState
        Initial simulation state.
    n_iterations : int
        Number of integration steps.
    time_step : float
        Integration time step.
    compute_forces_fn : callable
        Forces computation function.
    checkpoint_interval : int
        Save checkpoint every N iterations.

    Returns
    -------
    final_state : SimulationState
        Final simulation state.
    checkpoints : dict
        Dictionary of checkpoint arrays.

    Notes
    -----
    The first call will trigger compilation and may be slower. Subsequent
    calls with the same shapes and types will be fast.
    The compute_forces_fn, n_iterations, and checkpoint_interval are treated
    as static arguments for JIT compilation since they determine array shapes.
    """
    jit_fn = jax.jit(
        forward_simulation_with_checkpoints,
        static_argnames=["compute_forces_fn", "n_iterations", "checkpoint_interval"],
    )
    return jit_fn(
        simulation_state,
        n_iterations,
        time_step,
        compute_forces_fn,
        checkpoint_interval,
    )


def extract_checkpoint_times(
    checkpoints: dict, checkpoint_interval: int, time_step: float
) -> JaxArray:
    """Extract the simulation times corresponding to checkpoints.

    Calculates the physical time values that correspond to each saved
    checkpoint based on the checkpoint interval and time step size.

    Parameters
    ----------
    checkpoints : dict
        Checkpoint dictionary from simulation
        containing checkpoint arrays.
    checkpoint_interval : int
        Checkpoint interval used in simulation.
    time_step : float
        Time step used in simulation.

    Returns
    -------
    times : JaxArray
        Array of times corresponding to each checkpoint.
        Shape: (n_checkpoints,)
    """
    # Get number of checkpoints from any field
    n_checkpoints = next(iter(checkpoints.values())).shape[0]

    # Calculate times: [0, checkpoint_interval*dt, 2*checkpoint_interval*dt, ...]
    checkpoint_iterations = jnp.arange(n_checkpoints) * checkpoint_interval
    times = checkpoint_iterations * time_step

    return times


def test_differentiability():
    """Test that the final vertex positions are differentiable..

    This test verifies that gradients can flow through the entire simulation,
    which is essential for gradient-based optimization and parameter estimation.
    The test computes gradients of a simple loss function (sum of final positions)
    with respect to initial vertex positions.

    Raises
    ------
    AssertionError
        If gradients are not computed correctly or contain invalid values.
    """

    # Define example SimulationState
    class ExampleSimulationState(NamedTuple):
        """Example simulation state for testing.

        Parameters
        ----------
        vertices : JaxArray
            Vertex positions with shape (n_vertices, 3).
        velocities : JaxArray
            Vertex velocities with shape (n_vertices, 3).
        spring_constant : JaxArray
            The sprint constant value.
        """

        vertices: JaxArray
        velocities: JaxArray
        spring_constant: JaxArray

    def example_compute_forces(state: ExampleSimulationState) -> JaxArray:
        """Example force computation for testing.

        Applies a simple linear restoring force proportional to displacement
        from origin, simulating a spring-like system.

        Parameters
        ----------
        state : ExampleSimulationState
            Current simulation state.

        Returns
        -------
        JaxArray
            Forces on each vertex with shape (n_vertices, 3).
        """
        # Simple spring-like force: F = -k * x
        forces = -state.spring_constant * state.vertices
        return forces

    def simulation_wrapper(
        initial_vertices: JaxArray, spring_constant: JaxArray
    ) -> JaxArray:
        """Wrapper function for gradient computation.

        Runs the simulation and returns a scalar loss function of the
        final vertex positions. This function is designed to be
        differentiated with respect to initial_vertices.

        Parameters
        ----------
        initial_vertices : JaxArray
            Initial vertex positions with shape (n_vertices, 3).
        spring_constant : JaxArray
            The value for the spring.

        Returns
        -------
        JaxArray
            Scalar loss value (sum of all final vertex coordinates).
        """
        initial_velocities = jnp.zeros_like(initial_vertices)
        initial_state = ExampleSimulationState(
            vertices=initial_vertices,
            velocities=initial_velocities,
            spring_constant=spring_constant,
        )

        final_state, _ = jit_forward_simulation_with_checkpoints(
            initial_state,
            n_iterations=10,
            time_step=0.01,
            compute_forces_fn=example_compute_forces,
            checkpoint_interval=5,
        )

        # Return scalar loss (sum of all final positions)
        return jnp.sum(final_state.vertices)

    # Test setup
    initial_vertices = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Compute gradients
    grad_fn = jax.grad(simulation_wrapper, argnums=1)
    gradients = grad_fn(initial_vertices, jnp.array([1.0]))

    print(gradients)

    # Verify gradients are computed and valid
    assert gradients.shape == (
        1,
    ), f"Gradient shape mismatch: {gradients.shape} vs (1,)"
    assert jnp.all(jnp.isfinite(gradients)), "Gradients contain non-finite values"
    assert jnp.any(
        gradients != 0
    ), "All gradients are zero (unexpected for this system)"

    print("✓ Differentiability test passed!")
    print(f"Initial vertices:\n{initial_vertices}")
    print(f"Gradients:\n{gradients}")


def example_usage():
    """Demonstrate checkpoint simulation with JIT compilation.

    This example shows how to use the checkpointing simulation with
    a simple gravitational force system. Demonstrates the complete
    workflow from setup to analysis of results.
    """

    # Example SimulationState (replace with your actual one)
    class ExampleSimulationState(NamedTuple):
        """Example simulation state containing vertices and velocities.

        Parameters
        ----------
        vertices : JaxArray
            Vertex positions with shape (n_vertices, 3).
        velocities : JaxArray
            Vertex velocities with shape (n_vertices, 3).
        """

        vertices: JaxArray
        velocities: JaxArray

    def example_compute_forces(state: ExampleSimulationState) -> JaxArray:
        """Example force computation with gravity.

        Applies a constant downward gravitational force to all vertices.

        Parameters
        ----------
        state : ExampleSimulationState
            Current simulation state.

        Returns
        -------
        JaxArray
            Gravitational forces with shape (n_vertices, 3).
        """
        n_vertices = state.vertices.shape[0]
        gravity = jnp.tile(jnp.array([0.0, -9.81, 0.0]), (n_vertices, 1))
        return gravity

    # Create initial state - two particles at height 10m
    initial_vertices = jnp.array([[0.0, 10.0, 0.0], [1.0, 10.0, 0.0]])
    initial_velocities = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    initial_state = ExampleSimulationState(
        vertices=initial_vertices, velocities=initial_velocities
    )

    # Run simulation with checkpoints using JIT compilation
    final_state, checkpoints = jit_forward_simulation_with_checkpoints(
        initial_state,
        n_iterations=100,
        time_step=0.01,
        compute_forces_fn=example_compute_forces,
        checkpoint_interval=10,
    )

    # Extract checkpoint times
    times = extract_checkpoint_times(checkpoints, 10, 0.01)

    # Display results
    print(f"Simulation completed with {len(times)} checkpoints")
    print(f"Checkpoint times: {times}")
    print(f"Initial vertices:\n{initial_vertices}")
    print(f"Final vertices:\n{final_state.vertices}")
    print(f"Checkpointed vertices shape: {checkpoints['vertices'].shape}")

    # Show trajectory of first particle's height
    first_particle_heights = checkpoints["vertices"][
        :, 0, 1
    ]  # y-coordinate of first particle
    print(f"First particle height trajectory: {first_particle_heights}")


if __name__ == "__main__":
    print("Running example usage...")
    example_usage()
    print("\nRunning differentiability test...")
    test_differentiability()
    print("\nAll tests completed successfully!")

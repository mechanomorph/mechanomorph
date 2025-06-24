"""Euler integration with checkpointing for JAX simulations.

This module provides factory functions for creating JIT-compiled simulation loops
with periodic checkpointing. The implementation uses fixed-size checkpoint arrays
and conditional updates to maintain JAX compatibility while allowing gradient-based
optimization of simulation parameters.
"""

from typing import Callable, Dict, NamedTuple, Tuple

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax import Array as JaxArray


class LoopState(NamedTuple):
    """State container for the simulation loop.

    Parameters
    ----------
    simulation_state : NamedTuple
        Current state of the simulation containing positions, iteration, and
        other user-defined fields.
    checkpoints : dict
        Dictionary of pre-allocated checkpoint arrays for specified fields.
    checkpoint_index : int
        Index for the next checkpoint write position.
    """

    simulation_state: NamedTuple
    checkpoints: Dict[str, JaxArray]
    checkpoint_index: int


def make_simulation_loop(
    condition_function: Callable[[NamedTuple], bool],
    compute_forces_function: Callable[
        [NamedTuple, JaxArray], Tuple[NamedTuple, JaxArray]
    ],
    time_step: float,
    checkpoint_every_n: int,
    max_iterations: int,
    checkpoint_properties: Dict[str, Tuple[int, ...]],
) -> Callable[[NamedTuple, JaxArray], Tuple[NamedTuple, Dict[str, JaxArray]]]:
    """Create a JIT-compiled simulation function with checkpointing.

    Creates a simulation function that performs forward Euler integration
    with periodic state checkpointing. The returned function is JIT-compiled
    and supports gradient-based optimization of mechanical parameters.

    Parameters
    ----------
    condition_function : callable
        Function that determines whether to continue simulation.
        Signature: condition_function(state) -> bool
        Should return True to continue, False to stop.
    compute_forces_function : callable
        Function that computes forces for the current state.
        Signature: compute_forces_function(state, params) -> (updated_state, forces)
        The updated_state may contain mutations to non-position fields.
    time_step : float
        Time step size for Euler integration.
    checkpoint_every_n : int
        Number of iterations between checkpoints.
    max_iterations : int
        Maximum number of iterations allowed.
    checkpoint_properties : dict
        Dictionary mapping field names to their shapes for checkpointing.
        Example: {'positions': (100, 3), 'energy': ()}

    Returns
    -------
    simulation_function : callable
        JIT-compiled function with signature:
        simulation_function(initial_state, params) -> (final_state, checkpoints)
        Where checkpoints is a dict containing checkpoint arrays and metadata.
    """

    def simulation_function(
        initial_state: NamedTuple, params: JaxArray
    ) -> Tuple[NamedTuple, Dict[str, JaxArray]]:
        """Run the simulation with checkpointing.

        Parameters
        ----------
        initial_state : NamedTuple
            Initial simulation state. Must contain 'positions' and 'iteration' fields,
            plus any fields specified in checkpoint_properties.
        params : JaxArray
            Mechanical parameters passed to compute_forces_function.

        Returns
        -------
        final_state : NamedTuple
            Final simulation state after termination.
        checkpoints : dict
            Dictionary containing:
            - Field arrays with shape (n_max_checkpoints, *field_shape) for each
              field in checkpoint_properties
            - 'iterations': Array of iteration indices for each checkpoint
            - 'times': Array of physical times for each checkpoint
            - 'valid_mask': Boolean array indicating valid checkpoints
        """
        # Calculate maximum number of checkpoints
        n_max_checkpoints = (max_iterations // checkpoint_every_n) + 1

        # Pre-allocate checkpoint arrays
        checkpoints = _allocate_checkpoints(checkpoint_properties, n_max_checkpoints)

        # Store initial state in checkpoints
        checkpoints = _store_initial_checkpoint(
            checkpoints, initial_state, checkpoint_properties
        )

        # Create initial loop state
        initial_loop_state = LoopState(
            simulation_state=initial_state,
            checkpoints=checkpoints,
            checkpoint_index=0,
        )

        # Define loop condition
        def loop_condition(loop_state: LoopState) -> bool:
            """Check if simulation should continue.

            Parameters
            ----------
            loop_state : LoopState
                Current loop state.

            Returns
            -------
            bool
                True if should continue, False otherwise.
            """
            return condition_function(loop_state.simulation_state)

        # Define loop body
        def loop_body(loop_state: LoopState) -> LoopState:
            """Perform one integration step with conditional checkpointing.

            Parameters
            ----------
            loop_state : LoopState
                Current loop state.

            Returns
            -------
            LoopState
                Updated loop state after one integration step.
            """
            # Compute forces
            updated_state, forces = compute_forces_function(
                loop_state.simulation_state, params
            )

            # Euler integration step
            new_positions = updated_state.positions + time_step * forces
            new_state = updated_state._replace(
                positions=new_positions, iteration=updated_state.iteration + 1
            )

            # Check if we should checkpoint
            should_checkpoint = (new_state.iteration % checkpoint_every_n == 0) & (
                new_state.iteration <= max_iterations
            )

            # Conditionally update checkpoints
            new_checkpoints, new_checkpoint_index = jax.lax.cond(
                should_checkpoint,
                lambda: _save_checkpoint(
                    loop_state.checkpoints,
                    new_state,
                    loop_state.checkpoint_index + 1,
                    checkpoint_properties,
                    time_step,
                ),
                lambda: (loop_state.checkpoints, loop_state.checkpoint_index),
            )

            return LoopState(
                simulation_state=new_state,
                checkpoints=new_checkpoints,
                checkpoint_index=new_checkpoint_index,
            )

        # Run the simulation loop
        final_loop_state = eqxi.while_loop(
            loop_condition,
            loop_body,
            initial_loop_state,
            max_steps=max_iterations,
            kind="checkpointed",
        )

        return final_loop_state.simulation_state, final_loop_state.checkpoints

    # Return JIT-compiled function
    return jax.jit(simulation_function)


def _allocate_checkpoints(
    checkpoint_properties: Dict[str, Tuple[int, ...]], n_max_checkpoints: int
) -> Dict[str, JaxArray]:
    """Allocate checkpoint storage arrays.

    Parameters
    ----------
    checkpoint_properties : dict
        Dictionary mapping field names to their shapes.
    n_max_checkpoints : int
        Maximum number of checkpoints to allocate.

    Returns
    -------
    dict
        Dictionary containing pre-allocated checkpoint arrays and metadata arrays.
    """
    checkpoints = {}

    # Allocate arrays for each checkpointed field
    for field_name, shape in checkpoint_properties.items():
        checkpoint_shape = (n_max_checkpoints, *shape)
        checkpoints[field_name] = jnp.zeros(checkpoint_shape, dtype=jnp.float32)

    # Allocate metadata arrays
    checkpoints["iterations"] = jnp.zeros(n_max_checkpoints, dtype=jnp.int32)
    checkpoints["times"] = jnp.zeros(n_max_checkpoints, dtype=jnp.float32)
    checkpoints["valid_mask"] = jnp.zeros(n_max_checkpoints, dtype=bool)

    return checkpoints


def _store_initial_checkpoint(
    checkpoints: Dict[str, JaxArray],
    initial_state: NamedTuple,
    checkpoint_properties: Dict[str, Tuple[int, ...]],
) -> Dict[str, JaxArray]:
    """Store the initial state as the first checkpoint.

    Parameters
    ----------
    checkpoints : dict
        Pre-allocated checkpoint arrays.
    initial_state : NamedTuple
        Initial simulation state.
    checkpoint_properties : dict
        Dictionary mapping field names to their shapes.

    Returns
    -------
    dict
        Updated checkpoint dictionary with initial state stored.
    """
    updated_checkpoints = {}

    # Store each field from initial state
    for field_name in checkpoint_properties:
        field_value = getattr(initial_state, field_name)
        updated_checkpoints[field_name] = checkpoints[field_name].at[0].set(field_value)

    # Store metadata for initial checkpoint
    updated_checkpoints["iterations"] = (
        checkpoints["iterations"].at[0].set(initial_state.iteration)
    )
    updated_checkpoints["times"] = checkpoints["times"].at[0].set(0.0)
    updated_checkpoints["valid_mask"] = checkpoints["valid_mask"].at[0].set(True)

    # Copy over remaining metadata arrays unchanged
    for key in ["iterations", "times", "valid_mask"]:
        if key not in updated_checkpoints:
            updated_checkpoints[key] = checkpoints[key]

    return updated_checkpoints


def _save_checkpoint(
    checkpoints: Dict[str, JaxArray],
    state: NamedTuple,
    checkpoint_index: int,
    checkpoint_properties: Dict[str, Tuple[int, ...]],
    time_step: float,
) -> Tuple[Dict[str, JaxArray], int]:
    """Save current state to checkpoint arrays.

    Parameters
    ----------
    checkpoints : dict
        Current checkpoint arrays.
    state : NamedTuple
        Current simulation state to checkpoint.
    checkpoint_index : int
        Index where to store this checkpoint.
    checkpoint_properties : dict
        Dictionary mapping field names to their shapes.
    time_step : float
        Time step size for computing physical time.

    Returns
    -------
    updated_checkpoints : dict
        Updated checkpoint arrays with new checkpoint stored.
    checkpoint_index : int
        Same as input checkpoint_index (for consistency with skip case).
    """
    updated_checkpoints = {}

    # Store each checkpointed field
    for field_name in checkpoint_properties:
        field_value = getattr(state, field_name)
        updated_checkpoints[field_name] = (
            checkpoints[field_name].at[checkpoint_index].set(field_value)
        )

    # Update metadata
    updated_checkpoints["iterations"] = (
        checkpoints["iterations"].at[checkpoint_index].set(state.iteration)
    )
    updated_checkpoints["times"] = (
        checkpoints["times"].at[checkpoint_index].set(state.iteration * time_step)
    )
    updated_checkpoints["valid_mask"] = (
        checkpoints["valid_mask"].at[checkpoint_index].set(True)
    )

    return updated_checkpoints, checkpoint_index

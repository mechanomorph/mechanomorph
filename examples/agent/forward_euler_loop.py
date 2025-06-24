#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax",
#     "jaxlib",
#     "numpy",
#     "matplotlib",
#     "equinox",
# ]
# ///
"""Demonstration of gradient-based optimization with ABM integration.

This script shows how to use the mechanomorph ABM integration module to:
1. Run a simple particle simulation with parametric forces
2. Compute gradients of particle trajectories w.r.t. force parameters
3. Visualize the results with checkpointing
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array as JaxArray

# Import from mechanomorph (adjust path as needed for your setup)
from mechanomorph.jax.agent.integration import make_simulation_loop


class ParticleState(NamedTuple):
    """State of a particle system.

    Parameters
    ----------
    positions : JaxArray
        Particle positions with shape (n_particles, 3).
    iteration : int
        Current iteration number.
    """

    positions: JaxArray
    iteration: int


def compute_forces(
    state: ParticleState, params: JaxArray
) -> tuple[ParticleState, JaxArray]:
    """Compute constant forces based on parameters.

    Parameters
    ----------
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
        Each particle experiences force: [1, 0, 0] * param_a + param_b
    """
    param_a = params[0]
    param_b = params[1]

    # Unit vector in x-direction
    unit_vector = jnp.array([1.0, 0.0, 0.0])

    # Compute force: [1, 0, 0] * param_a + param_b
    # This creates force [param_a + param_b, param_b, param_b]
    force_vector = unit_vector * param_a + param_b

    # Apply same force to all particles
    n_particles = state.positions.shape[0]
    forces = jnp.tile(force_vector, (n_particles, 1))

    # Return unchanged state and forces
    return state, forces


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


def run_simulation_and_compute_gradients():
    """Run the demo simulation and compute parameter gradients."""
    # Set up initial conditions
    n_particles = 3
    initial_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    initial_state = ParticleState(positions=initial_positions, iteration=0)

    # Create simulation function
    sim_fn = make_simulation_loop(
        condition_function=condition_function,
        compute_forces_function=compute_forces,
        time_step=0.01,
        checkpoint_every_n=10,
        max_iterations=100,
        checkpoint_properties={"positions": (n_particles, 3)},
    )

    # Define a loss function that depends on final positions
    def loss_function(params: JaxArray) -> float:
        """Compute loss as sum of final x-positions.

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
        # Loss is sum of final x-positions
        return jnp.sum(final_state.positions[:, 0])

    # Initial parameters
    initial_params = jnp.array([1.0, 0.1])

    # Run simulation once to get trajectories
    print("Running simulation with initial parameters...")
    final_state, checkpoints = sim_fn(initial_state, initial_params)

    # Compute gradients
    print("\nComputing gradients...")
    grad_fn = jax.grad(loss_function)
    gradients = grad_fn(initial_params)

    # Display results
    print(
        f"\nInitial parameters: param_a={initial_params[0]:.3f},"
        f" param_b={initial_params[1]:.3f}"
    )
    print(f"Final positions:\n{final_state.positions}")
    print("\nGradients w.r.t. parameters:")
    print(f"  ∂loss/∂param_a = {gradients[0]:.6f}")
    print(f"  ∂loss/∂param_b = {gradients[1]:.6f}")

    # Extract valid checkpoints
    valid_mask = checkpoints["valid_mask"]
    valid_iterations = checkpoints["iterations"][valid_mask]
    valid_times = checkpoints["times"][valid_mask]
    valid_positions = checkpoints["positions"][valid_mask]

    print("\nCheckpoint summary:")
    print(f"  Total checkpoints: {jnp.sum(valid_mask)}")
    print(f"  Checkpoint iterations: {valid_iterations}")

    return valid_positions, valid_times, gradients


def visualize_trajectories(positions: JaxArray, times: JaxArray):
    """Visualize particle trajectories from checkpoints.

    Parameters
    ----------
    positions : JaxArray
        Checkpointed positions with shape (n_checkpoints, n_particles, 3).
    times : JaxArray
        Physical times for each checkpoint.
    """
    n_particles = positions.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot x-position vs time
    ax = axes[0, 0]
    for i in range(n_particles):
        ax.plot(times, positions[:, i, 0], "o-", label=f"Particle {i}")
    ax.set_xlabel("Time")
    ax.set_ylabel("X Position")
    ax.set_title("X Position vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot y-position vs time
    ax = axes[0, 1]
    for i in range(n_particles):
        ax.plot(times, positions[:, i, 1], "o-", label=f"Particle {i}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Y Position")
    ax.set_title("Y Position vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot z-position vs time
    ax = axes[1, 0]
    for i in range(n_particles):
        ax.plot(times, positions[:, i, 2], "o-", label=f"Particle {i}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Z Position")
    ax.set_title("Z Position vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3D trajectories
    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(224, projection="3d")

    for i in range(n_particles):
        ax.plot(
            positions[:, i, 0],
            positions[:, i, 1],
            positions[:, i, 2],
            "o-",
            label=f"Particle {i}",
        )
        # Mark start and end
        ax.scatter(*positions[0, i], color="green", s=100, marker="o")
        ax.scatter(*positions[-1, i], color="red", s=100, marker="s")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Particle Trajectories")
    ax.legend()

    plt.tight_layout()
    plt.savefig("particle_trajectories.png", dpi=150, bbox_inches="tight")
    plt.show()


def parameter_sensitivity_analysis():
    """Analyze how parameters affect the simulation."""
    # Set up initial conditions
    n_particles = 3
    initial_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    initial_state = ParticleState(positions=initial_positions, iteration=0)

    # Create simulation function
    sim_fn = make_simulation_loop(
        condition_function=condition_function,
        compute_forces_function=compute_forces,
        time_step=0.01,
        checkpoint_every_n=10,
        max_iterations=100,
        checkpoint_properties={"positions": (n_particles, 3)},
    )

    # Test different parameter values
    param_a_values = jnp.linspace(0.5, 2.0, 5)
    param_b_values = jnp.array([0.0, 0.1, 0.2])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Vary param_a
    ax = axes[0]
    for param_a in param_a_values:
        params = jnp.array([param_a, 0.1])
        final_state, _ = sim_fn(initial_state, params)
        mean_x = jnp.mean(final_state.positions[:, 0])
        ax.plot(param_a, mean_x, "o", markersize=8)

    ax.set_xlabel("param_a")
    ax.set_ylabel("Mean Final X Position")
    ax.set_title("Effect of param_a on Final Position")
    ax.grid(True, alpha=0.3)

    # Vary param_b
    ax = axes[1]
    for param_b in param_b_values:
        params = jnp.array([1.0, param_b])
        final_state, _ = sim_fn(initial_state, params)
        mean_y = jnp.mean(final_state.positions[:, 1])
        ax.plot(param_b, mean_y, "o", markersize=8)

    ax.set_xlabel("param_b")
    ax.set_ylabel("Mean Final Y Position")
    ax.set_title("Effect of param_b on Y-drift")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=== ABM forward simulation demo ===\n")

    # Run main simulation and compute gradients
    positions, times, gradients = run_simulation_and_compute_gradients()

    # Visualize results
    print("\nGenerating trajectory plots...")
    visualize_trajectories(positions, times)

    # Parameter sensitivity analysis
    print("\nPerforming parameter sensitivity analysis...")
    parameter_sensitivity_analysis()

    print("\nDemo complete! Check the generated plots.")

    # Interpretation of results
    print("\n=== Interpretation ===")
    print("1. The positive gradient w.r.t. param_a shows that increasing the")
    print("   x-direction force multiplier increases final x-positions.")
    print("2. The gradient w.r.t. param_b affects all dimensions since it's")
    print("   added to all force components.")
    print("3. The checkpoints show smooth particle trajectories over time.")
    print("4. This demonstrates that gradients flow correctly through the")
    print("   entire simulation, enabling parameter optimization.")

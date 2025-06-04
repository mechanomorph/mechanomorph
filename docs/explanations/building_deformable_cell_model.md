# Building a Deformable Cell Simulation with mechanomorph

## 1. Overview

This tutorial demonstrates how to build a physics-based deformable cell simulation using the mechanomorph framework. A deformable cell model represents cells a surface meshes and can simulate their mechanical behavior under various forces. In the context of this tutorial, we will simulate a single cell that starts as a cube and evolves into a spherical shape due to surface tension and pressure forces.


## 2. Geometry and physics

### 2.1 Geometry Representation

Our cell is represented as a **triangular mesh** - a collection of vertices connected by triangular faces that form a closed 3D surface.

```python
# Basic mesh components
vertices = jnp.array([[x, y, z], ...])  # (n_vertices, 3) coordinates
faces = jnp.array([[v1, v2, v3], ...])  # (n_faces, 3) vertex indices
```

The mechanomorph framework uses a **packed mesh format** where data is organized into fixed-size arrays with padding. This enables efficient vectorized computation across multiple cells:

```python
# Packed format for multiple cells
vertices_packed = jnp.zeros((max_cells, max_vertices_per_cell, 3))
faces_packed = jnp.zeros((max_cells, max_faces_per_cell, 3))
valid_masks = jnp.zeros((max_cells, max_vertices_per_cell), dtype=bool)
```

### 2.2 Physics Implementation

Surface tension acts to minimize the surface area of the cell. For each triangular face, surface tension creates forces on the three vertices that tend to shrink the triangle  Internal pressure maintains cell volume by pushing outward on all surfaces.

The simulation demonstrates how competing forces reach equilibrium:

1. Surface tension pulls vertices inward to minimize surface area
2. Pressure forces push vertices outward to maintain volume
3. Equilibrium occurs when these forces balance

For a uniform cell, the equilibrium shape is a sphere because it minimizes surface area for a given volume.

### 2.3 Time Integration

We use explicit Euler integration to evolve the system through time:

```python
# Euler integration step
new_position = old_position + timestep * force
```
At each time step, we update vertex positions based on the computed forces. We simulate for a fixed number of steps at a fixed time step size.

## 3. JAX-Specific Design Patterns

### 3.1 JIT Compilation Requirements

JAX's JIT compiler requires that array shapes and control flow be determined at compile time, not runtime. This means:

- **Timestep and iteration count** must be compile-time constants
- **Array sizes** must be fixed (hence the packed format with padding)
- **Control flow** cannot depend on data values

### 3.2 Factory Pattern for JIT Compilation

We use a factory function to create JIT-compiled simulations with baked-in parameters:

```python
def make_forward_simulation(time_step: float, n_iterations: int):
    """Create a JIT-compiled forward simulation.
    
    Parameters are baked in at compile time for optimal performance.
    """
    @jax.jit
    def forward_simulation(vertices, faces, ...):
        # Simulation implementation with time_step and n_iterations as constants
        pass
    
    return forward_simulation
```

This pattern allows us to:
- Compile once, run many times
- Ensure parameters are compile-time constants
- Create multiple simulation variants with different parameters

### 3.3 Vectorization Strategy

We use `jax.vmap` to batch operations over multiple cells efficiently:

```python
# Create batched versions of single-cell functions
batched_pressure_forces = jax.vmap(
    compute_cell_pressure_forces,
    in_axes=(0, 0, 0, 0, 0, None),  # Batch over first 5 args, broadcast last
    out_axes=0
)
```

## 4. Implementation Walkthrough

### 4.1 Data Structures

The simulation state is encapsulated in a `NamedTuple` for immutability and clarity:

```python
class SimulationState(NamedTuple):
    """Container for all simulation state variables."""
    vertices_packed: jax.Array          # (max_cells, max_vertices, 3)
    faces_packed: jax.Array             # (max_cells, max_faces, 3)
    valid_vertices_mask: jax.Array      # (max_cells, max_vertices)
    valid_faces_mask: jax.Array         # (max_cells, max_faces)
    valid_cells_mask: jax.Array         # (max_cells,)
    surface_tensions: jax.Array         # (max_cells, max_faces)
    pressures: jax.Array                # (max_cells,)
```

This structure:
- Groups related data together
- Supports JAX transformations (e.g., JIT, vmap)

### 4.2 Force Calculation Pipeline

Forces are computed by combining pressure and surface tension contributions:

```python
def calculate_forces(
    simulation_state: SimulationState,
    target_cell_volumes: JaxArray,
    bulk_modulus: float,
    min_norm: float = 1e-10,
) -> JaxArray:
    """Calculate forces acting on vertices."""
    
    # Compute pressure forces (outward, maintain volume)
    pressure_forces = batched_compute_cell_pressure_forces(
        simulation_state.vertices_packed,
        simulation_state.valid_vertices_mask,
        simulation_state.faces_packed,
        simulation_state.valid_faces_mask,
        target_cell_volumes,
        bulk_modulus,
    )

    # Compute surface tension forces (inward, minimize area)
    surface_tension_forces = batched_surface_tension_forces(
        simulation_state.vertices_packed,
        simulation_state.faces_packed,
        simulation_state.valid_vertices_mask,
        simulation_state.valid_faces_mask,
        simulation_state.surface_tensions,
        min_norm,
    )

    return pressure_forces + surface_tension_forces
```

The batched functions use `jax.vmap` to process multiple cells simultaneously, dramatically improving performance.

### 4.3 Time Integration with `lax.scan`

#### The Problem with Python Loops

A standard implementation of Euler time integration might use a Python for loop to execute the specified number of time steps:

```python
# For loops are slow both to compile and run in JAX
def simulate_with_python_loop(initial_vertices, forces_func, timestep, n_steps):
    vertices = initial_vertices
    for i in range(n_steps):
        forces = forces_func(vertices)
        vertices = vertices + timestep * forces
    return vertices
```

This approach has a couple of problems:
- **Performance**: Python loops are slow
- **JAX incompatibility**: JIT cannot optimize dynamic loop bounds

#### Introduction to `lax.scan`

JAX provides `lax.scan` as a high-performance alternative to Python loops. Here's a simple example:

```python
# Basic scan example: compute cumulative sum
def cumulative_sum_with_scan(array):
    def scan_fn(carry, x):
        new_carry = carry + x
        output = new_carry  # What we want to collect
        return new_carry, output
    
    init_carry = 0
    final_carry, outputs = jax.lax.scan(scan_fn, init_carry, array)
    return outputs

# Example usage
result = cumulative_sum_with_scan(jnp.array([1, 2, 3, 4]))
# result = [1, 3, 6, 10]
```

Key concepts:
- **Carry**: State that persists across iterations (like a loop variable)
- **Scan function**: Processes one element, updates carry, optionally produces output
- **Efficiency**: Compiles to fast, optimized code

#### Using `lax.scan` for Euler Integration

We can apply this pattern to time integration:

```python
def euler_integration_with_scan(initial_state, forces_func, timestep, n_steps):
    def scan_fn(state, step_idx):
        forces = forces_func(state)
        new_state = state + timestep * forces
        return new_state, None  # No output needed, just update state
    
    final_state, _ = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))
    return final_state
```

This approach:
- Compiles efficiently: JAX can optimize the entire loop
- Memory efficient: Doesn't store intermediate states
- Scalable: Works with complex state objects

#### Our Implementation

Thus, we use `lax.scan` to efficiently loop over our time step routine and perform the Euler integration:

```python
def scan_body(carry: SimulationState, step_idx: int) -> tuple[SimulationState, None]:
    """Single time step of the simulation loop."""
    state = carry
    
    # Calculate forces based on current state
    forces = calculate_forces(
        state,
        target_cell_volumes=target_cell_volumes,
        bulk_modulus=2500.0,
        min_norm=1e-10,
    )
    
    # Integrate vertex positions (Euler step)
    new_vertices = state.vertices_packed + time_step * forces
    
    # Update state with new positions (only for valid vertices)
    state = state._replace(
        vertices_packed=jnp.where(
            state.valid_vertices_mask[..., None],
            new_vertices,
            state.vertices_packed,
        )
    )
    
    return state, None

# Run the simulation loop
final_state, _ = jax.lax.scan(scan_body, initial_state, jnp.arange(n_iterations))
```

Key advantages of `lax.scan`:
- **Compiled efficiency**: Much faster than Python loops
- **Memory efficiency**: Doesn't store intermediate states
- **JAX compatibility**: Works seamlessly with JIT and other transformations

The `jnp.where` call ensures that only valid vertices are updated, leaving padding unchanged.

## 5. Step-by-Step Code Explanation

### 5.1 Complete Script Structure

Let's walk through the complete implementation:

```python
"""Forward simulation of a single cell."""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import napari
import numpy as np
from jax import Array as JaxArray

from mechanomorph.jax.dcm.forces import (
    compute_cell_pressure_forces,
    compute_cell_surface_tension_forces,
)
from mechanomorph.jax.dcm.utils import pack_mesh_to_cells
from mechanomorph.mesh import make_cube_doublet
```

### 5.2 Mesh Preparation

The simulation starts by creating and preparing the mesh data:

```python
# Create a cube doublet mesh (we'll use only one cube)
vertices, faces, vertex_cell_mapping, face_cell_mapping, _ = make_cube_doublet(
    edge_width=edge_width,
    target_area=target_area,
)

# Extract only the first cell (single cube)
vertices = vertices[vertex_cell_mapping == 0]
faces = faces[face_cell_mapping == 0]

# Convert to physical units
vertices = vertices * grid_size

# Pack into JAX-compatible format
(
    vertices_packed,
    faces_packed,
    valid_vertices_mask,
    valid_faces_mask,
    valid_cells_mask,
    vertex_overflow,
    face_overflow,
    cell_overflow,
) = pack_mesh_to_cells(
    vertices=jnp.array(vertices),
    faces=jnp.array(faces),
    vertex_cell_mapping=jnp.zeros(vertices.shape[0], dtype=int),
    face_cell_mapping=jnp.zeros(faces.shape[0], dtype=int),
    max_vertices_per_cell=int(vertices.shape[0]),
    max_faces_per_cell=int(faces.shape[0]),
    max_cells=1,
)
```

The `pack_mesh_to_cells` function converts the irregular mesh data into fixed-size arrays suitable for vectorized computation.

### 5.3 Setting Up Mechanical Properties

```python
# Define material properties
surface_tensions = surface_tension_value * jnp.ones(
    (faces_packed.shape[0], faces_packed.shape[1]), dtype=float
)
pressures = pressures_value * jnp.ones((1,))
target_cell_volumes = jnp.array([(edge_width * grid_size)**3])
```

These arrays define the mechanical properties:
- **Surface tensions**: Per-face values (uniform in this example)
- **Pressures**: Per-cell internal pressure
- **Target volumes**: Desired cell volumes for pressure calculation

### 5.4 Factory Function Implementation

The core simulation logic is wrapped in a factory function:

```python
def make_forward_simulation(time_step: float, n_iterations: int):
    """Create a JIT-compiled forward simulation."""
    
    @jax.jit
    def forward_simulation(
        vertices_packed,
        faces_packed,
        valid_vertices_mask,
        valid_faces_mask,
        valid_cells_mask,
        surface_tensions,
        pressures,
        target_cell_volumes
    ) -> SimulationState:
        # Bundle initial state
        initial_state = SimulationState(
            vertices_packed=vertices_packed,
            faces_packed=faces_packed,
            valid_vertices_mask=valid_vertices_mask,
            valid_faces_mask=valid_faces_mask,
            valid_cells_mask=valid_cells_mask,
            surface_tensions=surface_tensions,
            pressures=pressures,
        )
        
        # [scan_body function definition here]
        
        # Run simulation loop
        final_state, _ = jax.lax.scan(
            scan_body, initial_state, jnp.arange(n_iterations)
        )
        
        return final_state
    
    return forward_simulation
```

This pattern ensures that:
- `time_step` and `n_iterations` are compile-time constants
- The function is JIT-compiled for maximum performance
- We can create multiple simulation variants easily

### 5.5 Running the Simulation

```python
# Create the simulation function
simulate = make_forward_simulation(
    time_step=time_step, 
    n_iterations=n_forward_iterations
)

# Run the simulation
print("Running simulation...")
start_time = time.time()
final_state = simulate(
    vertices_packed,
    faces_packed,
    valid_vertices_mask,
    valid_faces_mask,
    valid_cells_mask,
    surface_tensions,
    pressures,
    target_cell_volumes=target_cell_volumes,
)
end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds")
```

### 5.6 Visualization

The results are visualized using napari:

```python
# Create 3D viewer
viewer = napari.Viewer()

# Add initial mesh
initial_mesh = viewer.add_surface(
    (vertices / grid_size, faces),
    name='Initial Mesh',
)
initial_mesh.wireframe.visible = True

# Add final mesh
final_vertices = np.concatenate(final_state.vertices_packed)
final_mesh = viewer.add_surface(
    (final_vertices / grid_size, faces),
    name='Final Mesh',
)
final_mesh.wireframe.visible = True

viewer.dims.ndisplay = 3
napari.run()
```

## 6. Key Takeaways

This tutorial demonstrates several important patterns for scientific computing with JAX:

1. **Factory functions** enable JIT compilation with compile-time parameters
2. **NamedTuples** provide clean, immutable state management
3. **lax.scan** replaces Python loops for performance and JAX compatibility
4. **vmap** enables efficient vectorization across multiple data items
5. **Packed arrays with masks** handle variable-size data efficiently

These patterns are broadly applicable to many types of scientific simulations, making this tutorial a useful foundation for more complex modeling work.

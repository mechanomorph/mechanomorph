# JAX Mesh Format Documentation

## Introduction

This document describes the mesh data format used in the Jax implementation of the deformable cell mesh simulation framework. The format is specifically designed to meet Jax JIT compilation requirements while enabling efficient vectorized operations across multiple cells.

## Core Concepts

### The Packed Cell Format

Jax JIT compilation requires arrays with static, compile-time known shapes. However, meshes may have cells with varying numbers of vertices and faces. Our solution is to pre-allocate fixed-size arrays with padding to accommodate the maximum expected cell size.

```
Unpacked (variable size):
Cell 0: [v0, v1, v2, v3, v4, v5, v6, v7]     (8 vertices)
Cell 1: [v8, v9, v10, v11, v12]              (5 vertices)

Packed (fixed size with padding):
Cell 0: [v0, v1, v2, v3, v4, v5, v6, v7, -, -]  (8 real + 2 padding)
Cell 1: [v8, v9, v10, v11, v12, -, -, -, -, -]  (5 real + 5 padding)
```

**Trade-offs:**
- JIT compatible with static shapes
- Enables vectorization with `vmap`
- Uses more memory when arrays aren't fully packed
- Requires careful sizing for dynamic scenarios (e.g., remeshing)

### Validity Masks

Since padded arrays contain unused elements, we use boolean masks to track which elements contain real data versus padding.

```
Data:     [v0, v1, v2, -, -]
Mask:     [ T,  T,  T, F, F]
```

### Cell-wise Batching

The format organizes data to enable `vmap` operations across all cells simultaneously. Arrays follow the structure `(max_cells, max_items_per_cell, dimensions)`, allowing parallel computation and automatic differentiation across the cell dimension.

## Data Structures

### Vertex Data

**`vertices_packed`**: `(max_cells, max_vertices_per_cell, 3)`
- Contains 3D coordinates for each vertex
- Padding elements are typically set to zero

**`valid_vertices_mask`**: `(max_cells, max_vertices_per_cell)`
- Boolean mask indicating valid vertex positions

```
Example with 2 cells, max 4 vertices per cell:

vertices_packed[0] = [[0.0, 0.0, 0.0],    # vertex 0
                      [1.0, 0.0, 0.0],    # vertex 1  
                      [0.0, 1.0, 0.0],    # vertex 2
                      [0.0, 0.0, 0.0]]    # padding

valid_vertices_mask[0] = [True, True, True, False]
```

### Face Data

**`faces_packed`**: `(max_cells, max_faces_per_cell, 3)`
- Contains vertex indices for each triangular face
- Uses **local indexing** within each cell's vertex array

**`valid_faces_mask`**: `(max_cells, max_faces_per_cell)`
- Boolean mask indicating valid faces

```
Example face referencing local vertices 0, 1, 2:

faces_packed[0] = [[0, 1, 2],    # face 0 (valid)
                   [1, 2, 3],    # face 1 (valid)
                   [0, 0, 0]]    # padding

valid_faces_mask[0] = [True, True, False]
```

### Cell Metadata

**`valid_cells_mask`**: `(max_cells,)`
- Boolean mask indicating which cells contain actual data

**Overflow flags**: Boolean indicators for debugging
- `vertex_overflow`: True if any cell exceeded `max_vertices_per_cell`
- `face_overflow`: True if any cell exceeded `max_faces_per_cell`
- `cell_overflow`: True if mesh exceeded `max_cells`

## The Packing Process

### Input: Unstructured Mesh

The packing process starts with a standard unstructured mesh representation:

```python
# Global vertex coordinates
vertices = jnp.array([[0,0,0], [1,0,0], [1,1,0], ...])  # (n_vertices, 3)

# Global face definitions  
faces = jnp.array([[0,1,2], [1,3,2], ...])              # (n_faces, 3)

# Cell membership
vertex_cell_mapping = jnp.array([0, 0, 0, 1, 1, ...])   # (n_vertices,)
face_cell_mapping = jnp.array([0, 0, 1, 1, ...])        # (n_faces,)
```

## Indexing Systems

### Global vs. Local Indexing

**Global indexing** refers to vertex indices in the original unstructured mesh:
```
Global vertices: [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, ...]
Global face: [2, 5, 8]  # references vertices 2, 5, 8 in global array
```

**Local indexing** refers to vertex indices within each cell's packed array:
```
Cell 0 vertices: [v2, v5, v8]     # local indices [0, 1, 2]
Cell 0 face: [0, 1, 2]            # same triangle, local indices
```

### Index Conversion

During packing, faces are automatically remapped from global to local indices:

```
Original mesh:
vertices = [[0,0,0], [1,0,0], [0,1,0], [1,1,0]]  # global indices 0,1,2,3
faces = [[0,1,2]]                                  # face uses global indices
vertex_cell_mapping = [0, 0, 0, 1]               # vertices 0,1,2 → cell 0

After packing cell 0:
vertices_packed[0] = [[0,0,0], [1,0,0], [0,1,0], [0,0,0]]  # local order
faces_packed[0] = [[0,1,2]]                                 # remapped to local indices
```

## Working with Packed Meshes

### Common Operations

**Accessing valid data:**
```python
# Get valid vertices for cell 0
valid_verts = vertices_packed[0][valid_vertices_mask[0]]

# Process all cells with vmap
cell_volumes = jax.vmap(compute_cell_volume)(
    vertices_packed, faces_packed, valid_faces_mask
)
```

**Handling padding in computations:**
```python
# Mask out padding elements
face_areas = compute_face_areas(vertices_packed, faces_packed)
face_areas = jnp.where(valid_faces_mask, face_areas, 0.0)
```

### Best Practices

1. **Array sizing**: Choose `max_*_per_cell` values based on expected mesh complexity
2. **Memory vs. performance**: Smaller arrays use less memory but may cause overflow
3. **Debugging**: Check overflow flags to detect undersized arrays
4. **Vectorization**: Use masks consistently to handle padding in all operations

## Complete Example: Two Adjacent Cubes

Let's walk through packing a mesh with two adjacent cubes sharing a face.

### Input Mesh

```python
# Each cube has 8 vertices, 12 triangular faces
vertices = jnp.array([
    # Cube 0 vertices (indices 0-7)
    [0,0,0], [1,0,0], [1,1,0], [0,1,0],  # bottom face
    [0,0,1], [1,0,1], [1,1,1], [0,1,1],  # top face
    # Cube 1 vertices (indices 8-15)  
    [1,0,0], [2,0,0], [2,1,0], [1,1,0],  # bottom face
    [1,0,1], [2,0,1], [2,1,1], [1,1,1],  # top face
])

vertex_cell_mapping = jnp.array([0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1])

# Faces for both cubes (24 total triangular faces)
faces = jnp.array([
    [0,1,2], [0,2,3], ...  # cube 0 faces (indices 0-11)
    [8,9,10], [8,10,11], ...  # cube 1 faces (indices 12-23)
])

face_cell_mapping = jnp.array([0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1])
```

### Packing Process
We can use the `pack_mesh_to_cells()` function to pack this mesh into the required format. This function will handle the remapping of vertices and faces, ensuring that each cell's data is correctly organized and padded as necessary.

```python
from mechanomorph.jax.utils import pack_mesh_to_cells

(vertices_packed, faces_packed, valid_vertices_mask, valid_faces_mask, 
 valid_cells_mask, vertex_overflow, face_overflow, cell_overflow) = pack_mesh_to_cells(
    vertices=vertices,
    faces=faces, 
    vertex_cell_mapping=vertex_cell_mapping,
    face_cell_mapping=face_cell_mapping,
    max_vertices_per_cell=8,
    max_faces_per_cell=12,
    max_cells=2
)
```

### Packed Result
The expected results of the packing operation are below:
```python
# Cell 0 gets vertices 0-7
vertices_packed[0] = [
    [0,0,0], [1,0,0], [1,1,0], [0,1,0],  # local indices 0-3
    [0,0,1], [1,0,1], [1,1,1], [0,1,1]   # local indices 4-7
]

# Cell 1 gets vertices 8-15  
vertices_packed[1] = [
    [1,0,0], [2,0,0], [2,1,0], [1,1,0],  # local indices 0-3
    [1,0,1], [2,0,1], [2,1,1], [1,1,1]   # local indices 4-7
]

# Faces remapped to use local vertex indices
# Cell 0 vertex indices are remapped to local indices 0-7
# Cell 1 vertex indices are remapped to local indices 0-7 
faces_packed[0] = [[0,1,2], [0,2,3], ...]  # cube 0 faces
faces_packed[1] = [[0,1,2], [0,2,3], ...]  # cube 1 faces (same pattern)

# All vertices and faces are valid (no padding needed)
valid_vertices_mask = jnp.array([[True]*8, [True]*8])
valid_faces_mask = jnp.array([[True]*12, [True]*12])  
valid_cells_mask = jnp.array([True, True])

# No overflow occurred
assert vertex_overflow is False
assert face_overflow is False
assert cell_overflow is False
```

This packed format is now ready for vectorized operations across both cells using JAX's `vmap` function.
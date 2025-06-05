"""Example of gradient based optimization through a remeshing operation using JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyvista as pv
from matplotlib import pyplot as plt

from mechanomorph.jax.dcm.remeshing import remesh_edge_split_single_cell
from mechanomorph.jax.dcm.utils import pack_mesh_to_cells

batched_edge_split = jax.vmap(
    remesh_edge_split_single_cell,
    in_axes=(0, 0, 0, 0, None),
    out_axes=(0, 0, 0, 0, 0),
)
jit_batched_edge_split = jax.jit(batched_edge_split)


def model_forward(
    params: jax.Array,
    vertices_packed: jax.Array,
    faces_packed: jax.Array,
    valid_vertices_mask: jax.Array,
    valid_faces_mask: jax.Array,
):
    """Simple model that scales and translates a mesh.

    This also performs remeshing
    """
    edge_length_threshold = 5.0

    vertices_packed = vertices_packed * params[0] + params[1]

    new_vertices, new_faces, new_vertex_mask, new_face_mask, overflow = (
        jit_batched_edge_split(
            vertices_packed,
            faces_packed,
            valid_vertices_mask,
            valid_faces_mask,
            edge_length_threshold,
        )
    )

    return new_vertices, new_vertex_mask


def loss(
    params: jax.Array,
    vertices_packed: jax.Array,
    faces_packed: jax.Array,
    valid_vertices_mask: jax.Array,
    valid_faces_mask: jax.Array,
):
    """Loss function for the model."""
    new_vertices, new_valid_vertices_mask = model_forward(
        params,
        vertices_packed,
        faces_packed,
        valid_vertices_mask,
        valid_faces_mask,
    )

    valid_new_vertices = new_vertices[0][new_valid_vertices_mask[0]]

    mesh_centroid = jnp.mean(valid_new_vertices, axis=0)
    mesh_radius = jnp.mean(jnp.linalg.norm(valid_new_vertices - mesh_centroid, axis=1))

    loss_value = (
        jnp.mean(jnp.abs(mesh_centroid - jnp.array([0, 0, 0])) ** 2)
        + (mesh_radius - 6) ** 2
    )

    return loss_value


if __name__ == "__main__":
    # Create a simple mesh for demonstration
    mesh = pv.Sphere(
        radius=10.0, center=(0, 0, 0), theta_resolution=10, phi_resolution=10
    )

    vertices = np.asarray(mesh.points)
    faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:]

    mesh_centroid = jnp.mean(vertices, axis=0)
    print(jnp.mean(jnp.linalg.norm(vertices - mesh_centroid, axis=1)))

    # convert the mesh to packed format
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
        max_vertices_per_cell=10 * int(vertices.shape[0]),
        max_faces_per_cell=10 * int(faces.shape[0]),
        max_cells=1,
    )

    # initialize the optimization
    params = jnp.array([1.0, 0.0])
    start_learning_rate = 0.01
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(params)

    # run the optimization
    losses = []
    for _ in range(100):
        loss_value, grads = jax.value_and_grad(loss)(
            params, vertices_packed, faces_packed, valid_vertices_mask, valid_faces_mask
        )
        losses.append(loss_value)
        print(f"    Grads: {grads}")
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    print(params)

    # convert the jax arrays to float values
    float_loss_values = np.asarray(losses)

    f, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
    ax.plot(float_loss_values)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    f.savefig("jax_remeshing.png", dpi=300)

    # make the napari viewer
    # viewer = napari.Viewer(ndisplay=3)
    # initial_mesh_layer = viewer.add_surface(
    #     (vertices, faces),
    #     name='Initial Mesh',
    # )
    # initial_mesh_layer.wireframe.visible = True
    #
    # remeshed_mesh_layer = viewer.add_surface(
    #     (new_vertices[0], new_faces[0]),
    #     name='Remeshed Mesh',
    # )
    # remeshed_mesh_layer.wireframe.visible = True
    #
    # napari.run()

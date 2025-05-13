"""Script to find performance bottlenecks in the contact detection."""

import time

import numpy as np
import torch
from skimage.measure import marching_cubes

from mechanomorph.sim._deformable_cell_model._contact import (
    average_vector_by_group,
    find_contacting_vertices_from_cell_map,
    group_contacting_vertices_union_find,
)
from mechanomorph.sim._deformable_cell_model._geometry_utils import (
    find_intersecting_bounding_boxes,
)


def group_contacting_vertices_pytorch(contact_matrix: torch.Tensor) -> torch.Tensor:
    """
    Group directly contacting vertices with connected components.

    Parameters
    ----------
    contact_matrix : (n_vertices, n_vertices) torch.Tensor
        Binary sparse matrix (0 or 1) indicating contacts.

    Returns
    -------
    group_labels : (n_vertices,) torch.Tensor
        Group ID assigned to each vertex.
    """
    n_vertices = contact_matrix.size(0)
    device = contact_matrix.device

    # # Ensure the matrix is symmetric
    # contact_matrix = contact_matrix + contact_matrix.T
    # contact_matrix = contact_matrix.clamp(max=1)

    # Initialize labels
    group_labels = torch.arange(n_vertices, device=device) + 1

    # Iteratively propagate labels
    prev_labels = group_labels.clone()
    for i in range(n_vertices):
        print(i)
        group_labels = (
            torch.max(
                group_labels.unsqueeze(0).expand(n_vertices, n_vertices)
                * contact_matrix,
                group_labels.unsqueeze(1).expand(n_vertices, n_vertices)
                * contact_matrix,
            )
            .max(dim=1)
            .values
        )

        # Check for convergence
        if torch.equal(group_labels, prev_labels):
            break
        prev_labels = group_labels.clone()

    return group_labels


def get_contact_index_set(vertex_labels: torch.Tensor) -> set[tuple[int, ...]]:
    """Get the set of the indices of contact groups.

    This is a utility function for comparing outputs.
    """
    vertex_labels = vertex_labels.numpy(force=True)
    unique_labels = np.unique(vertex_labels)

    contact_indices = set()
    for label_value in unique_labels:
        indices = np.atleast_1d(np.squeeze(np.argwhere(vertex_labels == label_value)))
        contact_indices.add(tuple(sorted(indices)))

    return contact_indices


def make_cube_mesh_data(edge_width: float):
    """Make a mesh of two cubes.

    This also produces the auxiliary data required for the simulation.
    """
    # make the first cube
    cube_mask = np.zeros((edge_width + 2, edge_width + 2, edge_width + 2), dtype=bool)
    cube_mask[1:-2, 1:-2, 1:-2] = True
    cube_vertices_0, cube_faces_0, _, _ = marching_cubes(cube_mask, 0)
    cube_faces_0 = cube_faces_0[:, [2, 1, 0]]  # make normals point out
    n_faces = cube_faces_0.shape[0]
    cell_indices_0 = np.arange(n_faces, dtype=int)

    # make the second cube by translating it along the 0th axis
    cube_vertices_1 = cube_vertices_0 + np.array([edge_width, 0, 0])
    n_vertices = cube_vertices_0.shape[0]
    cube_faces_1 = cube_faces_0 + n_vertices
    cell_indices_1 = np.arange(n_faces, dtype=int) + n_faces

    # concatenate the two cubes
    vertices = np.concatenate((cube_vertices_0, cube_vertices_1), axis=0)
    faces = np.concatenate((cube_faces_0, cube_faces_1), axis=0)
    face_cell_index = [cell_indices_0, cell_indices_1]

    # make the bounding boxes
    bounding_box_0 = np.concat(
        (np.min(cube_vertices_0, axis=0) - 1, np.max(cube_vertices_0, axis=0) + 1),
        axis=0,
    )
    bounding_box_1 = np.concat(
        (np.min(cube_vertices_1, axis=0) - 1, np.max(cube_vertices_1, axis=0) + 1),
        axis=0,
    )
    bounding_boxes = np.stack((bounding_box_0, bounding_box_1), axis=0)

    return (
        torch.from_numpy(vertices).float(),
        torch.from_numpy(faces),
        [torch.from_numpy(cell_face_indices) for cell_face_indices in face_cell_index],
        torch.from_numpy(bounding_boxes),
    )


# parameters
contact_distance_threshold = 0.3e-6
edge_width = 30
grid_spacing = 1e-6
device = "cuda:0"

vertices, faces, face_cell_index, bounding_boxes = make_cube_mesh_data(
    edge_width=edge_width
)

# move to GPU
vertex_coordinates = vertices.to(device) * grid_spacing
faces = faces.to("cuda:0")
face_cell_index = [index.to(device) for index in face_cell_index]
bounding_boxes = bounding_boxes.to(device)

start_contact_time = time.time()
# make the cell contact map
contacting_cells = find_intersecting_bounding_boxes(bounding_boxes)
upper_triangle_indices = contacting_cells.T
lower_triangle_indices = upper_triangle_indices.flip(0)
cell_contact_map = torch.sparse_coo_tensor(
    indices=torch.cat([upper_triangle_indices, lower_triangle_indices], dim=1),
    values=torch.ones(2 * contacting_cells.shape[0], device=vertex_coordinates.device),
    check_invariants=True,
    device=vertex_coordinates.device,
)

vertex_contact_map = find_contacting_vertices_from_cell_map(
    vertices=vertex_coordinates,
    faces=faces,
    face_cell_index=face_cell_index,
    cell_contact_map=cell_contact_map,
    distance_threshold=contact_distance_threshold,
)

# displace the vertices
vertex_labels = group_contacting_vertices_union_find(vertex_contact_map)
vertex_coordinates = average_vector_by_group(vertex_coordinates, vertex_labels)

print("Contact time:", time.time() - start_contact_time)

start_contact_time = time.time()
# make the cell contact map
contacting_cells = find_intersecting_bounding_boxes(bounding_boxes)
upper_triangle_indices = contacting_cells.T
lower_triangle_indices = upper_triangle_indices.flip(0)
cell_contact_map = torch.sparse_coo_tensor(
    indices=torch.cat([upper_triangle_indices, lower_triangle_indices], dim=1),
    values=torch.ones(2 * contacting_cells.shape[0], device=vertex_coordinates.device),
    check_invariants=True,
    device=vertex_coordinates.device,
)

print(f"cell contact time: {time.time() - start_contact_time}")

start_vertex_contact_time = time.time()
vertex_contact_map = find_contacting_vertices_from_cell_map(
    vertices=vertex_coordinates,
    faces=faces,
    face_cell_index=face_cell_index,
    cell_contact_map=cell_contact_map,
    distance_threshold=contact_distance_threshold,
)
print("Vertex contact time:", time.time() - start_vertex_contact_time)

# displace the vertices
start_relabel_vertices_time = time.time()
vertex_labels = group_contacting_vertices_union_find(vertex_contact_map)
print("Union find time:", time.time() - start_relabel_vertices_time)

contact_set_original = get_contact_index_set(vertex_labels)

vertex_coordinates = average_vector_by_group(vertex_coordinates, vertex_labels)
print(f"Relabel time: {time.time() - start_relabel_vertices_time}")

print("Contact time:", time.time() - start_contact_time)


vertex_labels = group_contacting_vertices_pytorch(vertex_contact_map.to_dense())
start_torch_contact_time = time.time()
vertex_labels = group_contacting_vertices_pytorch(vertex_contact_map.to_dense())
print("Pytorch time:", time.time() - start_torch_contact_time)

contact_set_new = get_contact_index_set(vertex_labels)

assert contact_set_original == contact_set_new

"""Demo of the deformable cell model contact detection.

The mesh used in this demo can be generated with make_double_cubes.py.
"""

import time

import h5py
import napari
import torch

from mechanomorph.sim._deformable_cell_model._contact import (
    average_vector_by_group,
    find_contacting_vertices_from_cell_map,
    group_contacting_vertices_union_find,
)
from mechanomorph.sim._deformable_cell_model._geometry_utils import (
    find_intersecting_bounding_boxes,
)


def load_mesh(file_path):
    """Load the demo mesh file."""
    with h5py.File(file_path, "r") as f:
        vertices = f["vertices"][:]
        faces = f["faces"][:]
        face_cell_index = f["face_cell_index"][:]
        bounding_boxes = f["bounding_boxes"][:]

    return (
        torch.from_numpy(vertices).float(),
        torch.from_numpy(faces),
        torch.from_numpy(face_cell_index),
        torch.from_numpy(bounding_boxes).float(),
    )


if __name__ == "__main__":
    edge_width = 30
    gap = 0.2
    contact_distance_threshold = 0.3

    vertices, faces, face_cell_index, bounding_boxes = load_mesh("double_cube_mesh.h5")
    # vertices.requires_grad = True

    # dummy_constant = torch.tensor(3.0, requires_grad=True)
    # vertices = vertices + dummy_constant

    # make the cell contact map
    contacting_cells = find_intersecting_bounding_boxes(bounding_boxes)
    upper_triangle_indices = contacting_cells.T
    lower_triangle_indices = upper_triangle_indices.flip(0)
    cell_contact_map = torch.sparse_coo_tensor(
        indices=torch.cat([upper_triangle_indices, lower_triangle_indices], dim=1),
        values=torch.ones(2 * contacting_cells.shape[0]),
        check_invariants=True,
    )

    print("Contacting cells:")
    print(cell_contact_map.to_dense())
    contact_start = time.time()
    vertex_contact_map = find_contacting_vertices_from_cell_map(
        vertices=vertices,
        faces=faces,
        face_cell_index=face_cell_index,
        cell_contact_map=cell_contact_map,
        distance_threshold=contact_distance_threshold,
    )
    print(f"contact time: {time.time() - contact_start:.4f} s")

    # displace the vertices
    vertex_labels = group_contacting_vertices_union_find(vertex_contact_map)
    displaced_vertices = average_vector_by_group(vertices, vertex_labels)

    # loss = torch.mean(displaced_vertices - vertices)
    # print(dummy_constant.requires_grad)
    # print(torch.autograd.grad(loss, dummy_constant))

    # get data from the contacting vertices for visualization
    contacting_vertex_mask = vertex_contact_map.to_dense().sum(1) > 0
    contacting_vertices = vertices[contacting_vertex_mask]

    contacting_indices = torch.argwhere(contacting_vertex_mask)
    first_contacting_index = contacting_indices[60]
    contacting_row = vertex_contact_map.to_dense()[first_contacting_index].squeeze()
    other_contacting_index = torch.argwhere(contacting_row > 0)[0]
    print(first_contacting_index, other_contacting_index)
    contact_pair = vertices[torch.cat((first_contacting_index, other_contacting_index))]

    displaced_mask = torch.linalg.norm(displaced_vertices - vertices, dim=1) > 0.001
    moved_vertices = displaced_vertices[displaced_mask]

    # make the clipping planes
    plane_parameters_neg = {
        "position": (-edge_width / 2, edge_width / 2, edge_width / 2),
        "normal": (1, 0, 0),
        "enabled": True,
    }

    plane_parameters_pos = {
        "position": (edge_width / 2, edge_width / 2, edge_width / 2),
        "normal": (-1, 0, 0),
        "enabled": True,
    }

    viewer = napari.Viewer()
    mesh_layer = viewer.add_surface(
        (vertices.numpy(force=True), faces.numpy(force=True)),
        experimental_clipping_planes=[plane_parameters_neg, plane_parameters_pos],
    )
    mesh_layer.wireframe.visible = True

    viewer.add_points(
        data=contacting_vertices.numpy(force=True),
        size=0.2,
        face_color="red",
        name="contacts",
    )
    viewer.add_points(
        data=moved_vertices.numpy(force=True),
        size=0.2,
        face_color="blue",
        name="displaced",
    )
    viewer.add_points(
        data=contact_pair.numpy(force=True),
        size=1,
        face_color="green",
        name="contact pair",
    )

    viewer.dims.ndisplay = 3

    napari.run()

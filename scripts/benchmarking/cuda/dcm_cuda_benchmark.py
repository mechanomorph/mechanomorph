"""Benchmark for the deformable cell model using the CUDA backend."""

import napari
import numpy as np
import torch
from skimage.measure import marching_cubes

from mechanomorph.sim._deformable_cell_model._contact import (
    average_vector_by_group,
    find_contacting_vertices_from_cell_map,
    group_contacting_vertices_union_find,
)
from mechanomorph.sim._deformable_cell_model._forces import (
    compute_cell_volume,
    compute_vertex_forces,
)
from mechanomorph.sim._deformable_cell_model._geometry_utils import (
    find_intersecting_bounding_boxes,
)
from mechanomorph.sim._deformable_cell_model._mesh_utils import (
    get_face_vertex_mapping,
    get_per_cell_face_vertex_mapping,
)


def forward_simulation(
    mesh_data: dict[str, torch.Tensor],
    n_forward_iterations: int,
    time_step: float,
    normalized_pressure: torch.Tensor,
    normalized_surface_tension: torch.Tensor,
    pressure_range: tuple[float, float],
    surface_tension_range: tuple[float, float],
    bulk_modulus: float,
    target_cell_volumes: torch.Tensor,
    contact_distance_threshold: float = 0.1,
):
    """Run the model.

    Parameters
    ----------
    mesh_data : dict
        A dictionary containing the mesh data. The keys are:
        - "vertices": the vertex coordinates of the mesh
        - "faces": the face indices of the mesh
        - "face_cell_index": list of the face indices for each cell.
        - "bounding_boxes": (n_cells, 6) array with the bounding boxes for each cell
    n_forward_iterations : int
        The number of forward iterations to run.
    time_step : float
        The normalized time step for the simulation.
    normalized_pressure : torch.Tensor
        The pressure in the normalized 0-1 range.
        This should be a Parameter that can be optimized.
    normalized_surface_tension : torch.Tensor
        The surface tension in the normalized 0-1 range.
        This should be a Parameter that can be optimized.
    pressure_range : tuple[float, float]
        The min and max value for the pressure.
        This is the range that the normalized pressure is scaled to.
    surface_tension_range : tuple[float, float]
        The min and max value for the surface tension.
        This is the range that the normalized surface tension is scaled to.
    bulk_modulus : float
        The bulk modulus of the material.
    target_cell_volumes : torch.Tensor
        (n_cells,) array of the target cell volumes.
    contact_distance_threshold : float
        The minimum distance two nodes have to be apart to be considered in contact.
    """
    vertex_coordinates = mesh_data["vertices"]
    faces = mesh_data["faces"]
    face_cell_index = mesh_data["face_cell_index"]
    cell_face_list = mesh_data["cell_face_list"]
    bounding_boxes = mesh_data["bounding_boxes"]

    # un-normalized parameters
    pressure = (
        normalized_pressure * (pressure_range[1] - pressure_range[0])
    ) + pressure_range[0]
    surface_tension = (
        normalized_surface_tension
        * (surface_tension_range[1] - surface_tension_range[0])
    ) + surface_tension_range[0]

    scaled_target_volume = target_cell_volumes * torch.exp(pressure / bulk_modulus)

    all_vertex_contact_maps = []
    for i in range(n_forward_iterations):
        print(f"iteration {i}")
        # make the cell contact map
        contacting_cells = find_intersecting_bounding_boxes(bounding_boxes)
        upper_triangle_indices = contacting_cells.T
        lower_triangle_indices = upper_triangle_indices.flip(0)
        cell_contact_map = torch.sparse_coo_tensor(
            indices=torch.cat([upper_triangle_indices, lower_triangle_indices], dim=1),
            values=torch.ones(2 * contacting_cells.shape[0]),
            check_invariants=True,
        )

        vertex_contact_map = find_contacting_vertices_from_cell_map(
            vertices=vertex_coordinates,
            faces=faces,
            face_cell_index=face_cell_index,
            cell_contact_map=cell_contact_map,
            distance_threshold=contact_distance_threshold,
        )
        all_vertex_contact_maps.append(vertex_contact_map)

        # displace the vertices
        vertex_labels = group_contacting_vertices_union_find(vertex_contact_map)
        vertex_coordinates = average_vector_by_group(vertex_coordinates, vertex_labels)

        # get the face-vertex mappings
        (face_vertex_mapping_v0, face_vertex_mapping_v1, face_vertex_mapping_v2) = (
            get_face_vertex_mapping(
                vertices=vertex_coordinates,
                faces=faces,
            )
        )

        (
            face_vertex_mapping_v0_all_cells,
            face_vertex_mapping_v1_all_cells,
            face_vertex_mapping_v2_all_cells,
        ) = get_per_cell_face_vertex_mapping(
            vertices=vertex_coordinates, cell_face_lst=cell_face_list
        )

        # # get the surface tensions
        face_surface_tension = surface_tension * torch.ones(
            faces.shape[0],
            dtype=vertex_coordinates.dtype,
            requires_grad=True,
        )

        # we don't have static nodes
        static_nodes_mask = torch.zeros(vertex_coordinates.shape[0], dtype=torch.bool)

        # # compute the forces on the vertices
        vertex_forces = compute_vertex_forces(
            vertices=vertex_coordinates,
            faces=faces,
            face_vertex_mapping_v0=face_vertex_mapping_v0,
            face_vertex_mapping_v1=face_vertex_mapping_v1,
            face_vertex_mapping_v2=face_vertex_mapping_v2,
            face_vertex_mapping_v0_all_cells=face_vertex_mapping_v0_all_cells,
            face_vertex_mapping_v1_all_cells=face_vertex_mapping_v1_all_cells,
            face_vertex_mapping_v2_all_cells=face_vertex_mapping_v2_all_cells,
            target_cell_volume=scaled_target_volume,
            bulk_modulus=bulk_modulus,
            face_surface_tension=face_surface_tension,
            static_nodes_mask=static_nodes_mask,
        )

        # average the forces of contacting nodes
        vertex_forces_with_contacts = average_vector_by_group(
            vertex_forces, vertex_labels
        )

        vertex_coordinates = (
            vertex_coordinates + time_step * vertex_forces_with_contacts
        )

        # if i % self.refresh_every_n == 0:
        #     self.vertex_coordinates, self.faces = remesh(
        #         vertices=self.vertex_coordinates,
        #         faces=self.faces,
        #         max_edge_length=2,
        #     )

    return {
        "vertices": vertex_coordinates,
        "faces": faces,
        "cell_face_list": cell_face_list,
        "vertex_contact_map": all_vertex_contact_maps,
    }


def compute_target_volumes(
    vertices: torch.Tensor, faces: torch.Tensor, face_cell_index: torch.Tensor
) -> torch.Tensor:
    """Compute the volume of each cell."""
    # get the vertex coordinates
    vertex_0_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 0])
    vertex_1_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 1])
    vertex_2_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 2])
    cell_volumes = []
    for cell_faces in face_cell_index:
        # get the vertices for the cell
        cell_vertex_indices = faces[cell_faces]

        cell_vertex_0_coordinates = torch.index_select(
            vertex_0_coordinates, dim=0, index=cell_vertex_indices[:, 0]
        )
        cell_vertex_1_coordinates = torch.index_select(
            vertex_1_coordinates, dim=0, index=cell_vertex_indices[:, 1]
        )
        cell_vertex_2_coordinates = torch.index_select(
            vertex_2_coordinates, dim=0, index=cell_vertex_indices[:, 2]
        )

        # compute the volume
        cell_volumes.append(
            compute_cell_volume(
                cell_vertex_0_coordinates,
                cell_vertex_1_coordinates,
                cell_vertex_2_coordinates,
            )
        )

    return torch.tensor(cell_volumes, dtype=torch.float32)


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


if __name__ == "__main__":
    # parameters
    edge_width = 30
    target_element_area = 5.0
    grid_spacing = 1e-6

    # numeric parameters
    n_forward_iterations = 50
    time_step = 30.0

    # mechanical parameters
    normalized_surface_tension = torch.tensor(0.40)
    normalized_pressure = torch.tensor(0.5)

    pressure_range = (10.0, 200.0)
    surface_tension_range = (0.002, 0.02)

    bulk_modulus = 2500.0

    contact_distance_threshold = 0.3e-6

    # make the mesh (CPU)
    vertices, faces, face_cell_index, bounding_boxes = make_cube_mesh_data(
        edge_width=edge_width
    )

    # make an initial cell target volume
    target_cell_volumes = torch.tensor([8e-14, 8e-14])

    # run the forward (CPU)
    cell_face_list = [faces[cell_faces, :] for cell_faces in face_cell_index]
    mesh_data = {
        "vertices": vertices * grid_spacing,
        "faces": faces,
        "cell_face_list": cell_face_list,
        "face_cell_index": face_cell_index,
        "bounding_boxes": bounding_boxes,
    }

    final_mesh = forward_simulation(
        mesh_data=mesh_data,
        n_forward_iterations=n_forward_iterations,
        time_step=time_step,
        normalized_pressure=normalized_pressure,
        normalized_surface_tension=normalized_surface_tension,
        pressure_range=pressure_range,
        surface_tension_range=surface_tension_range,
        bulk_modulus=bulk_modulus,
        target_cell_volumes=target_cell_volumes,
        contact_distance_threshold=contact_distance_threshold,
    )

    # visualize results
    plane_parameters = {"position": (0, 15, 0), "normal": (0, 1, 0), "enabled": True}
    viewer = napari.Viewer()
    initial_mesh_layer = viewer.add_surface(
        (vertices.numpy(force=True), faces.numpy(force=True)), visible=False
    )
    initial_mesh_layer.wireframe.visible = True
    initial_mesh_layer.normals.face.visible = True

    final_mesh_layer = viewer.add_surface(
        (
            final_mesh["vertices"].numpy(force=True) * 1e6,
            final_mesh["faces"].numpy(force=True),
        ),
        experimental_clipping_planes=[plane_parameters],
        name="final mesh",
    )
    final_mesh_layer.wireframe.visible = True
    final_mesh_layer.normals.face.visible = True

    viewer.dims.ndisplay = 3

    napari.run()

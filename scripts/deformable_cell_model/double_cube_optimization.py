"""Demo of a forward simulation of a cube double with contact.

The mesh used in this demo can be generated with make_double_cubes.py.
"""

import time

import h5py
import napari
import torch
from matplotlib import pyplot as plt

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


def load_mesh(file_path, grid_spacing: float = 1.0):
    """Load the demo mesh file."""
    with h5py.File(file_path, "r") as f:
        vertices = f["vertices"][:] * grid_spacing
        faces = f["faces"][:]
        face_cell_index = f["face_cell_index"][:]
        bounding_boxes = f["bounding_boxes"][:] * grid_spacing

    return (
        torch.from_numpy(vertices).float(),
        torch.from_numpy(faces),
        torch.from_numpy(face_cell_index),
        torch.from_numpy(bounding_boxes).float(),
    )


class SoapBubble(torch.nn.Module):
    """Model objects as soap bubbles.

    All objects have the same pressure and surface tension.
    Contacts are detected by proximity.
    """

    def __init__(
        self,
        n_forward_iterations: int = 10,
        time_step: float = 30.0,
        contact_distance_threshold: float = 0.3,
        normalized_pressure: float = 0.5,
        normalized_surface_tension: float = 0.5,
        pressure_range: tuple[float, float] = (10.0, 200.0),
        surface_tension_range: tuple[float, float] = (0.002, 0.02),
        target_cell_volume: float = 1.0,
        bulk_modulus: float = 2500.0,
    ):
        super().__init__()

        # numerical parameters
        self.n_forward_iterations = n_forward_iterations
        self.time_step = time_step

        # static parameters
        self.bulk_modulus = bulk_modulus
        self.contact_distance_threshold = contact_distance_threshold
        self.target_cell_volume = target_cell_volume
        self.pressure_range = pressure_range
        self.surface_tension_range = surface_tension_range

        # model parameters (these can be optimized)
        self.normalized_surface_tension = torch.nn.Parameter(
            torch.tensor(normalized_surface_tension, dtype=torch.float32)
        )
        self.normalized_pressure = torch.nn.Parameter(
            torch.tensor(normalized_pressure, dtype=torch.float32)
        )

    def forward(self, mesh_data):
        """Run the model.

        Parameters
        ----------
        mesh_data : dict
            A dictionary containing the mesh data. The keys are:
            - "vertices": the vertex coordinates of the mesh
            - "faces": the face indices of the mesh
            - "face_cell_index": list of the face indices for each cell.
            - "bounding_boxes": (n_cells, 6) array with the bounding boxes for each cell
        """
        self.vertex_coordinates = mesh_data["vertices"]
        self.faces = mesh_data["faces"]
        self.face_cell_index = mesh_data["face_cell_index"]
        self.cell_face_list = mesh_data["cell_face_list"]
        self.bounding_boxes = mesh_data["bounding_boxes"]

        # un-normalized parameters
        pressure = (
            self.normalized_pressure * (self.pressure_range[1] - self.pressure_range[0])
        ) + self.pressure_range[0]
        surface_tension = (
            self.normalized_surface_tension
            * (self.surface_tension_range[1] - self.surface_tension_range[0])
        ) + self.surface_tension_range[0]
        print(f"    pressure: {pressure} surface tension: {surface_tension}\n")

        scaled_target_volume = self.target_cell_volume * torch.exp(
            pressure / self.bulk_modulus
        )

        all_vertex_contact_maps = []
        for _ in range(self.n_forward_iterations):
            # make the cell contact map
            contacting_cells = find_intersecting_bounding_boxes(bounding_boxes)
            upper_triangle_indices = contacting_cells.T
            lower_triangle_indices = upper_triangle_indices.flip(0)
            cell_contact_map = torch.sparse_coo_tensor(
                indices=torch.cat(
                    [upper_triangle_indices, lower_triangle_indices], dim=1
                ),
                values=torch.ones(2 * contacting_cells.shape[0]),
                check_invariants=True,
            )

            vertex_contact_map = find_contacting_vertices_from_cell_map(
                vertices=self.vertex_coordinates,
                faces=self.faces,
                face_cell_index=self.face_cell_index,
                cell_contact_map=cell_contact_map,
                distance_threshold=self.contact_distance_threshold,
            )
            all_vertex_contact_maps.append(vertex_contact_map)

            # displace the vertices
            vertex_labels = group_contacting_vertices_union_find(vertex_contact_map)
            self.vertex_coordinates = average_vector_by_group(
                self.vertex_coordinates, vertex_labels
            )

            # get the face-vertex mappings
            (face_vertex_mapping_v0, face_vertex_mapping_v1, face_vertex_mapping_v2) = (
                get_face_vertex_mapping(
                    vertices=self.vertex_coordinates,
                    faces=self.faces,
                )
            )

            (
                face_vertex_mapping_v0_all_cells,
                face_vertex_mapping_v1_all_cells,
                face_vertex_mapping_v2_all_cells,
            ) = get_per_cell_face_vertex_mapping(
                vertices=self.vertex_coordinates, cell_face_lst=self.cell_face_list
            )

            # # get the surface tensions
            face_surface_tension = surface_tension * torch.ones(
                self.faces.shape[0],
                dtype=self.vertex_coordinates.dtype,
                requires_grad=True,
            )

            # we don't have static nodes
            static_nodes_mask = torch.zeros(
                self.vertex_coordinates.shape[0], dtype=torch.bool
            )

            # # compute the forces on the vertices
            vertex_forces = compute_vertex_forces(
                vertices=self.vertex_coordinates,
                faces=self.faces,
                face_vertex_mapping_v0=face_vertex_mapping_v0,
                face_vertex_mapping_v1=face_vertex_mapping_v1,
                face_vertex_mapping_v2=face_vertex_mapping_v2,
                face_vertex_mapping_v0_all_cells=face_vertex_mapping_v0_all_cells,
                face_vertex_mapping_v1_all_cells=face_vertex_mapping_v1_all_cells,
                face_vertex_mapping_v2_all_cells=face_vertex_mapping_v2_all_cells,
                target_cell_volume=scaled_target_volume,
                bulk_modulus=self.bulk_modulus,
                face_surface_tension=face_surface_tension,
                static_nodes_mask=static_nodes_mask,
            )

            # average the forces of contacting nodes
            vertex_forces_with_contacts = average_vector_by_group(
                vertex_forces, vertex_labels
            )

            self.vertex_coordinates = (
                self.vertex_coordinates + self.time_step * vertex_forces_with_contacts
            )

            # if i % self.refresh_every_n == 0:
            #     self.vertex_coordinates, self.faces = remesh(
            #         vertices=self.vertex_coordinates,
            #         faces=self.faces,
            #         max_edge_length=2,
            #     )

        return {
            "vertices": self.vertex_coordinates,
            "faces": self.faces,
            "cell_face_list": self.cell_face_list,
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


if __name__ == "__main__":
    # initial mesh parameters
    initial_radius = 15
    target_element_area = 5.0
    grid_spacing = 1e-6

    # numeric parameters
    n_forward_iterations = 1000
    n_optimization_iterations = 1000

    # mechanical parameters
    surface_tension = 0.20
    pressure = 0.7

    initial_vertices, initial_faces, face_cell_index, bounding_boxes = load_mesh(
        file_path="double_cube_mesh.h5",
        grid_spacing=grid_spacing,
    )
    # cell_volumes = compute_target_volumes(
    #     vertices=initial_vertices,
    #     faces=initial_faces,
    #     face_cell_index=face_cell_index
    #
    cell_volumes = torch.tensor([5e-14, 5e-14])

    expected_volume = (30e-6) ** 3
    print(f"computed volume: {cell_volumes} expected: {expected_volume}")

    model = SoapBubble(
        n_forward_iterations=n_forward_iterations,
        normalized_pressure=pressure,
        normalized_surface_tension=surface_tension,
        target_cell_volume=cell_volumes,
        contact_distance_threshold=0.3e-6,
    )

    # set up the optimizer
    print(list(model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    iteration_indices = []
    iteration_times = []
    losses = []
    pressures = []
    surface_tensions = []
    start_time = time.time()
    for optimization_index in range(n_optimization_iterations):
        print(f"\noptimization iteration: {optimization_index}")

        # list of faces for each cell
        # each element is a (n_cell_face, 3) array of indices
        # for the vertices of the faces
        cell_face_list = [
            initial_faces[cell_faces, :] for cell_faces in face_cell_index
        ]
        mesh_data = {
            "vertices": initial_vertices,
            "faces": initial_faces,
            "cell_face_list": cell_face_list,
            "face_cell_index": face_cell_index,
            "bounding_boxes": bounding_boxes,
        }

        # clear the gradients
        model.zero_grad()

        final_mesh = model(mesh_data)

        # compute a loss
        final_vertices = final_mesh["vertices"]
        mean_coordinate = torch.mean(final_vertices, dim=0)

        axis_1_length = (
            torch.max(final_vertices, dim=0)[0][1]
            - torch.min(final_vertices, dim=0)[0][1]
        )
        print(f"    axis 1 length: {axis_1_length * 1e6}")

        loss = (torch.abs(axis_1_length - 42e-6)) / (42e-6)
        loss.backward()
        print(f"    loss: {loss.item()}")
        print(f"    st grad before clip: {model.normalized_surface_tension.grad}")
        print(f"    p grad before clip: {model.normalized_pressure.grad}")
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # print(f"    st grad after clip: {model.normalized_surface_tension.grad}")
        # print(f"    p grad after clip: {model.normalized_pressure.grad}")

        optimizer.step()

        # record the data
        iteration_indices.append(optimization_index)
        losses.append(float(loss.item()))
        pressures.append(float(model.normalized_pressure.data))
        surface_tensions.append(float(model.normalized_surface_tension.item()))
        iteration_times.append(time.time() - start_time)

        if optimization_index % 10 == 0:
            # plot the data
            f, axs = plt.subplots(3, 1, sharex=True)

            axs[0].plot(
                iteration_indices,
                losses,
                color="black",
            )

            axs[0].set_title("loss")

            axs[1].plot(
                iteration_indices,
                pressures,
                color="blue",
            )

            axs[1].set_title("normalized pressure [-]")

            axs[2].plot(iteration_indices, surface_tensions, color="red")
            axs[2].set_title("normalized surface tension [-]")
            axs[2].set_xlabel("iteration")
            f.savefig("cube_doublet.png", bbox_inches="tight", dpi=300)

            f_2, ax_2 = plt.subplots(1, 1)
            ax_2.plot(iteration_indices, iteration_times)
            average_time = iteration_times[-1] / (iteration_indices[-1] + 1)
            ax_2.set_title(f"average time/iteration: {average_time}")
            ax_2.set_xlabel("iteration")
            ax_2.set_ylabel("time [s]")
            f_2.savefig("iteration_times.png", bbox_inches="tight", dpi=300)

    # make the viewer
    plane_parameters = {"position": (0, 15, 0), "normal": (0, 1, 0), "enabled": True}

    viewer = napari.Viewer()
    initial_mesh_layer = viewer.add_surface(
        (initial_vertices.numpy(force=True) * 1e6, initial_faces.numpy(force=True)),
        experimental_clipping_planes=[plane_parameters],
        name="initial mesh",
        visible=False,
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

    # add the contact map
    contact_mask = final_mesh["vertex_contact_map"][0].to_dense().sum(1) > 0
    contacting_vertices = final_mesh["vertices"][contact_mask]
    viewer.add_points(
        data=contacting_vertices.numpy(force=True) * 1e6,
        size=0.2,
        face_color="red",
        name="contacting vertices",
    )

    # visualize one contacting vertex pair
    contacting_indices = torch.argwhere(final_mesh["vertex_contact_map"][0].to_dense())
    first_contacting_index = contacting_indices[50]
    pair_coordinates_0 = final_mesh["vertices"][first_contacting_index[0]].unsqueeze(
        dim=0
    )
    pair_coordinates_1 = final_mesh["vertices"][first_contacting_index[1]].unsqueeze(
        dim=0
    )
    viewer.add_points(
        data=pair_coordinates_0.numpy(force=True) * 1e6,
        size=1,
        face_color="green",
        name=f"contact pair {first_contacting_index[0]}",
    )
    viewer.add_points(
        data=pair_coordinates_1.numpy(force=True) * 1e6,
        size=1,
        face_color="blue",
        name=f"contact pair {first_contacting_index[1]}",
    )

    viewer.dims.ndisplay = 3

    napari.run()

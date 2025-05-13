"""Forward simulation of a single soap bubble."""

import time

import napari
import numpy as np
import pyacvd
import pyvista as pv
import torch
from skimage.draw import ellipsoid
from skimage.measure import marching_cubes

from mechanomorph.sim._deformable_cell_model._forces import (
    compute_cell_volume,
    compute_vertex_forces,
)
from mechanomorph.sim._deformable_cell_model._mesh_utils import (
    get_face_vertex_mapping,
    get_per_cell_face_vertex_mapping,
)


def resample_mesh(
    vertices,
    faces,
    target_area: float = 1.0,
):
    """Resample a meshed surface using voronoi clustering.

    Parameters
    ----------
    vertices : np.ndarray
        (n x 3) array containing the coordinates of the vertices of the mesh
    faces : np.ndarray
        (n x 3) array containing the indices of the vertex for each face.
    target_area : float, optional
        The target area of each vertex in the mesh,
        by default 1.0

    Returns
    -------
    vertices : np.ndarray
        (n x 3) array containing the coordinates of the vertices of the mesh
    faces : np.ndarray
        (n x 3) array containing the indices of the vertex for each face.
    """
    # Prepend 3 for pyvista format
    faces = np.concatenate((np.ones((faces.shape[0], 1), dtype=int) * 3, faces), axis=1)

    # Create a mesh
    surf = pv.PolyData(vertices, faces)
    # surf = surf.smooth(n_iter=smoothing)

    # remesh to desired point size
    cluster_points = int(surf.area / target_area)
    # cluster_points = n_faces
    clus = pyacvd.Clustering(surf)
    # clus.subdivide(3)
    clus.cluster(cluster_points)
    remeshed = clus.create_mesh()

    verts = remeshed.points
    faces = remeshed.faces.reshape(-1, 4)[:, 1:]

    # switch face order to have outward normals
    faces = faces[:, [0, 2, 1]]

    return verts, faces


def make_mesh(
    radius: float = 15.0, target_element_area: float = 5.0, voxel_size: float = 1e-6
):
    """Make a sphere mesh with a given radius."""
    diameter = 2 * radius
    # we offset the sphere so that it doesn't touch the boundary of the image
    starting_coordinate = np.array([1, 1, 1])
    ellipsoid_mask = ellipsoid(diameter, diameter, diameter, levelset=False)

    # embed the mask in the image
    image_shape = 2 + np.array(ellipsoid_mask.shape)
    label_image = np.zeros(image_shape, dtype=bool)
    max_z = starting_coordinate[0] + ellipsoid_mask.shape[0]
    max_y = starting_coordinate[1] + ellipsoid_mask.shape[1]
    max_x = starting_coordinate[2] + ellipsoid_mask.shape[2]
    label_image[
        starting_coordinate[0] : max_z,
        starting_coordinate[1] : max_y,
        starting_coordinate[2] : max_x,
    ] = ellipsoid_mask

    # convert the mask to a mesh
    vertices, faces, _, _ = marching_cubes(label_image, 0)

    resampled_vertices, resampled_faces = resample_mesh(
        vertices=vertices, faces=faces, target_area=target_element_area
    )
    return (
        torch.from_numpy(np.asarray(resampled_vertices)).float() * voxel_size,
        torch.from_numpy(np.asarray(resampled_faces)),
    )


class SoapBubble(torch.nn.Module):
    """Model of a soap bubble."""

    def __init__(
        self,
        n_forward_iterations: int = 10,
        time_step: float = 30.0,
        surface_tension: float = 1.0,
        target_cell_volume: float = 1.0,
        bulk_modulus: float = 2500.0,
    ):
        super().__init__()

        # numerical parameters
        self.n_forward_iterations = n_forward_iterations
        self.time_step = time_step

        # static parameters
        self.bulk_modulus = bulk_modulus

        # model parameters (these can be optimized)
        self.surface_tension = torch.tensor(surface_tension, dtype=torch.float32)
        self.target_cell_volume = torch.tensor(
            [target_cell_volume], dtype=torch.float32
        )
        self.pressure = torch.tensor(100, dtype=torch.float32)

    def forward(self, mesh_data):
        """Run the forward simulation."""
        self.vertex_coordinates = mesh_data["vertices"]
        self.faces = mesh_data["faces"]

        scaled_target_volume = self.target_cell_volume * torch.exp(
            self.pressure / self.bulk_modulus
        )

        for _ in range(self.n_forward_iterations):
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
                vertices=self.vertex_coordinates, cell_face_lst=[self.faces]
            )

            # # get the surface tensions
            face_surface_tension = self.surface_tension * torch.ones(
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

            self.vertex_coordinates = (
                self.vertex_coordinates + self.time_step * vertex_forces
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
        }


if __name__ == "__main__":
    # initial mesh parameters
    initial_radius = 15
    target_element_area = 5.0
    voxel_size = 1e-6

    # numeric parameters
    n_forward_iterations = 100

    # mechanical parameters
    surface_tension = 0.001

    initial_vertices, initial_faces = make_mesh(
        radius=initial_radius,
        target_element_area=target_element_area,
        voxel_size=voxel_size,
    )
    print(initial_vertices.size())
    indexing_start = time.time()
    vertex_0_coordinates = torch.index_select(
        initial_vertices, dim=0, index=initial_faces[:, 0]
    )
    vertex_1_coordinates = torch.index_select(
        initial_vertices, dim=0, index=initial_faces[:, 1]
    )
    vertex_2_coordinates = torch.index_select(
        initial_vertices, dim=0, index=initial_faces[:, 2]
    )
    print(f"indexing time: {time.time() - indexing_start}")

    cell_volume = compute_cell_volume(
        vertex_0_coordinates,
        vertex_1_coordinates,
        vertex_2_coordinates,
    )
    expected_volume = 4 / 3 * np.pi * (voxel_size * initial_radius) ** 3
    print(f"computed volume: {cell_volume} expected: {expected_volume}")

    model = SoapBubble(
        n_forward_iterations=n_forward_iterations,
        surface_tension=surface_tension,
        target_cell_volume=cell_volume,
    )

    mesh_data = {
        "vertices": initial_vertices,
        "faces": initial_faces,
    }

    final_mesh = model(mesh_data)

    # make the viewer
    viewer = napari.Viewer()
    initial_mesh_layer = viewer.add_surface(
        (initial_vertices.numpy(force=True) * 1e6, initial_faces.numpy(force=True)),
        name="initial mesh",
    )
    initial_mesh_layer.wireframe.visible = True

    final_mesh_layer = viewer.add_surface(
        (
            final_mesh["vertices"].numpy(force=True) * 1e6,
            final_mesh["faces"].numpy(force=True),
        ),
        name="final mesh",
    )
    final_mesh_layer.wireframe.visible = True

    viewer.dims.ndisplay = 3

    napari.run()

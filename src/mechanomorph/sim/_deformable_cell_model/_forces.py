import torch

from mechanomorph.sim._deformable_cell_model._mesh_utils import (
    compute_face_unit_normals,
)


def compute_surface_tension_forces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_vertex_mapping_v0: torch.Tensor,
    face_vertex_mapping_v1: torch.Tensor,
    face_vertex_mapping_v2: torch.Tensor,
    face_surface_tension: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Compute the surface tension forces on each vertex.

    Parameters
    ----------
    vertices: torch.Tensor
        (n_vertices, 3) array of the coordinates of the vertices of the mesh.
    faces: torch.Tensor
        (n_faces, 3) array of the indices of the vertices of each face of the mesh.
    face_vertex_mapping_v0: torch.Tensor
        (n_faces, n_vertices) sparse tensor mapping the faces to the vertices.
        This is for the 0th vertex in each face.
    face_vertex_mapping_v1: torch.Tensor
        (n_faces, n_vertices) sparse tensor mapping the faces to the vertices.
        This is for the 1st vertex in each face.
    face_vertex_mapping_v2: torch.Tensor
        (n_faces, n_vertices) sparse tensor mapping the faces to the vertices.
        This is for the 2nd vertex in each face.
    face_surface_tension: torch.Tensor
        (n_faces,) array of the surface tension for each face.
    epsilon: float
        Small value to avoid division by zero.
    """
    # get the coordinates of each vertex
    # has shape (n_faces, 3)
    vertex_0_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 0])
    vertex_1_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 1])
    vertex_2_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 2])

    # compute the surface tension forces for each face
    # this computes it for each vertex in each element
    # each one has shape (n_faces, 3)
    unit_normals = compute_face_unit_normals(
        vertices=vertices,
        faces=faces,
        epsilon=epsilon,
    )
    surface_tension_unsqueezed = face_surface_tension.unsqueeze(1)

    surface_tension_force_face_v0 = (
        torch.linalg.cross(unit_normals, vertex_2_coordinates - vertex_1_coordinates)
        * surface_tension_unsqueezed
    )
    surface_tension_force_face_v1 = (
        torch.linalg.cross(unit_normals, vertex_0_coordinates - vertex_2_coordinates)
        * surface_tension_unsqueezed
    )
    surface_tension_force_face_v2 = (
        torch.linalg.cross(unit_normals, vertex_1_coordinates - vertex_0_coordinates)
        * surface_tension_unsqueezed
    )

    # some the contributions from each face to each vertex
    # each array has shape (n_vertices, 3)
    surface_tension_force_v0 = face_vertex_mapping_v0.T @ surface_tension_force_face_v0
    surface_tension_force_v1 = face_vertex_mapping_v1.T @ surface_tension_force_face_v1
    surface_tension_force_v2 = face_vertex_mapping_v2.T @ surface_tension_force_face_v2

    return -0.5 * (
        surface_tension_force_v0 + surface_tension_force_v1 + surface_tension_force_v2
    )


def compute_cell_volume(
    vertex_0_coordinates: torch.Tensor,
    vertex_1_coordinates: torch.Tensor,
    vertex_2_coordinates: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.abs(
            torch.linalg.vecdot(
                vertex_0_coordinates,
                torch.linalg.cross(vertex_1_coordinates, vertex_2_coordinates),
                dim=0,
            ).sum()
        )
        / 6.0
    )


def cell_volume_gradient_wrt_vertices(
    vertex_0_coordinates,
    vertex_1_coordinates,
    vertex_2_coordinates,
    face_vertex_mapping_v0,
    face_vertex_mapping_v1,
    face_vertex_mapping_v2,
) -> torch.Tensor:
    """Compute the gradient of the cell volume with respect to the vertices."""
    # These matrices have shape (N_face, 3)
    face_grad_cell_volume_ar_n1 = torch.linalg.cross(
        vertex_1_coordinates, vertex_2_coordinates
    )
    face_grad_cell_volume_ar_n2 = torch.linalg.cross(
        vertex_2_coordinates, vertex_0_coordinates
    )
    face_grad_cell_volume_ar_n3 = torch.linalg.cross(
        vertex_0_coordinates, vertex_1_coordinates
    )

    # These matrices have shape (N_node, 3)
    node_grad_cell_volume_n1 = face_vertex_mapping_v0.T @ face_grad_cell_volume_ar_n1
    node_grad_cell_volume_n2 = face_vertex_mapping_v1.T @ face_grad_cell_volume_ar_n2
    node_grad_cell_volume_n3 = face_vertex_mapping_v2.T @ face_grad_cell_volume_ar_n3

    return (
        node_grad_cell_volume_n1 + node_grad_cell_volume_n2 + node_grad_cell_volume_n3
    ) / 6.0


def compute_pressure_forces(
    vertices: torch.Tensor,
    face_vertex_mapping_v0_all_cells: list[torch.Tensor],
    face_vertex_mapping_v1_all_cells: list[torch.Tensor],
    face_vertex_mapping_v2_all_cells: list[torch.Tensor],
    target_cell_volume: torch.Tensor,
    bulk_modulus: float,
) -> torch.Tensor:
    """Compute the pressure forces on each vertex."""
    # Loop over the cells
    all_pressure_forces = []
    for cell_idx in range(target_cell_volume.shape[0]):
        # Retrieve the face information of the cell
        face_vertex_mapping_v0 = face_vertex_mapping_v0_all_cells[cell_idx]
        face_vertex_mapping_v1 = face_vertex_mapping_v1_all_cells[cell_idx]
        face_vertex_mapping_v2 = face_vertex_mapping_v2_all_cells[cell_idx]

        # Get the target volume of the cell
        cell_target_volume = target_cell_volume[cell_idx]

        # Extract the node positions of the faces in the given cell
        vertex_coordinates_v0 = face_vertex_mapping_v0 @ vertices
        vertex_coordinates_v1 = face_vertex_mapping_v1 @ vertices
        vertex_coordinates_v2 = face_vertex_mapping_v2 @ vertices

        # Compute the volume of the cell
        cell_volume = compute_cell_volume(
            vertex_coordinates_v0, vertex_coordinates_v1, vertex_coordinates_v2
        )

        # Compute the gradient of the cell volume with respect to the node positions
        grad_cell_volume_ar = cell_volume_gradient_wrt_vertices(
            vertex_coordinates_v0,
            vertex_coordinates_v1,
            vertex_coordinates_v2,
            face_vertex_mapping_v0,
            face_vertex_mapping_v1,
            face_vertex_mapping_v2,
        )

        # Compute the pressure forces
        pressure_forces = (
            -bulk_modulus
            * torch.log(cell_volume / cell_target_volume)
            * grad_cell_volume_ar
        )
        all_pressure_forces.append(pressure_forces)

    return torch.stack(all_pressure_forces).sum(dim=0)


def compute_vertex_forces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_vertex_mapping_v0: torch.Tensor,
    face_vertex_mapping_v1: torch.Tensor,
    face_vertex_mapping_v2: torch.Tensor,
    face_vertex_mapping_v0_all_cells: list[torch.Tensor],
    face_vertex_mapping_v1_all_cells: list[torch.Tensor],
    face_vertex_mapping_v2_all_cells: list[torch.Tensor],
    target_cell_volume: torch.Tensor,
    bulk_modulus: float,
    face_surface_tension: torch.Tensor,
    static_nodes_mask: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Compute the node forces.

    This includes contributions from pressure and surface tension.

    Parameters
    ----------
    vertices: torch.Tensor
        (n_vertices, 3) array of the coordinates of the vertices of the mesh.
    faces: torch.Tensor
        (n_faces, 3) array of the indices of the vertices of each face of the mesh.
    face_vertex_mapping_v0: torch.Tensor
        (n_faces, n_vertices) sparse tensor mapping the faces to the vertices.
        This is for the 0th vertex in each face.
    face_vertex_mapping_v1: torch.Tensor
        (n_faces, n_vertices) sparse tensor mapping the faces to the vertices.
        This is for the 1st vertex in each face.
    face_vertex_mapping_v2: torch.Tensor
        (n_faces, n_vertices) sparse tensor mapping the faces to the vertices.
        This is for the 2nd vertex in each face.
    face_vertex_mapping_v0_all_cells : list[torch.Tensor]
        A list of (n_faces, n_vertices) sparse tensors. Each tensor maps
        the faces to the 0th vertex of the faces for a cell.
    face_vertex_mapping_v1_all_cells : list[torch.Tensor]
        A list of (n_faces, n_vertices) sparse tensors. Each tensor maps
        the faces to the 1st vertex of the faces for a cell.
    face_vertex_mapping_v2_all_cells : list[torch.Tensor]
        A list of (n_faces, n_vertices) sparse tensors. Each tensor maps
        the faces to the 2nd vertex of the faces for a cell.
    face_surface_tension: torch.Tensor
        (n_faces,) array of the surface tension for each face.
    target_cell_volume: torch.Tensor
        (n_cells,) array of the target volume for each cell.
    bulk_modulus : float
        The bulk modulus of the cells.
    static_nodes_mask: torch.Tensor
        (n_vertices,) boolean array set to True when a node should be static.
    epsilon: float
        Small value to avoid division by zero.

    Returns
    -------
    node_forces: torch.Tensor
        (n_vertices, 3) array of the forces acting on each node.
    """
    # compute the surface tension forces
    surface_tension_forces = compute_surface_tension_forces(
        vertices=vertices,
        faces=faces,
        face_vertex_mapping_v0=face_vertex_mapping_v0,
        face_vertex_mapping_v1=face_vertex_mapping_v1,
        face_vertex_mapping_v2=face_vertex_mapping_v2,
        face_surface_tension=face_surface_tension,
        epsilon=epsilon,
    )

    # compute the pressure forces
    pressure_forces = compute_pressure_forces(
        vertices=vertices,
        face_vertex_mapping_v0_all_cells=face_vertex_mapping_v0_all_cells,
        face_vertex_mapping_v1_all_cells=face_vertex_mapping_v1_all_cells,
        face_vertex_mapping_v2_all_cells=face_vertex_mapping_v2_all_cells,
        target_cell_volume=target_cell_volume,
        bulk_modulus=bulk_modulus,
    )

    # compute the total forces acting on each node
    node_forces = surface_tension_forces + pressure_forces

    # mask the static nodes
    node_forces[static_nodes_mask, :] = 0

    return node_forces

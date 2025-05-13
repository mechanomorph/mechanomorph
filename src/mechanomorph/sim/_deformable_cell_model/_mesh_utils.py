"""Utility functions for mesh operations."""

import torch


def get_face_vertex_mapping(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the mapping between face and vertex indices.

    This produces 3 mappings, one for each vertex in a face.

    face_vertex_mapping_n1 =

        |   | n1 | n2 | n3 | ...
        |---|----|----|----|----
        | f1|  1 |  0 |  0 | ...
        | f2|  1 |  0 |  0 | ...
        | f3|  0 |  0 |  1 | ...

        In this example, the first vertex of f1 is n1,
        the first vertex of f2 is n1 and the first vertex of f3 is n3.

    Parameters
    ----------
    vertices: np.ndarray
        (n_vertices, 3) array of vertex positions.
    faces: np.ndarray
        (n_faces, 3) array of the indices of the vertices of each face of the mesh.
    dtype : torch.dtype
        The data type of the resulting matrices. Default is torch.float32.

    Returns
    -------
    3x adjacency_matrix : torch.sparse_coo (N_face_mesh, N_node)
        Returns 3 matrices. The first matrix contains the information
        of the first node of each face, the second matrix
        contains the information of the second node of each face, etc...
    """
    n_nodes = vertices.shape[0]
    n_faces = faces.shape[0]

    data_n1 = torch.ones((n_faces,), dtype=dtype, device=vertices.device)
    data_n2 = torch.ones((n_faces,), dtype=dtype, device=vertices.device)
    data_n3 = torch.ones((n_faces,), dtype=dtype, device=vertices.device)

    col_n1 = faces[:, 0]
    col_n2 = faces[:, 1]
    col_n3 = faces[:, 2]

    face_indices = torch.arange(n_faces, device=vertices.device)
    n1_indices = torch.column_stack([face_indices, col_n1]).T
    n2_indices = torch.column_stack([face_indices, col_n2]).T
    n3_indices = torch.column_stack([face_indices, col_n3]).T

    face_node_mapping_n1 = torch.sparse_coo_tensor(
        indices=n1_indices, values=data_n1, size=(n_faces, n_nodes)
    )
    face_node_mapping_n2 = torch.sparse_coo_tensor(
        indices=n2_indices, values=data_n2, size=(n_faces, n_nodes)
    )
    face_node_mapping_n3 = torch.sparse_coo_tensor(
        indices=n3_indices, values=data_n3, size=(n_faces, n_nodes)
    )

    return (
        face_node_mapping_n1,
        face_node_mapping_n2,
        face_node_mapping_n3,
    )


def get_per_cell_face_vertex_mapping(
    vertices: torch.Tensor,
    cell_face_lst: list[torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Get the face-vertex mapping for each cell.

    Parameters
    ----------
    vertices : torch.Tensor
        (n_vertices, 3) array of the coordinates of each vertex in the mesh.
    cell_face_lst : list[torch.Tensor]
        A list of (n_faces_in_cell, 3) arrays. Each array contains
        the indices of the vertices of each face in a given cell.
    dtype: torch.dtype
        The data type of the resulting matrices. Default is torch.float32.

    Returns
    -------
    face_vertex_v0_all_cells : list[torch.Tensor]
        A list of (n_faces, n_vertices) sparse tensors. Each tensor maps
        the faces to the 0th vertex of the faces for a cell.
    face_vertex_v1_all_cells : list[torch.Tensor]
        A list of (n_faces, n_vertices) sparse tensors. Each tensor maps
        the faces to the 1st vertex of the faces for a cell.
    face_vertex_v2_all_cells : list[torch.Tensor]
        A list of (n_faces, n_vertices) sparse tensors. Each tensor maps
        the faces to the 2nd vertex of the faces for a cell.
    """
    face_vertex_v0_all_cells = []
    face_vertex_v1_all_cells = []
    face_vertex_v2_all_cells = []

    # the cell face lst is just the faces since we have only one cell

    for cell_faces in cell_face_lst:
        (
            face_vertex_v0_cell,
            face_vertex_v1_cell,
            face_vertex_v2_cell,
        ) = get_face_vertex_mapping(vertices, cell_faces, dtype=dtype)

        face_vertex_v0_all_cells.append(face_vertex_v0_cell)
        face_vertex_v1_all_cells.append(face_vertex_v1_cell)
        face_vertex_v2_all_cells.append(face_vertex_v2_cell)

    return face_vertex_v0_all_cells, face_vertex_v1_all_cells, face_vertex_v2_all_cells


def compute_face_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
):
    """Compute the normals of each face of the mesh.

    Note: these are not unit normals.

    Parameters
    ----------
    vertices: torch.Tensor
        (n_vertices, 3) array of the coordinates of the vertices of the mesh.
    faces: torch.Tensor
        (n_faces, 3) array of the indices of the vertices of each face of the mesh.


    Returns
    -------
    normals: torch.Tensor
        (n_faces, 3) array of the normals of each face of the mesh.
    """
    vertex_0_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 0])
    vertex_1_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 1])
    vertex_2_coordinates = torch.index_select(vertices, dim=0, index=faces[:, 2])

    # Compute face normals
    return torch.linalg.cross(
        vertex_1_coordinates - vertex_0_coordinates,
        vertex_2_coordinates - vertex_0_coordinates,
    )


def compute_face_unit_normals(
    vertices: torch.Tensor, faces: torch.Tensor, epsilon: float = 1e-12
) -> torch.Tensor:
    """Compute the unit normal vector for each face of the mesh.

    Parameters
    ----------
    vertices: torch.Tensor
        (n_vertices, 3) array of the coordinates of the vertices of the mesh.
    faces: torch.Tensor
        (n_faces, 3) array of the indices of the vertices of each face of the mesh.
    epsilon: float
        Minimum normal vector magnitude. This is used to avoid a divide by zero when
        converting to unit vectors. Default value is 1e-12.

    Returns
    -------
    normals: torch.Tensor
        (n_faces, 3) array of the unit normal vectors of each face of the mesh.
    """
    normals = compute_face_normals(vertices, faces)
    return torch.nn.functional.normalize(normals, dim=1, p=2, eps=epsilon)

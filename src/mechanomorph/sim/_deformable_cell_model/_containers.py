import torch


class CellMesh:
    """A container for a triangular mesh of a cell.

    Parameters
    ----------
    vertices : torch.Tensor
        A tensor of shape (N, 3) containing the vertex positions.
    faces : torch.Tensor
        A tensor of shape (M, 3) containing the vertex indices
        of the triangular faces.
    """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        self._vertices = vertices
        self._faces = faces

    @property
    def vertices(self) -> torch.Tensor:
        """Get the vertex positions."""
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: torch.Tensor):
        """Set the vertex positions."""
        self._vertices = vertices

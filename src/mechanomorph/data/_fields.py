import torch


class ScalarField:
    """A scalar field that can be sampled for values.

    Parameters
    ----------
    field : torch.Tensor
        The field to sample from. Should be a 3D tensor.
    origin : tuple[float, float, float]
        The coordinate of the origin of the field in global coordinates.
        The default value is (0, 0, 0).
    scale : tuple[float, float, float]
        The size of the field voxels in global coordinates.
        This is used to convert the sampling coordinates to field coordinates.
        The sampling coordinates are divided by the scale.
        The default value is (1, 1, 1).
    """

    def __init__(
        self,
        field: torch.Tensor,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self._field = field

        # store the origin and scale
        self._origin = torch.tensor(origin, device=field.device)
        self._scale = torch.tensor(scale, device=field.device)

    @property
    def field(self) -> torch.Tensor:
        """Get the field."""
        return self._field

    @property
    def origin(self) -> torch.Tensor:
        """Get the origin."""
        return self._origin

    @property
    def scale(self) -> torch.Tensor:
        """Get the scale."""
        return self._scale

    def sample(self, positions: torch.Tensor, order: int = 0):
        """Sample the field at the given positions.

        Parameters
        ----------
        positions : torch.Tensor
            (n_points, 3) array containing positions to sample the field at.
            The positions should be given in global coordinates.
            They are transformed to the field coordinates using:
            field_coordinates = (positions - origin) / scale
        order : int
            The interpolation order to use in sampling.
            Currently only order 0 is implemented (nearest neighbor).
            Default is 0.
        """
        if order != 0:
            raise NotImplementedError("Only order 0 sampling is implemented.")

        # transform the positions to field coordinates
        positions = (positions - self.origin) / self.scale

        # round the positions to integer values to be
        # used as indices
        indices = positions.round().long()

        # clamp indices to the boundaries of the field
        max_indices = torch.tensor(self.field.shape, device=self.field.device) - 1
        indices = torch.clamp(
            indices, torch.tensor([0, 0, 0], device=self.field.device), max=max_indices
        )

        # sample the field at the given indices
        return self.field[indices[:, 0], indices[:, 1], indices[:, 2]]


class VectorField:
    """A vector field that can be sampled for values.

    Parameters
    ----------
    field : torch.Tensor
        The field to sample from. Should be a 4D tensor.
        The tensor (n_vector_dims, h, w, d).
    origin : tuple[float, float, float]
        The coordinate of the origin of the field in the global coordinates.
        The default value is (0, 0, 0).
    scale : tuple[float, float, float]
        The size of the field voxels in global coordinates.
        This is used to convert the sampling coordinates to field coordinates.
        The sampling coordinates are divided by the scale.
        The default value is (1, 1, 1).

    """

    def __init__(
        self,
        field: torch.Tensor,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self._field = field

        # store the origin and scale
        self._origin = torch.tensor(origin, device=field.device)
        self._scale = torch.tensor(scale, device=field.device)

    @property
    def field(self) -> torch.Tensor:
        """Get the field."""
        return self._field

    @property
    def origin(self) -> torch.Tensor:
        """Get the origin."""
        return self._origin

    @property
    def scale(self) -> torch.Tensor:
        """Get the scale."""
        return self._scale

    def sample(self, positions: torch.Tensor, order: int = 0):
        """Sample the field at the given positions.

        Parameters
        ----------
        positions : torch.Tensor
            (n_points, 3) array containing positions to sample the field at.
        order : int
            The interpolation order to use in sampling.
            Currently only order 0 is implemented (nearest neighbor).
            Default is 0.
        """
        if order != 0:
            raise NotImplementedError("Only order 0 sampling is implemented.")

        # transform the positions to field coordinates
        positions = (positions - self.origin) / self.scale

        # round the positions to integer values to be
        # used as indices
        indices = positions.round().long()

        # clamp indices to the boundaries of the field
        max_indices = (
            torch.tensor(self.field.shape, device=self.field.device)[1:, ...] - 1
        )
        indices = torch.clamp(
            indices, torch.tensor([0, 0, 0], device=self.field.device), max=max_indices
        )

        # sample the field at the given indices
        return self.field[:, indices[:, 0], indices[:, 1], indices[:, 2]].T

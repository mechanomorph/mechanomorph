"""Functions for working with fields.

These are generally used as parameters (e.g., distance fields).
"""

import torch


def field_gradient(field: torch.Tensor) -> torch.Tensor:
    """Compute the unit vector gradient field.

    Parameters
    ----------
    field : torch.Tensor
        (h, w, d) array containing the scalar field to compute the gradient of.

    Returns
    -------
    gradient_vectors : torch.Tensor
        (3, h, w, d) array containing the gradient vector field. All vectors
        are unit vectors.
    """
    # compute the gradient
    axis_gradients = torch.gradient(field)

    # stack the components into vectors
    gradients = torch.stack(axis_gradients, dim=0)

    # normalize to unit vectors
    return torch.nn.functional.normalize(gradients, p=2, dim=0)

"""Test performance of a grid partitioning scheme."""

from collections import defaultdict

import torch


def create_grid_partition(points, threshold):
    """
    Create a grid partition for 3D points using vectorized operations.

    Args:
        points (torch.Tensor): A tensor of shape (N, 3) containing 3D points
        threshold (float): The distance threshold which determines cell size

    Returns
    -------
        dict: A mapping from cell coordinates to the indices of points in that cell
    """
    # Ensure points is a PyTorch tensor
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    # Determine grid cell size based on threshold
    cell_size = threshold

    # Get min coordinates to normalize the grid
    min_coords, _ = torch.min(points, dim=0)

    # Calculate grid cell indices for all points (vectorized)
    cell_indices = ((points - min_coords) / cell_size).floor().int()

    # Create a dictionary mapping cell coordinates to point indices
    grid = defaultdict(list)

    # Build the grid dictionary using numpy's unique function
    unique_cells, inverse_indices = torch.unique(
        cell_indices, dim=0, return_inverse=True
    )

    # For each unique cell, find all points that belong to it
    for i, unique_cell in enumerate(unique_cells):
        # Find all points that map to this unique cell
        points_in_this_cell = torch.where(inverse_indices == i)[0]
        # Store the cell and its points
        grid[tuple(unique_cell.cpu().tolist())] = points_in_this_cell.cpu().tolist()

    return grid


# Example usage
if __name__ == "__main__":
    # Generate random points in 3D space
    n_points = 10000
    device = "cpu"
    points = torch.rand(n_points, 3, device=device)
    threshold = 0.125

    # Create the grid partition
    import time

    start_time = time.time()
    grid = create_grid_partition(points, threshold)
    end_time = time.time()

    # Count total points assigned and cells created
    total_points = sum(len(indices) for indices in grid.values())
    print(f"Grid partition created with {len(grid)} cells")
    print(f"Total points assigned: {total_points} (should equal {n_points})")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # Calculate average number of points per cell
    avg_points_per_cell = total_points / len(grid) if grid else 0
    print(f"Average points per cell: {avg_points_per_cell:.2f}")

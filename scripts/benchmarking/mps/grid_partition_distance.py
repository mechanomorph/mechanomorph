"""Test performance of a grid partitioning scheme with distance thresholding."""

import time

import torch


def find_points_within_threshold(points, threshold):
    """Find all pairs of points within a threshold distance.

    This uses a grid-based approach.

    Args:
        points (torch.Tensor): A tensor of shape (N, 3) containing 3D points
        threshold (float): The maximum distance threshold

    Returns
    -------
        torch.Tensor: A tensor of shape (M, 3) containing (i, j, distance) for each pair
    """
    # Ensure points is a PyTorch tensor
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)

    device = points.device
    n_points = points.shape[0]

    # --- Step 1: Create the grid partition ---
    cell_size = threshold

    # Get min coordinates to normalize the grid
    min_coords, _ = torch.min(points, dim=0)

    # Calculate grid cell indices for all points
    cell_indices = ((points - min_coords) / cell_size).floor().int()

    # Create a unique hash for each cell
    multipliers = torch.tensor([1, 10000, 100000000], device=device)
    cell_hashes = torch.sum(cell_indices * multipliers, dim=1)

    # --- Step 2: For each point, find all potential neighbors ---
    # Generate all 27 neighboring cell offsets
    neighbor_offsets = torch.tensor(
        [[dx, dy, dz] for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]],
        device=device,
    )

    # For each point, calculate the hash of all its neighboring cells
    neighbor_cell_indices = cell_indices.unsqueeze(1) + neighbor_offsets.unsqueeze(
        0
    )  # Shape: [n_points, 27, 3]
    neighbor_cell_hashes = torch.sum(
        neighbor_cell_indices * multipliers, dim=2
    )  # Shape: [n_points, 27]

    # --- Step 3: Find pairs efficiently ---
    all_pairs = []
    all_distances = []

    # For each point, find all other points in its neighboring cells
    for i in range(n_points):
        # Get the current point
        current_point = points[i]

        # Get hashes of all neighboring cells for this point
        current_neighbor_hashes = neighbor_cell_hashes[i]

        # Find all points in any of the neighboring cells
        # Use vectorized operations to find points in neighboring cells
        in_any_neighbor = torch.any(
            cell_hashes.unsqueeze(1) == current_neighbor_hashes.unsqueeze(0), dim=1
        )
        neighbor_indices = torch.where(in_any_neighbor)[0]

        # Only consider points with higher indices to avoid duplicates
        neighbor_indices = neighbor_indices[neighbor_indices > i]

        if len(neighbor_indices) > 0:
            # Calculate distances to all neighboring points at once
            neighbor_points = points[neighbor_indices]
            distances = torch.norm(neighbor_points - current_point.unsqueeze(0), dim=1)

            # Filter by threshold
            mask = distances <= threshold
            if mask.sum() > 0:
                valid_neighbors = neighbor_indices[mask]
                valid_distances = distances[mask]

                # Add pairs to results
                for j, d in zip(valid_neighbors.tolist(), valid_distances.tolist()):
                    all_pairs.append((i, j))
                    all_distances.append(d)

    # Convert to tensor for return
    if all_pairs:
        pair_tensor = torch.tensor(all_pairs, device=device)
        distances_tensor = torch.tensor(all_distances, device=device).unsqueeze(1)
        result = torch.cat([pair_tensor, distances_tensor], dim=1)
        return result
    else:
        return torch.zeros((0, 3), device=device)  # Empty tensor with shape (0, 3)


# Example usage
if __name__ == "__main__":
    # Generate random points in 3D space
    n_points = 10000

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate random points
    torch.manual_seed(42)  # For reproducibility
    points = torch.rand(n_points, 3, device=device)
    threshold = 0.05

    # Time the function
    start_time = time.time()
    result_tensor = find_points_within_threshold(points, threshold)
    end_time = time.time()

    print(f"Found {len(result_tensor)} pairs within threshold {threshold}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

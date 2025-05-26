"""Compare the performance of different methods to compute pairwise distances.

In particular this compares cdist, pdist, and a vectorized norm.

A few forum issues on the different functions:
https://discuss.pytorch.org/t/why-is-torch-pdist-so-slow/178947
https://discuss.pytorch.org/t/pdist-vs-cdist-performance/192843
"""

import time

import torch


def vectors_distances_between_agents(
    positions: torch.Tensor, epsilon: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the vector between all agents.

    Parameters
    ----------
    positions : torch.Tensor
        (n_agents, 3) array of the positions of the agents.
    epsilon : float
        Small value to prevent divide by zero when computing unit vectors.

    Returns
    -------
    distances : torch.Tensor
        (n_agents, n_agents) array of distances between all agents.
    unit_vectors : torch.Tensor
        (n_agents, n_agents, 3) array of unit vectors between all agents.
        unit_vectors[i, j, :] is the unit vector from agent i to agent j.
    """
    # compute the distances
    distances = torch.cdist(positions, positions)

    vectors = positions.unsqueeze(0) - positions.unsqueeze(1)
    return distances, torch.nn.functional.normalize(vectors, eps=epsilon, dim=2)


if __name__ == "__main__":
    n_coordinates = 10000

    # make 3D coordinates
    coordinates = torch.rand(n_coordinates, 3, device="cpu")

    print(f"coordinates shape: {coordinates.shape}")

    # use cdist to calculate pairwise distances
    start_time = time.time()
    distances = torch.cdist(coordinates, coordinates, p=2)
    print(f"cpu cdist took {time.time() - start_time:.4f} seconds")

    # use pdist to calculate pairwise distances
    start_time = time.time()
    distances_pdist = torch.pdist(coordinates, p=2)
    print(f"cpu pdist took {time.time() - start_time:.4f} seconds")

    # use vectorized norm
    start_time = time.time()
    distances_vectorized = torch.norm(
        coordinates.unsqueeze(1) - coordinates.unsqueeze(0), dim=2
    )
    print(f"cpu vectorized norm took {time.time() - start_time:.4f} seconds")

    # use our function to calculate pairwise distances
    _ = vectors_distances_between_agents(coordinates)
    start_time = time.time()
    distances, vectors = vectors_distances_between_agents(coordinates)
    print(f"cpu our func took {time.time() - start_time:.4f} seconds")

    # to mps
    coordinates_mps = coordinates.to("mps")

    # use our function to calculate pairwise distances
    _ = vectors_distances_between_agents(coordinates_mps)
    torch.mps.synchronize()
    start_time = time.time()
    distances, vectors = vectors_distances_between_agents(coordinates_mps)
    print(f"mps our func took {time.time() - start_time:.4f} seconds")

    # use cdist to calculate pairwise distances
    _ = torch.cdist(coordinates_mps, coordinates_mps, p=2)
    torch.mps.synchronize()
    start_time = time.time()
    distances = torch.cdist(coordinates_mps, coordinates_mps, p=2)
    print(f"mps cdist took {time.time() - start_time:.4f} seconds")

    # use vectorized norm
    torch.mps.synchronize()
    start_time = time.time()
    distances_vectorized = torch.norm(
        coordinates_mps.unsqueeze(1) - coordinates_mps.unsqueeze(0), dim=2
    )
    print(f"mps vectorized norm took {time.time() - start_time:.4f} seconds")

"""Script to test MPS speedup over CPU for different computations.

Note: MPS support is on hold as there is no sparse matrix support in MPS.
"""

import time

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def find_close_vertices(
    vertices_i: torch.Tensor,
    vertices_j: torch.Tensor,
    threshold: float = 0.1,
):
    """Proxy function for computing the distance between two sets of vertices."""
    diff = vertices_i[:, None, :] - vertices_j[None, :, :]  # (n_i, n_j, 3)
    dists = torch.norm(diff, dim=-1)  # (n_i, n_j)
    close = dists < threshold

    return close


# n_vertices_to_test = [100, 1000, 10000, 100000]
n_vertices_to_test = [100, 500, 1000, 5000, 10000]
threshold = 0.1
time_cpu = []
time_mps = []
for n_vertices in tqdm(n_vertices_to_test):
    # make the vertex arrays
    vertices_i = torch.randn(n_vertices, 3, device="cpu")
    vertices_j = torch.randn(n_vertices, 3, device="cpu")

    # run on CPU
    # warmup
    _ = find_close_vertices(vertices_i, vertices_j, threshold)
    start_time = time.time()
    result = find_close_vertices(vertices_i, vertices_j, threshold)
    time_cpu.append(time.time() - start_time)
    assert result.device == torch.device("cpu")

    # run on MPS
    vertices_i_mps = vertices_i.to("mps:0")
    vertices_j_mps = vertices_j.to("mps:0")
    # warmup
    _ = find_close_vertices(vertices_i_mps, vertices_j_mps, threshold)
    start_time = time.time()
    result = find_close_vertices(vertices_i_mps, vertices_j_mps, threshold)
    time_mps.append(time.time() - start_time)
    assert result.device == torch.device("mps:0")


f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.loglog(n_vertices_to_test, time_cpu, label="CPU")
ax.loglog(n_vertices_to_test, time_mps, label="MPS")
ax.legend()
ax.set_xlabel("number of vertices")
ax.set_ylabel("time (s)")

f.savefig("mps_benchmark.png", bbox_inches="tight", dpi=300)

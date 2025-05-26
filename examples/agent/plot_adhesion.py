"""Plot the results of the adhesion simulation."""

import numpy as np
from matplotlib import pyplot as plt

log_file_name = "n_5000_0_4_max.npy"
plot_file_name_base = "n_5000_0_4"

min_adhesion = 0.04
max_adhesion = 0.4

positions = np.load(log_file_name)

time_points = np.unique(positions[:, 0])

# get the data from the first time point
first_time_point = int(np.min(time_points))
first_time_point_positions = positions[positions[:, 0] == first_time_point]

# get the data from the last time point
last_time_point = int(np.max(time_points))
last_time_point_positions = positions[positions[:, 0] == last_time_point]

# get the adhesion strengths
min_x = np.min(positions[:, 1])
max_x = np.max(positions[:, 1])
adhesion_x = np.linspace(min_x, max_x, num=100)
adhesion_strength = []
adhesion_slope = (max_adhesion - min_adhesion) / (max_x - min_x)
for x in adhesion_x:
    adhesion_strength.append(min_adhesion + adhesion_slope * (x - min_x))

f, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axs[0].plot(adhesion_x, adhesion_strength)
axs[0].set_xlabel("x position")
axs[0].set_ylabel("adhesion strength")
axs[0].set_ylim(0, 0.5)

axs[1].hist(
    first_time_point_positions[:, 1], bins=np.linspace(10, 210, 41), label="0 hrs"
)
axs[1].set_title("t = 0")
axs[1].set_xlabel("x position")
axs[1].set_ylabel("n cells")
axs[1].set_ylim((0, 175))
axs[1].axhline(5000 / 41, c="gray")

axs[2].hist(
    last_time_point_positions[:, 1], bins=np.linspace(10, 210, 41), label="2 hrs"
)
axs[2].set_title("t = 1000")
axs[2].set_xlabel("x position")
axs[2].set_ylabel("n cells")
axs[2].set_ylim((0, 175))
axs[2].axhline(5000 / 41, c="gray")

plt.tight_layout()
f.savefig(plot_file_name_base + "_hist.png")

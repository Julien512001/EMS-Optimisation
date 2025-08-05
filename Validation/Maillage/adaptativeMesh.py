import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

d1 = 2

def compute_sizes(N, L):
    def equation(r):
        return d1 * (r**N - 1) / (r - 1) - L
    r_solution = fsolve(equation, 1.5)[0]
    sizes = d1 * r_solution ** np.arange(N)
    sizes = sizes[::-1]
    positions = np.concatenate(([0], np.cumsum(sizes)))
    return sizes, positions




sizes_10, pos_10 = compute_sizes(8, 80)
sizes_20, pos_20 = compute_sizes(16, 160)
pos_10 = pos_10 + 80
fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

def plot_cells(ax, sizes, positions, title):
    for i in range(len(sizes)):
        rect = Rectangle(
            (positions[i], 0), sizes[i], 1,
            facecolor='none', edgecolor='black', linewidth=1
        )
        ax.add_patch(rect)

    ax.set_xlim(0, positions[-1])
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)

plot_cells(axs[0], sizes_10, pos_10, "Maillage progressif pour N = 10")

plot_cells(axs[1], sizes_20, pos_20, "Maillage progressif pour N = 20")

axs[1].set_xlabel("$x \\;[mm]$")

plt.savefig("Validation/Maillage/adaptativeMesh.pdf")
plt.tight_layout()
plt.show()


sizes, pos = compute_sizes(80)
print(sizes)
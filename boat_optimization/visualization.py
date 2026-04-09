"""
Visualization and STL export utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from .hull import HullParams, cross_section_curve, beam_at_x
from .analysis import compute_stability_curve, find_waterline, center_of_mass, total_mass
from . import config


def plot_cross_section(params: HullParams, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the midship cross-section of the hull."""
    if ax is None:
        _, ax = plt.subplots()

    y = np.linspace(-params.beam / 2, params.beam / 2, 300)
    z_bottom = cross_section_curve(y, params)

    ax.fill_between(y * 100, z_bottom * 100, params.depth * 100, alpha=0.3, label="Hull")
    ax.plot(y * 100, z_bottom * 100, "b-", linewidth=2)
    ax.axhline(params.depth * 100, color="b", linestyle="--", alpha=0.5)

    # Waterline
    wl = find_waterline(params)
    ax.axhline(wl * 100, color="c", linestyle="-.", label=f"Waterline ({wl*100:.1f} cm)")

    ax.set_xlabel("y (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_title("Midship Cross-Section")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_hull_top_view(params: HullParams, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the top-down (plan) view of the hull outline."""
    if ax is None:
        _, ax = plt.subplots()

    x = np.linspace(-params.length / 2, params.length / 2, 300)
    half_b = beam_at_x(x, params) / 2

    ax.fill_between(x * 100, -half_b * 100, half_b * 100, alpha=0.3)
    ax.plot(x * 100, half_b * 100, "b-", linewidth=2)
    ax.plot(x * 100, -half_b * 100, "b-", linewidth=2)

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Plan View (Top)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def plot_stability_curve(params: HullParams, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot righting moment vs heel angle."""
    if ax is None:
        _, ax = plt.subplots()

    result = compute_stability_curve(params)

    ax.plot(result.heel_angles_deg, result.righting_moments, "b-", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(result.avs_deg, color="r", linestyle="--",
               label=f"AVS = {result.avs_deg:.1f}°")
    ax.axvline(config.MIN_AVS_DEG, color="orange", linestyle=":",
               label=f"Required AVS = {config.MIN_AVS_DEG}°")

    ax.set_xlabel("Heel Angle (°)")
    ax.set_ylabel("Righting Moment (N·m)")
    ax.set_title("Stability Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_3d_hull(params: HullParams, n_x: int = 60, n_y: int = 60) -> plt.Figure:
    """Render a 3D surface plot of the hull."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    x_vals = np.linspace(-params.length / 2, params.length / 2, n_x)
    y_max = params.beam / 2

    X, Y, Z = [], [], []
    for xi in x_vals:
        lb = float(beam_at_x(np.array([xi]), params)[0])
        y_vals = np.linspace(-lb / 2, lb / 2, n_y)
        z_vals = cross_section_curve(y_vals * (params.beam / max(lb, 1e-12)), params)
        X.append(np.full_like(y_vals, xi) * 100)
        Y.append(y_vals * 100)
        Z.append(z_vals * 100)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    ax.plot_surface(X, Y, Z, alpha=0.6, cmap="Blues")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title("Hull Bottom Surface")
    return fig


def plot_summary(params: HullParams) -> plt.Figure:
    """Generate a multi-panel summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    plot_cross_section(params, axes[0])
    plot_hull_top_view(params, axes[1])
    plot_stability_curve(params, axes[2])

    fig.suptitle("Boat Design Summary", fontsize=14)
    fig.tight_layout()
    return fig


def export_stl(params: HullParams, filename: str = "boatmodel.stl",
               n_x: int = 100, n_y: int = 100) -> None:
    """
    Export the hull as an STL file for CNC fabrication.

    Uses numpy-stl to create a triangulated mesh of the hull solid.
    """
    from stl import mesh as stl_mesh

    # Build hull surface points
    x_vals = np.linspace(-params.length / 2, params.length / 2, n_x)

    vertices = []
    faces = []

    # Build grid of bottom surface + top (deck) surface
    bottom_grid = np.zeros((n_x, n_y, 3))
    top_grid = np.zeros((n_x, n_y, 3))

    for i, xi in enumerate(x_vals):
        lb = float(beam_at_x(np.array([xi]), params)[0])
        y_vals = np.linspace(-lb / 2, lb / 2, n_y)
        z_bottom = cross_section_curve(y_vals * (params.beam / max(lb, 1e-12)), params)

        for j in range(n_y):
            bottom_grid[i, j] = [xi, y_vals[j], z_bottom[j]]
            top_grid[i, j] = [xi, y_vals[j], params.depth]

    # Create triangles for bottom surface
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            # Bottom face (two triangles per quad)
            faces.append([bottom_grid[i, j], bottom_grid[i+1, j], bottom_grid[i+1, j+1]])
            faces.append([bottom_grid[i, j], bottom_grid[i+1, j+1], bottom_grid[i, j+1]])
            # Top face
            faces.append([top_grid[i, j], top_grid[i+1, j+1], top_grid[i+1, j]])
            faces.append([top_grid[i, j], top_grid[i, j+1], top_grid[i+1, j+1]])

    # Side faces (connect bottom to top at the edges)
    for i in range(n_x - 1):
        for edge_j in [0, n_y - 1]:
            faces.append([bottom_grid[i, edge_j], top_grid[i, edge_j], top_grid[i+1, edge_j]])
            faces.append([bottom_grid[i, edge_j], top_grid[i+1, edge_j], bottom_grid[i+1, edge_j]])

    # Bow and stern caps
    for edge_i in [0, n_x - 1]:
        for j in range(n_y - 1):
            faces.append([bottom_grid[edge_i, j], top_grid[edge_i, j], top_grid[edge_i, j+1]])
            faces.append([bottom_grid[edge_i, j], top_grid[edge_i, j+1], bottom_grid[edge_i, j+1]])

    faces = np.array(faces)
    hull_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            hull_mesh.vectors[i][j] = f[j]

    hull_mesh.save(filename)
    print(f"STL exported to {filename} ({len(faces)} triangles)")

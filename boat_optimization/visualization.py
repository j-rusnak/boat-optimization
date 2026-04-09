"""
Visualization and STL export utilities.

Includes:
- Cross-section plot (midship, with waterline)
- Plan view (top-down hull outline)
- Stability curve (righting moment and righting arm vs heel angle, AVS marked)
- AVS heel-angle animation panels (boat at several heel angles with COM / COB)
- 3D hull surface
- Multi-panel summary
- STL export for CNC fabrication
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .hull import HullParams, cross_section_z, beam_at_x, build_cross_section_polygon
from .analysis import (
    compute_stability_curve, find_waterline_upright, center_of_mass_body,
    total_mass, StabilityResult, _rotate_polygon, _rotate_point,
    _clip_polygon_below_z,
)
from . import config


# ===================================================================
# Cross-section
# ===================================================================

def plot_cross_section(params: HullParams, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the midship cross-section with waterline, COM, and ballast."""
    if ax is None:
        _, ax = plt.subplots()

    y = np.linspace(-params.beam / 2, params.beam / 2, 300)
    z_bottom = cross_section_z(y, params.beam, params)

    # Fill hull solid
    ax.fill_between(y * 100, z_bottom * 100, params.depth * 100,
                    alpha=0.25, color="steelblue", label="Hull (foam)")
    ax.plot(y * 100, z_bottom * 100, "b-", linewidth=2)
    ax.axhline(params.depth * 100, color="b", linestyle="--", alpha=0.4)

    # Waterline
    wl = find_waterline_upright(params)
    ax.axhline(wl * 100, color="cyan", linestyle="-.",
               linewidth=1.5, label=f"Waterline ({wl*100:.1f} cm)")

    # COM marker
    com = center_of_mass_body(params)
    ax.plot(0, com[1] * 100, "r^", markersize=10, label="COM")

    # Ballast marker
    ax.plot(0, params.ballast_z * 100, "ks", markersize=8, label="Ballast")

    # Mast line
    ax.plot([0, 0], [0, config.MAST_LENGTH_M * 100], "k-",
            linewidth=1.5, alpha=0.5, label="Mast")

    ax.set_xlabel("y (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_title("Midship Cross-Section")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax


# ===================================================================
# Plan view
# ===================================================================

def plot_hull_top_view(params: HullParams, ax: plt.Axes | None = None) -> plt.Axes:
    """Top-down hull outline."""
    if ax is None:
        _, ax = plt.subplots()

    x = np.linspace(-params.length / 2, params.length / 2, 300)
    half_b = beam_at_x(x, params) / 2

    ax.fill_between(x * 100, -half_b * 100, half_b * 100, alpha=0.25, color="steelblue")
    ax.plot(x * 100, half_b * 100, "b-", linewidth=2)
    ax.plot(x * 100, -half_b * 100, "b-", linewidth=2)

    ax.set_xlabel("x (cm) — bow →")
    ax.set_ylabel("y (cm)")
    ax.set_title("Plan View")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


# ===================================================================
# Stability curve
# ===================================================================

def plot_stability_curve(params: HullParams, ax: plt.Axes | None = None,
                         stability: StabilityResult | None = None) -> plt.Axes:
    """Plot righting moment and righting arm vs heel angle, marking AVS."""
    if ax is None:
        _, ax = plt.subplots()

    if stability is None:
        stability = compute_stability_curve(params)

    # Primary axis: righting moment
    color_m = "steelblue"
    ax.plot(stability.heel_angles_deg, stability.righting_moments,
            "-", color=color_m, linewidth=2, label="Righting moment")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.fill_between(stability.heel_angles_deg, 0, stability.righting_moments,
                    where=stability.righting_moments > 0,
                    alpha=0.15, color=color_m)

    # AVS line
    ax.axvline(stability.avs_deg, color="red", linestyle="--", linewidth=1.5,
               label=f"AVS = {stability.avs_deg:.1f}°")
    # Required AVS line
    ax.axvline(config.MIN_AVS_DEG, color="orange", linestyle=":", linewidth=1.5,
               label=f"Required > {config.MIN_AVS_DEG:.0f}°")

    # Secondary axis: righting arm
    ax2 = ax.twinx()
    color_a = "forestgreen"
    ax2.plot(stability.heel_angles_deg, stability.righting_arms * 100,
             "--", color=color_a, linewidth=1.5, alpha=0.7, label="Righting arm")
    ax2.set_ylabel("Righting Arm GZ (cm)", color=color_a)
    ax2.tick_params(axis="y", labelcolor=color_a)

    ax.set_xlabel("Heel Angle (°)")
    ax.set_ylabel("Righting Moment (N·m)", color=color_m)
    ax.tick_params(axis="y", labelcolor=color_m)
    ax.set_title("Stability Curve")
    ax.set_xlim(0, 180)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax


# ===================================================================
# AVS visualisation — boat at several heel angles
# ===================================================================

def plot_avs_panels(params: HullParams,
                    angles_deg: list[float] | None = None,
                    stability: StabilityResult | None = None) -> plt.Figure:
    """Show the boat cross-section at several heel angles, with COM, COB,
    waterline, and righting-arm annotation.

    This is the main AVS-visualisation figure.
    """
    if stability is None:
        stability = compute_stability_curve(params)

    if angles_deg is None:
        # Pick a few illustrative angles including near-AVS
        avs = stability.avs_deg
        angles_deg = sorted(set([0, 30, 60, 90, min(avs, 170), min(avs + 15, 180)]))

    n_panels = len(angles_deg)
    fig, axes = plt.subplots(2, (n_panels + 1) // 2, figsize=(5 * ((n_panels + 1) // 2), 10))
    axes = axes.flatten()

    # Find the closest computed angle for each requested angle
    for i, target_deg in enumerate(angles_deg):
        ax = axes[i]
        # Find closest result
        idx = int(np.argmin(np.abs(stability.heel_angles_deg - target_deg)))
        res = stability.results[idx]
        actual_deg = stability.heel_angles_deg[idx]

        _draw_heeled_cross_section(ax, params, res, actual_deg)

    # Hide unused axes
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Boat Stability at Various Heel Angles", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def _draw_heeled_cross_section(ax: plt.Axes, params: HullParams,
                                res, actual_deg: float) -> None:
    """Draw one panel: rotated hull, waterline, COM, COB, righting arm."""
    theta = np.radians(actual_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Build midship polygon in body frame, rotate to world
    poly_body = build_cross_section_polygon(params.beam, params, n_pts=120)
    poly_world = _rotate_polygon(poly_body, cos_t, sin_t)

    wl = res.waterline_world_z

    # Submerged part (clipped polygon)
    clipped = _clip_polygon_below_z(poly_world, wl)

    # Scale to cm for plotting
    pw = poly_world * 100
    hull_patch = plt.Polygon(pw, closed=True, facecolor="steelblue",
                             edgecolor="navy", alpha=0.3, linewidth=1.5)
    ax.add_patch(hull_patch)

    if clipped is not None and len(clipped) >= 3:
        cp = clipped * 100
        sub_patch = plt.Polygon(cp, closed=True, facecolor="cyan",
                                edgecolor="teal", alpha=0.35, linewidth=1)
        ax.add_patch(sub_patch)

    # Waterline
    y_range = pw[:, 0]
    y_min, y_max = y_range.min() - 2, y_range.max() + 2
    ax.plot([y_min, y_max], [wl * 100, wl * 100], "c-", linewidth=1.5, alpha=0.8)

    # Water fill below waterline
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < wl * 100 else wl * 100 - 5,
               wl * 100, color="cyan", alpha=0.08)

    # COM marker
    com = res.com_world * 100
    ax.plot(com[0], com[1], "r^", markersize=10, zorder=5)
    ax.annotate("G", (com[0], com[1]), fontsize=8, fontweight="bold",
                color="red", xytext=(4, 4), textcoords="offset points")

    # COB marker
    cob = res.cob_world * 100
    ax.plot(cob[0], cob[1], "go", markersize=8, zorder=5)
    ax.annotate("B", (cob[0], cob[1]), fontsize=8, fontweight="bold",
                color="green", xytext=(4, -8), textcoords="offset points")

    # Righting arm (horizontal line between G and B y-coords)
    ax.annotate("", xy=(cob[0], com[1]), xytext=(com[0], com[1]),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))

    # Mast (rotated)
    mast_bottom_body = np.array([0.0, 0.0])
    mast_top_body = np.array([0.0, config.MAST_LENGTH_M])
    mast_bottom_w = _rotate_point(mast_bottom_body, cos_t, sin_t) * 100
    mast_top_w = _rotate_point(mast_top_body, cos_t, sin_t) * 100
    ax.plot([mast_bottom_w[0], mast_top_w[0]], [mast_bottom_w[1], mast_top_w[1]],
            "k-", linewidth=2, alpha=0.6)

    # Labels
    gz_cm = res.righting_arm_m * 100
    m_str = f"{res.righting_moment_Nm:.4f}"
    title = f"θ = {actual_deg:.0f}°\nGZ = {gz_cm:.2f} cm\nM = {m_str} N·m"
    ax.set_title(title, fontsize=9)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("y (cm)")
    ax.set_ylabel("z (cm)")

    # Auto-range
    all_z = pw[:, 1]
    z_min = min(all_z.min(), wl * 100) - 2
    z_max = max(all_z.max(), mast_top_w[1]) + 2
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(z_min, z_max)


# ===================================================================
# 3D hull surface
# ===================================================================

def plot_3d_hull(params: HullParams, n_x: int = 60, n_y: int = 60) -> plt.Figure:
    """Render a 3D surface plot of the hull bottom."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    x_vals = np.linspace(-params.length / 2, params.length / 2, n_x)

    X, Y, Z = [], [], []
    for xi in x_vals:
        lb = float(beam_at_x(np.array([xi]), params)[0])
        if lb < 1e-12:
            continue
        y_vals = np.linspace(-lb / 2, lb / 2, n_y)
        z_vals = cross_section_z(y_vals, lb, params)
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


# ===================================================================
# Summary figure
# ===================================================================

def plot_summary(params: HullParams,
                 stability: StabilityResult | None = None) -> plt.Figure:
    """Three-panel summary: cross-section, plan view, stability curve."""
    if stability is None:
        stability = compute_stability_curve(params)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    plot_cross_section(params, axes[0])
    plot_hull_top_view(params, axes[1])
    plot_stability_curve(params, axes[2], stability)

    fig.suptitle("Boat Design Summary", fontsize=14)
    fig.tight_layout()
    return fig


# ===================================================================
# STL export
# ===================================================================

def export_stl(params: HullParams, filename: str = "boatmodel.stl",
               n_x: int = 100, n_y: int = 100) -> None:
    """Export the hull solid as an STL file for CNC fabrication."""
    from stl import mesh as stl_mesh

    x_vals = np.linspace(-params.length / 2, params.length / 2, n_x)

    # Build bottom + top grids
    bottom_grid = np.zeros((n_x, n_y, 3))
    top_grid = np.zeros((n_x, n_y, 3))

    for i, xi in enumerate(x_vals):
        lb = float(beam_at_x(np.array([xi]), params)[0])
        if lb < 1e-12:
            lb = 1e-12
        y_vals = np.linspace(-lb / 2, lb / 2, n_y)
        z_bottom = cross_section_z(y_vals, lb, params)
        for j in range(n_y):
            bottom_grid[i, j] = [xi, y_vals[j], z_bottom[j]]
            top_grid[i, j] = [xi, y_vals[j], params.depth]

    faces = []

    # Bottom + top quads
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            faces.append([bottom_grid[i, j], bottom_grid[i+1, j], bottom_grid[i+1, j+1]])
            faces.append([bottom_grid[i, j], bottom_grid[i+1, j+1], bottom_grid[i, j+1]])
            faces.append([top_grid[i, j], top_grid[i+1, j+1], top_grid[i+1, j]])
            faces.append([top_grid[i, j], top_grid[i, j+1], top_grid[i+1, j+1]])

    # Side faces
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

#!/usr/bin/env python3
"""
Boat Optimization — Main entry point.

Usage:
    python main.py analyze        Analyse the default hull design
    python main.py optimize       Run the optimiser, then analyse the result
    python main.py export         Export the default design as STL
    python main.py mathematica    Generate Mathematica notebook code
"""

import sys
import numpy as np

from boat_optimization.hull import HullParams, hull_mass, hull_volume, is_within_foam_block
from boat_optimization.analysis import (
    total_mass, center_of_mass_body, find_waterline_upright,
    compute_stability_curve,
)
from boat_optimization.optimizer import optimize
from boat_optimization.visualization import (
    plot_summary, plot_avs_panels, export_stl, plot_3d_hull,
)
from boat_optimization.mathematica_export import export_mathematica
from boat_optimization import config


# ===================================================================
# Pretty-print summary
# ===================================================================

def print_design_summary(params: HullParams, stability=None) -> None:
    """Print a detailed textual analysis report."""
    if stability is None:
        print("Computing stability curve …")
        stability = compute_stability_curve(params)

    print("=" * 60)
    print("BOAT DESIGN SUMMARY")
    print("=" * 60)

    print(f"\n--- Dimensions ---")
    print(f"  Length:  {params.length * 100:.2f} cm  (max {config.MAX_LENGTH_M * 100:.2f})")
    print(f"  Beam:    {params.beam * 100:.2f} cm  (max {config.MAX_WIDTH_M * 100:.2f})")
    print(f"  Depth:   {params.depth * 100:.2f} cm  (max {config.MAX_HEIGHT_M * 100:.2f})")
    print(f"  Fits in foam block: {is_within_foam_block(params)}")

    print(f"\n--- Hull Shape ---")
    print(f"  Flare exponent (n): {params.flare_exp:.3f}")
    print(f"  Taper exponent (p): {params.taper_exp:.3f}")

    vol = hull_volume(params)
    m_hull = hull_mass(params)
    m_total = total_mass(params)
    print(f"\n--- Mass Budget ---")
    print(f"  Hull volume:  {vol * 1e6:.1f} cm³")
    print(f"  Hull mass:    {m_hull * 1000:.1f} g")
    print(f"  Ballast:      {params.ballast_mass * 1000:.0f} g  at z = {params.ballast_z * 100:.2f} cm")
    print(f"  Mast:         {config.MAST_MASS_KG * 1000:.1f} g")
    print(f"  Total:        {m_total * 1000:.1f} g")

    com = center_of_mass_body(params)
    wl = find_waterline_upright(params)
    print(f"\n--- Equilibrium (upright) ---")
    print(f"  Centre of mass  (z): {com[1] * 100:.2f} cm")
    print(f"  Waterline       (z): {wl * 100:.2f} cm")
    print(f"  Freeboard:           {(params.depth - wl) * 100:.2f} cm")

    print(f"\n--- Stability ---")
    print(f"  AVS:          {stability.avs_deg:.1f}°  (required > {config.MIN_AVS_DEG:.0f}°)")
    print(f"  Meets req:    {stability.is_stable}")
    print(f"  Peak moment:  {stability.righting_moments.max():.4f} N·m")
    print("=" * 60)

    return stability


# ===================================================================
# Commands
# ===================================================================

def cmd_analyze(params: HullParams) -> None:
    print("Running full stability analysis …\n")
    stability = compute_stability_curve(params)
    print_design_summary(params, stability)

    fig1 = plot_summary(params, stability)
    fig1.savefig("design_summary.png", dpi=150, bbox_inches="tight")
    print("Saved design_summary.png")

    fig2 = plot_avs_panels(params, stability=stability)
    fig2.savefig("avs_panels.png", dpi=150, bbox_inches="tight")
    print("Saved avs_panels.png")

    import matplotlib.pyplot as plt
    plt.show()


def cmd_optimize() -> None:
    print("Starting optimisation …\n")
    best = optimize()
    print()
    stability = compute_stability_curve(best)
    print_design_summary(best, stability)

    fig1 = plot_summary(best, stability)
    fig1.savefig("optimized_summary.png", dpi=150, bbox_inches="tight")
    print("Saved optimized_summary.png")

    fig2 = plot_avs_panels(best, stability=stability)
    fig2.savefig("optimized_avs_panels.png", dpi=150, bbox_inches="tight")
    print("Saved optimized_avs_panels.png")

    export_mathematica(best, "optimized_boat.nb")

    import matplotlib.pyplot as plt
    plt.show()


def cmd_export(params: HullParams) -> None:
    export_stl(params, "boatmodel.stl")


def cmd_mathematica(params: HullParams) -> None:
    export_mathematica(params, "boat_design.nb")


# ===================================================================
# Entry point
# ===================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "analyze":
        cmd_analyze(HullParams())
    elif command == "optimize":
        cmd_optimize()
    elif command == "export":
        cmd_export(HullParams())
    elif command == "mathematica":
        cmd_mathematica(HullParams())
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

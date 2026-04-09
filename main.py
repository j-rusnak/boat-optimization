#!/usr/bin/env python3
"""
Boat Optimization — Main entry point.

Usage:
    python main.py analyze       Analyze the default hull design
    python main.py optimize      Run the optimizer
    python main.py export        Export the current design as STL
"""

import sys
import numpy as np

from boat_optimization.hull import HullParams, hull_mass, hull_volume, is_within_foam_block
from boat_optimization.analysis import (
    total_mass, center_of_mass, find_waterline, compute_stability_curve,
)
from boat_optimization.optimizer import optimize
from boat_optimization.visualization import plot_summary, export_stl, plot_3d_hull
from boat_optimization import config


def print_design_summary(params: HullParams) -> None:
    """Print a full analysis summary for a given hull design."""
    print("=" * 60)
    print("BOAT DESIGN SUMMARY")
    print("=" * 60)

    print(f"\n--- Dimensions ---")
    print(f"  Length:  {params.length * 100:.2f} cm  (max {config.MAX_LENGTH_M * 100:.2f})")
    print(f"  Beam:    {params.beam * 100:.2f} cm  (max {config.MAX_WIDTH_M * 100:.2f})")
    print(f"  Depth:   {params.depth * 100:.2f} cm  (max {config.MAX_HEIGHT_M * 100:.2f})")
    print(f"  Fits in foam block: {is_within_foam_block(params)}")

    print(f"\n--- Hull Shape ---")
    print(f"  Flare exponent: {params.flare_exp:.2f}")
    print(f"  Taper exponent: {params.taper_exp:.2f}")

    vol = hull_volume(params)
    m_hull = hull_mass(params)
    print(f"\n--- Mass Budget ---")
    print(f"  Hull volume: {vol * 1e6:.1f} cm³")
    print(f"  Hull mass:   {m_hull * 1000:.1f} g")
    print(f"  Ballast:     {params.ballast_mass * 1000:.0f} g")
    print(f"  Mast:        {config.MAST_MASS_KG * 1000:.1f} g")
    print(f"  Total:       {total_mass(params) * 1000:.1f} g")

    com = center_of_mass(params)
    wl = find_waterline(params)
    print(f"\n--- Equilibrium ---")
    print(f"  Center of mass (z): {com[2] * 100:.2f} cm")
    print(f"  Waterline:          {wl * 100:.2f} cm")
    print(f"  Freeboard:          {(params.depth - wl) * 100:.2f} cm")

    stability = compute_stability_curve(params)
    print(f"\n--- Stability ---")
    print(f"  AVS:      {stability.avs_deg:.1f}°  (required > {config.MIN_AVS_DEG}°)")
    print(f"  Meets requirement: {stability.is_stable}")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "analyze":
        params = HullParams()
        print_design_summary(params)
        fig = plot_summary(params)
        fig.savefig("design_summary.png", dpi=150, bbox_inches="tight")
        print("\nSaved design_summary.png")
        import matplotlib.pyplot as plt
        plt.show()

    elif command == "optimize":
        best = optimize()
        print_design_summary(best)
        fig = plot_summary(best)
        fig.savefig("optimized_summary.png", dpi=150, bbox_inches="tight")
        print("\nSaved optimized_summary.png")
        import matplotlib.pyplot as plt
        plt.show()

    elif command == "export":
        params = HullParams()
        export_stl(params, "boatmodel.stl")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

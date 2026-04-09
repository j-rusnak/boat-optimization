"""
Optimization driver for the boat hull design.

Uses scipy.optimize.differential_evolution to search over hull shape
parameters while enforcing all design constraints.

Objective: maximise AVS and slenderness (for drag), while ensuring
the boat floats upright with positive freeboard.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import differential_evolution
from .hull import HullParams, is_within_foam_block, hull_mass
from .analysis import (
    compute_stability_curve, total_mass,
    find_waterline_upright, center_of_mass_body,
)
from . import config


@dataclass
class OptimizationOutcome:
    """Result of one optimization run."""

    params: HullParams
    objective_value: float
    iterations: int
    evaluations: int
    success: bool
    message: str


# ===================================================================
# Parameter vector <-> HullParams
# ===================================================================

PARAM_NAMES = [
    "length", "beam", "depth", "flare_exp", "taper_exp",
    "ballast_mass", "ballast_z",
]


def params_to_vector(p: HullParams) -> np.ndarray:
    return np.array([
        p.length, p.beam, p.depth, p.flare_exp, p.taper_exp,
        p.ballast_mass, p.ballast_z,
    ])


def vector_to_params(x: np.ndarray) -> HullParams:
    return HullParams(
        length=x[0], beam=x[1], depth=x[2],
        flare_exp=x[3], taper_exp=x[4],
        ballast_mass=x[5], ballast_z=x[6],
    )


# Parameter bounds
BOUNDS = [
    (0.15, config.MAX_LENGTH_M),                                # length
    (0.06, config.MAX_WIDTH_M),                                 # beam
    (0.04, config.MAX_HEIGHT_M),                                # depth
    (1.2, 5.0),                                                 # flare_exp
    (1.2, 5.0),                                                 # taper_exp
    (config.BALLAST_MASS_MIN_KG, config.BALLAST_MASS_MAX_KG),  # ballast_mass
    (0.005, 0.04),                                              # ballast_z
]


# ===================================================================
# Objective
# ===================================================================

def objective(x: np.ndarray) -> float:
    """Cost function (lower = better).

    Terms:
        1. Hard penalty if hull exceeds foam block.
        2. Hard penalty if boat cannot float (waterline >= depth).
        3. Large penalty if AVS < 100 deg.
        4. Reward higher AVS (most important).
        5. Reward slenderness (length / beam) for lower drag.
        6. Reward low COM (more stability margin).
    """
    params = vector_to_params(x)

    if not is_within_foam_block(params):
        return 1e6

    # Quick float check
    wl = find_waterline_upright(params, n_x=80)
    if wl >= params.depth - 1e-4:
        return 5e5

    # Stability (use coarser grid for speed during optimisation)
    stab = compute_stability_curve(
        params,
        heel_angles_deg=np.arange(0, 181, 10, dtype=float),
        n_x=25, n_poly=40,
    )

    # AVS penalty / reward
    avs = stab.avs_deg
    if avs < config.MIN_AVS_DEG:
        avs_cost = (config.MIN_AVS_DEG - avs) ** 2 * 50.0
    else:
        avs_cost = 0.0
    avs_reward = -avs * 3.0  # maximise AVS

    # Slenderness reward (length-to-beam ratio → lower drag)
    slenderness = -params.length / max(params.beam, 0.01) * 5.0

    # Low COM reward
    com = center_of_mass_body(params)
    com_reward = com[1] * 200.0  # penalise high COM

    # Freeboard check
    freeboard = params.depth - wl
    if freeboard < 0.005:
        freeboard_penalty = 1e4
    else:
        freeboard_penalty = 0.0

    return avs_cost + avs_reward + slenderness + com_reward + freeboard_penalty


# ===================================================================
# Runner
# ===================================================================

def optimize(seed: int = 42, maxiter: int = 60, popsize: int = 20,
             verbose: bool = True) -> OptimizationOutcome:
    """Run differential evolution to find optimal hull parameters."""
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-5,
        disp=verbose,
        workers=1,
    )

    best = vector_to_params(result.x)
    outcome = OptimizationOutcome(
        params=best,
        objective_value=float(result.fun),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )

    if verbose:
        print(f"\nOptimisation finished — cost = {result.fun:.4f}")
        print(f"  Length:   {best.length * 100:.2f} cm")
        print(f"  Beam:     {best.beam * 100:.2f} cm")
        print(f"  Depth:    {best.depth * 100:.2f} cm")
        print(f"  Flare n:  {best.flare_exp:.3f}")
        print(f"  Taper p:  {best.taper_exp:.3f}")
        print(f"  Ballast:  {best.ballast_mass * 1000:.0f} g  "
              f"at z = {best.ballast_z * 100:.2f} cm")
        print(f"  Iterations: {outcome.iterations}")
        print(f"  Evaluations: {outcome.evaluations}")

    return outcome

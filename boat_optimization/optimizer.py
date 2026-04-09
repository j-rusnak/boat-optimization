"""
Optimization driver for the boat hull design.

Uses scipy.optimize to search over hull shape parameters while enforcing
all design constraints (foam block size, ballast range, AVS > 100°, etc.).
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from .hull import HullParams, is_within_foam_block, hull_mass
from .analysis import compute_stability_curve, total_mass, find_waterline, center_of_mass
from . import config


def params_to_vector(p: HullParams) -> np.ndarray:
    """Pack tunable hull parameters into a flat array for the optimizer."""
    return np.array([
        p.length,
        p.beam,
        p.depth,
        p.flare_exp,
        p.taper_exp,
        p.ballast_mass,
        p.ballast_z,
    ])


def vector_to_params(x: np.ndarray) -> HullParams:
    """Unpack optimizer vector back into HullParams."""
    return HullParams(
        length=x[0],
        beam=x[1],
        depth=x[2],
        flare_exp=x[3],
        taper_exp=x[4],
        ballast_mass=x[5],
        ballast_z=x[6],
    )


# Parameter bounds: (min, max) for each element in the vector
BOUNDS = [
    (0.10, config.MAX_LENGTH_M),       # length
    (0.05, config.MAX_WIDTH_M),        # beam
    (0.03, config.MAX_HEIGHT_M),       # depth
    (1.0, 5.0),                        # flare_exp
    (1.0, 5.0),                        # taper_exp
    (config.BALLAST_MASS_MIN_KG, config.BALLAST_MASS_MAX_KG),  # ballast_mass
    (0.005, 0.05),                     # ballast_z
]


def objective(x: np.ndarray) -> float:
    """
    Objective function to minimize. Lower is better.

    Goals (weighted):
    - Maximize AVS (penalize if below 100°)
    - Minimize drag (reward slender hulls)
    - Ensure the boat floats upright

    Returns a scalar cost.
    """
    params = vector_to_params(x)

    # Hard constraint: must fit in foam block
    if not is_within_foam_block(params):
        return 1e6

    # Compute stability
    stability = compute_stability_curve(params, np.arange(0, 181, 10, dtype=float))

    # Penalty: AVS below requirement
    avs_penalty = max(0, config.MIN_AVS_DEG - stability.avs_deg) ** 2 * 100.0

    # Reward: higher AVS is better
    avs_reward = -stability.avs_deg

    # Reward: slenderness ratio for lower drag (length / beam)
    slenderness_reward = -params.length / max(params.beam, 0.01)

    # Penalty: boat doesn't float (waterline above hull depth)
    waterline = find_waterline(params)
    if waterline >= params.depth:
        float_penalty = 1e5
    else:
        float_penalty = 0.0

    # Penalty: center of mass above waterline (less stable)
    com = center_of_mass(params)
    com_penalty = max(0, com[2] - waterline) * 500.0

    cost = (
        avs_penalty
        + avs_reward * 2.0
        + slenderness_reward * 10.0
        + float_penalty
        + com_penalty
    )
    return cost


def optimize(seed: int = 42, maxiter: int = 50, popsize: int = 15) -> HullParams:
    """
    Run differential evolution to find optimal hull parameters.

    Args:
        seed: random seed for reproducibility
        maxiter: maximum optimizer iterations
        popsize: population size multiplier

    Returns:
        Optimized HullParams
    """
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        disp=True,
    )

    best_params = vector_to_params(result.x)
    print(f"\nOptimization finished: cost = {result.fun:.4f}")
    print(f"  Length: {best_params.length * 100:.1f} cm")
    print(f"  Beam:   {best_params.beam * 100:.1f} cm")
    print(f"  Depth:  {best_params.depth * 100:.1f} cm")
    print(f"  Flare:  {best_params.flare_exp:.2f}")
    print(f"  Taper:  {best_params.taper_exp:.2f}")
    print(f"  Ballast: {best_params.ballast_mass * 1000:.0f} g at z={best_params.ballast_z * 100:.1f} cm")

    return best_params

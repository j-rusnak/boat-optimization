"""
Hull geometry definition.

Defines the boat hull shape via a 2D cross-section curve that is extruded along
the length of the boat.  Parameters controlling the shape can be varied by the
optimizer.

Coordinate system (body frame, attached to hull):
    x – longitudinal, 0 at midship, positive toward bow
    y – transverse, 0 at centerline, positive to port
    z – vertical, 0 at the lowest point of the hull bottom, positive up

Cross-section equation (at a given x-station with local beam b):
    z_bottom(y) = D · |2y / b|^n        for |y| ≤ b/2
    The deck is at z = D everywhere.

Taper function:
    b(x) = B · (1 - |2x / L|^p)

Where:
    L = length, B = beam (max width), D = depth (hull height)
    n = flare_exp (shape of cross-section)
    p = taper_exp (how fast the bow/stern narrow)
"""

import numpy as np
from dataclasses import dataclass
from . import config


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class HullParams:
    """Parameters that define the hull shape.  These are the optimization variables."""

    # Overall dimensions (must stay within foam block limits)
    length: float = config.MAX_LENGTH_M       # L, boat length (x-axis), metres
    beam: float = config.MAX_WIDTH_M          # B, max width  (y-axis), metres
    depth: float = config.MAX_HEIGHT_M * 0.8  # D, hull depth  (z-axis), metres

    # Cross-section shape controls
    flare_exp: float = 2.0   # n: 1 = V-hull, 2 = parabolic, >2 = flatter bottom
    taper_exp: float = 2.0   # p: how quickly the beam narrows at bow/stern

    # Ballast placement
    ballast_mass: float = 0.850  # kg  (range 0.7 – 1.0)
    ballast_z: float = 0.01     # height of ballast centre above hull bottom, m
    ballast_x: float = 0.0      # longitudinal offset from midship, m

    # Mast longitudinal placement (x-position, 0 = midship)
    mast_x: float = 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def cross_section_z(y: np.ndarray, local_beam: float, params: HullParams) -> np.ndarray:
    """Hull bottom height z as a function of transverse position y at a station
    with the given *local_beam*.  Uses the same flare exponent everywhere."""
    half_b = local_beam / 2.0
    if half_b < 1e-12:
        return np.full_like(y, params.depth)
    normalized = np.clip(np.abs(y) / half_b, 0.0, 1.0)
    return params.depth * normalized ** params.flare_exp


def beam_at_x(x: np.ndarray, params: HullParams) -> np.ndarray:
    """Local beam (width) as a function of longitudinal position x."""
    half_L = params.length / 2.0
    normalized = np.clip(np.abs(x) / half_L, 0.0, 1.0)
    return params.beam * (1.0 - normalized ** params.taper_exp)


def build_cross_section_polygon(local_beam: float, params: HullParams,
                                n_pts: int = 80) -> np.ndarray:
    """Return a closed polygon (N×2 array of [y, z] vertices) for the hull
    cross-section at a station with the given *local_beam*.

    The polygon goes: bottom curve left-to-right, then top deck right-to-left.
    """
    if local_beam < 1e-12:
        return np.empty((0, 2))
    half_b = local_beam / 2.0
    y_bottom = np.linspace(-half_b, half_b, n_pts)
    z_bottom = cross_section_z(y_bottom, local_beam, params)
    # Deck corners (right-to-left to close polygon)
    deck = np.array([[half_b, params.depth], [-half_b, params.depth]])
    return np.vstack([np.column_stack([y_bottom, z_bottom]), deck])


# ---------------------------------------------------------------------------
# Volume / mass (upright, used for quick checks)
# ---------------------------------------------------------------------------

def cross_section_area(local_beam: float, params: HullParams) -> float:
    """Analytical area of one cross-section slice (hull solid)."""
    # Area = integral_{-b/2}^{b/2} [D - D|2y/b|^n] dy
    # = D * b * (1 - 1/(n+1))  = D * b * n/(n+1)
    return params.depth * local_beam * params.flare_exp / (params.flare_exp + 1.0)


def hull_volume(params: HullParams, n_x: int = 300) -> float:
    """Numerically integrate the hull solid volume over x."""
    x = np.linspace(-params.length / 2, params.length / 2, n_x)
    beams = beam_at_x(x, params)
    areas = np.array([cross_section_area(b, params) for b in beams])
    return float(np.trapezoid(areas, x))


def hull_mass(params: HullParams) -> float:
    """Mass of the foam hull in kg."""
    return hull_volume(params) * config.FOAM_DENSITY_KG_M3


def is_within_foam_block(params: HullParams) -> bool:
    """Check that the hull fits within the foam block constraints."""
    return (
        params.length <= config.MAX_LENGTH_M + 1e-9
        and params.beam <= config.MAX_WIDTH_M + 1e-9
        and params.depth <= config.MAX_HEIGHT_M + 1e-9
    )

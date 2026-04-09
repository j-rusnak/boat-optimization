"""
Hull geometry definition.

Defines the boat hull shape via a 2D cross-section curve that is extruded along
the length of the boat. Parameters controlling the shape can be varied by the
optimizer.
"""

import numpy as np
from dataclasses import dataclass, field
from . import config


@dataclass
class HullParams:
    """Parameters that define the hull shape. These are the optimization variables."""

    # Overall dimensions (must stay within foam block limits)
    length: float = config.MAX_LENGTH_M       # boat length (x-axis), meters
    beam: float = config.MAX_WIDTH_M          # max width (y-axis), meters
    depth: float = config.MAX_HEIGHT_M * 0.8  # hull depth (z-axis), meters

    # Cross-section shape controls
    # The cross-section is a power-law curve: z(y) = depth * |2y/beam|^flare_exp
    flare_exp: float = 2.0  # 1 = V-hull, 2 = parabolic, >2 = flatter bottom

    # Bow/stern taper exponent (how quickly the beam narrows at the ends)
    taper_exp: float = 2.0

    # Ballast placement
    ballast_mass: float = 0.850  # kg (within 0.7–1.0 range)
    ballast_z: float = 0.01     # height of ballast center above hull bottom, m
    ballast_x: float = 0.0      # longitudinal offset from center, m (0 = centered)

    # Mast placement (x-position along length, 0 = center)
    mast_x: float = 0.0


def cross_section_curve(y: np.ndarray, params: HullParams) -> np.ndarray:
    """
    Compute the hull bottom height z as a function of transverse position y.
    z = 0 at the lowest point (keel line), increasing upward.

    Args:
        y: transverse coordinates (centered at 0)
        params: hull shape parameters

    Returns:
        z values for the bottom surface of the hull
    """
    half_beam = params.beam / 2.0
    normalized = np.clip(np.abs(y) / half_beam, 0, 1)
    return params.depth * normalized ** params.flare_exp


def beam_at_x(x: np.ndarray, params: HullParams) -> np.ndarray:
    """
    Compute the local beam (width) as a function of longitudinal position x.
    The hull tapers toward the bow and stern.

    Args:
        x: longitudinal coordinates (centered at 0, ranges -L/2 to L/2)
        params: hull shape parameters

    Returns:
        Local beam width at each x position
    """
    half_length = params.length / 2.0
    normalized = np.clip(np.abs(x) / half_length, 0, 1)
    taper = 1.0 - normalized ** params.taper_exp
    return params.beam * taper


def hull_volume(params: HullParams, n_x: int = 200, n_y: int = 200) -> float:
    """
    Numerically compute the volume of the hull solid (the foam region).

    Args:
        params: hull shape parameters
        n_x: integration points along length
        n_y: integration points along beam

    Returns:
        Volume in m³
    """
    x = np.linspace(-params.length / 2, params.length / 2, n_x)
    dx = x[1] - x[0]
    total = 0.0

    for xi in x:
        local_beam = float(beam_at_x(np.array([xi]), params)[0])
        y = np.linspace(-local_beam / 2, local_beam / 2, n_y)
        # Scale y to full beam for cross-section computation
        z_bottom = cross_section_curve(y * (params.beam / max(local_beam, 1e-12)), params)
        z_top = params.depth
        integrand = np.maximum(z_top - z_bottom, 0)
        slice_area = np.trapz(integrand, y)
        total += slice_area * dx

    return total


def hull_mass(params: HullParams) -> float:
    """Compute mass of the foam hull."""
    return hull_volume(params) * config.FOAM_DENSITY_KG_M3


def is_within_foam_block(params: HullParams) -> bool:
    """Check that the hull fits within the foam block constraints."""
    return (
        params.length <= config.MAX_LENGTH_M + 1e-9
        and params.beam <= config.MAX_WIDTH_M + 1e-9
        and params.depth <= config.MAX_HEIGHT_M + 1e-9
    )

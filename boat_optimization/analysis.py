"""
Physics and engineering analysis for the boat.

Computes center of mass, center of buoyancy, waterline, righting moment,
and angle of vanishing stability (AVS).
"""

import numpy as np
from dataclasses import dataclass
from . import config
from .hull import HullParams, hull_volume, hull_mass, cross_section_curve, beam_at_x


@dataclass
class AnalysisResult:
    """Container for analysis outputs at a given heel angle."""
    heel_angle_deg: float
    center_of_mass: np.ndarray       # [x, y, z] in world frame
    center_of_buoyancy: np.ndarray   # [x, y, z] in world frame
    righting_arm: float              # GZ, meters
    righting_moment: float           # N·m
    displaced_volume: float          # m³


@dataclass
class StabilityResult:
    """Full stability curve results."""
    heel_angles_deg: np.ndarray
    righting_moments: np.ndarray
    avs_deg: float                   # angle of vanishing stability
    is_stable: bool                  # True if AVS > required minimum


def total_mass(params: HullParams) -> float:
    """Total mass of the loaded boat: hull + ballast + mast."""
    return hull_mass(params) + params.ballast_mass + config.MAST_MASS_KG


def center_of_mass(params: HullParams) -> np.ndarray:
    """
    Compute the center of mass of the fully loaded boat (hull + ballast + mast).
    Coordinate system: x = longitudinal (0 = center), y = transverse (0 = center),
    z = vertical (0 = hull bottom).

    Returns:
        np.ndarray [x, y, z] of center of mass
    """
    m_hull = hull_mass(params)
    m_ballast = params.ballast_mass
    m_mast = config.MAST_MASS_KG
    m_total = m_hull + m_ballast + m_mast

    # Hull COM (approximate as centroid of hull solid — symmetric in x and y)
    hull_com = np.array([0.0, 0.0, _hull_centroid_z(params)])

    # Ballast COM
    ballast_com = np.array([params.ballast_x, 0.0, params.ballast_z])

    # Mast COM: extends from hull bottom (z=0) upward for full mast length
    mast_com = np.array([params.mast_x, 0.0, config.MAST_LENGTH_M / 2.0])

    com = (m_hull * hull_com + m_ballast * ballast_com + m_mast * mast_com) / m_total
    return com


def find_waterline(params: HullParams, n_x: int = 100, n_y: int = 100) -> float:
    """
    Find the equilibrium waterline height (draft) by matching displaced water
    weight to total boat weight.

    Returns:
        Waterline z-coordinate (measured from hull bottom)
    """
    m_total = total_mass(params)
    target_volume = m_total / config.WATER_DENSITY_KG_M3

    # Binary search for waterline z
    z_low, z_high = 0.0, params.depth
    for _ in range(60):
        z_mid = (z_low + z_high) / 2.0
        vol = _submerged_volume(params, z_mid, n_x, n_y)
        if vol < target_volume:
            z_low = z_mid
        else:
            z_high = z_mid

    return (z_low + z_high) / 2.0


def compute_stability_curve(
    params: HullParams,
    heel_angles_deg: np.ndarray | None = None,
) -> StabilityResult:
    """
    Compute the righting moment as a function of heel angle.

    Args:
        params: hull parameters
        heel_angles_deg: array of heel angles to evaluate (default: 0–180° in 5° steps)

    Returns:
        StabilityResult with righting moments and AVS
    """
    if heel_angles_deg is None:
        heel_angles_deg = np.arange(0, 181, 5, dtype=float)

    moments = np.zeros_like(heel_angles_deg)

    for i, angle in enumerate(heel_angles_deg):
        result = _analyze_at_heel(params, angle)
        moments[i] = result.righting_moment

    # Find AVS: first angle > 0 where righting moment crosses zero from positive to negative
    avs = _find_avs(heel_angles_deg, moments)

    return StabilityResult(
        heel_angles_deg=heel_angles_deg,
        righting_moments=moments,
        avs_deg=avs,
        is_stable=avs > config.MIN_AVS_DEG,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hull_centroid_z(params: HullParams, n_x: int = 100, n_y: int = 100) -> float:
    """Compute the z-centroid of the hull solid."""
    x = np.linspace(-params.length / 2, params.length / 2, n_x)
    dx = x[1] - x[0]

    numerator = 0.0
    denominator = 0.0

    for xi in x:
        local_b = float(beam_at_x(np.array([xi]), params)[0])
        y = np.linspace(-local_b / 2, local_b / 2, n_y)
        z_bottom = cross_section_curve(y * (params.beam / max(local_b, 1e-12)), params)
        z_top = params.depth
        height = np.maximum(z_top - z_bottom, 0)
        z_mid = (z_bottom + z_top) / 2.0

        area = np.trapz(height, y)
        moment = np.trapz(height * z_mid, y)

        denominator += area * dx
        numerator += moment * dx

    return numerator / max(denominator, 1e-15)


def _submerged_volume(
    params: HullParams, waterline_z: float, n_x: int = 100, n_y: int = 100
) -> float:
    """Compute the volume of the hull below waterline_z."""
    x = np.linspace(-params.length / 2, params.length / 2, n_x)
    dx = x[1] - x[0]
    total = 0.0

    for xi in x:
        local_b = float(beam_at_x(np.array([xi]), params)[0])
        y = np.linspace(-local_b / 2, local_b / 2, n_y)
        z_bottom = cross_section_curve(y * (params.beam / max(local_b, 1e-12)), params)
        submerged_height = np.maximum(np.minimum(waterline_z, params.depth) - z_bottom, 0)
        total += np.trapz(submerged_height, y) * dx

    return total


def _analyze_at_heel(params: HullParams, heel_deg: float) -> AnalysisResult:
    """
    Analyze the boat at a given heel angle.
    This is a placeholder that will need a proper rotated-waterplane integration.
    For now, implements a simplified righting-arm calculation.
    """
    # TODO: Implement full rotated cross-section integration
    # For now, use a simplified metacentric approach for small angles
    # and extend to large angles with proper geometry

    m_total = total_mass(params)
    weight = m_total * config.G
    com = center_of_mass(params)

    heel_rad = np.radians(heel_deg)

    # Simplified: righting arm ≈ (BM - BG) * sin(heel) for small angles
    # This needs to be replaced with proper large-angle stability computation
    waterline = find_waterline(params)
    cob_z = _centroid_of_submerged(params, waterline)

    bg = com[2] - cob_z  # vertical distance between G and B
    bm = _metacentric_radius(params, waterline)  # transverse metacentric radius
    gm = bm - bg  # metacentric height

    # Simplified righting arm (valid for moderate angles)
    gz = gm * np.sin(heel_rad)
    righting_moment = weight * gz

    return AnalysisResult(
        heel_angle_deg=heel_deg,
        center_of_mass=com,
        center_of_buoyancy=np.array([0.0, 0.0, cob_z]),
        righting_arm=gz,
        righting_moment=righting_moment,
        displaced_volume=m_total / config.WATER_DENSITY_KG_M3,
    )


def _centroid_of_submerged(params: HullParams, waterline_z: float, n_y: int = 100) -> float:
    """Z-centroid of the submerged cross-section at midship."""
    y = np.linspace(-params.beam / 2, params.beam / 2, n_y)
    z_bottom = cross_section_curve(y, params)
    sub_h = np.maximum(waterline_z - z_bottom, 0)
    z_mid = z_bottom + sub_h / 2.0

    area = np.trapz(sub_h, y)
    if area < 1e-15:
        return 0.0
    return float(np.trapz(sub_h * z_mid, y) / area)


def _metacentric_radius(params: HullParams, waterline_z: float, n_x: int = 100, n_y: int = 100) -> float:
    """
    Compute the transverse metacentric radius BM = I_waterplane / V_displaced.
    I_waterplane = second moment of area of the waterplane about the centerline.
    """
    x = np.linspace(-params.length / 2, params.length / 2, n_x)
    dx = x[1] - x[0]

    i_wp = 0.0
    v_sub = 0.0

    for xi in x:
        local_b = float(beam_at_x(np.array([xi]), params)[0])
        # Find the width at the waterline for this cross section
        # For a power-law hull, solve: depth * |2y/beam|^exp = waterline_z
        # -> half_width_at_wl = (beam/2) * (waterline_z/depth)^(1/exp)
        if waterline_z > 0 and params.depth > 0:
            ratio = min(waterline_z / params.depth, 1.0)
            hw = (local_b / 2) * ratio ** (1.0 / params.flare_exp)
        else:
            hw = 0.0

        # Second moment of waterplane strip: (1/12) * dx * (2*hw)^3 ... but integrated as strip
        i_wp += (2 * hw) ** 3 / 12.0 * dx

        # Submerged volume strip
        y = np.linspace(-local_b / 2, local_b / 2, n_y)
        z_bottom = cross_section_curve(y * (params.beam / max(local_b, 1e-12)), params)
        sub_h = np.maximum(waterline_z - z_bottom, 0)
        v_sub += np.trapz(sub_h, y) * dx

    if v_sub < 1e-15:
        return 0.0
    return i_wp / v_sub


def _find_avs(angles: np.ndarray, moments: np.ndarray) -> float:
    """Find the angle of vanishing stability (first zero-crossing after peak)."""
    # Skip angle 0
    for i in range(1, len(moments) - 1):
        if moments[i] > 0 and moments[i + 1] <= 0:
            # Linear interpolation
            frac = moments[i] / max(moments[i] - moments[i + 1], 1e-15)
            return float(angles[i] + frac * (angles[i + 1] - angles[i]))
    # If no crossing found, return 180 (or 0 if never positive)
    if np.any(moments > 0):
        return 180.0
    return 0.0

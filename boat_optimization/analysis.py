"""
Physics and engineering analysis for the boat.

Implements proper large-angle stability analysis by:
1.  Building the hull cross-section polygon in body coordinates
2.  Rotating the polygon by the heel angle
3.  Clipping against a horizontal waterline (Sutherland-Hodgman)
4.  Finding the equilibrium waterline via binary search on displaced volume
5.  Computing centre of buoyancy via polygon centroid of the submerged region
6.  Computing righting arm GZ = y_B_world - y_G_world
7.  Righting moment = W * GZ

Coordinate frames
-----------------
Body frame  (y_b, z_b): fixed to the hull.  y_b transverse (+ port), z_b up.
World frame (y_w, z_w): z_w always points up, y_w always points to port.

Rotation (heel angle theta, positive = starboard heel, i.e. starboard side dips):
    y_w =  y_b cos(theta) + z_b sin(theta)
    z_w = -y_b sin(theta) + z_b cos(theta)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from . import config
from .hull import (
    HullParams, hull_mass, hull_volume,
    beam_at_x, build_cross_section_polygon, cross_section_area,
)


# ===================================================================
# Data containers
# ===================================================================

@dataclass
class AnalysisResult:
    """Analysis outputs at a single heel angle."""
    heel_angle_deg: float
    com_world: np.ndarray        # [y, z] in world frame
    cob_world: np.ndarray        # [y, z] in world frame
    righting_arm_m: float        # GZ (metres, + = restoring)
    righting_moment_Nm: float    # W * GZ (N*m)
    waterline_world_z: float     # equilibrium waterline in world z
    displaced_volume_m3: float


@dataclass
class StabilityResult:
    """Full stability curve."""
    heel_angles_deg: np.ndarray
    righting_arms: np.ndarray        # GZ at each angle
    righting_moments: np.ndarray     # W*GZ at each angle
    avs_deg: float                   # angle of vanishing stability
    is_stable: bool                  # AVS > requirement
    results: list                    # list[AnalysisResult], per-angle details


# ===================================================================
# Public API
# ===================================================================

def total_mass(params: HullParams) -> float:
    """Total mass: hull + ballast + mast (kg)."""
    return hull_mass(params) + params.ballast_mass + config.MAST_MASS_KG


def center_of_mass_body(params: HullParams) -> np.ndarray:
    """Centre of mass in the body frame [y_b, z_b].

    Symmetric hull => y_COM = 0 (ballast and mast on centreline).
    z_COM is the mass-weighted average of hull centroid, ballast, and mast.

    Hull z-centroid (analytical for power-law cross-section):
        Area of cross-section = D * b * n / (n + 1)
        integral of z dA      = D^2 * b * n / (2n + 1)
        z_bar                 = D * (n + 1) / (2n + 1)
    """
    m_hull = hull_mass(params)
    m_ball = params.ballast_mass
    m_mast = config.MAST_MASS_KG
    m_total = m_hull + m_ball + m_mast

    n = params.flare_exp
    z_hull = params.depth * (n + 1.0) / (2.0 * n + 1.0)
    z_ball = params.ballast_z
    z_mast = config.MAST_LENGTH_M / 2.0

    z_com = (m_hull * z_hull + m_ball * z_ball + m_mast * z_mast) / m_total
    return np.array([0.0, z_com])


def find_waterline_upright(params: HullParams, n_x: int = 200) -> float:
    """Equilibrium waterline height z (upright, body frame) via binary search.

    Uses a fully vectorised analytical formula (no polygon clipping) so
    this is very fast.
    """
    m_total = total_mass(params)
    target_vol = m_total / config.WATER_DENSITY_KG_M3

    x = np.linspace(-params.length / 2, params.length / 2, n_x)
    beams_arr = beam_at_x(x, params)          # (n_x,)
    n = params.flare_exp
    full_areas = beams_arr * params.depth * n / (n + 1.0)  # fully-submerged areas
    inv_n = 1.0 / n
    coeff = beams_arr * params.flare_exp / (params.flare_exp + 1.0) / (params.depth ** inv_n)

    def submerged_vol(h: float) -> float:
        if h <= 0:
            return 0.0
        if h >= params.depth:
            return float(np.trapezoid(full_areas, x))
        # area_i = 2 * y_max_i * h * n/(n+1)  with  y_max_i = (b_i/2)(h/D)^(1/n)
        areas = coeff * h ** (1.0 + inv_n)
        return float(np.trapezoid(areas, x))

    z_lo, z_hi = 0.0, params.depth
    for _ in range(40):
        z_mid = (z_lo + z_hi) / 2.0
        if submerged_vol(z_mid) < target_vol:
            z_lo = z_mid
        else:
            z_hi = z_mid
    return (z_lo + z_hi) / 2.0


def compute_stability_curve(
    params: HullParams,
    heel_angles_deg: np.ndarray | None = None,
    n_x: int = 50,
    n_poly: int = 80,
) -> StabilityResult:
    """Compute the righting-moment curve over a range of heel angles.

    Uses proper large-angle analysis: rotate hull cross-sections, clip against
    the waterline, find equilibrium, compute true COB.

    Parameters
    ----------
    params : HullParams
    heel_angles_deg : array of angles (default 0-180 in 5-deg steps)
    n_x : number of longitudinal stations for integration
    n_poly : points on the cross-section bottom curve
    """
    if heel_angles_deg is None:
        heel_angles_deg = np.arange(0, 181, 5, dtype=float)

    m_total = total_mass(params)
    weight = m_total * config.G
    target_vol = m_total / config.WATER_DENSITY_KG_M3
    com_body = center_of_mass_body(params)

    # Pre-build cross-section polygons for each x-station
    x_stations = np.linspace(-params.length / 2, params.length / 2, n_x)
    dx = x_stations[1] - x_stations[0] if n_x > 1 else params.length
    beams_arr = beam_at_x(x_stations, params)
    body_polys = []
    for b in beams_arr:
        body_polys.append(build_cross_section_polygon(b, params, n_poly))

    results = []
    gz_arr = np.zeros(len(heel_angles_deg))
    mom_arr = np.zeros(len(heel_angles_deg))

    for idx, angle_deg in enumerate(heel_angles_deg):
        res = _analyze_at_heel(
            params, angle_deg, m_total, weight, target_vol,
            com_body, x_stations, dx, beams_arr, body_polys,
        )
        results.append(res)
        gz_arr[idx] = res.righting_arm_m
        mom_arr[idx] = res.righting_moment_Nm

    avs = _find_avs(heel_angles_deg, mom_arr)

    return StabilityResult(
        heel_angles_deg=heel_angles_deg,
        righting_arms=gz_arr,
        righting_moments=mom_arr,
        avs_deg=avs,
        is_stable=avs > config.MIN_AVS_DEG,
        results=results,
    )


# ===================================================================
# Internal: per-angle analysis
# ===================================================================

def _analyze_at_heel(
    params, heel_deg, m_total, weight, target_vol,
    com_body, x_stations, dx, beams_arr, body_polys,
):
    """Full large-angle stability analysis at one heel angle."""
    theta = np.radians(heel_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotate COM to world frame
    com_w = _rotate_point(com_body, cos_t, sin_t)

    # Rotate all cross-section polygons to world frame
    rot_polys = []
    for poly in body_polys:
        if len(poly) == 0:
            rot_polys.append(poly)
        else:
            rot_polys.append(_rotate_polygon(poly, cos_t, sin_t))

    # Determine search bounds for waterline
    z_all_min = 1e30
    z_all_max = -1e30
    for rp in rot_polys:
        if len(rp) == 0:
            continue
        z_all_min = min(z_all_min, float(rp[:, 1].min()))
        z_all_max = max(z_all_max, float(rp[:, 1].max()))

    if z_all_min >= z_all_max:
        return AnalysisResult(heel_deg, com_w, np.array([0.0, 0.0]), 0.0, 0.0, 0.0, 0.0)

    # Check if the boat can float (max submerged vol >= target)
    max_vol = _total_submerged_volume(rot_polys, dx, z_all_max + 0.001)
    if max_vol < target_vol * 0.99:
        return AnalysisResult(heel_deg, com_w, np.array([0.0, 0.0]), 0.0, 0.0, z_all_max, max_vol)

    # Binary search for equilibrium waterline
    z_lo, z_hi = z_all_min, z_all_max
    for _ in range(30):
        h = (z_lo + z_hi) / 2.0
        vol = _total_submerged_volume(rot_polys, dx, h)
        if vol < target_vol:
            z_lo = h
        else:
            z_hi = h
    waterline = (z_lo + z_hi) / 2.0

    # Centre of buoyancy in world frame
    cob_w = _total_cob(rot_polys, dx, waterline)

    # Righting arm: positive GZ means restoring
    #   For starboard heel (theta > 0), G swings to port (+y).
    #   If B is further to port than G, there is a restoring moment.
    gz = cob_w[0] - com_w[0]
    righting_moment = weight * gz

    return AnalysisResult(
        heel_angle_deg=heel_deg,
        com_world=com_w,
        cob_world=cob_w,
        righting_arm_m=gz,
        righting_moment_Nm=righting_moment,
        waterline_world_z=waterline,
        displaced_volume_m3=target_vol,
    )


# ===================================================================
# Polygon geometry helpers
# ===================================================================

def _rotate_point(pt, cos_t, sin_t):
    """Rotate a [y, z] point from body to world frame."""
    y, z = pt
    return np.array([y * cos_t + z * sin_t,
                     -y * sin_t + z * cos_t])


def _rotate_polygon(poly, cos_t, sin_t):
    """Rotate an N*2 polygon from body to world frame."""
    y = poly[:, 0]
    z = poly[:, 1]
    return np.column_stack([y * cos_t + z * sin_t,
                            -y * sin_t + z * cos_t])


def _clip_polygon_below_z(poly, z_max):
    """Clip a polygon to the half-plane z <= z_max (Sutherland-Hodgman).

    Returns the clipped vertices (np.ndarray), or None if empty.
    """
    n = len(poly)
    if n < 3:
        return None

    output = []
    for i in range(n):
        curr = poly[i]
        nxt = poly[(i + 1) % n]
        c_in = curr[1] <= z_max
        n_in = nxt[1] <= z_max

        if c_in and n_in:
            output.append(nxt)
        elif c_in and not n_in:
            output.append(_intersect_z(curr, nxt, z_max))
        elif not c_in and n_in:
            output.append(_intersect_z(curr, nxt, z_max))
            output.append(nxt)
        # both outside: skip

    if len(output) < 3:
        return None
    return np.array(output)


def _intersect_z(a, b, z_val):
    """Intersection of segment a->b with horizontal line z = z_val."""
    dz = b[1] - a[1]
    if abs(dz) < 1e-15:
        return (a + b) / 2.0
    t = (z_val - a[1]) / dz
    t = np.clip(t, 0.0, 1.0)
    return a + t * (b - a)


def _polygon_area_and_centroid(poly):
    """Signed area and centroid via the shoelace formula.

    Returns (|area|, y_centroid, z_centroid).
    """
    n = len(poly)
    if n < 3:
        return 0.0, 0.0, 0.0

    y = poly[:, 0]
    z = poly[:, 1]
    y_next = np.roll(y, -1)
    z_next = np.roll(z, -1)

    cross = y * z_next - y_next * z
    area_2 = cross.sum()
    area = area_2 / 2.0

    if abs(area) < 1e-20:
        return 0.0, 0.0, 0.0

    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    cz = ((z + z_next) * cross).sum() / (6.0 * area)
    return abs(area), cy, cz


# ===================================================================
# Volume / COB integration over x-stations
# ===================================================================

def _clip_area_fast(poly, z_max):
    """Compute the area of a polygon clipped below z=z_max without allocating
    a full clipped-polygon array.  Inlines Sutherland-Hodgman + shoelace."""
    n = len(poly)
    if n < 3:
        return 0.0

    # Sutherland-Hodgman clipping, building output list
    out_y = []
    out_z = []
    py, pz = poly[:, 0], poly[:, 1]
    for i in range(n):
        cy, cz = py[i], pz[i]
        ny, nz = py[(i + 1) % n], pz[(i + 1) % n]
        c_in = cz <= z_max
        n_in = nz <= z_max
        if c_in:
            if n_in:
                out_y.append(ny); out_z.append(nz)
            else:
                dz = nz - cz
                t = (z_max - cz) / dz if abs(dz) > 1e-15 else 0.5
                t = max(0.0, min(1.0, t))
                out_y.append(cy + t * (ny - cy))
                out_z.append(cz + t * (nz - cz))
        elif n_in:
            dz = nz - cz
            t = (z_max - cz) / dz if abs(dz) > 1e-15 else 0.5
            t = max(0.0, min(1.0, t))
            out_y.append(cy + t * (ny - cy))
            out_z.append(cz + t * (nz - cz))
            out_y.append(ny); out_z.append(nz)

    m = len(out_y)
    if m < 3:
        return 0.0

    # Shoelace area (absolute value)
    area2 = 0.0
    for i in range(m):
        j = (i + 1) % m
        area2 += out_y[i] * out_z[j] - out_y[j] * out_z[i]
    return abs(area2) * 0.5


def _clip_area_and_centroid_fast(poly, z_max):
    """Compute area, centroid_y, centroid_z of polygon clipped below z_max.

    Same as _clip_area_fast but also returns the centroid.
    """
    n = len(poly)
    if n < 3:
        return 0.0, 0.0, 0.0

    out_y = []
    out_z = []
    py, pz = poly[:, 0], poly[:, 1]
    for i in range(n):
        cy, cz = py[i], pz[i]
        ny, nz = py[(i + 1) % n], pz[(i + 1) % n]
        c_in = cz <= z_max
        n_in = nz <= z_max
        if c_in:
            if n_in:
                out_y.append(ny); out_z.append(nz)
            else:
                dz = nz - cz
                t = (z_max - cz) / dz if abs(dz) > 1e-15 else 0.5
                t = max(0.0, min(1.0, t))
                out_y.append(cy + t * (ny - cy))
                out_z.append(cz + t * (nz - cz))
        elif n_in:
            dz = nz - cz
            t = (z_max - cz) / dz if abs(dz) > 1e-15 else 0.5
            t = max(0.0, min(1.0, t))
            out_y.append(cy + t * (ny - cy))
            out_z.append(cz + t * (nz - cz))
            out_y.append(ny); out_z.append(nz)

    m = len(out_y)
    if m < 3:
        return 0.0, 0.0, 0.0

    area2 = 0.0
    sy = 0.0
    sz = 0.0
    for i in range(m):
        j = (i + 1) % m
        cross = out_y[i] * out_z[j] - out_y[j] * out_z[i]
        area2 += cross
        sy += (out_y[i] + out_y[j]) * cross
        sz += (out_z[i] + out_z[j]) * cross
    area = area2 / 2.0
    if abs(area) < 1e-20:
        return 0.0, 0.0, 0.0
    return abs(area), sy / (6.0 * area), sz / (6.0 * area)


def _total_submerged_volume(rot_polys, dx, waterline):
    """Sum of clipped cross-section areas * dx."""
    vol = 0.0
    for rp in rot_polys:
        if len(rp) == 0:
            continue
        vol += _clip_area_fast(rp, waterline)
    return vol * dx


def _total_cob(rot_polys, dx, waterline):
    """Volume-weighted centroid of the submerged region in world coords."""
    sum_aY = 0.0
    sum_aZ = 0.0
    sum_a = 0.0
    for rp in rot_polys:
        if len(rp) == 0:
            continue
        a, cy, cz = _clip_area_and_centroid_fast(rp, waterline)
        sum_a += a
        sum_aY += a * cy
        sum_aZ += a * cz

    if sum_a < 1e-20:
        return np.array([0.0, 0.0])
    return np.array([sum_aY / sum_a, sum_aZ / sum_a])


# ===================================================================
# AVS finder
# ===================================================================

def _find_avs(angles, moments):
    """Find the angle of vanishing stability: first positive-to-negative
    zero-crossing of the righting moment after 0 deg."""
    for i in range(1, len(moments) - 1):
        if moments[i] > 0 and moments[i + 1] <= 0:
            denom = moments[i] - moments[i + 1]
            if denom < 1e-15:
                return float(angles[i])
            frac = moments[i] / denom
            return float(angles[i] + frac * (angles[i + 1] - angles[i]))
    if np.any(moments > 0):
        return 180.0
    return 0.0

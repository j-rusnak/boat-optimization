"""
Project constants and design constraints derived from the ENGR 1103 project requirements.
"""

import numpy as np

# --- Unit conversions ---
INCHES_TO_METERS = 0.0254
LB_PER_FT3_TO_KG_PER_M3 = 16.0185

# --- Foam block constraints (maximum dimensions) ---
MAX_LENGTH_IN = 11.0  # inches
MAX_WIDTH_IN = 6.0    # inches
MAX_HEIGHT_IN = 4.0   # inches

MAX_LENGTH_M = MAX_LENGTH_IN * INCHES_TO_METERS  # ~0.2794 m
MAX_WIDTH_M = MAX_WIDTH_IN * INCHES_TO_METERS     # ~0.1524 m
MAX_HEIGHT_M = MAX_HEIGHT_IN * INCHES_TO_METERS    # ~0.1016 m

# --- Material properties ---
FOAM_DENSITY_LB_FT3 = 10.0
FOAM_DENSITY_KG_M3 = FOAM_DENSITY_LB_FT3 * LB_PER_FT3_TO_KG_PER_M3  # ~160.185 kg/m³

WATER_DENSITY_KG_M3 = 997.0  # freshwater at ~25°C

# --- Mast properties ---
MAST_LENGTH_M = 0.5          # 0.5 m
MAST_DIAMETER_IN = 3 / 8     # 3/8 inch
MAST_DIAMETER_M = MAST_DIAMETER_IN * INCHES_TO_METERS
MAST_RADIUS_M = MAST_DIAMETER_M / 2.0
ALUMINUM_DENSITY_KG_M3 = 2700.0  # typical aluminum

# Mast mass (solid aluminum rod)
MAST_CROSS_SECTION_M2 = np.pi * MAST_RADIUS_M ** 2
MAST_VOLUME_M3 = MAST_CROSS_SECTION_M2 * MAST_LENGTH_M
MAST_MASS_KG = ALUMINUM_DENSITY_KG_M3 * MAST_VOLUME_M3

# --- Ballast constraints ---
BALLAST_MASS_MIN_KG = 0.700  # 700 g
BALLAST_MASS_MAX_KG = 1.000  # 1000 g

# --- Performance requirements ---
MIN_AVS_DEG = 100.0  # angle of vanishing stability must exceed this

# --- Gravity ---
G = 9.81  # m/s²

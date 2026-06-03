import os
import sys
from typing import Final

import numpy as np
from pathlib import Path

# Ensure `src` is importable when running scripts from this directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.util import create_material_map  # type: ignore


"""
Central location for standard simulation / optimization parameters used by
scripts in this directory.

Individual scripts are free to override any of these values locally to keep
their sweep logic and behavior unchanged. The goal is to avoid duplicating
the common defaults (energies, geometry, materials, etc.), not to enforce a
single configuration.
"""

# --- Material / geometry maps -------------------------------------------------

MATERIAL_DEFAULT: Final[str] = "au"
MATERIAL_MAP: Final = create_material_map(MATERIAL_DEFAULT)
MATERIAL_MAP_AU: Final = create_material_map("au")
MATERIAL_MAP_NI: Final = create_material_map("ni")
GAP_MAP_DEFAULT: Final = [np.array([0, np.inf]), np.array([1.0, 1.0])]
MEMBRANE_MAP_SI3N4: Final = create_material_map("si3n4")


# --- Common scalar defaults ---------------------------------------------------

N_WVL_DEFAULT: Final[int] = 7
CENTRAL_ENERGY_EV_DEFAULT: Final[float] = 10e3
BANDWIDTH_DEFAULT: Final[float] = 1e-4

NX_DEFAULT: Final[int] = 2**13
DX_DEFAULT: Final[float] = 10e-9
MIN_FEATURE_SIZE_DEFAULT: Final[float] = 50e-9
N_ELEMENTS_DEFAULT: Final[int] = 10

F_DEFAULT: Final[float] = 33e-3
INTER_ELEM_DIST_DEFAULT: Final[float] = 1e-2

MEMBRANE_THICKNESS_DEFAULT: Final[float] = 1e-6
ELEMENT_THICKNESS_DEFAULT: Final[float] = 400e-9

FOCUSING_THRESHOLD_DEFAULT: Final[float] = 8e-3
# FOCUSING_THRESHOLD_DEFAULT: Final[float] = 0.5
CROP_WIDTH_DEFAULT: Final[int] = 256
EFF_WIDTH_DEFAULT: Final[int] = 12

EPSILON_DEFAULT: Final[float] = 1e-6
TOLERANCE_DEFAULT: Final[float] = 1e-7
PARAM_TOLERANCE_DEFAULT: Final[float] = 1e-4
MAX_EVAL_DEFAULT: Final[int] = 100
MIN_BETA_DEFAULT: Final[float] = 2.0
CONSTRAINT_FAC_DEFAULT: Final[float] = 1.0
P_DEFAULT: Final[int] = 2

CONSTRAINT_METHOD_DEFAULT: Final[str] = "morphological"
CONSTRAINT_AGGREGATION_DEFAULT: Final[str] = "smooth_max"

# These are the values used by the bandwidth / thickness sweeps; scripts that
# need different morphology parameters should override them locally.
MORPH_BETA_DEFAULT: Final[float] = 45.0
MORPH_AGG_BETA_DEFAULT: Final[float] = 10.0

# Beta continuation schedule for optimizer projection sharpness.
# `min_beta` controls when constraints are turned on; this schedule controls
# the full continuation path.
BETA_SCHEDULE_DEFAULT: Final[list[int]] = [1, 2, 4, 8, 16, 32]

# Forward model selections:
# - optimization_model: 1D objective function used during optimization
# - metric_model_1d / metric_model_2d: models used for post-opt metrics
#   For metric_model_2d, options include "angular_2d" (2D angular spectrum) and
#   "coherent_2d_qdht" (cylindrically symmetric field, order-0 QDHT propagation)
#   for the same fully coherent workflow as angular_2d.
OPTIMIZATION_MODEL_DEFAULT: Final[str] = "angular_1d"
METRIC_MODEL_1D_DEFAULT: Final[str] = "angular_1d"
METRIC_MODEL_2D_DEFAULT: Final[str] = "coherent_2d_qdht"

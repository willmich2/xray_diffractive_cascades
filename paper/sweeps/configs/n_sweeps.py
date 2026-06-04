import numpy as np

from paper.sweeps.standard_params import NX_DEFAULT

SAVE_PREFIX = "fig1_N_sweeps"
SAVE_DIR = None
N_RUNS = 5
SWEEP_AXES = {
    "materials": np.array(["au", "ni", "ge", "si"]),
    "Nelems": np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=int),
}
MAX_PARAMS = int(int(np.max(SWEEP_AXES["Nelems"])) * NX_DEFAULT // 2)
NX_STORE = int(NX_DEFAULT)


def build_point_overrides(index_tuple, axis_values, base_params):
    return {
        "material": str(axis_values["materials"]),
        "Nelem": int(axis_values["Nelems"]),
    }


def task_cost_fn(index_tuple, axis_values, params):
    return float(axis_values["Nelems"])

"""Resolve paths to ``fig1_N_sweeps`` inputs (bundled ``paper_data/`` or timestamped runs)."""
from __future__ import annotations

import os
from pathlib import Path

FIG1_N_SWEEPS_PREFIX = "fig1_N_sweeps"

DEFAULT_DATA_DIR = os.environ.get("DIFFRACTIVE_CASCADES_DATA_DIR", "paper_data")
DEFAULT_BASE_SWEEP_ID: str | None = None


def resolve_fig1_n_sweep_paths(
    data_dir: str | os.PathLike[str],
    run_id: int,
    base_id: str | None = None,
) -> tuple[Path, Path, Path]:
    """
    Return ``(results, params, sweep_arrays)`` paths for a ``fig1_N_sweeps`` run.

    When *base_id* is omitted, uses bundled naming (e.g. ``paper_data/fig1_N_sweeps_params.npy``).
    When set, uses timestamped sweep output (``fig1_N_sweeps_params_<base_id>.npy``, etc.).
    """
    root = Path(data_dir)
    if base_id:
        return (
            root / f"{FIG1_N_SWEEPS_PREFIX}_results_{base_id}_run_{run_id}.npz",
            root / f"{FIG1_N_SWEEPS_PREFIX}_params_{base_id}.npy",
            root / f"{FIG1_N_SWEEPS_PREFIX}_sweep_arrays_{base_id}.npy",
        )
    return (
        root / f"{FIG1_N_SWEEPS_PREFIX}_results_run_{run_id}.npz",
        root / f"{FIG1_N_SWEEPS_PREFIX}_params.npy",
        root / f"{FIG1_N_SWEEPS_PREFIX}_sweep_arrays.npy",
    )


def output_id_label(base_id: str | None) -> str:
    """Value stored in robustness output NPZ ``ID`` metadata."""
    return base_id if base_id else FIG1_N_SWEEPS_PREFIX


def robustness_results_path(
    data_dir: str | os.PathLike[str],
    output_prefix: str,
    base_id: str | None,
    save_time: str,
) -> Path:
    """Build a timestamped robustness result path (matches notebook prefixes when *base_id* is unset)."""
    id_part = f"_{base_id}" if base_id else ""
    return Path(data_dir) / f"{output_prefix}{id_part}_{save_time}.npz"

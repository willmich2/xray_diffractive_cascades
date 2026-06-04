"""Binary packing for projected element densities in sweep and experiment outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

BINARY_DENSITY_DTYPE = np.bool_


def pack_binary_density(rho: Any) -> np.ndarray:
    """Pack a projected density as bool (solid=True, void=False)."""
    arr = np.asarray(rho)
    if arr.dtype == np.bool_:
        return arr
    return (arr > 0.5).astype(BINARY_DENSITY_DTYPE, copy=False)


def load_opt_rhos(rhos: Any) -> np.ndarray:
    """Normalize ``opt_rhos`` from disk (bool or legacy float) without promoting to float64."""
    return pack_binary_density(rhos)


def density_half_profile(
    rhos: np.ndarray,
    index: tuple | int,
    length: int,
    *,
    dtype: type = np.float64,
) -> np.ndarray:
    """Extract an active half-profile as float for simulation or postprocessing."""
    return np.asarray(load_opt_rhos(rhos)[index][:length], dtype=dtype)


def save_sweep_results(path: str, arrays: dict[str, Any]) -> None:
    """Write sweep outputs; element densities are stored as compressed bool arrays."""
    payload = dict(arrays)
    if "opt_rhos" in payload:
        payload["opt_rhos"] = load_opt_rhos(payload["opt_rhos"])
    np.savez_compressed(path, **payload)

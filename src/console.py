"""Standardized, informative console output for diffractive cascades runs."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Mapping, Sequence


def _enabled() -> bool:
    return os.environ.get("DIFFRACTIVE_CASCADES_QUIET", "").lower() not in ("1", "true", "yes")


def _write(component: str, level: str, message: str) -> None:
    if not _enabled():
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] [{component}] {level}: {message}", flush=True)


def info(component: str, message: str) -> None:
    _write(component, "INFO", message)


def warn(component: str, message: str) -> None:
    _write(component, "WARN", message)


def error(component: str, message: str) -> None:
    _write(component, "ERROR", message)


def banner(component: str, message: str) -> None:
    if not _enabled():
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] [{component}] ===== {message} =====", flush=True)


def kv(component: str, label: str, value: Any) -> None:
    info(component, f"{label}={value}")


def runtime_pool(
    component: str,
    *,
    n_gpus: int,
    workers_per_gpu: int,
    max_workers: int,
) -> None:
    if n_gpus == 0:
        device = "CPU-only"
    else:
        device = f"{n_gpus} GPU(s)"
    info(
        component,
        f"parallel pool: {device}, {workers_per_gpu} worker(s)/GPU, {max_workers} worker(s) total",
    )


def file_load(component: str, path: str, *, label: str = "loading") -> None:
    info(component, f"{label} {path}")


def file_saved(component: str, path: str) -> None:
    info(component, f"saved results to {path}")


def progress(component: str, completed: int, total: int, *, detail: str = "") -> None:
    pct = 100.0 * completed / total if total else 100.0
    suffix = f" — {detail}" if detail else ""
    info(component, f"progress {completed}/{total} ({pct:.1f}%){suffix}")


def elapsed(component: str, message: str, seconds: float) -> None:
    if seconds >= 120.0:
        info(component, f"{message} in {seconds / 60.0:.2f} min")
    else:
        info(component, f"{message} in {seconds:.1f} s")


def command(component: str, argv: Sequence[str]) -> None:
    info(component, "command: " + " ".join(argv))


def script_start(component: str, *, argv: Sequence[str] | None = None) -> float:
    banner(component, "starting")
    if argv is not None:
        info(component, f"argv={' '.join(argv)}")
    info(component, f"python={sys.executable}")
    info(component, f"cwd={os.getcwd()}")
    return time.time()


def script_done(component: str, start_time: float, *, message: str = "finished") -> None:
    elapsed(component, message, time.time() - start_time)


def describe_axes(component: str, axes: Mapping[str, Any]) -> None:
    parts = []
    for name, values in axes.items():
        arr = values
        try:
            n = len(arr)  # type: ignore[arg-type]
        except TypeError:
            n = "?"
        parts.append(f"{name}×{n}")
    info(component, "sweep axes: " + ", ".join(parts))


def describe_mapping(component: str, title: str, mapping: Mapping[str, Any]) -> None:
    info(component, title)
    for key, value in mapping.items():
        kv(component, f"  {key}", value)

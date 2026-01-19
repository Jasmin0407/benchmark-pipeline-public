from __future__ import annotations
# bench/core/analysis/viz/plots_cpu_single.py
"""
CPU utilization visualization for single-run benchmarks.

This module provides a compact bar plot for host CPU utilization
(mean and p95) for a single benchmark run.
"""

from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from .colors import COLOR_CPU_MEAN, COLOR_CPU_P95  # Important: keep palette consistent with multi-run plots


def _get_float(d: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """
    Safely extract a float value from a mapping.

    Args:
        d: Input mapping (e.g., metrics dict).
        key: Key to extract.
        default: Fallback value if key is missing or conversion fails.

    Returns:
        The extracted float value, or the default.
    """
    try:
        return float(d.get(key, default))
    except Exception:
        return default


def plot_cpu_util_single(metrics: Mapping[str, Any], outfile: Path) -> Path | None:
    """
    Plot host CPU utilization for a single benchmark run.

    Expected input structure:
        metrics["cpu_utilization_pct"] = {
            "mean": <float>,
            "p95": <float>
        }

    Args:
        metrics: Metrics dictionary for a single run.
        outfile: Output path for the generated plot image.

    Returns:
        The path to the saved plot file, or None if CPU data is missing.
    """
    cpu = metrics.get("cpu_utilization_pct")
    if not isinstance(cpu, Mapping):
        print(
            "[plot_cpu_util_single] No CPU utilization data available "
            "(cpu_utilization_pct missing)."
        )
        return None

    mean = _get_float(cpu, "mean", default=0.0)
    p95 = _get_float(cpu, "p95", default=0.0)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    ax.bar(["mean", "p95"], [mean, p95], color=[COLOR_CPU_MEAN, COLOR_CPU_P95])
    ax.set_ylabel("CPU utilization [%]")
    ax.set_title("CPU utilization (single run)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(outfile), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return outfile

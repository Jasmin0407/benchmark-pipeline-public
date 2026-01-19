from __future__ import annotations
# benchmark-pipeline/bench/core/analysis/viz/plots.py
"""Facade for visualization helpers.

Implementation lives in:
 - plots_latency.py
 - plots_memory.py
 - plots_cpu.py
 - plots_multi.py
 - plots_summary.py
"""

from pathlib import Path
from typing import Dict, Any, List

from .plots_latency import plot_inference_boxplot, plot_latency_series
from .plots_memory import plot_memory_over_time
from .plots_cpu import plot_cpu_utilization
from .plots_multi import plot_multi_device_latency
from .plots_summary import plot_device_summary

__all__ = [
    "plot_inference_boxplot",
    "plot_latency_series",
    "plot_memory_over_time",
    "plot_cpu_utilization",
    "plot_multi_device_latency",
    "plot_device_summary",
]


# Optional convenience wrappers for legacy call sites.


def plot_memory_over_time_from_json(json_data: Dict[str, Any], outfile: Path) -> None:
    """Bridge JSON-shaped memory blocks to :func:`plot_memory_over_time`.

    Supports both root-level ``json_data["memory"]`` and nested
    ``json_data["metrics"]["memory"]``.
    """
    mem = json_data.get("memory") or json_data.get("metrics", {}).get("memory")
    if not mem:
        print("[plot_memory_over_time_from_json] No memory samples found in JSON.")
        return
    plot_memory_over_time(mem, outfile)

# bench/core/analysis/viz/plots_single.py
"""
Single-run visualization facade.

This module exposes visualization functions that are specifically intended
for *single-run* benchmark results (as opposed to aggregated multi-run plots).

It acts as a thin re-export layer to provide a clean and stable import surface
for the rest of the analysis pipeline.
"""

from __future__ import annotations

from bench.core.analysis.viz.plots_cpu_single import plot_cpu_util_single

__all__ = [
    "plot_cpu_util_single",
]

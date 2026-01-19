"""
Single-run CPU visualization export.

This module provides a minimal re-export of CPU-related visualization
functions intended for single-run benchmark results.
"""

from bench.core.analysis.viz.plots_cpu_single import plot_cpu_util_single

__all__ = [
    "plot_cpu_util_single",
]

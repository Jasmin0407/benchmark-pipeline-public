from __future__ import annotations
"""
Latency visualizations.

This module provides:
    - A box plot for inference latency distributions (from JSON metrics).
    - A simple time series plot showing latency per iteration.

Location:
    bench/core/analysis/viz/plots_latency.py
"""

from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt

from .colors import color_for_framework


def plot_inference_boxplot(json_data: Dict[str, Any], outfile: Path) -> Path | None:
    """
    Create a box plot of inference latencies for a single device.

    Expected input structure:
        json_data["metrics"]["inference_time_ms"]["samples"] -> List[float]

    Notes:
        - At least two samples are required to create a meaningful box plot.
        - Outliers are intentionally hidden to improve readability and comparability
          across different backends.
        - The box color is determined by the framework reported in json_data["meta"].

    Args:
        json_data: Benchmark result data for a single device/run.
        outfile: Output path for the generated plot image.

    Returns:
        The path to the saved plot file, or None if insufficient data is available.
    """
    metrics = json_data.get("metrics", {})
    inf = metrics.get("inference_time_ms", {})
    samples = inf.get("samples")

    if not samples or len(samples) < 2:
        print("[plot_inference_boxplot] No latency samples available – plot skipped.")
        return None

    framework = json_data.get("meta", {}).get("framework", "model")
    color = color_for_framework(framework)

    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    bp = plt.boxplot(
        samples,
        vert=True,
        patch_artist=True,
        showfliers=False,
    )

    for box in bp["boxes"]:
        box.set(facecolor=color, edgecolor="#333333")

    plt.ylabel("Latency [ms]")
    plt.title("Inference latency distribution")
    plt.xticks([1], [framework])
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
    plt.close()
    return outfile


def plot_latency_series(latencies: List[float], outfile: Path) -> Path | None:
    """
    Plot a latency time series (iteration index vs. latency).

    Args:
        latencies: List of per-iteration latency values in milliseconds.
        outfile: Output path for the generated plot image.

    Returns:
        The path to the saved plot file, or None if the input list is empty.
    """
    if not latencies:
        print("[plot_latency_series] No latency data available.")
        return None

    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(latencies, linewidth=1)
    plt.title("Latency time series")
    plt.xlabel("Iteration")
    plt.ylabel("Latency [ms]")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
    plt.close()
    return outfile

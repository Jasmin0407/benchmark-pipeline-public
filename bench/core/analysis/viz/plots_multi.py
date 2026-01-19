from __future__ import annotations
# bench/core/analysis/viz/plots_multi.py
"""
Multi-device visualizations.

This module provides comparative plots across multiple devices
(e.g. CPU, GPU, NPU) for a single benchmark run or aggregated results.
"""

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt

from .colors import color_for_device


def plot_multi_device_latency(results: Dict[str, Any], outfile: Path) -> None:
    """
    Create a box plot comparing inference latencies across multiple devices.

    Expected input structure (either variant is supported):
        - results[device]["timing_ms"]["samples"]
        - results[device]["metrics"]["inference_time_ms"]["samples"]

    Notes:
        - Devices without latency samples are skipped silently.
        - Outliers are hidden to improve readability and comparability.
        - Colors are assigned per device using the shared color mapping.

    Args:
        results: Dictionary of benchmark results keyed by device name.
        outfile: Output path for the generated plot image.
    """
    devices: list[str] = []
    data: list[list[float]] = []
    colors: list[str] = []

    for dev, block in results.items():
        latency = block.get("timing_ms") or block.get("metrics", {}).get("inference_time_ms")
        if not latency:
            continue

        samples = latency.get("samples", [])
        if not samples:
            continue

        devices.append(dev.upper())
        data.append(samples)
        colors.append(color_for_device(dev))

    if not data:
        print("[plot_multi_device_latency] No latency data available.")
        return

    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, labels=devices, patch_artist=True, showfliers=False)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    plt.title("Inference latency – CPU / GPU / NPU comparison")
    plt.ylabel("Latency [ms]")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
    plt.close()

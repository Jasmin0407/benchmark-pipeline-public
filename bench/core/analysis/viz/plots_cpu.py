from __future__ import annotations
# bench/core/analysis/viz/plots_cpu.py
"""
CPU-related visualizations.

This module provides plots for host CPU utilization during inference,
aggregated per device in multi-device benchmark runs.
"""

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt

from .colors import COLOR_CPU_MEAN, COLOR_CPU_P95


def plot_cpu_utilization(results: Dict[str, Any], outfile: Path) -> Path | None:
    """
    Create a bar chart of host CPU utilization per device (mean and p95).

    Expected input structure:
        results[device]["metrics"]["cpu_utilization_pct"] = {
            "mean": <float>,
            "p95": <float>
        }

    Notes:
        - Devices with names starting with "auto" or "hetero" are intentionally skipped,
          as they represent composite execution modes rather than a single device.
        - If no valid CPU utilization data is found, the function returns None and
          no plot is generated.

    Args:
        results: Dictionary of benchmark results keyed by device name.
        outfile: Output path for the generated plot image.

    Returns:
        The path to the saved plot file, or None if no valid data was available.
    """
    devices: list[str] = []
    means: list[float] = []
    p95s: list[float] = []

    for dev, block in results.items():
        dev_l = dev.lower()

        # Explicitly skip composite execution modes (AUTO / HETERO)
        if dev_l.startswith("auto") or dev_l.startswith("hetero"):
            continue

        metrics = block.get("metrics", {})
        cpu_block = metrics.get("cpu_utilization_pct")
        if not cpu_block:
            print(f"[plot_cpu_utilization] No cpu_utilization_pct data for {dev}")
            continue

        mean_val = cpu_block.get("mean")
        p95_val = cpu_block.get("p95")

        if mean_val is None or p95_val is None:
            print(
                f"[plot_cpu_utilization] Incomplete CPU utilization data for {dev}: {cpu_block}"
            )
            continue

        devices.append(dev.upper())
        means.append(mean_val)
        p95s.append(p95_val)

    if not devices:
        print("[plot_cpu_utilization] No CPU utilization data available.")
        return None

    outfile.parent.mkdir(parents=True, exist_ok=True)

    x = range(len(devices))

    plt.figure(figsize=(10, 6))
    plt.bar(x, means, color=COLOR_CPU_MEAN, width=0.4, label="mean")
    plt.bar([i + 0.4 for i in x], p95s, color=COLOR_CPU_P95, width=0.4, label="p95")

    plt.xticks([i + 0.2 for i in x], devices)
    plt.ylabel("Host CPU utilization [%]")
    plt.title("Host CPU utilization during inference per device")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
    plt.close()

    return outfile

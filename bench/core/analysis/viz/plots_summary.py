from __future__ import annotations
# bench/core/analysis/viz/plots_summary.py
"""
Per-device summary plot.

Includes:
    - Inference latency distribution
    - Host CPU utilization (mean & p95)
    - Host memory trace (RSS delta)
    - Sampling interval and drift diagnostics
"""

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from .colors import (
    COLOR_CPU_MEAN,
    COLOR_CPU_P95,
    COLOR_RAM_DELTA,
    COLOR_SAMPLING_INTERVAL,
    COLOR_SAMPLING_TARGET,
    phase_color,
)

# Devices that are intentionally skipped in the summary view (management/meta devices).
SKIP_DEVICES = {"auto", "hetero", "hetero:gpu,cpu"}


def _compute_sampling_intervals(
    memory_block: Dict[str, Any],
) -> tuple[np.ndarray, float, float, float]:
    """
    Compute sampling intervals and drift metrics from a memory block.

    Returns:
        intervals_ms: Per-sample interval durations in milliseconds.
        target_interval_ms: Target sampling interval in milliseconds.
        drift_mean_ms: Mean drift relative to the target interval (ms).
        drift_std_ms: Standard deviation of the drift (ms).
    """
    ts = np.array(memory_block.get("timestamps_s", []), dtype=float)
    if ts.size < 2:
        raise ValueError("Insufficient timestamps for sampling-interval plot.")

    sample_hz = float(memory_block.get("sample_hz", 75.0))
    target_interval_s = 1.0 / sample_hz

    deltas_s = np.diff(ts)
    intervals_ms = deltas_s * 1000.0

    drift_mean_ms = float((deltas_s.mean() - target_interval_s) * 1000.0)
    drift_std_ms = float(deltas_s.std() * 1000.0)
    target_interval_ms = target_interval_s * 1000.0

    return intervals_ms, target_interval_ms, drift_mean_ms, drift_std_ms


def plot_device_summary(block: Dict[str, Any], dev: str, outfile: Path) -> None:
    """
    Create a consolidated summary figure for a single device.

    The summary consists of four panels:
        1) Latency distribution (histogram)
        2) Host CPU utilization (mean and p95)
        3) Host memory usage over time (RSS delta with phases)
        4) Sampling interval and drift diagnostics

    Args:
        block: Result block for a single device.
        dev: Device identifier string.
        outfile: Output path for the generated summary image.
    """
    dev_l = dev.lower()

    if any(dev_l.startswith(x) for x in SKIP_DEVICES):
        print(f"[INFO] Summary plot skipped for {dev.upper()} (management device).")
        return

    outfile.parent.mkdir(parents=True, exist_ok=True)

    timing = block.get("timing_ms") or block.get("metrics", {}).get("inference_time_ms")
    cpu = block.get("cpu_utilization_pct") or block.get("metrics", {}).get("cpu_utilization_pct")
    mem = block.get("memory") or block.get("metrics", {}).get("memory")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_lat = axes[0, 0]
    ax_cpu = axes[0, 1]
    ax_mem = axes[1, 0]
    ax_samp = axes[1, 1]

    # ---------------------------------------------------------
    # 1) LATENCY (top-left)
    # ---------------------------------------------------------
    if timing:
        samples = timing.get("samples", [])
        if samples:
            ax_lat.hist(samples, bins=30, color=COLOR_RAM_DELTA)
            ax_lat.set_title(f"Latency distribution [{dev.upper()}]")
            ax_lat.set_xlabel("Latency [ms]")
            ax_lat.set_ylabel("Frequency")
            ax_lat.grid(True, linestyle="--", alpha=0.4)
        else:
            ax_lat.text(0.5, 0.5, "No latency samples", ha="center", va="center")
    else:
        ax_lat.text(0.5, 0.5, "No latency data", ha="center", va="center")

    # ---------------------------------------------------------
    # 2) HOST CPU LOAD (top-right)
    # ---------------------------------------------------------
    if cpu and ("mean" in cpu and "p95" in cpu):
        ax_cpu.bar(
            ["Mean", "P95"],
            [cpu.get("mean", 0), cpu.get("p95", 0)],
            color=[COLOR_CPU_MEAN, COLOR_CPU_P95],
        )
        ax_cpu.set_title(f"Host CPU utilization [{dev.upper()}]")
        ax_cpu.set_ylabel("CPU utilization [%]")
        ax_cpu.grid(True, linestyle="--", alpha=0.4)
    else:
        ax_cpu.text(0.5, 0.5, "No CPU data", ha="center", va="center")

    # ---------------------------------------------------------
    # 3) MEMORY TRACE (bottom-left)
    # ---------------------------------------------------------
    if mem:
        ts = np.array(mem.get("timestamps_s", []), dtype=float)
        rss = np.array(mem.get("rss_samples", []), dtype=float)

        if ts.size > 0 and rss.size > 0:
            i0 = int(mem.get("infer_start_idx", 0))
            i1 = int(mem.get("infer_end_idx", 0))
            i0 = max(0, min(i0, len(ts) - 1))
            i1 = max(i0, min(i1, len(ts) - 1))

            rss_delta_mb = (rss - rss[0]) / 1e6

            ax_mem.plot(ts, rss_delta_mb, color=COLOR_RAM_DELTA, linewidth=1.6, label="ΔRSS")

            t_start = ts[0]
            t_infer_start = ts[i0]
            t_infer_end = ts[i1]
            t_end = ts[-1]

            if t_infer_start > t_start:
                ax_mem.axvspan(
                    t_start,
                    t_infer_start,
                    color=phase_color("preroll"),
                    alpha=0.12,
                    label="Pre-roll",
                )

            ax_mem.axvspan(
                t_infer_start,
                t_infer_end,
                color=phase_color("inference"),
                alpha=0.10,
                label="Inference phase",
            )

            if t_end > t_infer_end:
                ax_mem.axvspan(
                    t_infer_end,
                    t_end,
                    color=phase_color("post"),
                    alpha=0.12,
                    label="Post-run",
                )

            ax_mem.axvline(
                t_infer_start,
                color=phase_color("inference"),
                linestyle="--",
                linewidth=1.3,
                label="Inference start",
            )
            ax_mem.axvline(
                t_infer_end,
                color=phase_color("post"),
                linestyle="--",
                linewidth=1.3,
                label="Post-run start",
            )

            ax_mem.set_title(f"Host memory usage [MB] – device {dev.upper()}")
            ax_mem.set_xlabel("Time [s]")
            ax_mem.set_ylabel("ΔRSS [MB]")
            ax_mem.grid(True, linestyle="--", alpha=0.4)
            ax_mem.legend()
        else:
            ax_mem.text(0.5, 0.5, "No memory data", ha="center", va="center")
    else:
        ax_mem.text(0.5, 0.5, "No memory data", ha="center", va="center")

    # ---------------------------------------------------------
    # 4) SAMPLING INTERVAL / DRIFT (bottom-right)
    # ---------------------------------------------------------
    if mem and mem.get("timestamps_s"):
        try:
            intervals_ms, target_ms, drift_mean_ms, drift_std_ms = _compute_sampling_intervals(mem)
            idx = np.arange(1, len(intervals_ms) + 1)

            ts = np.array(mem.get("timestamps_s", []), dtype=float)
            i0 = int(mem.get("infer_start_idx", 0))
            i1 = int(mem.get("infer_end_idx", 0))
            i0 = max(0, min(i0, len(ts) - 1))
            i1 = max(i0, min(i1, len(ts) - 1))

            ax_samp.plot(
                idx,
                intervals_ms,
                color=COLOR_SAMPLING_INTERVAL,
                linewidth=0.8,
                label="Actual interval",
            )

            ax_samp.axhline(
                target_ms,
                color=COLOR_SAMPLING_TARGET,
                linestyle="--",
                linewidth=1.0,
                label=f"Target ({1000.0 / target_ms:.0f} Hz)",
            )

            # Optional phase background shading (index-based)
            if i0 > 0:
                ax_samp.axvspan(
                    1,
                    i0,
                    color=phase_color("preroll"),
                    alpha=0.06,
                    label="Pre-roll",
                )

            if i1 > i0:
                ax_samp.axvspan(
                    i0,
                    i1,
                    color=phase_color("inference"),
                    alpha=0.06,
                    label="Inference phase",
                )

            if len(intervals_ms) > i1:
                ax_samp.axvspan(
                    i1,
                    len(intervals_ms),
                    color=phase_color("post"),
                    alpha=0.06,
                    label="Post-run",
                )

            ax_samp.set_xlabel("Sample index")
            ax_samp.set_ylabel("Interval [ms]")
            ax_samp.set_title("Sampling interval and drift")
            ax_samp.grid(True, linestyle=":", linewidth=0.5)

            text = f"Drift mean: {drift_mean_ms:.3f} ms\nStd: {drift_std_ms:.3f} ms"
            ax_samp.text(
                0.99,
                0.01,
                text,
                ha="right",
                va="bottom",
                transform=ax_samp.transAxes,
                fontsize=8,
            )

            ax_samp.legend(loc="upper right")
        except ValueError as exc:
            ax_samp.text(0.5, 0.5, str(exc), ha="center", va="center")
    else:
        ax_samp.text(0.5, 0.5, "No timestamps for sampling", ha="center", va="center")

    fig.suptitle(f"Device summary – {dev.upper()}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(str(outfile), bbox_inches="tight")
    plt.close(fig)

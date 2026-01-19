# bench/core/analysis/viz/plots_memory.py
from __future__ import annotations
"""
Memory usage visualization (host RSS) over time with phase annotations.

This plot shows the relative change in resident set size (RSS) during
a benchmark run and highlights the different execution phases:
    - pre-roll
    - inference
    - post-run
"""

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from .colors import COLOR_RAM_DELTA, phase_color


def plot_memory_over_time(
    memory_block: Dict[str, Any] | list,
    outfile: Path | None = None,
    *,
    timestamps_s=None,
    infer_start_idx: int = 0,
    infer_end_idx: int = 0,
    sample_hz: float = 1.0,
    out_dir=None,
) -> Path | None:
    """
    Plot host memory usage (RSS) over time with colored phase markers.

    Expected memory_block structure:
        {
            "timestamps_s": [...],        # timestamps in seconds
            "rss_samples": [...],         # RSS samples in bytes
            "infer_start_idx": int,       # index where inference starts
            "infer_end_idx": int,         # index where inference ends
            "pre_roll_s": float,          # optional (informational)
            "post_delay_s": float,        # optional (informational)
            "mode": "static" | "dynamic"  # optional
        }

    Backward compatibility:
        If memory_block is not a dict (e.g. a raw list of RSS samples),
        timestamps_s and out_dir must be provided explicitly.

    Args:
        memory_block: Memory metrics block or legacy list of RSS samples.
        outfile: Optional explicit output file path.
        timestamps_s: Explicit timestamps (required for legacy call style).
        infer_start_idx: Inference start index (legacy call style).
        infer_end_idx: Inference end index (legacy call style).
        sample_hz: Sampling frequency in Hz (currently informational).
        out_dir: Output directory (used if outfile is not provided).

    Returns:
        Path to the saved plot image, or None if no valid data is available.
    """
    # --- Compatibility with legacy argument-based test calls ---
    if not isinstance(memory_block, dict):
        if timestamps_s is None or out_dir is None:
            raise TypeError("timestamps_s and out_dir are required for legacy calls")

        outfile = Path(out_dir) / "memory_over_time.png"
        memory_block = {
            "timestamps_s": timestamps_s,
            "rss_samples": memory_block,
            "infer_start_idx": infer_start_idx,
            "infer_end_idx": infer_end_idx,
        }

    ts = np.array(memory_block.get("timestamps_s", []), dtype=float)
    rss = np.array(memory_block.get("rss_samples", []), dtype=float)

    if ts.size == 0 or rss.size == 0:
        print("[plot_memory_over_time] No memory data available.")
        return None

    i0 = int(memory_block.get("infer_start_idx", 0))
    i1 = int(memory_block.get("infer_end_idx", 0))
    i0 = max(0, min(i0, len(ts) - 1))
    i1 = max(i0, min(i1, len(ts) - 1))

    # Convert to delta RSS in MB relative to the first sample
    rss_delta_mb = (rss - rss[0]) / 1e6

    if outfile is None:
        if out_dir is None:
            print("[plot_memory_over_time] No outfile or out_dir specified – plot skipped.")
            return None
        outfile = Path(out_dir) / "memory_over_time.png"
    else:
        outfile = Path(outfile)

    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(ts, rss_delta_mb, color=COLOR_RAM_DELTA, linewidth=1.6, label="ΔRSS")

    t_start = ts[0]
    t_infer_start = ts[i0]
    t_infer_end = ts[i1]
    t_end = ts[-1]

    # Pre-roll phase
    if t_infer_start > t_start:
        plt.axvspan(
            t_start,
            t_infer_start,
            color=phase_color("preroll"),
            alpha=0.12,
            label="Pre-roll",
        )

    # Inference phase
    plt.axvspan(
        t_infer_start,
        t_infer_end,
        color=phase_color("inference"),
        alpha=0.10,
        label="Inference phase",
    )

    # Post-run phase
    if t_end > t_infer_end:
        plt.axvspan(
            t_infer_end,
            t_end,
            color=phase_color("post"),
            alpha=0.12,
            label="Post-run",
        )

    # Phase boundaries
    plt.axvline(
        t_infer_start,
        color=phase_color("inference"),
        linestyle="--",
        linewidth=1.3,
        label=f"Inference start ({t_infer_start:.2f}s)",
    )
    plt.axvline(
        t_infer_end,
        color=phase_color("post"),
        linestyle="--",
        linewidth=1.3,
        label=f"Post-run start ({t_infer_end:.2f}s)",
    )

    plt.title("Host memory usage during inference")
    plt.xlabel("Time [s]")
    plt.ylabel("ΔRAM [MB]")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
    plt.close()
    return outfile

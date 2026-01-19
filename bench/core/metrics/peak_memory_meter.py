"""
PeakMemoryMeter

Lightweight peak memory sampling around repeated function calls.

Reports:
- rss_start_bytes / rss_end_bytes
- rss_peak_bytes
- rss_delta_bytes
- duration_s

Optional (Linux-only):
- PSS (Proportional Set Size) via /proc/<pid>/smaps_rollup (best-effort)

Optional (CUDA):
- Current/peak CUDA allocator usage via torch.cuda.* (best-effort)
  Note: this is process allocator memory, not total GPU memory usage.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import psutil

try:
    import torch  # optional dependency (CUDA allocator info)
except Exception:
    torch = None


class PeakMemoryMeter:
    """
    Parameters
    ----------
    sample_hz:
        Sampling frequency (samples per second). Used as sleep pacing between repeats.
        Set to <=0 to disable sleeps (not recommended for reproducibility).
    use_pss:
        Enable Linux PSS measurement (best-effort).
    use_cuda:
        Enable CUDA allocator measurement via torch.cuda (best-effort).
    """

    def __init__(self, sample_hz: int = 50, use_pss: bool = False, use_cuda: bool = False):
        self.sample_hz = int(sample_hz)
        self.use_pss = bool(use_pss)
        self.use_cuda = bool(use_cuda)

        self.process = psutil.Process()
        self.samples_rss: list[int] = []

    # ------------------------------------------------------------------
    # Linux-only: PSS via smaps_rollup
    # ------------------------------------------------------------------
    def _read_pss_bytes(self) -> Optional[int]:
        """
        Read PSS in bytes from /proc/<pid>/smaps_rollup (Linux only).
        Returns None if not available.
        """
        try:
            p = Path(f"/proc/{os.getpid()}/smaps_rollup")
            if not p.exists():
                return None

            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    # Example: "Pss:                12345 kB"
                    if line.startswith("Pss:"):
                        kb = int(line.split()[1])
                        return kb * 1024
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # CUDA allocator info (best-effort)
    # ------------------------------------------------------------------
    def _read_cuda_used_bytes(self) -> Optional[int]:
        """
        Read current CUDA allocator usage (bytes) from PyTorch (best-effort).
        This is NOT total device memory usage; it is process allocator usage.
        """
        if not self.use_cuda:
            return None
        if torch is None:
            return None
        try:
            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated())
        except Exception:
            return None
        return None

    def _read_cuda_peak_bytes(self) -> Optional[int]:
        """
        Read CUDA allocator peak usage (bytes) from PyTorch (best-effort).
        """
        if not self.use_cuda:
            return None
        if torch is None:
            return None
        try:
            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_allocated())
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Main measurement routine
    # ------------------------------------------------------------------
    def measure(self, fn: Callable[[], Any], repeats: int = 10) -> Dict[str, Any]:
        """
        Execute fn repeatedly and sample process RSS after each call.

        Parameters
        ----------
        fn:
            Function to execute (e.g., one inference).
        repeats:
            Number of executions.

        Returns
        -------
        Dict[str, Any]
            Serializable measurement dictionary.
        """
        repeats = max(1, int(repeats))
        self.samples_rss.clear()

        # Reset CUDA peak stats if enabled (best-effort)
        if self.use_cuda and torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        rss_start = int(self.process.memory_info().rss)
        pss_start = self._read_pss_bytes() if self.use_pss else None

        peak_rss = rss_start
        t0 = time.perf_counter()

        # Sleep pacing (guard against division by zero)
        interval_s = 0.0
        if self.sample_hz > 0:
            interval_s = 1.0 / float(self.sample_hz)

        for _ in range(repeats):
            fn()

            rss_now = int(self.process.memory_info().rss)
            self.samples_rss.append(rss_now)
            if rss_now > peak_rss:
                peak_rss = rss_now

            if interval_s > 0:
                time.sleep(interval_s)

        t1 = time.perf_counter()

        rss_end = int(self.process.memory_info().rss)
        pss_end = self._read_pss_bytes() if self.use_pss else None

        out: Dict[str, Any] = {
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_peak_bytes": int(peak_rss),
            "rss_delta_bytes": int(rss_end - rss_start),
            "duration_s": float(t1 - t0),
            "samples": list(self.samples_rss),
            "sample_hz": int(self.sample_hz),
            "repeats": int(repeats),
        }

        if self.use_pss:
            out["pss_start_bytes"] = int(pss_start) if pss_start is not None else None
            out["pss_end_bytes"] = int(pss_end) if pss_end is not None else None

        cuda_used = self._read_cuda_used_bytes()
        cuda_peak = self._read_cuda_peak_bytes()
        if cuda_used is not None or cuda_peak is not None:
            out["cuda_memory"] = {
                "used_bytes": int(cuda_used) if cuda_used is not None else None,
                "peak_bytes": int(cuda_peak) if cuda_peak is not None else None,
            }

        return out

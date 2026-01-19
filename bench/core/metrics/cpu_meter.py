# bench/core/metrics/cpu_meter.py
"""
CpuMeter

Samples CPU utilization of the *current process* over time while inference runs.

Notes
-----
- Uses psutil.Process().cpu_percent(interval=None) which requires a warm-up call
  to establish a baseline.
- Sampling runs in a daemon thread and stores:
  - samples: CPU percent values
  - intervals: actual sampling intervals (seconds) to quantify drift/jitter
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any, Dict, Optional, Sequence

import psutil
import numpy as np


class CpuMeter:
    def __init__(self, sample_hz: int = 75):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        self.sample_hz = int(sample_hz)

        self.samples: list[float] = []
        self.intervals: list[float] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Current process CPU utilization (process-level, not system-wide)
        self.process = psutil.Process()

    def _sample_loop(self) -> None:
        target_dt = 1.0 / self.sample_hz

        # Prime psutil cpu_percent to avoid a misleading first sample
        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass

        next_tick = time.perf_counter()
        last_tick = next_tick

        while self._running:
            now = time.perf_counter()

            try:
                cpu_pct = float(self.process.cpu_percent(interval=None))
            except Exception:
                cpu_pct = 0.0

            with self._lock:
                self.samples.append(cpu_pct)

                # Measure actual time delta between samples (jitter/drift diagnostics)
                delta = now - last_tick
                self.intervals.append(delta)
                last_tick = now

            # Drift-corrected schedule (reduces accumulated timing drift)
            next_tick += target_dt
            sleep_time = max(0.0, next_tick - time.perf_counter())
            time.sleep(sleep_time)

    def start(self) -> None:
        """
        Start CPU sampling in a background thread.

        Calling start() resets previous samples.
        """
        with self._lock:
            self.samples.clear()
            self.intervals.clear()

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop sampling and join the background thread.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def summary(self, infer_start_idx: Optional[int] = None, infer_end_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute summary statistics.

        Parameters
        ----------
        infer_start_idx / infer_end_idx:
            If provided, only the sample slice [infer_start_idx:infer_end_idx] is used,
            which should correspond to the inference window derived from MemoryMeter.

        Returns
        -------
        Dict[str, Any]
            mean, p95, and sampling stability diagnostics.
        """
        with self._lock:
            all_samples = list(self.samples)
            all_intervals = list(self.intervals)

        # Select inference window if indices are provided
        if infer_start_idx is not None and infer_end_idx is not None:
            i0 = max(0, int(infer_start_idx))
            i1 = max(i0, int(infer_end_idx))
            samples = all_samples[i0:i1]
        else:
            samples = all_samples

        if not samples:
            return {
                "mean": 0.0,
                "p95": 0.0,
                "sampling_interval_mean_s": 0.0,
                "sampling_stddev_s": 0.0,
                "sampling_drift_ms": 0.0,
            }

        arr = np.asarray(samples, dtype=float)
        cpu_mean = float(np.mean(arr))
        cpu_p95 = float(np.percentile(arr, 95)) if arr.size >= 2 else float(arr[0])

        # Sampling stability (measured over the entire sampling period)
        if all_intervals:
            mean_interval = float(statistics.mean(all_intervals))
            std_interval = float(statistics.pstdev(all_intervals)) if len(all_intervals) >= 2 else 0.0
            drift_ms = (mean_interval - (1.0 / self.sample_hz)) * 1000.0
        else:
            mean_interval = 0.0
            std_interval = 0.0
            drift_ms = 0.0

        return {
            "mean": round(cpu_mean, 2),
            "p95": round(cpu_p95, 2),
            "sampling_interval_mean_s": round(mean_interval, 6),
            "sampling_stddev_s": round(std_interval, 6),
            "sampling_drift_ms": round(drift_ms, 3),
        }

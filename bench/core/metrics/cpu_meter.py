"""CpuMeter

Samples CPU utilization of the current process while inference runs.

Design notes
------------
- Uses ``psutil.Process().cpu_percent(interval=None)`` and therefore primes the API
  once before collecting the first meaningful sample.
- Keeps sample timestamps and actual sampling intervals for auditability.
- Exposes both legacy summary keys (``mean``, ``p95``) and richer scope-aware fields
  for reporting and debugging.
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import psutil


class CpuMeter:
    def __init__(self, sample_hz: int = 75):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")

        self.sample_hz = int(sample_hz)
        self.samples: list[float] = []
        self.intervals: list[float] = []
        self.sample_timestamps_s: list[float] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.process = psutil.Process()
        self._sample_start_perf: Optional[float] = None
        self._sample_end_perf: Optional[float] = None
        self._cpu_time_start_s: Optional[float] = None
        self._cpu_time_end_s: Optional[float] = None

    def _read_process_cpu_time_s(self) -> float:
        """Return user+system CPU time for the current process."""
        cpu_times = self.process.cpu_times()
        return float(getattr(cpu_times, "user", 0.0) + getattr(cpu_times, "system", 0.0))

    def _sample_loop(self) -> None:
        target_dt = 1.0 / self.sample_hz

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
                self.sample_timestamps_s.append(float(now - (self._sample_start_perf or now)))
                delta = now - last_tick
                self.intervals.append(delta)
                last_tick = now

            next_tick += target_dt
            sleep_time = max(0.0, next_tick - time.perf_counter())
            time.sleep(sleep_time)

    def start(self) -> None:
        """Start CPU sampling in a background thread and reset previous state."""
        with self._lock:
            self.samples.clear()
            self.intervals.clear()
            self.sample_timestamps_s.clear()

        self._sample_start_perf = time.perf_counter()
        self._sample_end_perf = None
        try:
            self._cpu_time_start_s = self._read_process_cpu_time_s()
        except Exception:
            self._cpu_time_start_s = None
        self._cpu_time_end_s = None

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and finalize audit timestamps."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._sample_end_perf = time.perf_counter()
        try:
            self._cpu_time_end_s = self._read_process_cpu_time_s()
        except Exception:
            self._cpu_time_end_s = None

    def summary(
        self,
        infer_start_idx: Optional[int] = None,
        infer_end_idx: Optional[int] = None,
        scope: str = "inference_window",
        scope_label: str = "cpu_inference_window",
    ) -> Dict[str, Any]:
        """
        Compute a scope-aware CPU summary.

        Parameters
        ----------
        infer_start_idx / infer_end_idx:
            Sample slice boundaries for the inference window. The indices are treated
            with Python slicing semantics ``[start:end]``.
        scope / scope_label:
            Human-readable descriptors for downstream reporting.
        """
        with self._lock:
            all_samples = list(self.samples)
            all_intervals = list(self.intervals)
            all_timestamps = list(self.sample_timestamps_s)

        if infer_start_idx is not None and infer_end_idx is not None:
            i0 = max(0, int(infer_start_idx))
            i1 = max(i0, int(infer_end_idx))
            samples = all_samples[i0:i1]
            timestamps = all_timestamps[i0:i1]
        else:
            samples = all_samples
            timestamps = all_timestamps

        if all_intervals:
            mean_interval = float(statistics.mean(all_intervals))
            std_interval = float(statistics.pstdev(all_intervals)) if len(all_intervals) >= 2 else 0.0
            drift_ms = (mean_interval - (1.0 / self.sample_hz)) * 1000.0
        else:
            mean_interval = 0.0
            std_interval = 0.0
            drift_ms = 0.0

        wall_s = None
        cpu_s = None
        cpu_core_util = None
        if self._sample_start_perf is not None and self._sample_end_perf is not None:
            wall_s = float(max(0.0, self._sample_end_perf - self._sample_start_perf))
        if self._cpu_time_start_s is not None and self._cpu_time_end_s is not None:
            cpu_s = float(max(0.0, self._cpu_time_end_s - self._cpu_time_start_s))
        if wall_s is not None and wall_s > 0.0 and cpu_s is not None:
            cpu_core_util = float(cpu_s / wall_s)

        if not samples:
            return {
                "mean": 0.0,
                "p95": 0.0,
                "valid": False,
                "invalid_reason": "no_samples_in_selected_scope",
                "scope": str(scope),
                "scope_label": str(scope_label),
                "n_samples": 0,
                "mean_unweighted": 0.0,
                "mean_time_weighted": None,
                "core_util_mean_cores": 0.0,
                "core_util_p95_cores": 0.0,
                "sample_time_start_s": None,
                "sample_time_end_s": None,
                "sampling_interval_mean_s": round(mean_interval, 6),
                "sampling_stddev_s": round(std_interval, 6),
                "sampling_drift_ms": round(drift_ms, 3),
                "cpu_time": {
                    "wall_s": wall_s,
                    "cpu_s": cpu_s,
                    "cpu_core_util": cpu_core_util,
                },
            }

        arr = np.asarray(samples, dtype=float)
        cpu_mean = float(np.mean(arr))
        cpu_p95 = float(np.percentile(arr, 95)) if arr.size >= 2 else float(arr[0])

        sample_time_start_s = float(timestamps[0]) if timestamps else None
        sample_time_end_s = float(timestamps[-1]) if timestamps else None

        return {
            "mean": round(cpu_mean, 2),
            "p95": round(cpu_p95, 2),
            "valid": True,
            "invalid_reason": None,
            "scope": str(scope),
            "scope_label": str(scope_label),
            "n_samples": int(arr.size),
            "mean_unweighted": round(cpu_mean, 6),
            "mean_time_weighted": None,
            "core_util_mean_cores": round(cpu_mean / 100.0, 6),
            "core_util_p95_cores": round(cpu_p95 / 100.0, 6),
            "sample_time_start_s": sample_time_start_s,
            "sample_time_end_s": sample_time_end_s,
            "sampling_interval_mean_s": round(mean_interval, 6),
            "sampling_stddev_s": round(std_interval, 6),
            "sampling_drift_ms": round(drift_ms, 3),
            "cpu_time": {
                "wall_s": wall_s,
                "cpu_s": cpu_s,
                "cpu_core_util": cpu_core_util,
            },
        }

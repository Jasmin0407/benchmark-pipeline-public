# bench/core/metrics/timing_meter.py
"""
TimingMeter

Measures inference latency for a callable over multiple iterations and provides
summary statistics (mean + percentiles). This meter is CPU-timer based
(time.perf_counter) and does not assume any particular framework.

Design:
- Warmups are executed first and are excluded from reported statistics.
- Measured iterations are stored in `history_ms` for post-hoc inspection/export.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class TimingMeter:
    def __init__(self, warmups: int = 5, repeats: int = 50, batch_size: int = 1):
        self.warmups = int(warmups)
        self.repeats = int(repeats)
        self.batch_size = int(batch_size)
        self.history_ms: List[float] = []

    def reset(self) -> None:
        """Clear previously recorded measurements."""
        self.history_ms.clear()

    def measure(self, func: Callable[[], Any]) -> Dict[str, Any]:
        """
        Run warmups + measured repeats and return summary statistics.

        Parameters
        ----------
        func:
            Callable that performs exactly one inference.

        Returns
        -------
        Dict[str, Any]
            Statistics dict including raw samples in milliseconds.
        """
        # Warmup iterations (excluded from statistics)
        for _ in range(max(0, self.warmups)):
            func()

        n = max(1, self.repeats)
        samples: List[float] = []

        for _ in range(n):
            t0 = time.perf_counter()
            func()
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)

        # Keep full history (useful for debugging / export)
        self.history_ms.extend(samples)

        arr = np.asarray(samples, dtype=float)
        mean_ms = float(arr.mean()) if arr.size else 0.0

        # Throughput:
        # - inferences per second (ips): 1000 / mean_ms
        # - samples per second (sps): ips * batch_size
        ips = (1000.0 / mean_ms) if mean_ms > 0 else None
        sps = (ips * self.batch_size) if ips is not None else None

        return {
            "mean": mean_ms,
            "p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
            "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
            "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "p99": float(np.percentile(arr, 99)) if arr.size else 0.0,
            "throughput_ips": ips,
            "throughput_sps": sps,
            "samples": arr.tolist(),
        }

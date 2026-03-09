"""
MemoryMeter

Measures host RAM usage (RSS) over time and reports:
- start/end/peak RSS
- RSS delta (total and inference-window)
- RSS time series + timestamps aligned to the true inference start

Supports an optional post-delay window to capture memory settling after inference.

Design notes
------------
- ``time.perf_counter()`` is the canonical source for duration fields.
- Sample timestamps are preserved for diagnostics, plots, and audit trails, but they
  are not used as the primary source of truth for exported durations.
- ``infer_start_idx`` / ``infer_end_idx`` follow Python slicing semantics:
  - ``infer_start_idx`` is inclusive
  - ``infer_end_idx`` is exclusive
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import psutil

try:
    import torch  # Optional dependency
except Exception:
    torch = None


class MemoryMeter:
    """
    Samples RSS over time and optionally adds GPU memory info (best-effort).

    Parameters
    ----------
    sample_hz:
        Sampling frequency (samples per second).
    gpu_index:
        GPU index to query via NVML (default: 0). If you need multi-GPU reporting,
        pass the desired index (or extend to a list).
    """

    def __init__(self, sample_hz: int = 75, gpu_index: int = 0):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")

        self.sample_hz = int(sample_hz)
        self.gpu_index = int(gpu_index)

        self.samples_rss: list[int] = []
        self._process = psutil.Process()

        self._rss_start = 0
        self._rss_peak = 0
        self._rss_end = 0

        # NVML state: lazy import + one-time init (no module-level side effects)
        self._pynvml = None
        self._nvml_initialized = False

    # ---------------------------------------------------------------------
    # NVML LAZY IMPORT / INIT (GPU memory - optional)
    # ---------------------------------------------------------------------
    def _get_pynvml(self):
        """Lazy import for NVML (pynvml). Returns module or None if unavailable."""
        if self._pynvml is not None:
            return self._pynvml
        try:
            import pynvml  # local import by design

            self._pynvml = pynvml
            return self._pynvml
        except Exception:
            self._pynvml = None
            return None

    def _ensure_nvml_initialized(self) -> bool:
        """Initialize NVML exactly once per instance. Returns True if ready."""
        pynvml = self._get_pynvml()
        if pynvml is None:
            return False
        if self._nvml_initialized:
            return True
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            return True
        except Exception:
            self._nvml_initialized = False
            return False

    # ---------------------------------------------------------------------
    # GPU MEMORY HELPERS
    # ---------------------------------------------------------------------
    def _read_gpu_bytes(self) -> Optional[int]:
        """
        Read current GPU memory usage in bytes (best-effort).

        Priority:
        1) NVML (pynvml) if available
        2) torch.cuda.memory_allocated() as fallback (process allocator only)

        Returns None if unavailable.
        """
        if self._ensure_nvml_initialized():
            try:
                pynvml = self._pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return int(mem.used)
            except Exception:
                pass

        if torch is not None and torch.cuda.is_available():
            try:
                return int(torch.cuda.memory_allocated())
            except Exception:
                pass

        return None

    def _read_gpu_peak(self) -> Optional[int]:
        """
        Read peak GPU memory (bytes) from PyTorch if available.

        NVML does not provide an allocator peak for the current process.
        torch peak refers to the process allocator peak.
        """
        if torch is not None and torch.cuda.is_available():
            try:
                return int(torch.cuda.max_memory_allocated())
            except Exception:
                return None
        return None

    # ---------------------------------------------------------------------
    # INTERNAL SAMPLING
    # ---------------------------------------------------------------------
    def _sample_rss(self) -> int:
        """Read RSS once (bytes)."""
        return int(self._process.memory_info().rss)

    def _run_sampling_window(
        self,
        duration_s: float,
        interval_s: float,
        t0_perf: float,
        timestamps_raw_s: list[float],
    ) -> None:
        """
        Sample RSS for a fixed duration using a drift-corrected schedule.
        """
        next_tick = time.perf_counter()
        end_time = next_tick + max(0.0, float(duration_s))

        while time.perf_counter() < end_time:
            now_perf = time.perf_counter()
            rss = self._sample_rss()

            self.samples_rss.append(rss)
            timestamps_raw_s.append(float(now_perf - t0_perf))

            if rss > self._rss_peak:
                self._rss_peak = rss

            next_tick += interval_s
            sleep_time = max(0.0, next_tick - time.perf_counter())
            time.sleep(sleep_time)

    # ---------------------------------------------------------------------
    # MAIN MEASUREMENT ROUTINE
    # ---------------------------------------------------------------------
    def measure(
        self,
        fn: Callable[[], Any],
        repeats: int = 1,
        pre_roll_s: float = 5.0,
        post_delay_s: float = 2.0,
        mode: str = "static",
        dynamic_infer_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Measure RSS over time around an inference function.

        Duration semantics
        ------------------
        ``inference_duration_s`` and ``duration_s`` are derived from ``time.perf_counter()``
        and therefore remain stable even if the sampler is coarse or slightly phase-shifted.
        ``timestamps_s`` and ``timestamps_raw_s`` are retained for plots and audits.
        """
        if mode not in ("static", "dynamic"):
            raise ValueError("mode must be 'static' or 'dynamic'")

        repeats = max(1, int(repeats))
        pre_roll_s = max(0.0, float(pre_roll_s))
        post_delay_s = max(0.0, float(post_delay_s))

        if dynamic_infer_s is not None:
            dynamic_infer_s = max(0.0, float(dynamic_infer_s))

        self.samples_rss.clear()
        timestamps_raw_s: list[float] = []

        self._rss_start = self._sample_rss()
        self._rss_peak = self._rss_start

        interval_s = 1.0 / max(float(self.sample_hz), 1.0)
        t0_perf = time.perf_counter()

        infer_start_idx = 0
        infer_end_idx = 0
        infer_start_perf = t0_perf
        infer_end_perf = t0_perf

        if mode == "static":
            self._run_sampling_window(pre_roll_s, interval_s, t0_perf, timestamps_raw_s)

            infer_start_idx = len(self.samples_rss)
            infer_start_perf = time.perf_counter()

            for _ in range(repeats):
                fn()

            infer_end_perf = time.perf_counter()
            inference_duration_s = float(max(0.0, infer_end_perf - infer_start_perf))

            rss_end_inf = self._sample_rss()
            self.samples_rss.append(rss_end_inf)
            timestamps_raw_s.append(float(time.perf_counter() - t0_perf))
            self._rss_peak = max(self._rss_peak, rss_end_inf)

            infer_end_idx = len(self.samples_rss)
            self._run_sampling_window(post_delay_s, interval_s, t0_perf, timestamps_raw_s)

        else:
            self._run_sampling_window(pre_roll_s, interval_s, t0_perf, timestamps_raw_s)

            infer_start_idx = len(self.samples_rss)
            infer_window_s = pre_roll_s if dynamic_infer_s is None else dynamic_infer_s

            infer_start_perf = time.perf_counter()
            infer_end_target = infer_start_perf + infer_window_s
            next_tick = time.perf_counter()

            while time.perf_counter() < infer_end_target:
                fn()

                now_perf = time.perf_counter()
                rss = self._sample_rss()
                self.samples_rss.append(rss)
                timestamps_raw_s.append(float(now_perf - t0_perf))
                self._rss_peak = max(self._rss_peak, rss)

                next_tick += interval_s
                sleep_time = max(0.0, next_tick - time.perf_counter())
                time.sleep(sleep_time)

            infer_end_perf = time.perf_counter()
            inference_duration_s = float(max(0.0, infer_end_perf - infer_start_perf))
            infer_end_idx = len(self.samples_rss)

            self._run_sampling_window(post_delay_s, interval_s, t0_perf, timestamps_raw_s)

        measure_end_perf = time.perf_counter()
        duration_s = float(max(0.0, measure_end_perf - t0_perf))

        infer_start_time_raw_s = float(max(0.0, infer_start_perf - t0_perf))
        infer_end_time_raw_s = float(max(infer_start_time_raw_s, infer_end_perf - t0_perf))

        timestamps_s = [float(ts - infer_start_time_raw_s) for ts in timestamps_raw_s]

        self._rss_end = self._sample_rss()
        rss_delta = int(self._rss_end - self._rss_start)
        rss_array = np.asarray(self.samples_rss, dtype=float)

        if infer_start_idx < len(rss_array):
            slice_end = min(infer_end_idx, len(rss_array))
            rss_slice = rss_array[infer_start_idx:slice_end]
            if rss_slice.size > 0:
                rss_peak_inference = float(np.max(rss_slice))
                rss_delta_inference = float(rss_peak_inference - float(rss_array[infer_start_idx]))
            else:
                rss_peak_inference = float(self._rss_peak)
                rss_delta_inference = 0.0
        else:
            rss_peak_inference = float(self._rss_peak)
            rss_delta_inference = 0.0

        result: Dict[str, Any] = {
            "rss_start_bytes": int(self._rss_start),
            "rss_end_bytes": int(self._rss_end),
            "rss_delta_bytes": int(rss_delta),
            "rss_peak_bytes": int(self._rss_peak),
            "rss_peak_inference_bytes": float(rss_peak_inference),
            "rss_delta_inference_bytes": float(rss_delta_inference),
            "duration_s": float(round(duration_s, 6)),
            "duration_source": "perf_counter",
            "pre_roll_s": float(pre_roll_s),
            "inference_duration_s": float(round(inference_duration_s, 6)),
            "inference_duration_source": "perf_counter",
            "post_delay_s": float(post_delay_s),
            "infer_start_idx": int(infer_start_idx),
            "infer_end_idx": int(infer_end_idx),
            "infer_start_time_raw_s": float(round(infer_start_time_raw_s, 6)),
            "infer_end_time_raw_s": float(round(infer_end_time_raw_s, 6)),
            "infer_start_time_s": 0.0,
            "infer_end_time_s": float(round(inference_duration_s, 6)),
            "rss_samples": list(self.samples_rss),
            "timestamps_raw_s": [float(round(ts, 6)) for ts in timestamps_raw_s],
            "timestamps_s": [float(round(ts, 6)) for ts in timestamps_s],
            "sample_hz": int(self.sample_hz),
            "mode": str(mode),
        }

        if mode == "dynamic" and dynamic_infer_s is not None:
            result["dynamic_infer_s"] = float(dynamic_infer_s)

        gpu_used = self._read_gpu_bytes()
        gpu_peak = self._read_gpu_peak()
        if gpu_used is not None or gpu_peak is not None:
            gpu_block: Dict[str, int] = {}
            if gpu_used is not None:
                gpu_block["used_bytes"] = int(gpu_used)
            if gpu_peak is not None:
                gpu_block["peak_bytes"] = int(gpu_peak)
            result["gpu_memory"] = gpu_block

        return result

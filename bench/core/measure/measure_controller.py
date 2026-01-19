# bench/core/measure/measure_controller.py
from __future__ import annotations
"""
MeasureController

Standardized measurement orchestration for:
- latency timing (warmups + repeats via TimingMeter)
- aligned CPU utilization sampling
- host RAM time series (RSS) with pre-roll and post-delay
- optional Jetson GPU utilization via jtop (best-effort)

Single source of truth for iteration counts:
- run.warmups
- run.repeats
"""

from time import sleep
from typing import Any, Dict, List, Optional

import numpy as np

from bench.core.metrics.timing_meter import TimingMeter
from bench.core.metrics.cpu_meter import CpuMeter
from bench.core.metrics.memory_meter import MemoryMeter
from bench.core.metrics.gpu_meter_jetson import JetsonGpuMeter


def _deep_get(cfg: Dict[str, Any], keys: List[str], default: Any) -> Any:
    """
    Robust config getter:
    - Try nested access (e.g., ["run", "warmups"])
    - Fall back to flat keys for backward compatibility (best-effort)

    This keeps the controller resilient while you migrate schemas.
    """
    cur: Any = cfg
    try:
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        # Flat fallbacks (legacy compatibility)
        flat_candidates = [
            keys[-1],
            "sampler_hz" if keys[-1] == "sampler_hz" else None,
            "pre_roll_s" if keys[-1] == "pre_roll_s" else None,
            "post_delay_s" if keys[-1] == "post_delay_s" else None,
            "warmups" if keys[-1] == "warmups" else None,
            "repeats" if keys[-1] == "repeats" else None,
            "memory_mode" if keys[-1] == "memory_mode" else None,
        ]
        for cand in flat_candidates:
            if cand and cand in cfg:
                return cfg[cand]
        return default


class MeasureController:
    """
    Executes standardized benchmark measurements in a reproducible way.

    Expected runner API:
    - load()
    - prepare(input_spec) -> dummy_input
    - warmup(n, input_spec)
    - infer(dummy_input)
    - teardown()
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # Unified schema: run.warmups / run.repeats
        self.sample_hz = int(_deep_get(cfg, ["metrics", "sampler_hz"], 75))
        self.warmups = int(_deep_get(cfg, ["run", "warmups"], 10))
        self.repeats = int(_deep_get(cfg, ["run", "repeats"], 100))

        # Memory meter configuration
        self.pre_roll_s = float(_deep_get(cfg, ["metrics", "pre_roll_s"], 5.0))
        self.post_delay_s = float(_deep_get(cfg, ["metrics", "post_delay_s"], 2.0))
        self.memory_mode = str(_deep_get(cfg, ["metrics", "memory_mode"], "static")).lower()

        if self.memory_mode not in ("static", "dynamic"):
            raise ValueError("memory_mode must be 'static' or 'dynamic'")

        # Meters
        # IMPORTANT: TimingMeter performs warmups + repeats internally in one call.
        self.timing_meter = TimingMeter(warmups=self.warmups, repeats=self.repeats)
        self.cpu_meter = CpuMeter(sample_hz=self.sample_hz)
        self.mem_meter = MemoryMeter(sample_hz=self.sample_hz)

        # Optional Jetson GPU Meter (best-effort). Use lower hz for slower sampling overhead.
        gpu_hz_cfg = _deep_get(cfg, ["metrics", "gpu_sampler_hz"], min(self.sample_hz, 10))
        self.gpu_meter = JetsonGpuMeter(sample_hz=int(gpu_hz_cfg))

    def run_benchmark(self, runner, dummy_input: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run a full benchmark cycle and return a serializable result dict.

        Returns a top-level structure for existing callers plus an adapter "metrics" block.
        """
        # ------------------------------------------------------------------
        # 1) Prepare dummy input (deterministic input spec)
        # ------------------------------------------------------------------
        if dummy_input is None:
            input_spec = _deep_get(self.cfg, ["input"], {}) or {}
            dummy_input = runner.prepare(input_spec)
        else:
            # If the caller supplies dummy_input, derive input_spec from it
            if isinstance(dummy_input, dict):
                arr = next(iter(dummy_input.values()))
            else:
                arr = dummy_input

            dtype_str = str(getattr(arr, "dtype", "float32"))
            if dtype_str.startswith("torch."):
                dtype_str = dtype_str.replace("torch.", "")

            input_spec = {
                "shape": list(getattr(arr, "shape", [])),
                "dtype": dtype_str,
            }

        # Determine batch size from input shape (best-effort)
        batch_size = 1
        try:
            shp = input_spec.get("shape")
            if isinstance(shp, list) and len(shp) >= 1:
                b = int(shp[0])
                batch_size = b if b > 0 else 1
        except Exception:
            batch_size = 1

        # ------------------------------------------------------------------
        # 2) Runner warmup (explicit)
        # ------------------------------------------------------------------
        # Keep this warmup because runners might need setup / compilation.
        # This warmup is separate from TimingMeter warmups which are focused on timing stability.
        runner.warmup(self.warmups, input_spec)
        sleep(0.2)

        # ------------------------------------------------------------------
        # 3) Latency measurement (single TimingMeter call)
        # ------------------------------------------------------------------
        self.timing_meter.reset()

        def _infer_once() -> None:
            runner.infer(dummy_input)

        timing_stats = self.timing_meter.measure(_infer_once)

        # Normalize for your downstream schema expectations
        stats = {
            "mean": float(timing_stats.get("mean") or 0.0),
            "p50": float(timing_stats.get("p50") or 0.0),
            "p90": float(timing_stats.get("p90") or 0.0),
            "p95": float(timing_stats.get("p95") or 0.0),
            "p99": float(timing_stats.get("p99") or 0.0),
            "samples": list(timing_stats.get("samples") or []),
        }

        # -------------------------------------------------
        # Per-sample normalization
        # -------------------------------------------------
        # The TimingMeter measures latency per inference call. If the input has batch_size > 1,
        # the measured mean corresponds to "ms per batch". For fair cross-hardware/model comparison
        # we also report "ms per sample" by dividing by batch_size.
        stats["batch_size"] = int(batch_size)

        mean_ms = stats["mean"]
        if mean_ms > 0 and batch_size > 0:
            stats["mean_per_sample"] = float(mean_ms / float(batch_size))
        else:
            # Keep explicit None instead of 0.0 when normalization is undefined.
            stats["mean_per_sample"] = None

        throughput_ips = (1000.0 / mean_ms) if mean_ms > 0 else None
        throughput_sps = (throughput_ips * batch_size) if throughput_ips is not None else None

        # ------------------------------------------------------------------
        # 4) Memory + aligned CPU/GPU monitoring (single inference block)
        # ------------------------------------------------------------------
        def _measure_once() -> None:
            runner.infer(dummy_input)

        gpu_block = None
        if self.gpu_meter.available():
            self.gpu_meter.start()

        self.cpu_meter.start()
        try:
            mem_result = self.mem_meter.measure(
                fn=_measure_once,
                repeats=1,
                pre_roll_s=self.pre_roll_s,
                post_delay_s=self.post_delay_s,
                mode=self.memory_mode,
            )
        finally:
            self.cpu_meter.stop()

        if self.gpu_meter.available():
            self.gpu_meter.stop()
            gpu_block = self.gpu_meter.summary()

        cpu_result = self.cpu_meter.summary(
            infer_start_idx=mem_result.get("infer_start_idx", 0),
            infer_end_idx=mem_result.get("infer_end_idx", 0),
        )

        # ------------------------------------------------------------------
        # 5) Duration calculations (consistent exported values)
        # ------------------------------------------------------------------
        timestamps = mem_result.get("timestamps_s") or []
        i0 = int(mem_result.get("infer_start_idx", 0) or 0)
        i1 = int(mem_result.get("infer_end_idx", 0) or 0)

        inference_duration_s = 0.0
        if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
            i0 = max(0, min(i0, len(timestamps) - 1))
            i1 = max(i0, min(i1, len(timestamps) - 1))
            inference_duration_s = float(timestamps[i1] - timestamps[i0])

        duration_s = 0.0
        if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
            duration_s = float(timestamps[-1] - timestamps[0])
        else:
            pre = float(mem_result.get("pre_roll_s", self.pre_roll_s) or 0.0)
            post = float(mem_result.get("post_delay_s", self.post_delay_s) or 0.0)
            total = pre + float(inference_duration_s) + post
            duration_s = float(total) if total > 0 else 0.0

        # Optional strict consistency check
        strict = bool(_deep_get(self.cfg, ["run", "strict_metrics"], False))
        meter_duration = float(mem_result.get("duration_s", 0.0) or 0.0)
        if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
            ts_dur = float(timestamps[-1] - timestamps[0])
            if meter_duration > 0 and abs(ts_dur - meter_duration) > 0.25:
                msg = (
                    "[WARN] memory duration mismatch: "
                    f"from_timestamps={ts_dur:.3f}s vs meter_duration={meter_duration:.3f}s"
                )
                if strict:
                    raise ValueError(msg)
                print(msg)

        # -------------------------------------------------
        # Average RAM usage (overall and inference window)
        # -------------------------------------------------
        rss_samples = mem_result.get("rss_samples") or []
        infer_start = int(mem_result.get("infer_start_idx", 0) or 0)
        infer_end = int(mem_result.get("infer_end_idx", 0) or 0)

        rss_mean_bytes = None
        rss_mean_inference_bytes = None

        if isinstance(rss_samples, (list, tuple)) and len(rss_samples) > 0:
            rss_mean_bytes = float(np.mean(rss_samples))

            # Mean RAM during inference window only
            if 0 <= infer_start < infer_end <= len(rss_samples):
                window = rss_samples[infer_start:infer_end]
                if len(window) > 0:
                    rss_mean_inference_bytes = float(np.mean(window))
        # -------------------------------------------------
        # Derived, model-specific runtime memory metrics
        # -------------------------------------------------
        # Baseline is estimated from the pre-roll window (samples before inference starts).
        # This is more stable than using a single rss_start_bytes snapshot.
        baseline_mean_bytes = None
        model_specific_runtime_memory_bytes = None

        if isinstance(rss_samples, (list, tuple)) and len(rss_samples) > 0:
            pre_end = max(0, min(infer_start, len(rss_samples)))
            pre_window = rss_samples[:pre_end]
            if len(pre_window) > 0:
                baseline_mean_bytes = float(np.mean(pre_window))

            # "Model-specific runtime memory" is the additional RAM during inference above baseline.
            if baseline_mean_bytes is not None and rss_mean_inference_bytes is not None:
                delta = float(rss_mean_inference_bytes - baseline_mean_bytes)
                model_specific_runtime_memory_bytes = float(max(0.0, delta))

        # Safety margin for edge sizing (default 15%). Configurable via metrics.memory_safety_margin_pct.
        safety_margin_pct = float(_deep_get(self.cfg, ["metrics", "memory_safety_margin_pct"], 15.0))
        peak_infer = float(mem_result.get("rss_peak_inference_bytes", 0.0) or 0.0)

        minimal_required_ram_bytes = None
        if peak_infer > 0:
            minimal_required_ram_bytes = float(peak_infer * (1.0 + safety_margin_pct / 100.0))

        # Human-readable recommendation (optional, useful for reports)
        memory_recommendation = None
        if minimal_required_ram_bytes is not None:
            gib = minimal_required_ram_bytes / (1024.0 ** 3)
            memory_recommendation = (
                f"Recommended minimum RAM: {gib:.2f} GiB "
                f"(peak_inference_rss + {safety_margin_pct:.1f}% safety margin)."
            )


        # ------------------------------------------------------------------
        # 6) Compact memory block (stable contract for JSON/schema)
        # ------------------------------------------------------------------
        memory_block = {
            "rss_start_bytes": float(mem_result.get("rss_start_bytes", 0.0)),
            "rss_end_bytes": float(mem_result.get("rss_end_bytes", 0.0)),
            "rss_peak_bytes": float(mem_result.get("rss_peak_bytes", 0.0)),
            "rss_delta_bytes": float(mem_result.get("rss_delta_bytes", 0.0)),
            "rss_peak_inference_bytes": float(mem_result.get("rss_peak_inference_bytes", 0.0)),
            "rss_delta_inference_bytes": float(mem_result.get("rss_delta_inference_bytes", 0.0)),
            # NEW: average RAM usage
            "rss_mean_bytes": rss_mean_bytes,
            "rss_mean_inference_bytes": rss_mean_inference_bytes,
            # NEW: explicit, model-specific runtime memory requirements
            "rss_baseline_mean_bytes": baseline_mean_bytes,
            "model_specific_runtime_memory_bytes": model_specific_runtime_memory_bytes,
            # NEW: edge sizing recommendation (peak + safety margin)
            "memory_safety_margin_pct": float(safety_margin_pct),
            "minimal_required_ram_bytes": minimal_required_ram_bytes,
            "memory_recommendation": memory_recommendation, 
            "pre_roll_s": float(mem_result.get("pre_roll_s", self.pre_roll_s)),
            "post_delay_s": float(mem_result.get("post_delay_s", self.post_delay_s)),
            "inference_duration_s": float(inference_duration_s),
            "duration_s": float(duration_s),
            "infer_start_idx": int(mem_result.get("infer_start_idx", 0)),
            "infer_end_idx": int(mem_result.get("infer_end_idx", 0)),
            "timestamps_s": mem_result.get("timestamps_s", []),
            "rss_samples": mem_result.get("rss_samples", []),
            "sample_hz": float(mem_result.get("sample_hz", self.sample_hz)),
            "mode": mem_result.get("mode", self.memory_mode),
        }

        # GPU utilization summary (if available)
        gpu_util = None
        if isinstance(gpu_block, dict):
            gpu_util = gpu_block.get("gpu_utilization_pct")

        # ------------------------------------------------------------------
        # 7) Return result dict (top-level + adapter block)
        # ------------------------------------------------------------------
        return {
            # Top-level keys used by existing callers (e.g., bench/core/main.py)
            "timing_ms": stats,
            "throughput_bps": throughput_ips,
            "throughput_sps": throughput_sps,
            "cpu_utilization_pct": cpu_result,
            "gpu_utilization_pct": gpu_util,
            "jetson_gpu": gpu_block,
            "memory": memory_block,

            # Adapter block (tests / schemas can rely on this)
            "metrics": {
                "inference_time_ms": stats,
                "throughput_sps": throughput_sps,
                "cpu_utilization_pct": cpu_result,
                "gpu_utilization_pct": gpu_util,
                "memory": memory_block,
            },
        }

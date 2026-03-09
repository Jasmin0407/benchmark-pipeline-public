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

import os
from time import sleep
from typing import Any, Dict, List, Optional

import numpy as np

from bench.core.metrics.cpu_meter import CpuMeter
from bench.core.metrics.gpu_meter_jetson import JetsonGpuMeter
from bench.core.metrics.memory_meter import MemoryMeter
from bench.core.metrics.timing_meter import TimingMeter


THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _deep_get(cfg: Dict[str, Any], keys: List[str], default: Any) -> Any:
    """Robust config getter with nested-first and legacy flat-key fallback."""
    cur: Any = cfg
    try:
        for key in keys:
            cur = cur[key]
        return cur
    except Exception:
        flat_candidates = [
            keys[-1],
            "sampler_hz" if keys[-1] == "sampler_hz" else None,
            "pre_roll_s" if keys[-1] == "pre_roll_s" else None,
            "post_delay_s" if keys[-1] == "post_delay_s" else None,
            "warmups" if keys[-1] == "warmups" else None,
            "repeats" if keys[-1] == "repeats" else None,
            "memory_mode" if keys[-1] == "memory_mode" else None,
        ]
        for candidate in flat_candidates:
            if candidate and candidate in cfg:
                return cfg[candidate]
        return default


class MeasureController:
    """Execute standardized benchmark measurements in a reproducible way."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.sample_hz = int(_deep_get(cfg, ["metrics", "sampler_hz"], 75))
        self.warmups = int(_deep_get(cfg, ["run", "warmups"], 10))
        self.repeats = int(_deep_get(cfg, ["run", "repeats"], 100))

        self.pre_roll_s = float(_deep_get(cfg, ["metrics", "pre_roll_s"], 5.0))
        self.post_delay_s = float(_deep_get(cfg, ["metrics", "post_delay_s"], 2.0))
        self.memory_mode = str(_deep_get(cfg, ["metrics", "memory_mode"], "static")).lower()
        dyn_infer = _deep_get(cfg, ["metrics", "dynamic_infer_s"], None)
        self.dynamic_infer_s = None if dyn_infer is None else float(dyn_infer)

        if self.memory_mode not in ("static", "dynamic"):
            raise ValueError("memory_mode must be 'static' or 'dynamic'")

        self.timing_meter = TimingMeter(warmups=self.warmups, repeats=self.repeats)
        self.cpu_meter = CpuMeter(sample_hz=self.sample_hz)
        self.mem_meter = MemoryMeter(sample_hz=self.sample_hz)

        gpu_hz_cfg = _deep_get(cfg, ["metrics", "gpu_sampler_hz"], min(self.sample_hz, 10))
        self.gpu_meter = JetsonGpuMeter(sample_hz=int(gpu_hz_cfg))

    def _resolve_duration_fields(self, mem_result: Dict[str, Any]) -> Dict[str, Any]:
        timestamps = mem_result.get("timestamps_s") or []
        i0 = int(mem_result.get("infer_start_idx", 0) or 0)
        i1 = int(mem_result.get("infer_end_idx", 0) or 0)

        inference_duration_s = float(mem_result.get("inference_duration_s", 0.0) or 0.0)
        inference_duration_source = str(
            mem_result.get("inference_duration_source") or ("perf_counter" if inference_duration_s > 0 else "")
        )

        duration_s = float(mem_result.get("duration_s", 0.0) or 0.0)
        duration_source = str(mem_result.get("duration_source") or ("perf_counter" if duration_s > 0 else ""))

        if inference_duration_s <= 0.0 and isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
            i0 = max(0, min(i0, len(timestamps) - 1))
            i1 = max(i0, min(i1, len(timestamps) - 1))
            inference_duration_s = float(max(0.0, timestamps[i1] - timestamps[i0]))
            inference_duration_source = "timestamps"

        if duration_s <= 0.0:
            if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
                duration_s = float(max(0.0, timestamps[-1] - timestamps[0]))
                duration_source = "timestamps"
            else:
                pre = float(mem_result.get("pre_roll_s", self.pre_roll_s) or 0.0)
                post = float(mem_result.get("post_delay_s", self.post_delay_s) or 0.0)
                duration_s = float(max(0.0, pre + inference_duration_s + post))
                duration_source = "fallback_sum"

        return {
            "inference_duration_s": float(inference_duration_s),
            "inference_duration_source": inference_duration_source or "unknown",
            "duration_s": float(duration_s),
            "duration_source": duration_source or "unknown",
        }

    def _build_thread_config(self, runner: Any) -> Dict[str, Any]:
        requested_threads_raw = _deep_get(self.cfg, ["run", "threads"], None)
        requested_threads = None if requested_threads_raw in (None, "") else int(requested_threads_raw)

        requested_inter_op_raw = _deep_get(self.cfg, ["run", "inter_op_threads"], None)
        requested_inter_op = None if requested_inter_op_raw in (None, "") else int(requested_inter_op_raw)

        thread_env = {key: os.environ.get(key) for key in THREAD_ENV_KEYS}
        runner_thread_audit = getattr(runner, "get_thread_audit", None)
        runner_audit = runner_thread_audit() if callable(runner_thread_audit) else {}

        note = (
            "Requested thread settings describe intended backend configuration. "
            "Observed CPU parallelism may be higher or lower depending on runtime behavior, "
            "kernel libraries, provider internals, and environment variables."
        )

        return {
            "requested_threads": requested_threads,
            "requested_inter_op_threads": requested_inter_op,
            "backend_thread_control_supported": runner_audit.get("backend_thread_control_supported", False),
            "applied_intra_op_threads": runner_audit.get("applied_intra_op_threads"),
            "applied_inter_op_threads": runner_audit.get("applied_inter_op_threads"),
            "execution_mode": runner_audit.get("execution_mode"),
            "thread_env": thread_env,
            "provider_chain": runner_audit.get("provider_chain"),
            "active_provider": runner_audit.get("active_provider"),
            "fallback_occurred": runner_audit.get("fallback_occurred"),
            "note": note,
        }

    @staticmethod
    def _build_cpu_audit_consistency_note(cpu_result: Dict[str, Any]) -> str:
        cpu_time = cpu_result.get("cpu_time") or {}
        cpu_core_util = cpu_time.get("cpu_core_util")
        sampled_cores = cpu_result.get("core_util_mean_cores")

        if cpu_core_util is None or sampled_cores is None:
            return "insufficient_data"

        diff = abs(float(sampled_cores) - float(cpu_core_util))
        if diff <= 0.35:
            return "plausible_agreement"
        if diff <= 1.0:
            return "mild_deviation"
        return "investigate_mismatch"

    def run_benchmark(self, runner, dummy_input: Optional[Any] = None) -> Dict[str, Any]:
        if dummy_input is None:
            input_spec = _deep_get(self.cfg, ["input"], {}) or {}
            dummy_input = runner.prepare(input_spec)
        else:
            if isinstance(dummy_input, dict):
                arr = next(iter(dummy_input.values()))
            else:
                arr = dummy_input

            dtype_str = str(getattr(arr, "dtype", "float32"))
            if dtype_str.startswith("torch."):
                dtype_str = dtype_str.replace("torch.", "")

            input_spec = {"shape": list(getattr(arr, "shape", [])), "dtype": dtype_str}

        batch_size = 1
        try:
            shape = input_spec.get("shape")
            if isinstance(shape, list) and len(shape) >= 1:
                batch_size = max(1, int(shape[0]))
        except Exception:
            batch_size = 1

        actual_input_shape: Optional[List[int]] = None
        input_num_samples: Optional[int] = None

        try:
            arr_any = next(iter(dummy_input.values())) if isinstance(dummy_input, dict) and dummy_input else dummy_input
            shape_any = getattr(arr_any, "shape", None)
            if shape_any is not None:
                actual_input_shape = list(shape_any)
                if len(actual_input_shape) >= 1:
                    samples_axis = -1
                    try:
                        inp_cfg = _deep_get(self.cfg, ["input"], {}) or {}
                        if isinstance(inp_cfg, dict):
                            samples_axis = int(inp_cfg.get("samples_axis", -1))
                    except Exception:
                        samples_axis = -1

                    if samples_axis < 0:
                        samples_axis = len(actual_input_shape) + samples_axis

                    if 0 <= samples_axis < len(actual_input_shape):
                        n_dim = actual_input_shape[samples_axis]
                        if isinstance(n_dim, (int, np.integer)) and int(n_dim) > 0:
                            input_num_samples = int(n_dim)
        except Exception:
            actual_input_shape = None
            input_num_samples = None

        if actual_input_shape is None:
            shape_cfg = input_spec.get("shape")
            if isinstance(shape_cfg, list) and shape_cfg:
                actual_input_shape = [int(x) for x in shape_cfg if isinstance(x, (int, float, np.integer))]

        fs_hz: Optional[float] = None
        try:
            fs_val = None
            inp_cfg = _deep_get(self.cfg, ["input"], {}) or {}
            if isinstance(inp_cfg, dict):
                fs_val = inp_cfg.get("fs_hz")
                if fs_val is None and isinstance(inp_cfg.get("signal"), dict):
                    fs_val = inp_cfg["signal"].get("fs_hz")
            if fs_val is not None:
                fs_hz = float(fs_val)
        except Exception:
            fs_hz = None

        input_duration_s: Optional[float] = None
        if input_num_samples is not None and fs_hz is not None and fs_hz > 0:
            input_duration_s = float(input_num_samples) / float(fs_hz)

        runner.warmup(self.warmups, input_spec)
        sleep(0.2)

        self.timing_meter.reset()

        def _infer_once() -> None:
            runner.infer(dummy_input)

        timing_stats = self.timing_meter.measure(_infer_once)
        stats = {
            "mean": float(timing_stats.get("mean") or 0.0),
            "p50": float(timing_stats.get("p50") or 0.0),
            "p90": float(timing_stats.get("p90") or 0.0),
            "p95": float(timing_stats.get("p95") or 0.0),
            "p99": float(timing_stats.get("p99") or 0.0),
            "samples": list(timing_stats.get("samples") or []),
        }
        stats["batch_size"] = int(batch_size)

        mean_ms = stats["mean"]
        stats["mean_per_sample"] = float(mean_ms / float(batch_size)) if mean_ms > 0 and batch_size > 0 else None
        throughput_ips = (1000.0 / mean_ms) if mean_ms > 0 else None
        throughput_sps = (throughput_ips * batch_size) if throughput_ips is not None else None

        ms_per_signal_s: Optional[float] = None
        if input_duration_s is not None and input_duration_s > 0 and mean_ms > 0:
            ms_per_signal_s = float(mean_ms) / float(input_duration_s)

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
                dynamic_infer_s=self.dynamic_infer_s,
            )
        finally:
            self.cpu_meter.stop()

        if self.gpu_meter.available():
            self.gpu_meter.stop()
            gpu_block = self.gpu_meter.summary()

        cpu_result = self.cpu_meter.summary(
            infer_start_idx=mem_result.get("infer_start_idx", 0),
            infer_end_idx=mem_result.get("infer_end_idx", 0),
            scope="inference_window",
            scope_label="cpu_inference_window",
        )
        cpu_result["cpu_audit_consistency_note"] = self._build_cpu_audit_consistency_note(cpu_result)

        duration_fields = self._resolve_duration_fields(mem_result)

        strict = bool(_deep_get(self.cfg, ["run", "strict_metrics"], False))
        timestamps = mem_result.get("timestamps_s") or []
        meter_duration = float(mem_result.get("duration_s", 0.0) or 0.0)
        if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
            ts_dur = float(max(0.0, timestamps[-1] - timestamps[0]))
            if meter_duration > 0 and abs(ts_dur - meter_duration) > 0.25:
                msg = (
                    "[WARN] memory duration mismatch: "
                    f"from_timestamps={ts_dur:.3f}s vs meter_duration={meter_duration:.3f}s"
                )
                if strict:
                    raise ValueError(msg)
                print(msg)

        rss_samples = mem_result.get("rss_samples") or []
        infer_start = int(mem_result.get("infer_start_idx", 0) or 0)
        infer_end = int(mem_result.get("infer_end_idx", 0) or 0)

        rss_mean_bytes = None
        rss_mean_inference_bytes = None
        if isinstance(rss_samples, (list, tuple)) and len(rss_samples) > 0:
            rss_mean_bytes = float(np.mean(rss_samples))
            if 0 <= infer_start < infer_end <= len(rss_samples):
                window = rss_samples[infer_start:infer_end]
                if len(window) > 0:
                    rss_mean_inference_bytes = float(np.mean(window))

        baseline_mean_bytes = None
        model_specific_runtime_memory_bytes = None
        if isinstance(rss_samples, (list, tuple)) and len(rss_samples) > 0:
            pre_end = max(0, min(infer_start, len(rss_samples)))
            pre_window = rss_samples[:pre_end]
            if len(pre_window) > 0:
                baseline_mean_bytes = float(np.mean(pre_window))
            if baseline_mean_bytes is not None and rss_mean_inference_bytes is not None:
                delta = float(rss_mean_inference_bytes - baseline_mean_bytes)
                model_specific_runtime_memory_bytes = float(max(0.0, delta))

        safety_margin_pct = float(_deep_get(self.cfg, ["metrics", "memory_safety_margin_pct"], 15.0))
        peak_infer = float(mem_result.get("rss_peak_inference_bytes", 0.0) or 0.0)
        observed_peak_process_rss_bytes = float(mem_result.get("rss_peak_bytes", 0.0) or 0.0)

        minimal_required_ram_bytes = None
        if peak_infer > 0:
            minimal_required_ram_bytes = float(peak_infer * (1.0 + safety_margin_pct / 100.0))

        runtime_overhead_estimate_bytes = None
        if observed_peak_process_rss_bytes > 0 and model_specific_runtime_memory_bytes is not None:
            runtime_overhead_estimate_bytes = float(
                max(0.0, observed_peak_process_rss_bytes - model_specific_runtime_memory_bytes)
            )

        memory_recommendation_scope = "observed_process_runtime"
        memory_recommendation = None
        if minimal_required_ram_bytes is not None:
            gib = minimal_required_ram_bytes / (1024.0 ** 3)
            memory_recommendation = (
                f"Recommended empirical host RAM budget: {gib:.2f} GiB "
                f"(observed inference-window RSS with {safety_margin_pct:.1f}% safety margin; "
                "includes runtime and host-process overhead, not a hard model minimum)."
            )

        memory_interpretation_note = (
            "Process RSS metrics describe empirical benchmark-process memory observations. "
            "They include runtime/framework overhead and should not be interpreted as a pure model-memory requirement."
        )

        memory_block = {
            "rss_start_bytes": float(mem_result.get("rss_start_bytes", 0.0)),
            "rss_end_bytes": float(mem_result.get("rss_end_bytes", 0.0)),
            "rss_peak_bytes": observed_peak_process_rss_bytes,
            "rss_delta_bytes": float(mem_result.get("rss_delta_bytes", 0.0)),
            "rss_peak_inference_bytes": float(mem_result.get("rss_peak_inference_bytes", 0.0)),
            "rss_delta_inference_bytes": float(mem_result.get("rss_delta_inference_bytes", 0.0)),
            "rss_mean_bytes": rss_mean_bytes,
            "rss_mean_inference_bytes": rss_mean_inference_bytes,
            "rss_baseline_mean_bytes": baseline_mean_bytes,
            "model_specific_runtime_memory_bytes": model_specific_runtime_memory_bytes,
            "runtime_overhead_estimate_bytes": runtime_overhead_estimate_bytes,
            "memory_safety_margin_pct": float(safety_margin_pct),
            "minimal_required_ram_bytes": minimal_required_ram_bytes,
            "memory_recommendation_scope": memory_recommendation_scope,
            "memory_recommendation": memory_recommendation,
            "memory_interpretation_note": memory_interpretation_note,
            "observed_peak_process_rss_bytes": observed_peak_process_rss_bytes,
            "observed_ram_estimate_bytes": minimal_required_ram_bytes,
            "pre_roll_s": float(mem_result.get("pre_roll_s", self.pre_roll_s)),
            "post_delay_s": float(mem_result.get("post_delay_s", self.post_delay_s)),
            "inference_duration_s": float(duration_fields["inference_duration_s"]),
            "inference_duration_source": duration_fields["inference_duration_source"],
            "duration_s": float(duration_fields["duration_s"]),
            "duration_source": duration_fields["duration_source"],
            "infer_start_idx": int(mem_result.get("infer_start_idx", 0)),
            "infer_end_idx": int(mem_result.get("infer_end_idx", 0)),
            "infer_start_time_raw_s": mem_result.get("infer_start_time_raw_s"),
            "infer_end_time_raw_s": mem_result.get("infer_end_time_raw_s"),
            "infer_start_time_s": mem_result.get("infer_start_time_s"),
            "infer_end_time_s": mem_result.get("infer_end_time_s"),
            "timestamps_raw_s": mem_result.get("timestamps_raw_s", []),
            "timestamps_s": mem_result.get("timestamps_s", []),
            "rss_samples": mem_result.get("rss_samples", []),
            "sample_hz": float(mem_result.get("sample_hz", self.sample_hz)),
            "mode": mem_result.get("mode", self.memory_mode),
        }

        gpu_util = gpu_block.get("gpu_utilization_pct") if isinstance(gpu_block, dict) else None
        thread_config = self._build_thread_config(runner)

        return {
            "actual_input_shape": actual_input_shape,
            "input_num_samples": input_num_samples,
            "input_fs_hz": fs_hz,
            "input_duration_s": input_duration_s,
            "ms_per_signal_s": ms_per_signal_s,
            "timing_ms": stats,
            "throughput_bps": throughput_ips,
            "throughput_sps": throughput_sps,
            "cpu_utilization_pct": cpu_result,
            "gpu_utilization_pct": gpu_util,
            "jetson_gpu": gpu_block,
            "memory": memory_block,
            "thread_config": thread_config,
            "metrics": {
                "inference_time_ms": stats,
                "ms_per_signal_s": ms_per_signal_s,
                "throughput_sps": throughput_sps,
                "cpu_utilization_pct": cpu_result,
                "gpu_utilization_pct": gpu_util,
                "memory": memory_block,
            },
        }

# bench/core/analysis/viz/report_markdown.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from bench.core.utils.path_sanitizer import sanitize_device_key


THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _is_single_run_payload(results: Dict[str, Any]) -> bool:
    return isinstance(results, dict) and "meta" in results and "metrics" in results and "model" in results


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _fmt_mb_from_bytes(b: Any) -> str:
    if b is None:
        return "-"
    try:
        return f"{float(b) / 1_000_000:.3f}"
    except Exception:
        return "-"


def _fmt_gb_from_bytes(b: Any) -> str:
    if b is None:
        return "-"
    try:
        return f"{float(b) / 1_000_000_000:.3f}"
    except Exception:
        return "-"


def _fmt_cores_equiv(aggregate_pct: Any) -> str:
    if aggregate_pct is None:
        return "-"
    try:
        return f"{float(aggregate_pct) / 100.0:.2f}"
    except Exception:
        return "-"


def _safe_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _extract_core_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    inf = res.get("metrics", {}).get("inference_time_ms") or res.get("inference_time_ms") or {}
    cpu = res.get("metrics", {}).get("cpu_utilization_pct") or res.get("cpu_utilization_pct") or {}
    mem = res.get("metrics", {}).get("memory") or res.get("memory") or {}
    thread_cfg = res.get("thread_config") or {}
    cpu_time = cpu.get("cpu_time") or {}

    return {
        "inf_mean_ms": inf.get("mean"),
        "inf_p50_ms": inf.get("p50"),
        "inf_p90_ms": inf.get("p90"),
        "inf_p95_ms": inf.get("p95"),
        "inf_p99_ms": inf.get("p99"),
        "inf_batch_size": inf.get("batch_size"),
        "inf_mean_per_sample_ms": inf.get("mean_per_sample"),
        "throughput_sps": res.get("metrics", {}).get("throughput_sps") or res.get("throughput_sps"),
        "ms_per_signal_s": res.get("metrics", {}).get("ms_per_signal_s") or res.get("ms_per_signal_s"),
        "cpu_mean_pct": cpu.get("mean"),
        "cpu_p95_pct": cpu.get("p95"),
        "cpu_mean_cores": cpu.get("core_util_mean_cores"),
        "cpu_p95_cores": cpu.get("core_util_p95_cores"),
        "cpu_wall_s": cpu_time.get("wall_s"),
        "cpu_process_s": cpu_time.get("cpu_s"),
        "cpu_time_based_core_util": cpu_time.get("cpu_core_util"),
        "cpu_audit_consistency_note": cpu.get("cpu_audit_consistency_note"),
        "rss_peak_infer_bytes": mem.get("rss_peak_inference_bytes"),
        "rss_peak_bytes": mem.get("rss_peak_bytes"),
        "rss_baseline_mean_bytes": mem.get("rss_baseline_mean_bytes"),
        "rss_mean_infer_bytes": mem.get("rss_mean_inference_bytes"),
        "model_specific_runtime_memory_bytes": mem.get("model_specific_runtime_memory_bytes"),
        "runtime_overhead_estimate_bytes": mem.get("runtime_overhead_estimate_bytes"),
        "memory_safety_margin_pct": mem.get("memory_safety_margin_pct"),
        "minimal_required_ram_bytes": mem.get("minimal_required_ram_bytes"),
        "memory_recommendation_scope": mem.get("memory_recommendation_scope"),
        "memory_recommendation": mem.get("memory_recommendation"),
        "memory_interpretation_note": mem.get("memory_interpretation_note"),
        "requested_threads": thread_cfg.get("requested_threads"),
        "backend_thread_control_supported": thread_cfg.get("backend_thread_control_supported"),
        "applied_intra_op_threads": thread_cfg.get("applied_intra_op_threads"),
        "applied_inter_op_threads": thread_cfg.get("applied_inter_op_threads"),
        "execution_mode": thread_cfg.get("execution_mode"),
        "thread_env": thread_cfg.get("thread_env") or {},
    }


def _append_metric_sections(lines: List[str], results: Dict[str, Any]) -> None:
    model = results.get("model", {}) or {}
    core = _extract_core_metrics(results)

    lines.append("\n## Metrics\n")

    lines.append("### Latency\n")
    lines.append(f"- **Inference mean (ms)**: `{_fmt(core.get('inf_mean_ms'))}`")
    lines.append(f"- **Inference p50 (ms)**: `{_fmt(core.get('inf_p50_ms'))}`")
    lines.append(f"- **Inference p90 (ms)**: `{_fmt(core.get('inf_p90_ms'))}`")
    lines.append(f"- **Inference p95 (ms)**: `{_fmt(core.get('inf_p95_ms'))}`")
    lines.append(f"- **Inference p99 (ms)**: `{_fmt(core.get('inf_p99_ms'))}`")
    lines.append(f"- **Inference batch size**: `{_fmt(core.get('inf_batch_size'))}`")
    lines.append(f"- **Inference mean per sample (ms)**: `{_fmt(core.get('inf_mean_per_sample_ms'))}`")

    lines.append("\n### Throughput\n")
    lines.append(f"- **Throughput (samples/s)**: `{_fmt(core.get('throughput_sps'))}`")
    if core.get("ms_per_signal_s") is not None:
        lines.append(f"- **ms per second of signal**: `{_fmt(core.get('ms_per_signal_s'))}`")
        lines.append("- Note: This value is derived from the measured inference mean and the runtime-derived input duration.")

    lines.append("\n### CPU\n")
    lines.append(f"- **Sample-based mean CPU utilization (aggregate %)**: `{_fmt(core.get('cpu_mean_pct'))}`")
    lines.append(f"- **Sample-based p95 CPU utilization (aggregate %)**: `{_fmt(core.get('cpu_p95_pct'))}`")
    lines.append(f"- **Effective mean cores (sample-based)**: `{_fmt(core.get('cpu_mean_cores') or _fmt_cores_equiv(core.get('cpu_mean_pct')))}`")
    lines.append(f"- **Effective p95 cores (sample-based)**: `{_fmt(core.get('cpu_p95_cores') or _fmt_cores_equiv(core.get('cpu_p95_pct')))}`")
    lines.append(f"- **CPU wall time (audit, s)**: `{_fmt(core.get('cpu_wall_s'))}`")
    lines.append(f"- **CPU process time (audit, s)**: `{_fmt(core.get('cpu_process_s'))}`")
    lines.append(f"- **CPU core utilization (time-based audit)**: `{_fmt(core.get('cpu_time_based_core_util'))}`")
    lines.append(f"- **CPU audit consistency**: `{_fmt(core.get('cpu_audit_consistency_note'))}`")
    lines.append("- Note: Sample-based CPU values reflect observed CPU load samples over the benchmark window.")
    lines.append("- Note: Time-based CPU audit reflects accumulated process CPU time over wall time.")
    if core.get("requested_threads") is not None or core.get("backend_thread_control_supported") is not None:
        lines.append("\n#### Thread configuration\n")
        lines.append(f"- **Requested threads**: `{_fmt(core.get('requested_threads'))}`")
        lines.append(f"- **Backend thread control supported**: `{_fmt(core.get('backend_thread_control_supported'))}`")
        lines.append(f"- **Applied intra-op threads**: `{_fmt(core.get('applied_intra_op_threads'))}`")
        lines.append(f"- **Applied inter-op threads**: `{_fmt(core.get('applied_inter_op_threads'))}`")
        lines.append(f"- **Execution mode**: `{_fmt(core.get('execution_mode'))}`")
        thread_env = core.get("thread_env") or {}
        for key in THREAD_ENV_KEYS:
            if key in thread_env:
                lines.append(f"- **{key}**: `{_fmt(thread_env.get(key))}`")
        lines.append("- Note: Requested thread settings may differ from observed effective CPU parallelism depending on backend/runtime behavior.")

    lines.append("\n### Memory\n")
    lines.append(f"- **Model size on disk (MB)**: `{_fmt_mb_from_bytes(model.get('size_on_disk_bytes'))}`")
    lines.append(f"- **Observed peak process RSS (MB)**: `{_fmt_mb_from_bytes(core.get('rss_peak_bytes'))}`")
    lines.append(f"- **Inference-window RSS peak (MB)**: `{_fmt_mb_from_bytes(core.get('rss_peak_infer_bytes'))}`")
    lines.append(f"- **Baseline RSS mean (MB)**: `{_fmt_mb_from_bytes(core.get('rss_baseline_mean_bytes'))}`")
    lines.append(f"- **RSS mean during inference (MB)**: `{_fmt_mb_from_bytes(core.get('rss_mean_infer_bytes'))}`")
    lines.append(f"- **Model-specific runtime memory (MB)**: `{_fmt_mb_from_bytes(core.get('model_specific_runtime_memory_bytes'))}`")
    lines.append(f"- **Runtime overhead estimate (MB)**: `{_fmt_mb_from_bytes(core.get('runtime_overhead_estimate_bytes'))}`")
    lines.append(f"- **Memory safety margin (%)**: `{_fmt(core.get('memory_safety_margin_pct'))}`")
    lines.append(f"- **Recommended empirical RAM (MB)**: `{_fmt_mb_from_bytes(core.get('minimal_required_ram_bytes'))}`")
    lines.append(f"- **Recommended empirical RAM (GB)**: `{_fmt_gb_from_bytes(core.get('minimal_required_ram_bytes'))}`")
    lines.append(f"- **Memory recommendation scope**: `{_fmt(core.get('memory_recommendation_scope'))}`")
    if core.get("memory_recommendation") is not None:
        lines.append(f"- **Memory recommendation**: `{_fmt(core.get('memory_recommendation'))}`")
    if core.get("memory_interpretation_note") is not None:
        lines.append(f"- **Memory interpretation note**: `{_fmt(core.get('memory_interpretation_note'))}`")

    lines.append("\n### Method notes\n")
    lines.append("- CPU values above 100% indicate multi-core utilization.")
    lines.append("- Process RSS includes runtime overhead and should not be interpreted as pure model memory.")
    lines.append("- Recommended RAM is an empirical process-level estimate with safety margin.")

    lines.append("\n### Interpretation\n")
    lines.append(
        f"- Observed performance: mean latency `{_fmt(core.get('inf_mean_ms'))}` ms and throughput `{_fmt(core.get('throughput_sps'))}` samples/s."
    )
    lines.append(
        f"- CPU parallelism: sample-based mean `{_fmt(core.get('cpu_mean_pct'))}`% "
        f"(about `{_fmt(core.get('cpu_mean_cores') or _fmt_cores_equiv(core.get('cpu_mean_pct')))}` cores) "
        f"with audit status `{_fmt(core.get('cpu_audit_consistency_note'))}`."
    )
    lines.append(
        f"- Memory interpretation: process RSS peak `{_fmt_mb_from_bytes(core.get('rss_peak_bytes'))}` MB versus "
        f"model-specific runtime delta `{_fmt_mb_from_bytes(core.get('model_specific_runtime_memory_bytes'))}` MB."
    )


def _append_common_sections(lines: List[str], results: Dict[str, Any]) -> None:
    meta = results.get("meta", {}) or {}
    model = results.get("model", {}) or {}
    cfg = results.get("config", {}) or {}
    hw = results.get("hardware", {}) or {}
    env = results.get("env", {}) or {}

    lines.append("## Run\n")
    lines.append(f"- **Framework**: `{_fmt(meta.get('framework'))}`")
    lines.append(f"- **Device**: `{_fmt(meta.get('device_target'))}`")
    if meta.get("timestamp"):
        lines.append(f"- **Timestamp**: `{_fmt(meta.get('timestamp'))}`")
    if meta.get("run_id"):
        lines.append(f"- **Run ID**: `{_fmt(meta.get('run_id'))}`")

    lines.append("\n## Model\n")
    if model.get("path"):
        lines.append(f"- **Path**: `{_fmt(model.get('path'))}`")
    if model.get("input_shape") is not None:
        lines.append(f"- **Input shape**: `{_fmt(model.get('input_shape'))}`")
    if model.get("input_duration_s") is not None:
        lines.append(f"- **Input duration (s)**: `{_fmt(model.get('input_duration_s'))}`")
    if model.get("fs_hz") is not None:
        lines.append(f"- **Sampling rate (Hz)**: `{_fmt(model.get('fs_hz'))}`")
    if model.get("input_num_samples") is not None:
        lines.append(f"- **Input samples**: `{_fmt(model.get('input_num_samples'))}`")
    if model.get("dtype"):
        lines.append(f"- **DType**: `{_fmt(model.get('dtype'))}`")
    if model.get("parameters") is not None:
        lines.append(f"- **Parameters**: `{_fmt(model.get('parameters'))}`")

    _append_metric_sections(lines, results)

    lines.append("\n## Config\n")
    if cfg.get("input"):
        lines.append(f"- **Input**: `{_fmt(cfg.get('input'))}`")
    if cfg.get("run"):
        lines.append(f"- **Run**: `{_fmt(cfg.get('run'))}`")
    if cfg.get("metrics"):
        lines.append(f"- **Metrics**: `{_fmt(cfg.get('metrics'))}`")
    if cfg.get("output"):
        lines.append(f"- **Output**: `{_fmt(cfg.get('output'))}`")

    lines.append("\n## Hardware\n")
    table = hw.get("table") if isinstance(hw, dict) else None
    if isinstance(table, dict) and table:
        for key, value in table.items():
            lines.append(f"- **{key}**: `{_fmt(value)}`")
    else:
        lines.append("- `-`")

    lines.append("\n## Environment\n")
    if isinstance(env, dict) and env:
        for key in ("python", "torch", "onnx", "onnxruntime", "openvino", "tensorrt", "numpy"):
            if key in env and env.get(key) is not None:
                lines.append(f"- **{key}**: `{_fmt(env.get(key))}`")
        env_thread_block = env.get("thread_env") or {}
        if env_thread_block:
            for key in THREAD_ENV_KEYS:
                if key in env_thread_block:
                    lines.append(f"- **env.{key}**: `{_fmt(env_thread_block.get(key))}`")
    else:
        lines.append("- `-`")


def _write_single_run_report(run_dir: Path, results: Dict[str, Any]) -> None:
    md_path = run_dir / "bench.md"
    lines: List[str] = ["# Benchmark Report\n", f"Directory: `{run_dir}`\n"]
    _append_common_sections(lines, results)

    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        pngs = sorted(plots_dir.glob("*.png"))
        if pngs:
            lines.append("\n## Plots\n")
            for path in pngs:
                lines.append(f"![{path.name}]({Path('plots') / path.name})\n")

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_multi_run_reports(run_dir: Path, results_by_device: Mapping[str, Dict[str, Any]]) -> None:
    run_md = run_dir / "bench.md"
    devices_dir = run_dir / "devices"
    plots_dir = run_dir / "plots"

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for device, result in results_by_device.items():
        rows.append((str(device), _extract_core_metrics(result)))
    rows.sort(key=lambda item: item[0].lower())

    lines: List[str] = ["# Multi-Device Benchmark Report\n", f"Directory: `{run_dir}`\n"]
    any_res = next(iter(results_by_device.values()), {})
    fw = _safe_get(any_res, ["meta", "framework"])
    model_path = _safe_get(any_res, ["model", "path"])
    if fw or model_path:
        lines.append("## Run\n")
        if fw:
            lines.append(f"- **Framework**: `{_fmt(fw)}`")
        if model_path:
            lines.append(f"- **Model**: `{_fmt(model_path)}`")

    lines.append("\n## Summary\n")
    lines.append("| Device | Mean (ms) | p95 (ms) | Throughput (sps) | CPU mean (agg %) | CPU audit | RSS peak infer (MB) | Report |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---|")
    for device, metrics in rows:
        safe_dev = sanitize_device_key(device)
        per_dev_md_rel = Path("devices") / safe_dev / "bench.md"
        lines.append(
            "| {device} | {mean} | {p95} | {thr} | {cpu} | {audit} | {rss_mb} | {link} |".format(
                device=device,
                mean=_fmt(metrics.get("inf_mean_ms")),
                p95=_fmt(metrics.get("inf_p95_ms")),
                thr=_fmt(metrics.get("throughput_sps")),
                cpu=_fmt(metrics.get("cpu_mean_pct")),
                audit=_fmt(metrics.get("cpu_audit_consistency_note")),
                rss_mb=_fmt_mb_from_bytes(metrics.get("rss_peak_infer_bytes")),
                link=f"[bench.md]({per_dev_md_rel.as_posix()})",
            )
        )

    if plots_dir.exists():
        global_pngs = sorted(plots_dir.glob("*.png"))
        if global_pngs:
            lines.append("\n## Plots\n")
            for path in global_pngs:
                lines.append(f"![{path.name}]({Path('plots') / path.name})\n")

    run_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    devices_dir.mkdir(parents=True, exist_ok=True)
    for device, result in results_by_device.items():
        safe_dev = sanitize_device_key(str(device))
        dev_out_dir = devices_dir / safe_dev
        dev_out_dir.mkdir(parents=True, exist_ok=True)

        lines = [f"# Device Report: `{device}`\n", f"Run directory: `{run_dir}`\n"]
        _append_common_sections(lines, result)

        if plots_dir.exists():
            candidates = [
                plots_dir / f"summary_{safe_dev}.png",
                plots_dir / f"ram_over_time_{safe_dev}.png",
            ]
            existing = [path for path in candidates if path.exists()]
            if existing:
                lines.append("\n## Plots\n")
                for path in existing:
                    rel = Path("..") / ".." / "plots" / path.name
                    lines.append(f"![{path.name}]({rel.as_posix()})\n")

        (dev_out_dir / "bench.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def create_markdown_report(run_dir: Path, results: Dict[str, Any]) -> None:
    run_dir = Path(run_dir)

    if _is_single_run_payload(results):
        _write_single_run_report(run_dir, results)
        return

    if isinstance(results, dict):
        device_map: Dict[str, Dict[str, Any]] = {}
        for key, value in results.items():
            if isinstance(value, dict):
                device_map[str(key)] = value
        _write_multi_run_reports(run_dir, device_map)
        return

    raise TypeError("create_markdown_report: results must be a dict (single payload or device mapping).")

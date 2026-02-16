# bench/core/analysis/viz/report_markdown.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, List, Tuple

from bench.core.utils.path_sanitizer import sanitize_device_key


def _is_single_run_payload(results: Dict[str, Any]) -> bool:
    """Heuristic: single-run bench.json payload has these top-level keys."""
    return isinstance(results, dict) and "meta" in results and "metrics" in results and "model" in results


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# ------------------------------------------------------------------
# Reporting helpers (human-friendly units + derived interpretation)
# ------------------------------------------------------------------
# We intentionally use SI units (MB/GB, base-10) in the Markdown report.
# Raw bytes remain available in bench.json for audit / reproducibility.
def _fmt_mb_from_bytes(b: Any) -> str:
    """Format a bytes value as MB (SI, 1 MB = 1_000_000 bytes)."""
    if b is None:
        return "-"
    try:
        return f"{float(b) / 1_000_000:.3f}"
    except Exception:
        return "-"


def _fmt_gb_from_bytes(b: Any) -> str:
    """Format a bytes value as GB (SI, 1 GB = 1_000_000_000 bytes)."""
    if b is None:
        return "-"
    try:
        return f"{float(b) / 1_000_000_000:.3f}"
    except Exception:
        return "-"


def _fmt_cores_equiv(aggregate_pct: Any) -> str:
    """
    Convert aggregate CPU utilization (%) into a 'cores-equivalent' number.
    Example: 200% ≈ 2.0 cores worth of compute time.
    """
    if aggregate_pct is None:
        return "-"
    try:
        return f"{float(aggregate_pct) / 100.0:.2f}"
    except Exception:
        return "-"


def _safe_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _extract_core_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts a compact set of metrics for tables/summary.
    Supports both:
      - res["metrics"]["inference_time_ms"]
      - res["inference_time_ms"] (if legacy ever appears)
    """
    inf = res.get("metrics", {}).get("inference_time_ms") or res.get("inference_time_ms") or {}
    cpu = res.get("metrics", {}).get("cpu_utilization_pct") or res.get("cpu_utilization_pct") or {}
    mem = res.get("metrics", {}).get("memory") or res.get("memory") or {}

    return {
        "inf_mean_ms": inf.get("mean"),
        "inf_p50_ms": inf.get("p50"),
        "inf_p90_ms": inf.get("p90"),
        "inf_p95_ms": inf.get("p95"),
        "inf_p99_ms": inf.get("p99"),
        # Batch-aware latency reporting:
        # - batch_size is derived from input tensor shape at runtime
        # - mean_per_sample normalizes batch latency for fair cross-config comparison
        "inf_batch_size": inf.get("batch_size"),
        "inf_mean_per_sample_ms": inf.get("mean_per_sample"),
        "throughput_sps": res.get("metrics", {}).get("throughput_sps") or res.get("throughput_sps"),
        "ms_per_signal_s": res.get("metrics", {}).get("ms_per_signal_s") or res.get("ms_per_signal_s"),
        "cpu_mean_pct": cpu.get("mean"),
        "cpu_p95_pct": cpu.get("p95"),
        "rss_peak_infer_bytes": mem.get("rss_peak_inference_bytes"),
        "rss_peak_bytes": mem.get("rss_peak_bytes"),
        # Runtime memory (RSS) derivations:
        # These fields make "model-specific runtime memory requirements" explicit.
        "rss_baseline_mean_bytes": mem.get("rss_baseline_mean_bytes"),
        "rss_mean_infer_bytes": mem.get("rss_mean_inference_bytes"),
        "model_specific_runtime_memory_bytes": mem.get("model_specific_runtime_memory_bytes"),
        "memory_safety_margin_pct": mem.get("memory_safety_margin_pct"),
        "minimal_required_ram_bytes": mem.get("minimal_required_ram_bytes"),
        "memory_recommendation": mem.get("memory_recommendation"),
    }


def _write_single_run_report(run_dir: Path, results: Dict[str, Any]) -> None:
    md_path = run_dir / "bench.md"

    meta = results.get("meta", {}) or {}
    model = results.get("model", {}) or {}
    metrics = results.get("metrics", {}) or {}
    cfg = results.get("config", {}) or {}
    hw = results.get("hardware", {}) or {}
    env = results.get("env", {}) or {}

    lines: List[str] = []
    lines.append("# Benchmark Report\n")
    lines.append(f"Directory: `{run_dir}`\n")

    # Run
    lines.append("## Run\n")
    lines.append(f"- **Framework**: `{_fmt(meta.get('framework'))}`")
    lines.append(f"- **Device**: `{_fmt(meta.get('device_target'))}`")
    if meta.get("timestamp"):
        lines.append(f"- **Timestamp**: `{_fmt(meta.get('timestamp'))}`")
    if meta.get("run_id"):
        lines.append(f"- **Run ID**: `{_fmt(meta.get('run_id'))}`")

    # Model
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
    if model.get("size_on_disk_bytes") is not None:
        # Keep the report human-readable: show MB instead of raw bytes.
        lines.append(f"- **Size on disk (MB)**: `{_fmt_mb_from_bytes(model.get('size_on_disk_bytes'))}`")

    # Metrics (compact)
    core = _extract_core_metrics({"metrics": metrics, **results})
    lines.append("\n## Metrics\n")
    lines.append(f"- **Inference mean (ms)**: `{_fmt(core.get('inf_mean_ms'))}`")
    lines.append(f"- **Inference p50 (ms)**: `{_fmt(core.get('inf_p50_ms'))}`")
    lines.append(f"- **Inference p90 (ms)**: `{_fmt(core.get('inf_p90_ms'))}`")
    lines.append(f"- **Inference p95 (ms)**: `{_fmt(core.get('inf_p95_ms'))}`")
    lines.append(f"- **Inference p99 (ms)**: `{_fmt(core.get('inf_p99_ms'))}`")
    lines.append(f"- **Inference batch size**: `{_fmt(core.get('inf_batch_size'))}`")
    lines.append(f"- **Inference mean per sample (ms)**: `{_fmt(core.get('inf_mean_per_sample_ms'))}`")
    lines.append(f"- **Throughput (samples/s)**: `{_fmt(core.get('throughput_sps'))}`")
    if core.get("ms_per_signal_s") is not None:
        lines.append(f"- **ms per second of signal**: `{_fmt(core.get('ms_per_signal_s'))}`")
        lines.append("- Note: This value is derived from the measured inference mean and the runtime-derived input duration.")

    # CPU utilization is reported as aggregate % (sum across logical cores).
    # Add a derived "cores-equivalent" value for quick interpretation.
    lines.append(
        f"- **CPU util mean (aggregate %)**: `{_fmt(core.get('cpu_mean_pct'))}` "
        f"(≈ `{_fmt_cores_equiv(core.get('cpu_mean_pct'))}` cores)"
    )
    lines.append(
        f"- **CPU util p95 (aggregate %)**: `{_fmt(core.get('cpu_p95_pct'))}` "
        f"(≈ `{_fmt_cores_equiv(core.get('cpu_p95_pct'))}` cores)"
    )

    # Memory/RSS: present SI MB/GB to keep the report readable.
    lines.append(f"- **RSS peak inference (MB)**: `{_fmt_mb_from_bytes(core.get('rss_peak_infer_bytes'))}`")
    lines.append(f"- **RSS peak (MB)**: `{_fmt_mb_from_bytes(core.get('rss_peak_bytes'))}`")
    # Make model-specific runtime memory requirements explicit (optional fields)
    lines.append(f"- **RSS baseline mean (MB)**: `{_fmt_mb_from_bytes(core.get('rss_baseline_mean_bytes'))}`")
    lines.append(f"- **RSS mean during inference (MB)**: `{_fmt_mb_from_bytes(core.get('rss_mean_infer_bytes'))}`")
    lines.append(f"- **Model-specific runtime memory (MB)**: `{_fmt_mb_from_bytes(core.get('model_specific_runtime_memory_bytes'))}`")
    lines.append(f"- **Memory safety margin (%)**: `{_fmt(core.get('memory_safety_margin_pct'))}`")
    lines.append(f"- **Recommended minimum RAM (MB)**: `{_fmt_mb_from_bytes(core.get('minimal_required_ram_bytes'))}`")
    lines.append(f"- **Recommended minimum RAM (GB)**: `{_fmt_gb_from_bytes(core.get('minimal_required_ram_bytes'))}`")
    if core.get("memory_recommendation") is not None:
        lines.append(f"- **Memory recommendation**: `{_fmt(core.get('memory_recommendation'))}`")

    # Config (keep readable)
    lines.append("\n## Config\n")
    if cfg.get("input"):
        lines.append(f"- **Input**: `{_fmt(cfg.get('input'))}`")
    if cfg.get("run"):
        lines.append(f"- **Run**: `{_fmt(cfg.get('run'))}`")
    if cfg.get("metrics"):
        lines.append(f"- **Metrics**: `{_fmt(cfg.get('metrics'))}`")
    if cfg.get("output"):
        lines.append(f"- **Output**: `{_fmt(cfg.get('output'))}`")

    # Hardware (table block if present)
    lines.append("\n## Hardware\n")
    table = hw.get("table") if isinstance(hw, dict) else None
    if isinstance(table, dict) and table:
        for k, v in table.items():
            lines.append(f"- **{k}**: `{_fmt(v)}`")
    else:
        lines.append("- `-`")

    # Env (selected keys)
    lines.append("\n## Environment\n")
    if isinstance(env, dict) and env:
        for k in ("python", "torch", "onnx", "onnxruntime", "openvino", "tensorrt", "numpy"):
            if k in env and env.get(k) is not None:
                lines.append(f"- **{k}**: `{_fmt(env.get(k))}`")
    else:
        lines.append("- `-`")

    # Embed plots if any exist
    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        pngs = sorted(plots_dir.glob("*.png"))
        if pngs:
            lines.append("\n## Plots\n")
            for p in pngs:
                lines.append(f"![{p.name}]({Path('plots') / p.name})\n")

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_multi_run_reports(run_dir: Path, results_by_device: Mapping[str, Dict[str, Any]]) -> None:
    run_md = run_dir / "bench.md"
    devices_dir = run_dir / "devices"
    plots_dir = run_dir / "plots"

    # -------- Root summary (table) --------
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for dev, res in results_by_device.items():
        rows.append((str(dev), _extract_core_metrics(res)))

    # Stable sort by device label
    rows.sort(key=lambda x: x[0].lower())

    lines: List[str] = []
    lines.append("# Multi-Device Benchmark Report\n")
    lines.append(f"Directory: `{run_dir}`\n")

    # Optional top meta (if present per-device)
    # If all devices share same model/framework, show it once.
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
    lines.append(
        "| Device | Mean (ms) | p95 (ms) | p99 (ms) | Throughput (sps) | CPU mean (agg %) | CPU mean (cores) | RSS peak infer (MB) | Report |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for dev, m in rows:
        safe_dev = sanitize_device_key(dev)
        per_dev_md_rel = Path("devices") / safe_dev / "bench.md"
        lines.append(
            "| {dev} | {mean} | {p95} | {p99} | {thr} | {cpu} | {cores} | {rss_mb} | {link} |".format(
                dev=dev,
                mean=_fmt(m.get("inf_mean_ms")),
                p95=_fmt(m.get("inf_p95_ms")),
                p99=_fmt(m.get("inf_p99_ms")),
                thr=_fmt(m.get("throughput_sps")),
                cpu=_fmt(m.get("cpu_mean_pct")),
                cores=_fmt_cores_equiv(m.get("cpu_mean_pct")),
                rss_mb=_fmt_mb_from_bytes(m.get("rss_peak_infer_bytes")),
                link=f"[bench.md]({per_dev_md_rel.as_posix()})",
            )
        )

    # Embed global plots (if present)
    if plots_dir.exists():
        global_pngs = sorted(plots_dir.glob("*.png"))
        if global_pngs:
            lines.append("\n## Plots\n")
            for p in global_pngs:
                lines.append(f"![{p.name}]({Path('plots') / p.name})\n")

    run_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    # -------- Per-device reports --------
    devices_dir.mkdir(parents=True, exist_ok=True)
    for dev, res in results_by_device.items():
        dev_str = str(dev)
        safe_dev = sanitize_device_key(dev_str)
        dev_out_dir = devices_dir / safe_dev
        dev_out_dir.mkdir(parents=True, exist_ok=True)

        dev_md = dev_out_dir / "bench.md"

        meta = res.get("meta", {}) or {}
        model = res.get("model", {}) or {}
        cfg = res.get("config", {}) or {}
        hw = res.get("hardware", {}) or {}
        env = res.get("env", {}) or {}

        core = _extract_core_metrics(res)

        dlines: List[str] = []
        dlines.append(f"# Device Report: `{dev_str}`\n")
        dlines.append(f"Run directory: `{run_dir}`\n")

        dlines.append("## Run\n")
        dlines.append(f"- **Framework**: `{_fmt(meta.get('framework'))}`")
        dlines.append(f"- **Device**: `{_fmt(meta.get('device_target'))}`")
        if meta.get("timestamp"):
            dlines.append(f"- **Timestamp**: `{_fmt(meta.get('timestamp'))}`")
        if meta.get("run_id"):
            dlines.append(f"- **Run ID**: `{_fmt(meta.get('run_id'))}`")

        dlines.append("\n## Model\n")
        if model.get("path"):
            dlines.append(f"- **Path**: `{_fmt(model.get('path'))}`")
        if model.get("input_shape") is not None:
            dlines.append(f"- **Input shape**: `{_fmt(model.get('input_shape'))}`")
        if model.get("input_duration_s") is not None:
            dlines.append(f"- **Input duration (s)**: `{_fmt(model.get('input_duration_s'))}`")
        if model.get("fs_hz") is not None:
            dlines.append(f"- **Sampling rate (Hz)**: `{_fmt(model.get('fs_hz'))}`")
        if model.get("input_num_samples") is not None:
            dlines.append(f"- **Input samples**: `{_fmt(model.get('input_num_samples'))}`")
        if model.get("dtype"):
            dlines.append(f"- **DType**: `{_fmt(model.get('dtype'))}`")

        dlines.append("\n## Metrics\n")
        dlines.append(f"- **Inference mean (ms)**: `{_fmt(core.get('inf_mean_ms'))}`")
        dlines.append(f"- **Inference p50 (ms)**: `{_fmt(core.get('inf_p50_ms'))}`")
        dlines.append(f"- **Inference p95 (ms)**: `{_fmt(core.get('inf_p95_ms'))}`")
        dlines.append(f"- **Inference p99 (ms)**: `{_fmt(core.get('inf_p99_ms'))}`")
        dlines.append(f"- **Inference batch size**: `{_fmt(core.get('inf_batch_size'))}`")
        dlines.append(f"- **Inference mean per sample (ms)**: `{_fmt(core.get('inf_mean_per_sample_ms'))}`")
        dlines.append(f"- **Throughput (samples/s)**: `{_fmt(core.get('throughput_sps'))}`")
        if core.get("ms_per_signal_s") is not None:
            dlines.append(f"- **ms per second of signal**: `{_fmt(core.get('ms_per_signal_s'))}`")
            dlines.append("- Note: This value is derived from the measured inference mean and the runtime-derived input duration.")

        dlines.append(
            f"- **CPU util mean (aggregate %)**: `{_fmt(core.get('cpu_mean_pct'))}` "
            f"(≈ `{_fmt_cores_equiv(core.get('cpu_mean_pct'))}` cores)"
        )
        dlines.append(
            f"- **CPU util p95 (aggregate %)**: `{_fmt(core.get('cpu_p95_pct'))}` "
            f"(≈ `{_fmt_cores_equiv(core.get('cpu_p95_pct'))}` cores)"
        )

        dlines.append(f"- **RSS peak inference (MB)**: `{_fmt_mb_from_bytes(core.get('rss_peak_infer_bytes'))}`")
        dlines.append(f"- **RSS peak (MB)**: `{_fmt_mb_from_bytes(core.get('rss_peak_bytes'))}`")
        # Make model-specific runtime memory requirements explicit (optional fields)
        dlines.append(f"- **RSS baseline mean (MB)**: `{_fmt_mb_from_bytes(core.get('rss_baseline_mean_bytes'))}`")
        dlines.append(f"- **RSS mean during inference (MB)**: `{_fmt_mb_from_bytes(core.get('rss_mean_infer_bytes'))}`")
        dlines.append(f"- **Model-specific runtime memory (MB)**: `{_fmt_mb_from_bytes(core.get('model_specific_runtime_memory_bytes'))}`")
        dlines.append(f"- **Memory safety margin (%)**: `{_fmt(core.get('memory_safety_margin_pct'))}`")
        dlines.append(f"- **Recommended minimum RAM (MB)**: `{_fmt_mb_from_bytes(core.get('minimal_required_ram_bytes'))}`")
        dlines.append(f"- **Recommended minimum RAM (GB)**: `{_fmt_gb_from_bytes(core.get('minimal_required_ram_bytes'))}`")
        if core.get("memory_recommendation") is not None:
            dlines.append(f"- **Memory recommendation**: `{_fmt(core.get('memory_recommendation'))}`")

        dlines.append("\n## Config\n")
        if cfg.get("input"):
            dlines.append(f"- **Input**: `{_fmt(cfg.get('input'))}`")
        if cfg.get("run"):
            dlines.append(f"- **Run**: `{_fmt(cfg.get('run'))}`")

        dlines.append("\n## Hardware\n")
        table = hw.get("table") if isinstance(hw, dict) else None
        if isinstance(table, dict) and table:
            for k, v in table.items():
                dlines.append(f"- **{k}**: `{_fmt(v)}`")

        dlines.append("\n## Environment\n")
        if isinstance(env, dict) and env:
            for k in ("python", "torch", "onnx", "onnxruntime", "openvino", "tensorrt", "numpy"):
                if k in env and env.get(k) is not None:
                    dlines.append(f"- **{k}**: `{_fmt(env.get(k))}`")

        # Device plot embedding (uses relative path back to run_root/plots)
        # Example: devices/<safe_dev>/bench.md -> ../../plots/<file>.png
        if plots_dir.exists():
            # Prefer summary plot if it exists
            candidates = [
                plots_dir / f"summary_{safe_dev}.png",
                plots_dir / f"ram_over_time_{safe_dev}.png",
            ]
            existing = [p for p in candidates if p.exists()]
            if existing:
                dlines.append("\n## Plots\n")
                for p in existing:
                    rel = Path("..") / ".." / "plots" / p.name
                    dlines.append(f"![{p.name}]({rel.as_posix()})\n")

        dev_md.write_text("\n".join(dlines).rstrip() + "\n", encoding="utf-8")


def create_markdown_report(run_dir: Path, results: Dict[str, Any]) -> None:
    """
    Writes Markdown report(s) for both single- and multi-device runs.

    Single-run:
      - run_dir/bench.md

    Multi-run (results is mapping device->bench_json_dict):
      - run_dir/bench.md (summary + global plots)
      - run_dir/devices/<device>/bench.md (per-device detail + device plots)
    """
    run_dir = Path(run_dir)

    if _is_single_run_payload(results):
        _write_single_run_report(run_dir, results)
        return

    # Multi-run case: treat as mapping device -> dict
    if isinstance(results, dict):
        # Filter only dict-like device results
        device_map: Dict[str, Dict[str, Any]] = {}
        for k, v in results.items():
            if isinstance(v, dict):
                device_map[str(k)] = v
        _write_multi_run_reports(run_dir, device_map)
        return

    raise TypeError("create_markdown_report: results must be a dict (single payload or device mapping).")

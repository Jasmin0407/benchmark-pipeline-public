# bench/core/io/result_writer.py
"""
ResultWriter

Persists benchmark results in a per-run folder layout:

runs/<run_name>/<timestamp>/
  bench.json
  bench.csv (optional)
  (additional artifacts generated elsewhere, e.g., bench.md, plots/*.png)

Design goals
------------
- One run == one directory (easy to archive, compare, and keep clean).
- No "loose" JSON/CSV files in the run group root.
- Cross-platform safe folder names (Windows-safe, no path traversal).
- Minimal side effects: only creates directories and writes files.

Security notes
--------------
- Folder names are sanitized to avoid invalid characters and path traversal.
- Avoid printing absolute local paths; log relative paths where possible.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from bench.core.schemas.run_schema import RunSchema


def _timestamp() -> str:
    """Return a stable timestamp string used as the run folder name."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _sanitize_component(value: str) -> str:
    """
    Sanitize a string so it can be safely used as a folder/file name.

    - Replaces unsupported characters with underscore.
    - Strips leading/trailing underscores.
    - Prevents empty names.
    """
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value))
    safe = safe.strip("._-")
    return safe or "run"


def _safe_join(base: Path, *parts: str) -> Path:
    """
    Join paths safely and prevent path traversal.

    This ensures the resulting path stays within `base` even if input components
    contained unexpected sequences.
    """
    out = base
    for p in parts:
        out = out / _sanitize_component(p)
    # Resolve to absolute paths for containment check
    base_resolved = base.resolve()
    out_resolved = out.resolve()

    if base_resolved not in out_resolved.parents and base_resolved != out_resolved:
        raise ValueError("Unsafe output path detected (path traversal).")

    return out


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    """
    Normalize Pydantic objects and dicts into a Mapping without transforming values.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):  # Pydantic v1 fallback
        return obj.dict()
    return {}


def _resolve_base_dir(run: RunSchema) -> Path:
    """
    Resolve the output base directory.

    Priority:
    1) run.config.output.base_dir
    2) run.config.output.dir (legacy)
    3) default: "runs"
    """
    out_cfg = (run.config.get("output", {}) or {}) if getattr(run, "config", None) else {}
    base = out_cfg.get("base_dir") or out_cfg.get("dir") or "runs"
    return Path(str(base))


def _resolve_run_name(run: RunSchema) -> str:
    """
    Determine the run group folder name (runs/<run_name>/...).

    Priority:
    1) run.config.output.run_name (explicit override)
    2) "<framework>_<device>" (legacy-compatible and most informative)
    3) "<framework>"
    4) "runs"
    """
    out_cfg = (run.config.get("output", {}) or {}) if getattr(run, "config", None) else {}
    if out_cfg.get("run_name"):
        return _sanitize_component(str(out_cfg["run_name"]))

    framework = getattr(run.meta, "framework", None) or "bench"
    device = getattr(run.meta, "device_target", None) or ""

    if device:
        # Example: onnx_cpu, openvino_npu, trt_gpu
        return _sanitize_component(f"{framework}_{device}".replace(":", "_").replace("/", "_"))

    return _sanitize_component(str(framework))


def _write_json(out_path: Path, run: RunSchema) -> None:
    """Write the full run object as JSON."""
    payload: Dict[str, Any] = run.model_dump() if hasattr(run, "model_dump") else dict(run)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(out_path: Path, run: RunSchema) -> None:
    """
    Write a compact CSV row for quick spreadsheet ingestion.

    The CSV intentionally contains a small, stable subset of fields.
    """
    inf = _as_mapping(run.metrics.inference_time_ms)
    cpu = _as_mapping(run.metrics.cpu_utilization_pct)
    mem = _as_mapping(run.metrics.memory)

    # Prefer current schema keys; fall back to legacy keys where needed.
    peak_bytes = mem.get("peak_ram_process_bytes")
    if peak_bytes is None:
        peak_bytes = mem.get("rss_peak_bytes") or mem.get("peak_bytes") or 0

    # Timestamp: prefer run.meta.timestamp if present; else use folder timestamp.
    ts = getattr(run.meta, "timestamp", None)

    flat = {
        "timestamp": ts,
        "framework": getattr(run.meta, "framework", None),
        "device": getattr(run.meta, "device_target", None),
        "macs": run.metrics.macs,
        "inference_mean_ms": inf.get("mean"),
        "cpu_mean_pct": cpu.get("mean"),
        "peak_ram_mb": float(peak_bytes) / 1e6 if peak_bytes is not None else None,
    }

    pd.DataFrame([flat]).to_csv(out_path, index=False)


def write_result(run: RunSchema, ts: Optional[str] = None) -> str:
    """
    Persist the benchmark result into a per-run folder and return the JSON path.

    Result layout
    -------------
    runs/<run_name>/<timestamp>/bench.json

    Notes
    -----
    - The 'plots/' folder and Markdown report are generated by the CLI layer,
      using the returned JSON path's parent directory.
    - This function does not generate plots or reports; it only writes results.
    """
    base_dir = _resolve_base_dir(run)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_name = _resolve_run_name(run)
    ts_str = ts or _timestamp()

    run_dir = _safe_join(base_dir, run_name, ts_str)
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / "bench.json"
    _write_json(json_path, run)

    # Optional CSV output (same folder)
    out_cfg = (run.config.get("output", {}) or {}) if getattr(run, "config", None) else {}
    formats = out_cfg.get("formats", []) or []
    if isinstance(formats, (list, tuple)) and "csv" in formats:
        csv_path = run_dir / "bench.csv"
        _write_csv(csv_path, run)

    # Log relative path where possible (avoid leaking absolute local paths)
    try:
        rel = json_path.relative_to(Path.cwd())
        print(f"[OK] Result written: {rel.as_posix()}")
    except Exception:
        print("[OK] Result written: bench.json")

    return str(json_path)

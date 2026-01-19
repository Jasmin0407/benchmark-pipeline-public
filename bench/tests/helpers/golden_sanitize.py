# bench/tests/helpers/golden_sanitize.py

from __future__ import annotations

from typing import Any, Dict, List, Optional


REDACTED = "redacted"


def _shorten_list(values: Any, keep: int = 2) -> Any:
    """
    Shorten long list/tuple values to keep diffs small and avoid leaking details.
    Keeps first `keep` and last `keep` elements if length is larger than 2*keep.
    """
    if not isinstance(values, (list, tuple)):
        return values
    if keep <= 0:
        return []
    if len(values) <= 2 * keep:
        return list(values)
    return list(values[:keep]) + list(values[-keep:])


def sanitize_golden_run(run_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a RunSchema dump for committing as a golden file.

    Goals:
    - Remove or redact system-identifying / environment-specific details
      (absolute paths, hostnames, fingerprints, verbose build logs).
    - Keep schema structure stable enough for regression tests.
    - Keep arrays small to avoid noisy diffs and repo bloat.
    """
    d: Dict[str, Any] = dict(run_dict)  # shallow copy

    # --- meta ---
    meta = dict(d.get("meta") or {})
    if "run_id" in meta:
        meta["run_id"] = REDACTED
    if "timestamp" in meta:
        # Use a stable timestamp to avoid noisy diffs.
        meta["timestamp"] = "2025-01-01T00:00:00Z"
    d["meta"] = meta

    # --- model ---
    model = dict(d.get("model") or {})
    if "path" in model:
        # Avoid leaking local absolute paths or temp locations.
        model["path"] = "inline"
    d["model"] = model

    # --- hardware ---
    hardware = dict(d.get("hardware") or {})
    if "fingerprint" in hardware:
        hardware["fingerprint"] = REDACTED

    # Keep hardware.table but make it generic if desired.
    table = dict(hardware.get("table") or {})
    if table:
        table["cpu"] = table.get("cpu") and "Generic CPU" or "Generic CPU"
        table["gpu"] = "Generic GPU"
        table["npu"] = "Generic NPU"
        # Normalize OS label to avoid embedding exact build numbers.
        if "os" in table:
            table["os"] = "Windows" if str(table["os"]).lower().startswith("windows") else str(table["os"]).split()[0]
    hardware["table"] = table

    detail = dict(hardware.get("detail") or {})
    # Drop hostname and deep inventory details.
    detail.pop("hostname", None)
    detail.pop("ram_modules", None)
    detail.pop("storage_modules", None)

    # Normalize OS string (remove exact build numbers).
    if "os" in detail:
        os_str = str(detail["os"])
        detail["os"] = "Windows" if os_str.lower().startswith("windows") else os_str.split()[0]

    # Make device names generic to avoid exposing exact models.
    cpu = dict(detail.get("cpu") or {})
    if cpu:
        cpu["name"] = "Generic CPU"
    detail["cpu"] = cpu

    gpus = detail.get("gpus")
    if isinstance(gpus, list):
        new_gpus: List[Dict[str, Any]] = []
        for g in gpus:
            gg = dict(g or {})
            gg["name"] = "Generic GPU"
            new_gpus.append(gg)
        detail["gpus"] = new_gpus

    npus = detail.get("npus")
    if isinstance(npus, list):
        new_npus: List[Dict[str, Any]] = []
        for n in npus:
            nn = dict(n or {})
            nn["name"] = "Generic NPU"
            new_npus.append(nn)
        detail["npus"] = new_npus

    hardware["detail"] = detail

    # Capabilities: keep but normalize provider list to CPU-only if present
    caps = dict(hardware.get("capabilities") or {})
    if "onnx_providers" in caps and isinstance(caps["onnx_providers"], list):
        # Golden should not depend on machine-specific provider availability.
        caps["onnx_providers"] = ["CPUExecutionProvider"]
    hardware["capabilities"] = caps

    d["hardware"] = hardware

    # --- env ---
    env = dict(d.get("env") or {})
    # Remove/normalize verbose environment details that often contain file paths.
    for k in ["numpy_blas", "cuda_toolkit", "mkl"]:
        if k in env:
            env[k] = REDACTED

    # Normalize python patch version to avoid diffs across machines.
    if "python" in env and isinstance(env["python"], str):
        env["python"] = "3.10.x"

    # Normalize provider list to CPU-only
    if "onnxruntime_providers" in env and isinstance(env["onnxruntime_providers"], list):
        env["onnxruntime_providers"] = ["CPUExecutionProvider"]

    d["env"] = env

    # --- metrics.memory arrays ---
    metrics = dict(d.get("metrics") or {})
    memory = dict(metrics.get("memory") or {})
    if "rss_samples" in memory:
        memory["rss_samples"] = _shorten_list(memory["rss_samples"], keep=2)
    if "timestamps_s" in memory:
        memory["timestamps_s"] = _shorten_list(memory["timestamps_s"], keep=2)

    # Infer index ranges become meaningless after shortening; keep them minimal.
    if "infer_start_idx" in memory:
        memory["infer_start_idx"] = 1
    if "infer_end_idx" in memory:
        memory["infer_end_idx"] = 2

    metrics["memory"] = memory
    d["metrics"] = metrics

    return d

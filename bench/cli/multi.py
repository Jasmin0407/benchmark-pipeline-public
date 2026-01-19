# bench/cli/multi.py
"""
Multi-device benchmark CLI (CPU / GPU / NPU / AUTO / HETERO).

Responsibilities
---------------
- Parse CLI arguments
- Load YAML config
- Apply runtime-only overrides (--verbose/--log-level)
- Validate minimal config contract (including unified run.* schema)
- Run benchmarks across multiple target devices via DeviceLoop
- Persist per-device JSON results under a clean run folder layout:
    runs/<run_name>/<timestamp>/
      bench.md
      plots/
        *.png
      devices/
        <device_key>/
          bench.json
- Optionally generate plots (--plot) and a Markdown report (--report)

Security / hygiene
------------------
- Sanitize device identifiers before using them in filenames.
- Avoid path traversal by using sanitized components only.
- Keep file I/O localized to the computed run directory.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from bench.core.analysis.viz.plots_cpu import plot_cpu_utilization
from bench.core.analysis.viz.plots_memory import plot_memory_over_time
from bench.core.analysis.viz.plots_multi import plot_multi_device_latency
from bench.core.analysis.viz.plots_summary import plot_device_summary
from bench.core.analysis.viz.report_markdown import create_markdown_report
from bench.core.config.config_loader import load_config
from bench.core.orchestrator.device_loop import DeviceLoop
from bench.core.utils.path_sanitizer import sanitize_component, sanitize_device_key


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the multi-device benchmark CLI."""
    parser = argparse.ArgumentParser(description="Multi-Device Benchmark CLI")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots")
    parser.add_argument("--report", action="store_true", help="Generate Markdown report (bench.md)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging")
    parser.add_argument("--log-level", type=str, help="Explicit log level (DEBUG/INFO/WARNING/ERROR)")
    return parser.parse_args()


def _apply_runtime_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Apply CLI runtime overrides to the loaded configuration.

    CLI flags are runtime-only and override YAML values without modifying the file.
    """
    run_cfg = cfg.setdefault("run", {})

    if args.verbose:
        run_cfg["verbose"] = True

    if args.log_level:
        lvl = str(args.log_level).upper()
        if lvl not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("Invalid --log-level. Use: DEBUG/INFO/WARNING/ERROR")
        run_cfg["log_level"] = lvl

    return cfg


def validate_config_contract(cfg: Dict[str, Any]) -> None:
    """
    Validate the minimal config contract required by the multi-device CLI.

    Enforces the unified schema:
    - model.path (required)
    - model.backend (required)
    - devices_to_test (non-empty list)
    - run.warmups (>= 0)
    - run.repeats (>= 1)
    """
    model = cfg.get("model") or {}
    if not model.get("path"):
        raise ValueError("Invalid configuration: model.path is required")
    if not model.get("backend"):
        raise ValueError("Invalid configuration: model.backend is required")

    devices = cfg.get("devices_to_test")
    if not isinstance(devices, (list, tuple)) or len(devices) == 0:
        raise ValueError("Invalid configuration: devices_to_test must be a non-empty list")

    run_cfg = cfg.get("run") or {}
    warmups = run_cfg.get("warmups")
    repeats = run_cfg.get("repeats")
    if warmups is None or int(warmups) < 0:
        raise ValueError("Invalid configuration: run.warmups must be an integer >= 0")
    if repeats is None or int(repeats) < 1:
        raise ValueError("Invalid configuration: run.repeats must be an integer >= 1")


def _make_run_dirs(base_runs_dir: Path, run_name: str) -> Tuple[Path, Path]:
    """
    Create the run output directory structure.

    Result
    ------
    runs/<run_name>/<timestamp>/
      plots/
      devices/
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_dir = base_runs_dir / sanitize_component(run_name) / ts
    plots_dir = base_dir / "plots"

    (base_dir / "devices").mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, plots_dir


def _write_results_json(base_dir: Path, results: Dict[str, Dict[str, Any]]) -> None:
    """
    Persist one JSON file per device.

    Layout
    ------
    <base_dir>/devices/<safe_device>/bench.json
    """
    devices_dir = base_dir / "devices"
    devices_dir.mkdir(parents=True, exist_ok=True)

    for dev, res in results.items():
        safe_dev = sanitize_device_key(str(dev))
        dev_dir = devices_dir / safe_dev
        dev_dir.mkdir(parents=True, exist_ok=True)

        out_file = dev_dir / "bench.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        print(f"[OK] Wrote JSON: {out_file.as_posix()}")


def _plot_outputs(results: Dict[str, Dict[str, Any]], plots_dir: Path) -> None:
    """
    Generate plot outputs for the multi-device run.

    Conventions
    -----------
    - Multi latency and multi CPU plots are created once for the entire run.
    - Memory plots are created per device; management devices (AUTO/HETERO) are skipped.
    - Summary plots are created per device if any relevant metrics exist.
    - File names are aligned with single-run naming where feasible.
    """
    plot_multi_device_latency(results, plots_dir / "latency_multi_device.png")
    plot_cpu_utilization(results, plots_dir / "cpu_multi_device.png")

    for dev, res in results.items():
        dev_str = str(dev)
        dev_up = dev_str.upper()
        safe_dev = sanitize_device_key(dev_str)

        # Management devices often do not expose stable memory metrics.
        if dev_up.startswith("AUTO") or dev_up.startswith("HETERO"):
            print(f"[INFO] Skipping memory plot for {dev_up} (management device).")
            continue

        mem_block = res.get("memory") or res.get("metrics", {}).get("memory")
        if not isinstance(mem_block, dict) or not mem_block:
            print(f"[INFO] No memory block for {dev_up}; skipping RAM plot.")
            continue

        try:
            plot_memory_over_time(mem_block, plots_dir / f"ram_over_time_{safe_dev}.png")
        except Exception as exc:
            print(f"[WARN] RAM plot failed for {dev_up}: {exc}")

    for dev, res in results.items():
        dev_str = str(dev)
        safe_dev = sanitize_device_key(dev_str)

        has_any_metrics = any(
            [
                res.get("timing_ms"),
                res.get("metrics", {}).get("inference_time_ms"),
                res.get("cpu_utilization_pct"),
                res.get("memory"),
                res.get("metrics", {}).get("memory"),
            ]
        )
        if not has_any_metrics:
            continue

        try:
            plot_device_summary(res, dev_str, plots_dir / f"summary_{safe_dev}.png")
        except Exception as exc:
            print(f"[WARN] Summary plot failed for {dev_str}: {exc}")


def main() -> None:
    """CLI main entry point."""
    args = parse_args()
    cfg = load_config(args.config)
    cfg = _apply_runtime_overrides(cfg, args)
    validate_config_contract(cfg)

    model_path = cfg["model"]["path"]
    backend = str(cfg["model"]["backend"]).lower()
    devices = cfg["devices_to_test"]

    # Base output dir (defaults to "runs")
    base_runs_dir = Path(str((cfg.get("output") or {}).get("base_dir", "runs")))

    # Run name:
    # - Prefer explicit output.run_name
    # - Else derive something informative and stable
    run_name = (cfg.get("output") or {}).get("run_name") or f"{backend}_multi"

    base_dir, plots_dir = _make_run_dirs(base_runs_dir, run_name)

    print("\n========================")
    print(" Starting multi-device run")
    print("========================\n")

    loop = DeviceLoop(model_path, backend, devices, cfg)
    results = loop.run_all()

    _write_results_json(base_dir, results)

    if args.plot:
        _plot_outputs(results, plots_dir)

    if args.report:
        # Writes <base_dir>/bench.md
        create_markdown_report(base_dir, results)

    print("\n[FINISHED] Multi-device benchmark completed.")
    print(f"Results -> {base_dir.as_posix()}")


if __name__ == "__main__":
    main()

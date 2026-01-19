# bench/cli/bench.py
"""
CLI entry point for the single-run benchmark pipeline.

Responsibilities:
- Parse CLI arguments
- Load YAML config
- Apply runtime-only overrides (--verbose, --log-level)
- Delegate execution to `bench.core.main.main`
- If --plot is set:
  - Generate plots into the run output directory
  - Generate a Markdown report

Security notes
--------------
- We do not trust device strings for file paths. Filenames are sanitized.
- We avoid relying on "latest directory" heuristics. Instead, we use the JSON path
  returned by bench.core.main.main() as the source of truth.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bench.core.config.config_loader import load_config
from bench.core.main import main

# Viz/report utilities (single-run)
from bench.core.analysis.viz.plots_latency import plot_inference_boxplot
from bench.core.analysis.viz.plots_memory import plot_memory_over_time
from bench.core.analysis.viz.plots_cpu_single import plot_cpu_util_single
from bench.core.analysis.viz.plots_summary import plot_device_summary
from bench.core.analysis.viz.report_markdown import create_markdown_report


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for a single benchmark run.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with:
        - config: Path to the YAML configuration file
        - plot: Boolean flag to generate plots and a markdown report
    """
    parser = argparse.ArgumentParser(description="Benchmark pipeline (single run)")
    parser.add_argument(
        "config",
        help="Path to the YAML configuration (e.g., bench/configs/local/onnx_cpu.yaml)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate PNG plots and Markdown report in the run output directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="DEBUG/INFO/WARNING/ERROR",
    )
    return parser.parse_args()


def _apply_runtime_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Apply CLI-level runtime overrides to the loaded configuration.

    Notes
    -----
    - Overrides are applied without modifying the YAML file.
    - Only shallow overrides are applied (cfg["run"][...]).
    """
    run_cfg = cfg.setdefault("run", {})

    if args.verbose:
        run_cfg["verbose"] = True

    if args.log_level:
        run_cfg["log_level"] = str(args.log_level).upper()

    return cfg


def _sanitize_device_key(device: str) -> str:
    """
    Sanitize a device identifier so it can be used safely in filenames.

    This prevents invalid Windows characters and reduces path traversal risk.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(device))
    return safe if safe else "_"


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file as dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _generate_single_run_outputs(json_path: Path) -> None:
    """
    Generate plots and a Markdown report for a single-run JSON result.

    We treat json_path as the single source of truth:
    - plots are stored in <run_dir>/plots/
    - report is stored as <run_dir>/bench.md (as per report_markdown)
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Result JSON not found: {json_path}")

    run_dir = json_path.parent
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = _load_json(json_path)

    # Defensive access patterns: tolerate older JSON shapes
    meta = data.get("meta") or {}
    metrics = data.get("metrics") or {}

    dev = meta.get("device_target") or "device"
    safe_dev = _sanitize_device_key(str(dev))

    # 1) Latency boxplot (requires inference_time_ms.samples)
    try:
        plot_inference_boxplot(data, plots_dir / f"latency_boxplot_{safe_dev}.png")
    except Exception as e:
        print(f"[WARN] Failed to generate latency plot: {e}")

    # 2) Memory plot (host RSS timeline)
    mem_block = metrics.get("memory") or data.get("memory")
    if isinstance(mem_block, dict) and mem_block.get("rss_samples"):
        try:
            plot_memory_over_time(mem_block, plots_dir / f"ram_over_time_{safe_dev}.png")
        except Exception as e:
            print(f"[WARN] Failed to generate memory plot: {e}")
    else:
        print("[INFO] No memory samples found; skipping RAM plot.")

    # 3) CPU utilization (single-run)
    cpu_block = metrics.get("cpu_utilization_pct") or data.get("cpu_utilization_pct")
    if isinstance(cpu_block, dict) and ("mean" in cpu_block or "p95" in cpu_block):
        try:
            plot_cpu_util_single(metrics, plots_dir / f"cpu_utilization_{safe_dev}.png")
        except Exception as e:
            print(f"[WARN] Failed to generate CPU plot: {e}")
    else:
        print("[INFO] No CPU utilization block found; skipping CPU plot.")

    # 4) Summary figure
    try:
        plot_device_summary(data, str(dev), plots_dir / f"summary_{safe_dev}.png")
    except Exception as e:
        print(f"[WARN] Failed to generate summary plot: {e}")

    # 5) Markdown report
    try:
        create_markdown_report(run_dir, data)
        
    except Exception as e:
        print(f"[WARN] Failed to generate Markdown report: {e}")

    print(f" Plots/Report generated in: {run_dir}")


def _run() -> None:
    """
    Execute the benchmark based on CLI arguments.

    We keep CLI logic minimal and use bench.core.main.main() for execution.
    Plot/report generation is triggered here (CLI layer), not inside core/main.py.
    """
    args = parse_args()
    cfg = load_config(args.config)
    cfg = _apply_runtime_overrides(cfg, args)

    # Run benchmark; main() returns a list of (device, json_path)
    written: List[Tuple[str, str]] = main(cfg_path=args.config, generate_plots=args.plot, cfg=cfg)

    if args.plot:
        # Single-run can still produce multiple devices if device.target is a list.
        # We generate per-device plots in the corresponding run directories.
        for dev, jp in written:
            try:
                _generate_single_run_outputs(Path(jp))
            except Exception as e:
                print(f"[WARN] Plot/report generation failed for device={dev}: {e}")


if __name__ == "__main__":
    _run()

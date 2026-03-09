# bench/core/main.py
"""
Main entry point for the benchmark pipeline (Torch / ONNX / OpenVINO / TensorRT).

Responsibilities:
- Load configuration (unless cfg is injected)
- Configure logging
- Collect hardware and environment metadata
- Run benchmarks across one or multiple device targets
- Compute model-level metrics (MACs / model size) once per run
- Persist structured benchmark results via result_writer

Design notes
------------
- Plot generation is intentionally NOT implemented here.
  The CLI layer (bench/cli/bench.py, bench/cli/multi.py) decides whether plots/reports
  should be generated and calls the visualization utilities accordingly.
"""

from __future__ import annotations

import logging
import platform
from copy import deepcopy
from typing import Any, Dict, Optional, List, Tuple

from pydantic import ValidationError

from bench.core.config.config_loader import load_config
from bench.core.io.result_writer import write_result
from bench.core.measure.measure_controller import MeasureController
from bench.core.metrics.macs_meter import MacsMeter
from bench.core.metrics.model_size_meter import ModelSizeMeter
from bench.core.system.hardware_probe import collect_system_snapshot
from bench.core.schemas.run_schema import (
    RunSchema,
    MetaSchema,
    ModelSchema,
    MetricsSchema,
    HardwareSchema,
    EnvSchema,
)
from bench.core.runner.torch_runner import TorchRunner

logger = logging.getLogger(__name__)

TensorRTRunner = None
if platform.system() != "Windows":
    try:
        from bench.core.runner.tensorrt_runner import TensorRTRunner  # type: ignore
    except Exception:
        TensorRTRunner = None


def _setup_logging(cfg: Dict[str, Any]) -> None:
    """
    Configure logging for the benchmark run.

    Supported config options:
    - run.log_level: DEBUG | INFO | WARNING | ERROR
    - run.verbose: bool (forces DEBUG)
    """
    run_cfg = cfg.get("run", {}) or {}
    verbose = bool(run_cfg.get("verbose", False))

    level_str = str(run_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    if verbose:
        level = logging.DEBUG

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root.setLevel(level)


def _validate_single_contract(cfg: Dict[str, Any]) -> None:
    """
    Validate the minimal config contract required by the single-run pipeline.

    Enforces the unified schema:
    - model.path (required)
    - model.backend (required)
    - device.target (required; string or list)
    - run.warmups (>= 0)
    - run.repeats (>= 1)
    """
    model = cfg.get("model") or {}
    if not model.get("path"):
        raise ValueError("Invalid config: model.path is required")
    if not model.get("backend"):
        raise ValueError("Invalid config: model.backend is required")

    run_cfg = cfg.get("run") or {}
    warmups = run_cfg.get("warmups")
    repeats = run_cfg.get("repeats")

    if warmups is None or int(warmups) < 0:
        raise ValueError("Invalid config: run.warmups must be an integer >= 0")
    if repeats is None or int(repeats) < 1:
        raise ValueError("Invalid config: run.repeats must be an integer >= 1")

    device_cfg = cfg.get("device") or {}
    if "target" not in device_cfg:
        raise ValueError("Invalid config: device.target is required (string or list of strings)")


def _patch_cfg_for_device(cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    """
    Create a per-device config view.

    This prevents cross-device leakage if any component reads cfg["device"]["target"].
    """
    cfg2 = deepcopy(cfg)
    cfg2.setdefault("device", {})
    cfg2["device"]["target"] = device
    return cfg2


def _ensure_list_targets(targets: Any) -> List[str]:
    """Normalize device targets to a list of strings."""
    if isinstance(targets, str):
        return [targets]
    if isinstance(targets, list):
        return [str(t) for t in targets]
    raise ValueError("Invalid config: device.target must be a string or a list of strings")


def main(
    cfg_path: Optional[str] = None,
    generate_plots: bool = False,  # kept for backward-compatibility (ignored here)
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """
    Execute a benchmark run based on the provided configuration.

    Parameters
    ----------
    cfg_path:
        YAML path, used only if cfg is not injected.
    generate_plots:
        Backward-compatibility flag. Plot generation is handled by the CLI layer.
    cfg:
        Injected configuration dict (useful for tests / no I/O).

    Returns
    -------
    List[Tuple[str, str]]
        A list of (device, json_path) for each successfully executed device.
        This return value enables the CLI to reliably locate outputs for plotting/reporting.
    """
    cfg = cfg if cfg is not None else load_config(cfg_path)
    _validate_single_contract(cfg)
    _setup_logging(cfg)

    logger.info("Starting benchmark pipeline")
    logger.debug("Config path: %s", cfg_path)

    # ------------------------------------------------------------------
    # Hardware & environment snapshot (once per run)
    # ------------------------------------------------------------------
    system_info = collect_system_snapshot(model_path=cfg["model"]["path"])
    try:
        hardware_schema = HardwareSchema(**system_info["hardware"])
        env_schema = EnvSchema(**system_info["env"])
    except ValidationError:
        logger.error("Schema validation failed", exc_info=True)
        raise

    backend = str(cfg["model"]["backend"]).lower()
    targets = _ensure_list_targets(cfg["device"]["target"])
    threads = int(cfg.get("run", {}).get("threads", 1))

    results_per_device: List[Tuple[str, Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Per-device execution loop
    # ------------------------------------------------------------------
    for dev in targets:
        logger.info("Running benchmark for device=%s backend=%s", dev, backend)

        per_dev_cfg = _patch_cfg_for_device(cfg, dev)
        controller = MeasureController(per_dev_cfg)

        # Runner selection
        if backend == "torch":
            runner = TorchRunner(
                model_path=per_dev_cfg["model"]["path"],
                device=dev,
                threads=threads,
            )

        elif backend == "onnx":
            from bench.core.runner.onnx_runner import OnnxRunner  # lazy import
            runner = OnnxRunner(
                per_dev_cfg["model"]["path"],
                dev,
                threads=threads,
                inter_op_threads=per_dev_cfg.get("run", {}).get("inter_op_threads"),
                execution_mode=per_dev_cfg.get("run", {}).get("execution_mode"),
            )

        elif backend == "openvino":
            from bench.core.runner.openvino_runner import OpenVinoRunner  # lazy import
            runner = OpenVinoRunner(per_dev_cfg["model"]["path"], dev)

        elif backend in ("tensorrt", "trt"):
            if TensorRTRunner is None:
                logger.warning("TensorRT not available – skipping device=%s", dev)
                continue

            runner = TensorRTRunner(
                model_path=per_dev_cfg["model"]["path"],
                device=dev,
                input_shape=per_dev_cfg["input"]["shape"],
                fp16=bool(per_dev_cfg.get("precision", {}).get("fp16", False)),
            )
        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

        try:
            runner.load()
            measure_results = controller.run_benchmark(runner)
            results_per_device.append((dev, measure_results))
        finally:
            runner.teardown()

    # ------------------------------------------------------------------
    # Model-level metrics (once per run)
    # ------------------------------------------------------------------
    meter_backend = "onnx" if backend in ("tensorrt", "trt") else backend

    macs_result = MacsMeter(meter_backend).analyze(
        cfg["model"]["path"],
        cfg["input"]["shape"],
    ) or {}

    size_result = ModelSizeMeter(meter_backend).analyze(cfg["model"]["path"]) or {}

    # ------------------------------------------------------------------
    # Persist results (one JSON per device)
    # ------------------------------------------------------------------
    written: List[Tuple[str, str]] = []

    for dev, measure_results in results_per_device:
        mem = measure_results.get("memory")

        # Prefer runtime-derived input facts to avoid drifting away from what was actually benchmarked.
        actual_input_shape = measure_results.get("actual_input_shape") or cfg["input"]["shape"]
        input_fs_hz = measure_results.get("input_fs_hz")
        input_num_samples = measure_results.get("input_num_samples")
        input_duration_s = measure_results.get("input_duration_s")
        ms_per_signal_s = measure_results.get("ms_per_signal_s")

        run = RunSchema(
            meta=MetaSchema(framework=backend, device_target=dev),
            model=ModelSchema(
                path=cfg["model"]["path"],
                input_shape=actual_input_shape,
                dtype=cfg["model"].get("dtype", "fp32"),
                fs_hz=input_fs_hz,
                input_num_samples=input_num_samples,
                input_duration_s=input_duration_s,
                parameters=size_result.get("parameters_total"),
                size_on_disk_bytes=size_result.get("total_on_disk_bytes"),
            ),
            metrics=MetricsSchema(
                macs=macs_result,
                inference_time_ms=measure_results.get("timing_ms"),
                throughput_sps=measure_results.get("throughput_sps"),
                cpu_utilization_pct=measure_results.get("cpu_utilization_pct"),
                memory=mem,
                ms_per_signal_s=ms_per_signal_s,
            ),
            hardware=hardware_schema,
            env=env_schema,
            config=cfg,
            thread_config=measure_results.get("thread_config"),
        )

        json_path = write_result(run)
        written.append((dev, json_path))
        logger.info("Result written: %s", json_path)

    logger.info("Benchmark pipeline finished successfully")
    return written

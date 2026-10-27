# bench/core/config/config_loader.py
"""
Configuration loader for the benchmark pipeline.

Responsibilities:
- Load YAML configuration files
- Merge user config over defaults (deep merge)
- Provide a stable config contract for the rest of the pipeline

Single source of truth:
- run.warmups / run.repeats control warm-up and measured iterations
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {"path": None, "backend": "torch", "dtype": "fp32"},
    "input": {"shape": [1, 1, 3600], "batch": 1, "fs_hz": None},
    "device": {"target": "cpu", "allow_fallback": False},
    # Single schema: warmups/repeats live here for both single- and multi-device flows
    "run": {"warmups": 5, "repeats": 10, "threads": 4, "log_level": "INFO", "verbose": False},
    "metrics": {
        "sampler_hz": 75,
        "pre_roll_s": 5.0,
        "post_delay_s": 2.0,
        "memory_mode": "static",
        "tegrastats": False,
        "energy": {
            "enabled": False,
            "min_duration_s": 2.0,
            "max_repeats": 10000,
        },
    },
    "experiment": {
        "context": {
            "enabled": False,
            "preflight": {
                "enabled": False,
                "duration_s": 5.0,
                "sample_hz": 2.0,
                "max_background_cpu_pct": None,
                "max_start_temp_c": None,
                "require_ac_power": False,
            },
            "manual": {
                "dell_thermal_mode": None,
                "cooling_setup": None,
                "room_temperature_c": None,
                "notes": None,
            },
        }
    },
    "output": {"dir": "runs", "base_dir": "runs", "formats": ["json"], "profile_level": "basic", "store_json": True, "store_plots": False},
}


def _deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge dicts: values in `new` override values in `base`.

    This is intentionally conservative:
    - dict + dict -> recursive merge
    - otherwise   -> overwrite
    """
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _validate_config_contract(cfg: Dict[str, Any]) -> None:
    """
    Validate the minimal contract required by the pipeline.

    This keeps failures early and actionable. Raise ValueError on issues.
    """
    model = cfg.get("model") or {}
    if not model.get("backend"):
        raise ValueError("Invalid config: model.backend is required")
    if not model.get("path"):
        raise ValueError("Invalid config: model.path is required")

    run = cfg.get("run") or {}
    warmups = run.get("warmups")
    repeats = run.get("repeats")

    if warmups is None or int(warmups) < 0:
        raise ValueError("Invalid config: run.warmups must be an integer >= 0")
    if repeats is None or int(repeats) < 1:
        raise ValueError("Invalid config: run.repeats must be an integer >= 1")

    metrics = cfg.get("metrics") or {}
    energy = metrics.get("energy") or {}

    enabled = energy.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("Invalid config: metrics.energy.enabled must be a boolean")

    min_duration_s = energy.get("min_duration_s", 2.0)
    if isinstance(min_duration_s, bool):
        raise ValueError("Invalid config: metrics.energy.min_duration_s must be a number > 0")
    try:
        min_duration_value = float(min_duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid config: metrics.energy.min_duration_s must be a number > 0"
        ) from exc
    if min_duration_value <= 0:
        raise ValueError("Invalid config: metrics.energy.min_duration_s must be a number > 0")


    backend = str(energy.get("backend", "auto")).strip().lower()
    if backend not in {
        "auto",
        "windows_emi",
        "emi",
        "rapl",
        "nvml",
        "jetson",
        "jetson_ina3221",
    }:
        raise ValueError(
            "Invalid config: metrics.energy.backend must be one of "
            "auto, windows_emi/emi, rapl, nvml, jetson_ina3221/jetson"
        )

    emi_channel_raw = energy.get("emi_channel", "RAPL_Package0_PKG")
    if emi_channel_raw is None or not str(emi_channel_raw).strip():
        raise ValueError("Invalid config: metrics.energy.emi_channel must not be empty")

    baseline = energy.get("baseline_correction", {})
    # Backward compatibility: an earlier planning document used
    # ``baseline_correction: false`` as the disabled form.
    if baseline is None or baseline is False:
        baseline = {}
    if not isinstance(baseline, dict):
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction must be a mapping or false"
        )
    baseline_enabled = baseline.get("enabled", False)
    if not isinstance(baseline_enabled, bool):
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction.enabled must be a boolean"
        )
    if baseline_enabled and not enabled:
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction requires metrics.energy.enabled=true"
        )

    baseline_duration_s = baseline.get("duration_s", 2.0)
    if isinstance(baseline_duration_s, bool):
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction.duration_s must be a number > 0"
        )
    try:
        baseline_duration_value = float(baseline_duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction.duration_s must be a number > 0"
        ) from exc
    if not math.isfinite(baseline_duration_value) or baseline_duration_value <= 0:
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction.duration_s must be a finite number > 0"
        )

    baseline_repetitions = baseline.get("repetitions", 3)
    if isinstance(baseline_repetitions, bool):
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction.repetitions must be an integer >= 1"
        )
    try:
        baseline_repetitions_value = int(baseline_repetitions)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction.repetitions must be an integer >= 1"
        ) from exc
    if (
        baseline_repetitions_value < 1
        or float(baseline_repetitions_value) != float(baseline_repetitions)
    ):
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction.repetitions must be an integer >= 1"
        )

    baseline_method = str(baseline.get("method", "post_energy_idle")).strip().lower()
    if baseline_method != "post_energy_idle":
        raise ValueError(
            "Invalid config: metrics.energy.baseline_correction.method must be post_energy_idle"
        )

    max_repeats = energy.get("max_repeats", 10000)
    if isinstance(max_repeats, bool):
        raise ValueError("Invalid config: metrics.energy.max_repeats must be an integer >= 1")
    try:
        max_repeats_value = int(max_repeats)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid config: metrics.energy.max_repeats must be an integer >= 1"
        ) from exc
    if max_repeats_value < 1 or float(max_repeats_value) != float(max_repeats):
        raise ValueError("Invalid config: metrics.energy.max_repeats must be an integer >= 1")

    experiment = cfg.get("experiment") or {}
    context = experiment.get("context") or {}
    context_enabled = context.get("enabled", False)
    if not isinstance(context_enabled, bool):
        raise ValueError("Invalid config: experiment.context.enabled must be a boolean")

    preflight = context.get("preflight") or {}
    preflight_enabled = preflight.get("enabled", False)
    if not isinstance(preflight_enabled, bool):
        raise ValueError("Invalid config: experiment.context.preflight.enabled must be a boolean")

    duration_s = preflight.get("duration_s", 5.0)
    sample_hz = preflight.get("sample_hz", 2.0)
    for name, value in (("duration_s", duration_s), ("sample_hz", sample_hz)):
        if isinstance(value, bool):
            raise ValueError(f"Invalid config: experiment.context.preflight.{name} must be > 0")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid config: experiment.context.preflight.{name} must be > 0"
            ) from exc
        if numeric <= 0:
            raise ValueError(f"Invalid config: experiment.context.preflight.{name} must be > 0")

    for name in ("max_background_cpu_pct", "max_start_temp_c"):
        value = preflight.get(name)
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(
                f"Invalid config: experiment.context.preflight.{name} must be a number or null"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid config: experiment.context.preflight.{name} must be a number or null"
            ) from exc
        if numeric < 0:
            raise ValueError(
                f"Invalid config: experiment.context.preflight.{name} must be >= 0 or null"
            )

    require_ac = preflight.get("require_ac_power", False)
    if not isinstance(require_ac, bool):
        raise ValueError(
            "Invalid config: experiment.context.preflight.require_ac_power must be a boolean"
        )


def load_config(cfg_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file and merge it with DEFAULT_CONFIG.

    Notes:
    - If cfg_path is None or missing, defaults are used.
    - Config validation is enforced to prevent ambiguous runtime behavior.
    """
    cfg: Dict[str, Any] = deepcopy(DEFAULT_CONFIG)

    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    else:
        print("[WARN] No YAML config found; using defaults.")

    _validate_config_contract(cfg)
    return cfg

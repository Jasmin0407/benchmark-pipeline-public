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

import os
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {"path": None, "backend": "torch", "dtype": "fp32"},
    "input": {"shape": [1, 1, 3600], "batch": 1},
    "device": {"target": "cpu", "allow_fallback": False},
    # Single schema: warmups/repeats live here for both single- and multi-device flows
    "run": {"warmups": 5, "repeats": 10, "threads": 4, "log_level": "INFO", "verbose": False},
    "metrics": {"sampler_hz": 75, "pre_roll_s": 5.0, "post_delay_s": 2.0, "memory_mode": "static", "tegrastats": False},
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


def load_config(cfg_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file and merge it with DEFAULT_CONFIG.

    Notes:
    - If cfg_path is None or missing, defaults are used.
    - Config validation is enforced to prevent ambiguous runtime behavior.
    """
    cfg: Dict[str, Any] = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}

    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    else:
        print("[WARN] No YAML config found; using defaults.")

    _validate_config_contract(cfg)
    return cfg

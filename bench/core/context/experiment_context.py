"""High-level experiment-context collection for benchmark reproducibility."""

from __future__ import annotations

import platform
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from .preflight import collect_preflight_observations, evaluate_preflight
from .windows_power_context import collect_windows_power_context


def collect_experiment_context(
    cfg: Dict[str, Any],
    *,
    platform_name: Optional[str] = None,
    power_collector: Optional[Callable[[], Dict[str, Any]]] = None,
    preflight_collector: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Collect optional benchmark context before the measured benchmark begins."""
    context_cfg = (((cfg.get("experiment") or {}).get("context")) or {})
    if not bool(context_cfg.get("enabled", False)):
        return None

    system_name = platform_name or platform.system()
    manual = deepcopy(context_cfg.get("manual") or {})

    if power_collector is not None:
        power = power_collector()
    else:
        power = collect_windows_power_context(platform_name=system_name)

    result: Dict[str, Any] = {
        "platform": str(system_name).lower(),
        "power": power,
        "manual": manual,
    }

    preflight_cfg = context_cfg.get("preflight") or {}
    if not bool(preflight_cfg.get("enabled", False)):
        result["preflight"] = {
            "passed": None,
            "status": "disabled",
            "warnings": [],
        }
        return result

    collector = preflight_collector or collect_preflight_observations
    observations = collector(
        duration_s=float(preflight_cfg.get("duration_s", 5.0)),
        sample_hz=float(preflight_cfg.get("sample_hz", 2.0)),
    )
    result.update(observations)
    result["preflight"] = evaluate_preflight(
        rules=preflight_cfg,
        power=power,
        background=observations.get("background") or {},
        thermal=observations.get("thermal") or {},
    )
    return result

def finalize_experiment_context(
    cfg: Dict[str, Any],
    context: Optional[Dict[str, Any]],
    *,
    platform_name: Optional[str] = None,
    power_collector: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Finalize audit context after all measured benchmark passes have finished.

    Only a read-only power-state snapshot is collected here. Keeping this step after
    the Energy pass prevents power/context queries from perturbing timing or energy
    measurements while still allowing the result to record whether AC state or the
    active Windows power scheme changed during the run.
    """
    context_cfg = (((cfg.get("experiment") or {}).get("context")) or {})
    if not bool(context_cfg.get("enabled", False)) or context is None:
        return context

    result = deepcopy(context)
    system_name = platform_name or platform.system()
    if power_collector is not None:
        power_end = power_collector()
    else:
        power_end = collect_windows_power_context(platform_name=system_name)

    result["power_end"] = power_end

    warnings = []
    compared = 0
    power_start = result.get("power") or {}
    for key, label in (
        ("active_scheme_guid", "Windows power scheme"),
        ("ac_connected", "AC connection state"),
    ):
        start_value = power_start.get(key)
        end_value = power_end.get(key)
        if start_value is None or end_value is None:
            continue
        compared += 1
        if start_value != end_value:
            warnings.append(
                f"{label} changed during the benchmark: {start_value!r} -> {end_value!r}."
            )

    result["finalization"] = {
        "power_state_consistent": (not warnings) if compared else None,
        "compared_fields": compared,
        "warnings": warnings,
    }
    return result


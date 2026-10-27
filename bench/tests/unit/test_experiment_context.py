from __future__ import annotations

from bench.core.context.experiment_context import (
    collect_experiment_context,
    finalize_experiment_context,
)


def test_context_disabled_performs_no_collectors():
    def fail():
        raise AssertionError("collector must not be called")

    cfg = {"experiment": {"context": {"enabled": False}}}
    assert collect_experiment_context(cfg, power_collector=fail) is None


def test_context_enabled_without_preflight_preserves_manual_metadata():
    cfg = {
        "experiment": {
            "context": {
                "enabled": True,
                "preflight": {"enabled": False},
                "manual": {
                    "dell_thermal_mode": "Ultra Performance",
                    "cooling_setup": "Laptop on desk",
                },
            }
        }
    }
    result = collect_experiment_context(
        cfg,
        platform_name="Windows",
        power_collector=lambda: {"available": True, "ac_connected": True},
    )
    assert result["platform"] == "windows"
    assert result["manual"]["dell_thermal_mode"] == "Ultra Performance"
    assert result["preflight"]["status"] == "disabled"


def test_context_enabled_with_preflight_is_evaluated():
    cfg = {
        "experiment": {
            "context": {
                "enabled": True,
                "preflight": {
                    "enabled": True,
                    "duration_s": 1.0,
                    "sample_hz": 1.0,
                    "max_background_cpu_pct": 5.0,
                    "require_ac_power": True,
                },
            }
        }
    }
    observations = {
        "background": {"available": True, "cpu_max_pct": 2.0},
        "cpu_frequency": {"available": True, "mean_mhz": 3000.0},
        "thermal": {"available": False},
    }
    result = collect_experiment_context(
        cfg,
        platform_name="Windows",
        power_collector=lambda: {"available": True, "ac_connected": True},
        preflight_collector=lambda **_kwargs: observations,
    )
    assert result["preflight"]["passed"] is True
    assert result["background"]["cpu_max_pct"] == 2.0
    assert result["cpu_frequency"]["mean_mhz"] == 3000.0

def test_context_finalization_detects_stable_power_state():
    cfg = {"experiment": {"context": {"enabled": True}}}
    initial = {
        "platform": "windows",
        "power": {
            "available": True,
            "active_scheme_guid": "scheme-a",
            "ac_connected": True,
        },
    }

    result = finalize_experiment_context(
        cfg,
        initial,
        platform_name="Windows",
        power_collector=lambda: {
            "available": True,
            "active_scheme_guid": "scheme-a",
            "ac_connected": True,
        },
    )

    assert result is not None
    assert result["power_end"]["ac_connected"] is True
    assert result["finalization"]["power_state_consistent"] is True
    assert result["finalization"]["compared_fields"] == 2
    assert result["finalization"]["warnings"] == []


def test_context_finalization_flags_power_scheme_change():
    cfg = {"experiment": {"context": {"enabled": True}}}
    initial = {
        "power": {
            "active_scheme_guid": "scheme-a",
            "ac_connected": True,
        }
    }

    result = finalize_experiment_context(
        cfg,
        initial,
        platform_name="Windows",
        power_collector=lambda: {
            "active_scheme_guid": "scheme-b",
            "ac_connected": True,
        },
    )

    assert result is not None
    assert result["finalization"]["power_state_consistent"] is False
    assert "power scheme" in result["finalization"]["warnings"][0].lower()


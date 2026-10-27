from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bench.core.config.config_loader import load_config


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _minimal_config() -> dict:
    return {
        "model": {
            "path": "model.onnx",
            "backend": "onnx",
        },
        "run": {
            "warmups": 1,
            "repeats": 2,
        },
    }


def test_energy_defaults_are_disabled(tmp_path):
    """Energy measurement must remain opt-in for backward compatibility."""
    config_path = _write_config(tmp_path / "config.yaml", _minimal_config())

    cfg = load_config(str(config_path))

    assert cfg["metrics"]["energy"] == {
        "enabled": False,
        "min_duration_s": 2.0,
        "max_repeats": 10000,
    }


def test_legacy_config_without_energy_section_remains_valid(tmp_path):
    """Existing YAML files must load without adding an explicit energy block."""
    payload = _minimal_config()
    payload["metrics"] = {
        "sampler_hz": 50,
        "memory_mode": "dynamic",
    }
    config_path = _write_config(tmp_path / "legacy.yaml", payload)

    cfg = load_config(str(config_path))

    assert cfg["metrics"]["sampler_hz"] == 50
    assert cfg["metrics"]["memory_mode"] == "dynamic"
    assert cfg["metrics"]["energy"]["enabled"] is False


def test_windows_emi_backend_and_channel_are_accepted(tmp_path):
    payload = _minimal_config()
    payload["metrics"] = {
        "energy": {
            "enabled": True,
            "backend": "windows_emi",
            "emi_channel": "RAPL_Package0_PKG",
        }
    }
    config_path = _write_config(tmp_path / "emi.yaml", payload)

    cfg = load_config(str(config_path))

    assert cfg["metrics"]["energy"]["backend"] == "windows_emi"
    assert cfg["metrics"]["energy"]["emi_channel"] == "RAPL_Package0_PKG"


def test_energy_override_does_not_mutate_defaults_between_loads(tmp_path):
    """Nested user overrides must never leak into subsequent config loads."""
    enabled_payload = _minimal_config()
    enabled_payload["metrics"] = {
        "energy": {
            "enabled": True,
            "min_duration_s": 3.5,
            "max_repeats": 2500,
        }
    }
    enabled_path = _write_config(tmp_path / "energy_enabled.yaml", enabled_payload)
    legacy_path = _write_config(tmp_path / "legacy.yaml", _minimal_config())

    enabled_cfg = load_config(str(enabled_path))
    legacy_cfg = load_config(str(legacy_path))

    assert enabled_cfg["metrics"]["energy"]["enabled"] is True
    assert enabled_cfg["metrics"]["energy"]["min_duration_s"] == 3.5
    assert enabled_cfg["metrics"]["energy"]["max_repeats"] == 2500
    assert legacy_cfg["metrics"]["energy"] == {
        "enabled": False,
        "min_duration_s": 2.0,
        "max_repeats": 10000,
    }


@pytest.mark.parametrize(
    ("energy_config", "message"),
    [
        ({"enabled": "false"}, "metrics.energy.enabled"),
        ({"min_duration_s": 0}, "metrics.energy.min_duration_s"),
        ({"min_duration_s": -1.0}, "metrics.energy.min_duration_s"),
        ({"max_repeats": 0}, "metrics.energy.max_repeats"),
        ({"max_repeats": 1.5}, "metrics.energy.max_repeats"),
        ({"backend": "not-a-backend"}, "metrics.energy.backend"),
        ({"backend": "windows_emi", "emi_channel": "   "}, "metrics.energy.emi_channel"),
        ({"backend": "windows_emi", "emi_channel": None}, "metrics.energy.emi_channel"),
    ],
)
def test_invalid_energy_config_is_rejected(tmp_path, energy_config, message):
    """Invalid energy settings must fail early with an actionable error."""
    payload = _minimal_config()
    payload["metrics"] = {"energy": energy_config}
    config_path = _write_config(tmp_path / "invalid.yaml", payload)

    with pytest.raises(ValueError, match=message):
        load_config(str(config_path))


def test_phase13_baseline_correction_config_is_optional_and_valid(tmp_path):
    payload = _minimal_config()
    payload["metrics"] = {
        "energy": {
            "enabled": True,
            "backend": "windows_emi",
            "emi_channel": "RAPL_Package0_PKG",
            "baseline_correction": {
                "enabled": True,
                "method": "post_energy_idle",
                "duration_s": 2.0,
                "repetitions": 5,
            },
        }
    }
    config_path = _write_config(tmp_path / "baseline.yaml", payload)

    cfg = load_config(str(config_path))

    baseline = cfg["metrics"]["energy"]["baseline_correction"]
    assert baseline["enabled"] is True
    assert baseline["method"] == "post_energy_idle"
    assert baseline["duration_s"] == 2.0
    assert baseline["repetitions"] == 5


@pytest.mark.parametrize(
    ("baseline_config", "message"),
    [
        ({"enabled": "true"}, "baseline_correction.enabled"),
        ({"enabled": True, "duration_s": 0}, "baseline_correction.duration_s"),
        ({"enabled": True, "repetitions": 0}, "baseline_correction.repetitions"),
        ({"enabled": True, "repetitions": 1.5}, "baseline_correction.repetitions"),
        ({"enabled": True, "method": "pre_energy_idle"}, "baseline_correction.method"),
    ],
)
def test_invalid_phase13_baseline_config_is_rejected(
    tmp_path,
    baseline_config,
    message,
):
    payload = _minimal_config()
    payload["metrics"] = {
        "energy": {
            "enabled": True,
            "baseline_correction": baseline_config,
        }
    }
    config_path = _write_config(tmp_path / "invalid-baseline.yaml", payload)

    with pytest.raises(ValueError, match=message):
        load_config(str(config_path))


def test_phase13_baseline_correction_requires_energy_enabled(tmp_path):
    payload = _minimal_config()
    payload["metrics"] = {
        "energy": {
            "enabled": False,
            "baseline_correction": {
                "enabled": True,
            },
        }
    }
    config_path = _write_config(tmp_path / "baseline-without-energy.yaml", payload)

    with pytest.raises(ValueError, match="requires metrics.energy.enabled=true"):
        load_config(str(config_path))


@pytest.mark.parametrize("baseline_config", [[], "enabled", 1, True])
def test_phase13_baseline_correction_must_be_mapping(tmp_path, baseline_config):
    payload = _minimal_config()
    payload["metrics"] = {
        "energy": {
            "enabled": True,
            "baseline_correction": baseline_config,
        }
    }
    config_path = _write_config(tmp_path / "invalid-baseline-mapping.yaml", payload)

    with pytest.raises(ValueError, match="baseline_correction must be a mapping or false"):
        load_config(str(config_path))


def test_phase13_baseline_duration_must_be_finite(tmp_path):
    payload = _minimal_config()
    payload["metrics"] = {
        "energy": {
            "enabled": True,
            "baseline_correction": {
                "enabled": True,
                "duration_s": float("nan"),
            },
        }
    }
    config_path = _write_config(tmp_path / "nan-baseline.yaml", payload)

    with pytest.raises(ValueError, match="baseline_correction.duration_s"):
        load_config(str(config_path))


def test_legacy_boolean_false_baseline_config_is_accepted(tmp_path):
    payload = _minimal_config()
    payload["metrics"] = {
        "energy": {
            "enabled": True,
            "baseline_correction": False,
        }
    }
    config_path = _write_config(tmp_path / "legacy-baseline-false.yaml", payload)

    cfg = load_config(str(config_path))

    assert cfg["metrics"]["energy"]["baseline_correction"] is False

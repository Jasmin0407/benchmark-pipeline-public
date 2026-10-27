from __future__ import annotations

from bench.core.context.preflight import collect_preflight_observations, evaluate_preflight


def test_preflight_collects_background_and_frequency_together(monkeypatch):
    monkeypatch.setattr("bench.core.context.preflight.psutil.pids", lambda: [1, 2, 3])
    cpu_values = iter([1.0, 2.0, 3.0, 4.0])
    freq_values = iter([2000.0, 2500.0, 3000.0, 3500.0])
    result = collect_preflight_observations(
        duration_s=2.0,
        sample_hz=2.0,
        cpu_reader=lambda _interval: next(cpu_values),
        frequency_reader=lambda: next(freq_values),
    )
    assert result["background"]["cpu_mean_pct"] == 2.5
    assert result["background"]["cpu_max_pct"] == 4.0
    assert result["background"]["process_count"] == 3
    assert result["cpu_frequency"]["mean_mhz"] == 2750.0
    assert result["thermal"]["available"] is False


def test_preflight_low_background_and_ac_passes():
    result = evaluate_preflight(
        rules={"max_background_cpu_pct": 5.0, "require_ac_power": True},
        power={"ac_connected": True},
        background={"cpu_max_pct": 4.0},
        thermal={"available": False},
    )
    assert result["passed"] is True
    assert result["status"] == "passed"


def test_preflight_high_background_fails():
    result = evaluate_preflight(
        rules={"max_background_cpu_pct": 5.0},
        power={},
        background={"cpu_max_pct": 12.0},
        thermal={"available": False},
    )
    assert result["passed"] is False
    assert "exceeds" in result["warnings"][0]


def test_preflight_requires_ac_power():
    result = evaluate_preflight(
        rules={"require_ac_power": True},
        power={"ac_connected": False},
        background={},
        thermal={"available": False},
    )
    assert result["passed"] is False
    assert "AC power" in result["warnings"][0]


def test_preflight_temperature_limit_requires_valid_temperature():
    result = evaluate_preflight(
        rules={"max_start_temp_c": 55.0},
        power={},
        background={},
        thermal={"available": False},
    )
    assert result["passed"] is False
    assert "temperature is unavailable" in result["warnings"][0]


def test_preflight_without_thresholds_requires_manual_review():
    result = evaluate_preflight(
        rules={}, power={}, background={}, thermal={"available": False}
    )
    assert result["passed"] is None
    assert result["status"] == "manual_review_required"

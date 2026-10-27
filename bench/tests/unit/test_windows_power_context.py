from __future__ import annotations

from bench.core.context.windows_power_context import (
    collect_windows_power_context,
    parse_powercfg_active_scheme,
)


def test_parse_powercfg_active_scheme_english():
    parsed = parse_powercfg_active_scheme(
        "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"
    )
    assert parsed == {
        "guid": "381b4222-f694-41f0-9685-ff5bb260df2e",
        "name": "Balanced",
    }


def test_parse_powercfg_active_scheme_localized_prefix():
    parsed = parse_powercfg_active_scheme(
        "Aktives Energieschema: 381b4222-f694-41f0-9685-ff5bb260df2e  (Ausbalanciert)"
    )
    assert parsed["guid"] == "381b4222-f694-41f0-9685-ff5bb260df2e"
    assert parsed["name"] == "Ausbalanciert"


def test_collect_windows_power_context_ac_power():
    result = collect_windows_power_context(
        platform_name="Windows",
        powercfg_reader=lambda: "GUID: 381b4222-f694-41f0-9685-ff5bb260df2e (Balanced)",
        power_status_reader=lambda: {
            "ac_connected": True,
            "battery_percent": 87,
            "battery_saver": False,
        },
    )
    assert result["available"] is True
    assert result["ac_connected"] is True
    assert result["battery_percent"] == 87
    assert result["battery_saver"] is False


def test_collect_windows_power_context_battery_power():
    result = collect_windows_power_context(
        platform_name="Windows",
        powercfg_reader=lambda: "GUID: 381b4222-f694-41f0-9685-ff5bb260df2e (Balanced)",
        power_status_reader=lambda: {
            "ac_connected": False,
            "battery_percent": 55,
            "battery_saver": True,
        },
    )
    assert result["available"] is True
    assert result["ac_connected"] is False
    assert result["battery_saver"] is True


def test_collect_windows_power_context_query_failures_are_nonfatal():
    def fail():
        raise OSError("simulated failure")

    result = collect_windows_power_context(
        platform_name="Windows",
        powercfg_reader=fail,
        power_status_reader=fail,
    )
    assert result["available"] is False
    assert "reason" in result
    assert len(result["warnings"]) == 2


def test_collect_windows_power_context_non_windows_is_unavailable():
    result = collect_windows_power_context(platform_name="Linux")
    assert result["available"] is False
    assert "only available on Windows" in result["reason"]

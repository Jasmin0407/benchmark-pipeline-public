"""Read-only Windows power-state collection.

The helpers in this module never change the active Windows power scheme. They
only collect audit metadata for benchmark reproducibility.
"""

from __future__ import annotations

import ctypes
import platform
import re
import subprocess
from ctypes import wintypes
from typing import Any, Callable, Dict, Optional

_POWER_GUID_RE = re.compile(
    r"(?P<guid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", wintypes.BYTE),
        ("BatteryFlag", wintypes.BYTE),
        ("BatteryLifePercent", wintypes.BYTE),
        ("SystemStatusFlag", wintypes.BYTE),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


def parse_powercfg_active_scheme(output: str) -> Dict[str, Optional[str]]:
    """Parse ``powercfg /getactivescheme`` without depending on UI language."""
    text = str(output or "").strip()
    match = _POWER_GUID_RE.search(text)
    if match is None:
        return {"guid": None, "name": None}

    guid = match.group("guid").lower()
    name = None
    after_guid = text[match.end() :]
    paren_matches = re.findall(r"\(([^()]*)\)", after_guid)
    if paren_matches:
        candidate = paren_matches[-1].strip()
        name = candidate or None

    return {"guid": guid, "name": name}


def _run_powercfg() -> str:
    completed = subprocess.run(
        ["powercfg", "/getactivescheme"],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return completed.stdout or completed.stderr or ""


def _read_system_power_status() -> Dict[str, Any]:
    status = _SystemPowerStatus()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ok = kernel32.GetSystemPowerStatus(ctypes.byref(status))
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetSystemPowerStatus failed")

    ac_connected: Optional[bool]
    if status.ACLineStatus == 1:
        ac_connected = True
    elif status.ACLineStatus == 0:
        ac_connected = False
    else:
        ac_connected = None

    battery_percent = None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent)
    battery_saver = bool(status.SystemStatusFlag) if status.SystemStatusFlag in (0, 1) else None

    return {
        "ac_connected": ac_connected,
        "battery_percent": battery_percent,
        "battery_saver": battery_saver,
    }


def collect_windows_power_context(
    *,
    platform_name: Optional[str] = None,
    powercfg_reader: Optional[Callable[[], str]] = None,
    power_status_reader: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Collect Windows power context without changing system settings."""
    system_name = platform_name or platform.system()
    if str(system_name).lower() != "windows":
        return {
            "available": False,
            "source": "windows_power_context",
            "reason": "Windows power context is only available on Windows.",
        }

    result: Dict[str, Any] = {
        "available": False,
        "source": "powercfg /getactivescheme + GetSystemPowerStatus",
        "active_scheme_guid": None,
        "active_scheme_name": None,
        "ac_connected": None,
        "battery_percent": None,
        "battery_saver": None,
        "warnings": [],
    }

    try:
        parsed = parse_powercfg_active_scheme((powercfg_reader or _run_powercfg)())
        result["active_scheme_guid"] = parsed["guid"]
        result["active_scheme_name"] = parsed["name"]
        if parsed["guid"] is None:
            result["warnings"].append("Active Windows power scheme could not be parsed.")
    except Exception as exc:  # pragma: no cover - exact OS exceptions vary
        result["warnings"].append(f"Power scheme query failed: {type(exc).__name__}: {exc}")

    try:
        result.update((power_status_reader or _read_system_power_status)())
    except Exception as exc:  # pragma: no cover - exact OS exceptions vary
        result["warnings"].append(f"System power status query failed: {type(exc).__name__}: {exc}")

    result["available"] = bool(
        result.get("active_scheme_guid") is not None
        or result.get("ac_connected") is not None
        or result.get("battery_percent") is not None
    )
    if not result["available"]:
        result["reason"] = "Windows power information could not be read."

    return result

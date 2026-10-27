"""Energy-backend selection for the dedicated inference energy window.

This module keeps platform and device-selection details out of
``MeasureController``. Backend objects are created without probing hardware;
actual availability checks are deferred until the energy pass runs, after all
existing timing and resource measurements have completed.
"""

from __future__ import annotations

import platform
import re
from typing import Any, Dict, Iterable, List, Optional

from .energy_backend import EnergyBackend, EnergyBackendError
from .jetson_energy_backend import JetsonEnergyBackend
from .nvml_energy_backend import NvmlEnergyBackend
from .rapl_energy_backend import RaplEnergyBackend
from .windows_emi_energy_backend import WindowsEmiEnergyBackend


class AutoEnergyBackend(EnergyBackend):
    """Select the first available backend from an ordered candidate list."""

    def __init__(self, candidates: Iterable[EnergyBackend]) -> None:
        self._candidates = list(candidates)
        self._selected: Optional[EnergyBackend] = None
        self._reasons: List[str] = []

    @property
    def name(self) -> str:
        """Expose the selected backend name, or ``auto`` before selection."""
        return self._selected.name if self._selected is not None else "auto"

    def available(self) -> bool:
        """Select and cache the first available candidate."""
        if self._selected is not None:
            return True

        self._reasons = []
        for backend in self._candidates:
            try:
                if backend.available():
                    self._selected = backend
                    return True
                self._reasons.append(
                    f"{backend.name}: {backend.unavailable_reason()}"
                )
            except (EnergyBackendError, OSError) as exc:
                self._reasons.append(f"{backend.name}: {exc}")
        return False

    def unavailable_reason(self) -> str:
        if self._selected is not None:
            return ""
        if self._reasons:
            return "No supported energy backend available. " + " | ".join(self._reasons)
        return "No supported energy backend available for the configured device target."

    def read_energy(self) -> float:
        if self._selected is None and not self.available():
            raise EnergyBackendError(self.unavailable_reason())
        assert self._selected is not None
        return self._selected.read_energy()

    def metadata(self) -> Dict[str, Any]:
        if self._selected is None:
            return {"selection": "auto"}
        return {
            "selection": "auto",
            "selected_backend": self._selected.name,
            **self._selected.metadata(),
        }

    def last_sample_time_s(self) -> Optional[float]:
        """Delegate backend-native sample time when the selected source provides it."""
        if self._selected is None and not self.available():
            return None
        assert self._selected is not None
        return self._selected.last_sample_time_s()

    def domains(self) -> Dict[str, Any]:
        if self._selected is None:
            return {}
        return self._selected.domains()


def create_energy_backend(cfg: Dict[str, Any]) -> EnergyBackend:
    """Create a backend for the configured execution target without probing it.

    Explicit ``metrics.energy.backend`` values are supported for controlled
    experiments. When omitted, ``auto`` uses a target-aware candidate order:

    - Jetson is considered first for Linux CPU/GPU/DLA targets because it
      measures module rails and can represent Jetson accelerators that
      NVML/RAPL cannot.
    - Discrete NVIDIA GPU targets then fall back to NVML.
    - Windows CPU targets use the Windows Energy Metering Interface (EMI).
    - Linux CPU targets use Intel RAPL.
    - Generic NPU targets have no safe non-Jetson energy backend yet.
    """
    metrics = cfg.get("metrics") if isinstance(cfg.get("metrics"), dict) else {}
    energy_cfg = metrics.get("energy") if isinstance(metrics.get("energy"), dict) else {}
    requested = str(energy_cfg.get("backend", "auto")).strip().lower()

    if requested == "rapl":
        return RaplEnergyBackend()
    if requested in {"windows_emi", "emi"}:
        return WindowsEmiEnergyBackend(
            channel_name=str(energy_cfg.get("emi_channel", "RAPL_Package0_PKG")),
        )
    if requested == "nvml":
        return NvmlEnergyBackend(device_index=_device_index(cfg))
    if requested in {"jetson", "jetson_ina3221"}:
        return JetsonEnergyBackend()
    if requested != "auto":
        raise ValueError(
            "metrics.energy.backend must be one of: auto, windows_emi, rapl, nvml, "
            "jetson_ina3221"
        )

    target = _device_target(cfg)
    candidates: List[EnergyBackend] = []
    system_name = platform.system().lower()

    # Jetson module rails are only meaningful on Linux Jetson systems. Keeping
    # the platform check here avoids pointless Windows hardware probes.
    if system_name == "linux" and (
        _is_cpu_target(target)
        or _is_gpu_target(target)
        or _is_dla_target(target)
        or _is_npu_target(target)
    ):
        candidates.append(JetsonEnergyBackend())

    if _is_gpu_target(target):
        candidates.append(NvmlEnergyBackend(device_index=_device_index(cfg)))
    elif _is_cpu_target(target):
        if system_name == "windows":
            candidates.append(
                WindowsEmiEnergyBackend(
                    channel_name=str(
                        energy_cfg.get("emi_channel", "RAPL_Package0_PKG")
                    ),
                )
            )
        elif system_name == "linux":
            candidates.append(RaplEnergyBackend())

    return AutoEnergyBackend(candidates)


def _device_target(cfg: Dict[str, Any]) -> str:
    device = cfg.get("device") if isinstance(cfg.get("device"), dict) else {}
    return str(device.get("target", "cpu")).strip().lower()


def _device_index(cfg: Dict[str, Any]) -> int:
    target = _device_target(cfg)
    match = re.search(r"(?:cuda|gpu)(?::|/)?(\d+)$", target)
    if match is not None:
        return int(match.group(1))

    device = cfg.get("device") if isinstance(cfg.get("device"), dict) else {}
    configured = device.get("index", 0)
    try:
        index = int(configured)
    except (TypeError, ValueError):
        return 0
    return max(0, index)


def _is_cpu_target(target: str) -> bool:
    return "cpu" in target


def _is_gpu_target(target: str) -> bool:
    return any(token in target for token in ("cuda", "gpu"))


def _is_dla_target(target: str) -> bool:
    return "dla" in target


def _is_npu_target(target: str) -> bool:
    return "npu" in target

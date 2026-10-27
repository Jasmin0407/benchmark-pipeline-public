"""NVIDIA NVML energy backend using the total GPU energy counter.

The backend exposes NVIDIA's cumulative total-energy counter through the common
:class:`EnergyBackend` contract. NVML reports total energy in millijoules; this
module normalizes the value to joules before returning it to ``EnergyMeter``.

The NVML Python module is imported lazily so the benchmark package can still be
imported on systems without NVIDIA drivers or NVML bindings. Unit tests inject a
fake NVML module and therefore do not require NVIDIA hardware.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .energy_backend import EnergyBackend, EnergyBackendError

_MILLIJOULES_PER_JOULE = 1000.0


class NvmlEnergyBackend(EnergyBackend):
    """Read cumulative energy for one NVIDIA GPU through NVML.

    Parameters
    ----------
    device_index:
        Zero-based NVML device index.
    nvml_module:
        Optional injected module implementing the small subset of the pynvml
        API used here. Production code leaves this as ``None``; tests inject a
        deterministic fake module.
    """

    name = "nvml"

    def __init__(
        self,
        device_index: int = 0,
        *,
        nvml_module: Optional[Any] = None,
    ) -> None:
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise ValueError("device_index must be an integer >= 0")
        if device_index < 0:
            raise ValueError("device_index must be an integer >= 0")

        self._device_index = device_index
        self._nvml = nvml_module
        self._initialized = False
        self._handle: Any = None
        self._import_error: Optional[str] = None
        self._last_unavailable_reason: Optional[str] = None

    def available(self) -> bool:
        """Return whether this GPU exposes NVML total-energy readings."""
        try:
            self._ensure_device()
            self._read_energy_mj()
        except EnergyBackendError as exc:
            self._last_unavailable_reason = str(exc)
            return False

        self._last_unavailable_reason = None
        return True

    def unavailable_reason(self) -> str:
        """Return the most specific known reason for NVML unavailability."""
        if self._last_unavailable_reason:
            return self._last_unavailable_reason
        if self._import_error:
            return self._import_error
        return (
            "NVML total-energy measurement is unavailable for the selected GPU."
        )

    def read_energy(self) -> float:
        """Return cumulative GPU energy in joules.

        NVML's ``nvmlDeviceGetTotalEnergyConsumption`` returns millijoules since
        the NVIDIA driver was last reloaded. The common energy interface uses
        joules, so conversion happens here rather than in ``EnergyMeter``.
        """
        self._ensure_device()
        energy_mj = self._read_energy_mj()
        return energy_mj / _MILLIJOULES_PER_JOULE

    def metadata(self) -> Dict[str, object]:
        """Return non-fatal descriptive metadata for the selected GPU."""
        result: Dict[str, object] = {
            "source": "nvml_total_energy_counter",
            "native_unit": "millijoule",
            "normalized_unit": "joule",
            "device_index": self._device_index,
        }

        try:
            self._ensure_device()
        except EnergyBackendError:
            return result

        name = self._safe_nvml_call("nvmlDeviceGetName", self._handle)
        if name is not None:
            result["device_name"] = self._decode_if_bytes(name)

        uuid = self._safe_nvml_call("nvmlDeviceGetUUID", self._handle)
        if uuid is not None:
            result["device_uuid"] = self._decode_if_bytes(uuid)

        return result

    def _ensure_module(self) -> Any:
        if self._nvml is not None:
            return self._nvml

        try:
            import pynvml  # type: ignore
        except ImportError as exc:
            self._import_error = (
                "Official NVIDIA NVML Python bindings are not installed "
                "(install 'nvidia-ml-py')."
            )
            raise EnergyBackendError(self._import_error) from exc

        self._nvml = pynvml
        return self._nvml

    def _ensure_device(self) -> None:
        if self._handle is not None:
            return

        nvml = self._ensure_module()
        error_type = getattr(nvml, "NVMLError", Exception)

        try:
            if not self._initialized:
                nvml.nvmlInit()
                self._initialized = True
            self._handle = nvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except error_type as exc:
            raise EnergyBackendError(
                f"Could not initialize NVML device {self._device_index}: {exc}"
            ) from exc
        except (AttributeError, OSError) as exc:
            raise EnergyBackendError(
                f"NVML is not usable on this system: {exc}"
            ) from exc

    def _read_energy_mj(self) -> float:
        nvml = self._ensure_module()
        error_type = getattr(nvml, "NVMLError", Exception)

        if self._handle is None:
            raise EnergyBackendError("NVML device is not initialized")

        try:
            value = nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except error_type as exc:
            not_supported_type = getattr(nvml, "NVMLError_NotSupported", None)
            if not_supported_type is not None and isinstance(exc, not_supported_type):
                raise EnergyBackendError(
                    "Selected NVIDIA GPU does not support the NVML total-energy counter."
                ) from exc
            raise EnergyBackendError(f"Could not read NVML energy counter: {exc}") from exc
        except (AttributeError, OSError) as exc:
            raise EnergyBackendError(f"Could not read NVML energy counter: {exc}") from exc

        try:
            energy_mj = float(value)
        except (TypeError, ValueError) as exc:
            raise EnergyBackendError(
                f"NVML returned an invalid total-energy value: {value!r}"
            ) from exc

        if energy_mj < 0.0:
            raise EnergyBackendError(
                f"NVML returned a negative total-energy value: {energy_mj}"
            )

        return energy_mj

    def _safe_nvml_call(self, function_name: str, *args: Any) -> Any:
        nvml = self._ensure_module()
        function = getattr(nvml, function_name, None)
        if function is None:
            return None
        try:
            return function(*args)
        except Exception:
            # Metadata must never make an otherwise valid energy measurement fail.
            return None

    @staticmethod
    def _decode_if_bytes(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

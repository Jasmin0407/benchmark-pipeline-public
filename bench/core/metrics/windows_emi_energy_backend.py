"""Windows Energy Metering Interface (EMI) backend.

This backend uses the documented Windows Energy Metering Interface directly via
SetupAPI and DeviceIoControl. It does not parse GUI or command-line output and
requires no third-party Intel utility.

For the Dell Windows reference system used in Phase 11, the primary channel is
``RAPL_Package0_PKG``. EMI exposes cumulative ``AbsoluteEnergy`` counters in
picowatt-hours; values are normalized to cumulative joules before they are
returned through the common :class:`EnergyBackend` contract.
"""

from __future__ import annotations

import ctypes
import math
import platform
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from .energy_backend import EnergyBackend, EnergyBackendError


EMI_DEVICE_INTERFACE_GUID = "45BD8344-7ED6-49CF-A440-C276C933B053"
EMI_VERSION_V1 = 1
EMI_VERSION_V2 = 2
EMI_MEASUREMENT_UNIT_PICOWATT_HOURS = 0
EMI_NAME_MAX = 16
PWH_TO_JOULE = 3.6e-9
HUNDRED_NS_TO_SECONDS = 1.0e-7
DEFAULT_PACKAGE_CHANNEL = "RAPL_Package0_PKG"

# emi.h uses FILE_DEVICE_UNKNOWN, METHOD_BUFFERED and FILE_READ_ACCESS.
_FILE_DEVICE_UNKNOWN = 0x22
_METHOD_BUFFERED = 0
_FILE_READ_ACCESS = 0x0001


def _ctl_code(device_type: int, function: int, method: int, access: int) -> int:
    return (device_type << 16) | (access << 14) | (function << 2) | method


IOCTL_EMI_GET_VERSION = _ctl_code(
    _FILE_DEVICE_UNKNOWN, 0, _METHOD_BUFFERED, _FILE_READ_ACCESS
)
IOCTL_EMI_GET_METADATA_SIZE = _ctl_code(
    _FILE_DEVICE_UNKNOWN, 1, _METHOD_BUFFERED, _FILE_READ_ACCESS
)
IOCTL_EMI_GET_METADATA = _ctl_code(
    _FILE_DEVICE_UNKNOWN, 2, _METHOD_BUFFERED, _FILE_READ_ACCESS
)
IOCTL_EMI_GET_MEASUREMENT = _ctl_code(
    _FILE_DEVICE_UNKNOWN, 3, _METHOD_BUFFERED, _FILE_READ_ACCESS
)


@dataclass(frozen=True)
class EmiChannel:
    """Metadata for one EMI energy channel."""

    name: str
    measurement_unit: int


@dataclass(frozen=True)
class EmiDevice:
    """Metadata required to read one EMI device."""

    path: str
    version: int
    metadata_size: int
    hardware_oem: str
    hardware_model: str
    hardware_revision: int
    channels: Tuple[EmiChannel, ...]


@dataclass(frozen=True)
class EmiMeasurement:
    """One absolute EMI channel sample."""

    absolute_energy_pwh: int
    absolute_time_100ns: int


class EmiApi(Protocol):
    """Small seam around Windows API calls so backend logic is unit-testable."""

    def discover_devices(self) -> Sequence[EmiDevice]:
        ...

    def read_measurements(self, device: EmiDevice) -> Sequence[EmiMeasurement]:
        ...


def _decode_utf16le_fixed(data: bytes, offset: int, char_count: int) -> str:
    size = char_count * 2
    if offset < 0 or offset + size > len(data):
        raise EnergyBackendError("EMI metadata contains an invalid fixed string range")
    text = data[offset : offset + size].decode("utf-16-le", errors="strict")
    return text.split("\x00", 1)[0].strip()


def _decode_utf16le_sized(data: bytes, offset: int, byte_count: int) -> str:
    if byte_count <= 0 or byte_count % 2 != 0:
        raise EnergyBackendError("EMI metadata contains an invalid UTF-16 byte count")
    if offset < 0 or offset + byte_count > len(data):
        raise EnergyBackendError("EMI metadata channel name exceeds the metadata buffer")
    text = data[offset : offset + byte_count].decode("utf-16-le", errors="strict")
    return text.split("\x00", 1)[0].strip()


def parse_emi_metadata(version: int, metadata: bytes) -> Tuple[str, str, int, Tuple[EmiChannel, ...]]:
    """Parse EMI V1/V2 metadata using the layouts documented in ``emi.h``."""

    if version == EMI_VERSION_V1:
        # EMI_METADATA_V1:
        # ULONG MeasurementUnit; WCHAR OEM[16]; WCHAR Model[16];
        # USHORT Revision; USHORT NameSize; WCHAR Name[]
        fixed_size = 72
        if len(metadata) < fixed_size:
            raise EnergyBackendError("EMI V1 metadata buffer is too small")
        measurement_unit = struct.unpack_from("<I", metadata, 0)[0]
        oem = _decode_utf16le_fixed(metadata, 4, EMI_NAME_MAX)
        model = _decode_utf16le_fixed(metadata, 36, EMI_NAME_MAX)
        revision = struct.unpack_from("<H", metadata, 68)[0]
        name_size = struct.unpack_from("<H", metadata, 70)[0]
        name = _decode_utf16le_sized(metadata, 72, name_size)
        if not name:
            name = "EMI_V1_Channel0"
        return oem, model, revision, (EmiChannel(name, measurement_unit),)

    if version == EMI_VERSION_V2:
        # EMI_METADATA_V2 fixed prefix:
        # WCHAR OEM[16]; WCHAR Model[16]; USHORT Revision; USHORT ChannelCount
        fixed_size = 68
        if len(metadata) < fixed_size:
            raise EnergyBackendError("EMI V2 metadata buffer is too small")
        oem = _decode_utf16le_fixed(metadata, 0, EMI_NAME_MAX)
        model = _decode_utf16le_fixed(metadata, 32, EMI_NAME_MAX)
        revision = struct.unpack_from("<H", metadata, 64)[0]
        channel_count = struct.unpack_from("<H", metadata, 66)[0]
        if channel_count < 1:
            raise EnergyBackendError("EMI V2 metadata exposes no channels")

        channels: List[EmiChannel] = []
        offset = fixed_size
        for index in range(channel_count):
            # EMI_CHANNEL_V2:
            # ULONG MeasurementUnit; USHORT ChannelNameSize; WCHAR ChannelName[]
            if offset + 6 > len(metadata):
                raise EnergyBackendError(
                    f"EMI V2 channel {index} header exceeds the metadata buffer"
                )
            measurement_unit = struct.unpack_from("<I", metadata, offset)[0]
            name_size = struct.unpack_from("<H", metadata, offset + 4)[0]
            name_offset = offset + 6
            name = _decode_utf16le_sized(metadata, name_offset, name_size)
            if not name:
                name = f"EMI_V2_Channel{index}"
            channels.append(EmiChannel(name, measurement_unit))
            offset = name_offset + name_size

        return oem, model, revision, tuple(channels)

    raise EnergyBackendError(f"Unsupported EMI version: {version}")


def parse_emi_measurements(raw: bytes, channel_count: int) -> Tuple[EmiMeasurement, ...]:
    """Parse contiguous ``EMI_CHANNEL_MEASUREMENT_DATA`` records."""

    if channel_count < 1:
        raise EnergyBackendError("EMI measurement requires at least one channel")
    expected = channel_count * 16
    if len(raw) != expected:
        raise EnergyBackendError(
            f"EMI measurement size mismatch: expected {expected} bytes, got {len(raw)}"
        )

    values: List[EmiMeasurement] = []
    for index in range(channel_count):
        energy, absolute_time = struct.unpack_from("<QQ", raw, index * 16)
        values.append(EmiMeasurement(energy, absolute_time))
    return tuple(values)


class _Guid(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _SpDeviceInterfaceData(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("InterfaceClassGuid", _Guid),
        ("Flags", ctypes.c_uint32),
        ("Reserved", ctypes.c_void_p),
    ]


def _guid_from_text(value: str) -> _Guid:
    import uuid

    parsed = uuid.UUID(value)
    data4 = (ctypes.c_ubyte * 8)(*parsed.bytes[8:])
    return _Guid(parsed.time_low, parsed.time_mid, parsed.time_hi_version, data4)


class _CtypesWindowsEmiApi:
    """Direct SetupAPI/Kernel32 implementation of the Windows EMI API."""

    _DIGCF_PRESENT = 0x00000002
    _DIGCF_DEVICEINTERFACE = 0x00000010
    _ERROR_INSUFFICIENT_BUFFER = 122
    _ERROR_NO_MORE_ITEMS = 259
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_NORMAL = 0x00000080

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise EnergyBackendError("Windows EMI API requires Windows")

        self._setupapi = ctypes.WinDLL("setupapi", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._setupapi.SetupDiGetClassDevsW.argtypes = [
            ctypes.POINTER(_Guid),
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self._setupapi.SetupDiGetClassDevsW.restype = ctypes.c_void_p

        self._setupapi.SetupDiEnumDeviceInterfaces.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_Guid),
            ctypes.c_uint32,
            ctypes.POINTER(_SpDeviceInterfaceData),
        ]
        self._setupapi.SetupDiEnumDeviceInterfaces.restype = ctypes.c_int

        self._setupapi.SetupDiGetDeviceInterfaceDetailW.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_SpDeviceInterfaceData),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        self._setupapi.SetupDiGetDeviceInterfaceDetailW.restype = ctypes.c_int

        self._setupapi.SetupDiDestroyDeviceInfoList.argtypes = [ctypes.c_void_p]
        self._setupapi.SetupDiDestroyDeviceInfoList.restype = ctypes.c_int

        self._kernel32.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._kernel32.CreateFileW.restype = ctypes.c_void_p

        self._kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._kernel32.CloseHandle.restype = ctypes.c_int

        self._kernel32.DeviceIoControl.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        self._kernel32.DeviceIoControl.restype = ctypes.c_int

    @staticmethod
    def _invalid_handle_value() -> int:
        return ctypes.c_void_p(-1).value

    @staticmethod
    def _win_error(prefix: str) -> EnergyBackendError:
        code = ctypes.get_last_error()
        return EnergyBackendError(f"{prefix}: Windows error {code} ({ctypes.FormatError(code)})")

    def discover_devices(self) -> Sequence[EmiDevice]:
        paths = self._discover_paths()
        devices: List[EmiDevice] = []
        errors: List[str] = []
        for path in paths:
            try:
                devices.append(self._read_device_metadata(path))
            except EnergyBackendError as exc:
                errors.append(f"{path}: {exc}")

        if not devices and errors:
            raise EnergyBackendError("EMI devices were found but metadata could not be read: " + " | ".join(errors))
        return tuple(devices)

    def read_measurements(self, device: EmiDevice) -> Sequence[EmiMeasurement]:
        handle = self._open_device(device.path)
        try:
            raw = self._ioctl(handle, IOCTL_EMI_GET_MEASUREMENT, len(device.channels) * 16)
        finally:
            self._kernel32.CloseHandle(handle)
        return parse_emi_measurements(raw, len(device.channels))

    def _discover_paths(self) -> Tuple[str, ...]:
        guid = _guid_from_text(EMI_DEVICE_INTERFACE_GUID)
        handle = self._setupapi.SetupDiGetClassDevsW(
            ctypes.byref(guid),
            None,
            None,
            self._DIGCF_PRESENT | self._DIGCF_DEVICEINTERFACE,
        )
        if handle == self._invalid_handle_value():
            raise self._win_error("SetupDiGetClassDevsW failed")

        paths: List[str] = []
        try:
            index = 0
            while True:
                interface = _SpDeviceInterfaceData()
                interface.cbSize = ctypes.sizeof(_SpDeviceInterfaceData)
                ok = self._setupapi.SetupDiEnumDeviceInterfaces(
                    handle,
                    None,
                    ctypes.byref(guid),
                    index,
                    ctypes.byref(interface),
                )
                if not ok:
                    error = ctypes.get_last_error()
                    if error == self._ERROR_NO_MORE_ITEMS:
                        break
                    raise self._win_error("SetupDiEnumDeviceInterfaces failed")

                required_size = ctypes.c_uint32(0)
                ctypes.set_last_error(0)
                ok = self._setupapi.SetupDiGetDeviceInterfaceDetailW(
                    handle,
                    ctypes.byref(interface),
                    None,
                    0,
                    ctypes.byref(required_size),
                    None,
                )
                error = ctypes.get_last_error()
                if ok or error != self._ERROR_INSUFFICIENT_BUFFER or required_size.value == 0:
                    raise self._win_error("Could not determine EMI device-path buffer size")

                buffer = ctypes.create_string_buffer(required_size.value)
                cb_size = 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6
                struct.pack_into("<I", buffer, 0, cb_size)

                ok = self._setupapi.SetupDiGetDeviceInterfaceDetailW(
                    handle,
                    ctypes.byref(interface),
                    ctypes.byref(buffer),
                    required_size.value,
                    ctypes.byref(required_size),
                    None,
                )
                if not ok:
                    raise self._win_error("SetupDiGetDeviceInterfaceDetailW failed")

                path = ctypes.wstring_at(ctypes.addressof(buffer) + 4)
                if path:
                    paths.append(path)
                index += 1
        finally:
            self._setupapi.SetupDiDestroyDeviceInfoList(handle)

        return tuple(paths)

    def _read_device_metadata(self, path: str) -> EmiDevice:
        handle = self._open_device(path)
        try:
            version_raw = self._ioctl(handle, IOCTL_EMI_GET_VERSION, 2)
            if len(version_raw) != 2:
                raise EnergyBackendError("EMI version response has an unexpected size")
            version = struct.unpack_from("<H", version_raw, 0)[0]
            if version not in (EMI_VERSION_V1, EMI_VERSION_V2):
                raise EnergyBackendError(f"Unsupported EMI version: {version}")

            size_raw = self._ioctl(handle, IOCTL_EMI_GET_METADATA_SIZE, 4)
            if len(size_raw) != 4:
                raise EnergyBackendError("EMI metadata-size response has an unexpected size")
            metadata_size = struct.unpack_from("<I", size_raw, 0)[0]
            if metadata_size < 1 or metadata_size > 1024 * 1024:
                raise EnergyBackendError(f"Invalid EMI metadata size: {metadata_size}")

            metadata = self._ioctl(handle, IOCTL_EMI_GET_METADATA, metadata_size)
        finally:
            self._kernel32.CloseHandle(handle)

        oem, model, revision, channels = parse_emi_metadata(version, metadata)
        return EmiDevice(
            path=path,
            version=version,
            metadata_size=metadata_size,
            hardware_oem=oem,
            hardware_model=model,
            hardware_revision=revision,
            channels=channels,
        )

    def _open_device(self, path: str) -> int:
        handle = self._kernel32.CreateFileW(
            path,
            self._GENERIC_READ,
            self._FILE_SHARE_READ | self._FILE_SHARE_WRITE,
            None,
            self._OPEN_EXISTING,
            self._FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if handle == self._invalid_handle_value() or handle is None:
            raise self._win_error(f"CreateFileW failed for EMI device {path}")
        return handle

    def _ioctl(self, handle: int, code: int, output_size: int) -> bytes:
        if output_size < 1:
            raise EnergyBackendError("EMI IOCTL output buffer must be non-empty")
        output = ctypes.create_string_buffer(output_size)
        returned = ctypes.c_uint32(0)
        ok = self._kernel32.DeviceIoControl(
            handle,
            code,
            None,
            0,
            ctypes.byref(output),
            output_size,
            ctypes.byref(returned),
            None,
        )
        if not ok:
            raise self._win_error(f"DeviceIoControl 0x{code:08X} failed")
        if returned.value != output_size:
            raise EnergyBackendError(
                f"EMI IOCTL 0x{code:08X} returned {returned.value} bytes; expected {output_size}"
            )
        return bytes(output.raw[: returned.value])


class WindowsEmiEnergyBackend(EnergyBackend):
    """Read cumulative Windows EMI package energy and normalize it to joules."""

    name = "windows_emi"

    def __init__(
        self,
        *,
        channel_name: str = DEFAULT_PACKAGE_CHANNEL,
        system_name: Optional[str] = None,
        api: Optional[EmiApi] = None,
    ) -> None:
        channel_name = str(channel_name).strip()
        if not channel_name:
            raise ValueError("EMI channel_name must not be empty")

        self._channel_name = channel_name
        self._system_name = system_name if system_name is not None else platform.system()
        self._api = api
        self._selected_device: Optional[EmiDevice] = None
        self._selected_channel_index: Optional[int] = None
        self._last_measurements: Optional[Tuple[EmiMeasurement, ...]] = None
        self._previous_measurements: Optional[Tuple[EmiMeasurement, ...]] = None
        self._last_unavailable_reason: Optional[str] = None

    def available(self) -> bool:
        if self._system_name.lower() != "windows":
            self._last_unavailable_reason = "Windows EMI energy measurement requires Windows."
            return False

        try:
            device, channel_index = self._select_device_and_channel()
        except (EnergyBackendError, OSError) as exc:
            self._last_unavailable_reason = str(exc)
            return False

        self._selected_device = device
        self._selected_channel_index = channel_index
        self._last_unavailable_reason = None
        return True

    def unavailable_reason(self) -> str:
        if self._last_unavailable_reason:
            return self._last_unavailable_reason
        if self._system_name.lower() != "windows":
            return "Windows EMI energy measurement requires Windows."
        return (
            f"No usable Windows EMI device exposing channel '{self._channel_name}' "
            "with picowatt-hour measurements was found."
        )

    def read_energy(self) -> float:
        device, channel_index = self._ensure_selected()
        try:
            measurements = tuple(self._api_instance().read_measurements(device))
        except (EnergyBackendError, OSError) as exc:
            raise EnergyBackendError(f"Could not read Windows EMI energy: {exc}") from exc

        if len(measurements) != len(device.channels):
            raise EnergyBackendError(
                "Windows EMI measurement count does not match metadata channel count"
            )

        measurement = measurements[channel_index]
        if measurement.absolute_energy_pwh < 0 or measurement.absolute_time_100ns < 0:
            raise EnergyBackendError("Windows EMI returned a negative absolute counter")

        if self._last_measurements is not None:
            previous = self._last_measurements[channel_index]
            if measurement.absolute_energy_pwh < previous.absolute_energy_pwh:
                raise EnergyBackendError(
                    "Windows EMI AbsoluteEnergy decreased during the measurement window"
                )
            if measurement.absolute_time_100ns <= previous.absolute_time_100ns:
                raise EnergyBackendError(
                    "Windows EMI AbsoluteTime did not increase during the measurement window"
                )

        self._previous_measurements = self._last_measurements
        self._last_measurements = measurements

        energy_j = measurement.absolute_energy_pwh * PWH_TO_JOULE
        if not math.isfinite(energy_j) or energy_j < 0.0:
            raise EnergyBackendError("Windows EMI produced an invalid normalized energy value")
        return energy_j

    def last_sample_time_s(self) -> Optional[float]:
        """Return the selected channel's last EMI AbsoluteTime in seconds."""
        if self._last_measurements is None or self._selected_channel_index is None:
            return None
        value = (
            self._last_measurements[self._selected_channel_index].absolute_time_100ns
            * HUNDRED_NS_TO_SECONDS
        )
        return value if math.isfinite(value) and value >= 0.0 else None

    def metadata(self) -> Dict[str, object]:
        device = self._selected_device
        if device is None:
            return {
                "source": "windows_emi",
                "interface_guid": EMI_DEVICE_INTERFACE_GUID,
                "requested_channel": self._channel_name,
                "native_unit": "picowatt_hours",
                "normalized_unit": "joule",
            }

        return {
            "source": "windows_emi",
            "interface_guid": EMI_DEVICE_INTERFACE_GUID,
            "emi_version": device.version,
            "hardware_oem": device.hardware_oem,
            "hardware_model": device.hardware_model,
            "hardware_revision": device.hardware_revision,
            "metadata_size_bytes": device.metadata_size,
            "primary_domain": self._channel_name,
            "native_unit": "picowatt_hours",
            "normalized_unit": "joule",
            "counter_semantics": "absolute_energy_with_arbitrary_zero_point",
            "time_semantics": "absolute_time_100ns",
            "available_channels": [channel.name for channel in device.channels],
        }

    def domains(self) -> Dict[str, object]:
        device = self._selected_device
        if device is None:
            return {}

        result: Dict[str, object] = {}
        for index, channel in enumerate(device.channels):
            entry: Dict[str, object] = {
                "measurement_unit": (
                    "picowatt_hours"
                    if channel.measurement_unit == EMI_MEASUREMENT_UNIT_PICOWATT_HOURS
                    else f"enum_{channel.measurement_unit}"
                ),
                "is_primary": channel.name == self._channel_name,
            }
            if self._last_measurements is not None:
                current = self._last_measurements[index]
                entry["absolute_time_100ns"] = current.absolute_time_100ns
            if self._previous_measurements is not None and self._last_measurements is not None:
                previous = self._previous_measurements[index]
                current = self._last_measurements[index]
                if current.absolute_energy_pwh >= previous.absolute_energy_pwh:
                    entry["interval_energy_j"] = (
                        current.absolute_energy_pwh - previous.absolute_energy_pwh
                    ) * PWH_TO_JOULE
            result[channel.name] = entry
        return result

    def _ensure_selected(self) -> Tuple[EmiDevice, int]:
        if self._selected_device is None or self._selected_channel_index is None:
            if not self.available():
                raise EnergyBackendError(self.unavailable_reason())
        assert self._selected_device is not None
        assert self._selected_channel_index is not None
        return self._selected_device, self._selected_channel_index

    def _select_device_and_channel(self) -> Tuple[EmiDevice, int]:
        try:
            devices = tuple(self._api_instance().discover_devices())
        except (EnergyBackendError, OSError) as exc:
            raise EnergyBackendError(f"Windows EMI discovery failed: {exc}") from exc

        if not devices:
            raise EnergyBackendError("No present Windows EMI device interface was found")

        matches: List[Tuple[EmiDevice, int]] = []
        wrong_unit_matches: List[int] = []
        for device in devices:
            for index, channel in enumerate(device.channels):
                if channel.name != self._channel_name:
                    continue
                if channel.measurement_unit != EMI_MEASUREMENT_UNIT_PICOWATT_HOURS:
                    wrong_unit_matches.append(channel.measurement_unit)
                    continue
                matches.append((device, index))

        if wrong_unit_matches and not matches:
            units = ", ".join(str(value) for value in sorted(set(wrong_unit_matches)))
            raise EnergyBackendError(
                f"EMI channel '{self._channel_name}' uses unsupported measurement unit(s): {units}"
            )

        if not matches:
            available_channels = sorted(
                {channel.name for device in devices for channel in device.channels}
            )
            raise EnergyBackendError(
                f"EMI channel '{self._channel_name}' was not found. Available channels: "
                + (", ".join(available_channels) if available_channels else "none")
            )

        if len(matches) > 1:
            raise EnergyBackendError(
                f"EMI channel '{self._channel_name}' is exposed by multiple devices; "
                "automatic aggregation is intentionally disabled to avoid double counting."
            )

        return matches[0]

    def _api_instance(self) -> EmiApi:
        if self._api is None:
            self._api = _CtypesWindowsEmiApi()
        return self._api

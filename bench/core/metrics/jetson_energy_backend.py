"""NVIDIA Jetson INA3221 energy backend.

Jetson power monitors expose rail voltage/current readings rather than a
portable cumulative energy counter. This backend therefore samples individual
rails during the dedicated energy window and numerically integrates power over
time. The result is presented through the common :class:`EnergyBackend`
contract as measurement-relative cumulative joules.

Rails are never blindly summed. Each rail is integrated independently and one
explicit ``primary_domain`` is used for the scalar value consumed by
``EnergyMeter``. This avoids double counting overlapping Jetson power domains.
"""

from __future__ import annotations

import math
import platform
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Protocol

from .energy_backend import EnergyBackend, EnergyBackendError


_LABEL_RE = re.compile(r"^in(?P<channel>\d+)_label$")
_DEFAULT_INA_ROOT = Path("/sys/bus/i2c/drivers/ina3221")
_DEFAULT_SAMPLE_INTERVAL_S = 0.02


@dataclass(frozen=True)
class _JetsonRail:
    """INA3221 files required to read one Jetson power rail."""

    label: str
    voltage_path: Path
    current_path: Path
    source_path: Path


class _Integrator(Protocol):
    """Small protocol used to keep the sampling engine injectable in tests."""

    def start(self) -> None:
        ...

    def stop(self) -> Dict[str, float]:
        ...


PowerReader = Callable[[], Dict[str, float]]
IntegratorFactory = Callable[[PowerReader, float], _Integrator]




def _trapezoid_energy_j(previous_power_w: float, current_power_w: float, dt_s: float) -> float:
    """Return one trapezoidal integration increment in joules."""
    values = (float(previous_power_w), float(current_power_w), float(dt_s))
    if not all(math.isfinite(value) for value in values):
        raise EnergyBackendError("Jetson integration values must be finite")
    if previous_power_w < 0.0 or current_power_w < 0.0 or dt_s < 0.0:
        raise EnergyBackendError("Jetson integration values must be non-negative")
    return 0.5 * (previous_power_w + current_power_w) * dt_s

class _ThreadedPowerIntegrator:
    """Integrate rail power with trapezoidal sampling in a background thread."""

    def __init__(
        self,
        read_powers: PowerReader,
        sample_interval_s: float,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._read_powers = read_powers
        self._sample_interval_s = sample_interval_s
        self._clock = clock
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_time_s: Optional[float] = None
        self._last_power_w: Dict[str, float] = {}
        self._energy_j: Dict[str, float] = {}
        self._error: Optional[BaseException] = None

    def start(self) -> None:
        if self._thread is not None:
            raise EnergyBackendError("Jetson power integrator is already running")

        self._sample(initial=True)
        self._thread = threading.Thread(
            target=self._run,
            name="jetson-energy-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if self._thread is None:
            raise EnergyBackendError("Jetson power integrator is not running")

        self._stop_event.set()
        self._thread.join(timeout=max(1.0, 5.0 * self._sample_interval_s))
        if self._thread.is_alive():
            raise EnergyBackendError("Jetson power sampler did not stop cleanly")

        if self._error is not None:
            raise EnergyBackendError(
                f"Jetson power sampling failed: {self._error}"
            ) from self._error

        self._sample(initial=False)
        return dict(self._energy_j)

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self._sample_interval_s):
                self._sample(initial=False)
        except BaseException as exc:  # captured and re-raised synchronously in stop()
            self._error = exc
            self._stop_event.set()

    def _sample(self, *, initial: bool) -> None:
        now_s = float(self._clock())
        powers_w = self._read_powers()

        with self._lock:
            if initial or self._last_time_s is None:
                self._last_time_s = now_s
                self._last_power_w = dict(powers_w)
                self._energy_j = {name: 0.0 for name in powers_w}
                return

            dt_s = now_s - self._last_time_s
            if not math.isfinite(dt_s) or dt_s < 0.0:
                raise EnergyBackendError("Jetson sampling clock moved backwards")

            all_domains = set(self._last_power_w) | set(powers_w)
            for domain in all_domains:
                if domain not in self._last_power_w or domain not in powers_w:
                    raise EnergyBackendError(
                        f"Jetson rail set changed during measurement: {domain}"
                    )
                previous_w = self._last_power_w[domain]
                current_w = powers_w[domain]
                self._energy_j[domain] = self._energy_j.get(domain, 0.0) + (
                    _trapezoid_energy_j(previous_w, current_w, dt_s)
                )

            self._last_time_s = now_s
            self._last_power_w = dict(powers_w)


class JetsonEnergyBackend(EnergyBackend):
    """Integrate NVIDIA Jetson INA3221 power rails into joules.

    Parameters
    ----------
    ina_root:
        Root of the Linux INA3221 driver tree. Tests can inject a temporary
        directory containing hwmon-style rail files.
    primary_domain:
        Optional rail to use as the scalar energy value. If omitted, ``VDD_IN``
        is preferred because NVIDIA documents it as total module power on Orin
        NX/Nano and Xavier NX. ``SUMMATION`` is the next preference but is
        explicitly reported as a partial/composite scope rather than total
        module power.
    sample_interval_s:
        Background power-sampling interval in seconds.
    system_name:
        Optional platform override used by unit tests.
    integrator_factory:
        Test seam for replacing the threaded sampler with deterministic fake
        integration.
    """

    name = "jetson_ina3221"

    def __init__(
        self,
        ina_root: str | Path = _DEFAULT_INA_ROOT,
        *,
        primary_domain: Optional[str] = None,
        sample_interval_s: float = _DEFAULT_SAMPLE_INTERVAL_S,
        system_name: Optional[str] = None,
        integrator_factory: Optional[IntegratorFactory] = None,
    ) -> None:
        if not math.isfinite(float(sample_interval_s)) or sample_interval_s <= 0.0:
            raise ValueError("sample_interval_s must be a finite number > 0")

        self._root = Path(ina_root)
        self._requested_primary = primary_domain
        self._sample_interval_s = float(sample_interval_s)
        self._system_name = system_name if system_name is not None else platform.system()
        self._integrator_factory = integrator_factory or self._make_default_integrator
        self._rails_cache: Optional[Dict[str, _JetsonRail]] = None
        self._primary_domain: Optional[str] = None
        self._active_integrator: Optional[_Integrator] = None
        self._last_domain_energy_j: Dict[str, float] = {}
        self._last_unavailable_reason: Optional[str] = None

    def available(self) -> bool:
        if self._system_name != "Linux":
            self._last_unavailable_reason = (
                "NVIDIA Jetson INA3221 energy measurement requires Linux."
            )
            return False

        try:
            rails = self._rails()
            primary = self._resolve_primary_domain(rails)
        except (OSError, EnergyBackendError) as exc:
            self._last_unavailable_reason = str(exc)
            return False

        if not rails:
            self._last_unavailable_reason = "No readable Jetson INA3221 rails were found."
            return False
        if primary is None:
            self._last_unavailable_reason = "No usable Jetson primary power rail was found."
            return False

        try:
            self._read_powers_w()
        except EnergyBackendError as exc:
            self._last_unavailable_reason = str(exc)
            return False

        self._primary_domain = primary
        self._last_unavailable_reason = None
        return True

    def unavailable_reason(self) -> str:
        if self._last_unavailable_reason:
            return self._last_unavailable_reason
        if self._system_name != "Linux":
            return "NVIDIA Jetson INA3221 energy measurement requires Linux."
        return "NVIDIA Jetson INA3221 power rails are unavailable."

    def read_energy(self) -> float:
        """Start or stop one sampling window and return primary-domain joules.

        ``EnergyMeter`` reads a cumulative counter before and after inference.
        Jetson exposes power rather than a cumulative energy counter, so the
        first call starts background integration and returns ``0.0``. The
        second call stops integration and returns the integrated energy of the
        selected primary rail. A subsequent pair starts a fresh window.
        """
        if self._active_integrator is None:
            if not self.available():
                raise EnergyBackendError(self.unavailable_reason())
            integrator = self._integrator_factory(
                self._read_powers_w,
                self._sample_interval_s,
            )
            integrator.start()
            self._active_integrator = integrator
            self._last_domain_energy_j = {}
            return 0.0

        integrator = self._active_integrator
        self._active_integrator = None
        try:
            energies = integrator.stop()
        except (OSError, EnergyBackendError) as exc:
            raise EnergyBackendError(f"Could not integrate Jetson power rails: {exc}") from exc

        self._validate_domain_energies(energies)
        self._last_domain_energy_j = dict(energies)

        primary = self._primary_domain or self._resolve_primary_domain(self._rails())
        if primary is None or primary not in energies:
            raise EnergyBackendError("Selected Jetson primary rail was not measured")
        return energies[primary]

    def metadata(self) -> Dict[str, object]:
        rails = self._safe_rails()
        primary = self._primary_domain or self._resolve_primary_domain(rails)
        dla_domains = [name for name in rails if "CV" in name.upper()]
        return {
            "source": "jetson_ina3221_power_sampling",
            "native_measurement": "voltage_mV_x_current_mA",
            "normalized_unit": "joule",
            "sample_interval_s": self._sample_interval_s,
            "primary_domain": primary,
            "primary_domain_scope": self._primary_scope(primary),
            "dla_relevant_domains": sorted(dla_domains),
        }

    def domains(self) -> Dict[str, object]:
        rails = self._safe_rails()
        result: Dict[str, object] = {}
        for name, rail in sorted(rails.items()):
            entry: Dict[str, object] = {
                "source_path": str(rail.source_path),
                "is_primary": name == (self._primary_domain or self._resolve_primary_domain(rails)),
                "dla_relevant": "CV" in name.upper(),
            }
            if name in self._last_domain_energy_j:
                entry["energy_j"] = self._last_domain_energy_j[name]
            result[name] = entry
        return result

    def _rails(self) -> Dict[str, _JetsonRail]:
        if self._rails_cache is not None:
            return self._rails_cache

        rails: Dict[str, _JetsonRail] = {}
        if not self._root.exists() or not self._root.is_dir():
            self._rails_cache = rails
            return rails

        for label_path in sorted(self._root.rglob("in*_label")):
            match = _LABEL_RE.fullmatch(label_path.name)
            if match is None:
                continue
            channel = match.group("channel")
            try:
                label = label_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not label or label.upper() == "NC":
                continue

            hwmon_dir = label_path.parent
            voltage_path = hwmon_dir / f"in{channel}_input"
            current_path = hwmon_dir / f"curr{channel}_input"
            if not voltage_path.is_file() or not current_path.is_file():
                continue

            if label in rails:
                raise EnergyBackendError(
                    f"Duplicate Jetson rail label discovered: {label}"
                )
            rails[label] = _JetsonRail(
                label=label,
                voltage_path=voltage_path,
                current_path=current_path,
                source_path=hwmon_dir,
            )

        self._rails_cache = rails
        return rails

    def _safe_rails(self) -> Dict[str, _JetsonRail]:
        try:
            return self._rails()
        except (OSError, EnergyBackendError):
            return {}

    def _read_powers_w(self) -> Dict[str, float]:
        rails = self._rails()
        if not rails:
            raise EnergyBackendError("No readable Jetson INA3221 rails were found.")

        result: Dict[str, float] = {}
        for name, rail in rails.items():
            try:
                voltage_mv = float(rail.voltage_path.read_text(encoding="utf-8").strip())
                current_ma = float(rail.current_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError) as exc:
                raise EnergyBackendError(
                    f"Could not read Jetson rail {name}: {exc}"
                ) from exc

            if (
                not math.isfinite(voltage_mv)
                or not math.isfinite(current_ma)
                or voltage_mv < 0.0
                or current_ma < 0.0
            ):
                raise EnergyBackendError(
                    f"Jetson rail {name} returned invalid voltage/current data"
                )

            # mV * mA = microwatt; divide by 1e6 to normalize to watt.
            result[name] = (voltage_mv * current_ma) / 1_000_000.0
        return result

    def _resolve_primary_domain(
        self,
        rails: Mapping[str, _JetsonRail],
    ) -> Optional[str]:
        if not rails:
            return None
        if self._requested_primary is not None:
            if self._requested_primary not in rails:
                raise EnergyBackendError(
                    f"Requested Jetson primary rail is unavailable: {self._requested_primary}"
                )
            return self._requested_primary

        for preferred in ("VDD_IN", "POM_5V_IN", "SUMMATION"):
            if preferred in rails:
                return preferred
        return None

    @staticmethod
    def _primary_scope(primary: Optional[str]) -> str:
        if primary in {"VDD_IN", "POM_5V_IN"}:
            return "total_module_power"
        if primary == "SUMMATION":
            return "hardware_summation_channel_not_guaranteed_total_module_power"
        if primary is None:
            return "unavailable"
        return "single_power_rail"

    @staticmethod
    def _validate_domain_energies(energies: Mapping[str, float]) -> None:
        if not energies:
            raise EnergyBackendError("Jetson integrator returned no domain energies")
        for name, value in energies.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise EnergyBackendError(
                    f"Jetson integrator returned invalid energy for {name}: {value!r}"
                )

    @staticmethod
    def _make_default_integrator(
        read_powers: PowerReader,
        sample_interval_s: float,
    ) -> _Integrator:
        return _ThreadedPowerIntegrator(read_powers, sample_interval_s)

"""Hardware-independent energy measurement for a controlled inference window.

The meter deliberately knows nothing about Intel RAPL, NVIDIA NVML, Jetson
power rails, or any other hardware API. It consumes a callable that returns a
cumulative energy counter in joules. Hardware-specific backends introduced in
later phases are responsible for exposing values in that unit.

Keeping the meter hardware-independent makes the measurement mathematics fully
testable with deterministic fake counters.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional

from .energy_backend import EnergyBackend, EnergyBackendError

EnergyReader = Callable[[], float]
Clock = Callable[[], float]


class EnergyMeter:
    """Measure energy consumed by repeated calls to an inference function.

    Parameters
    ----------
    energy_reader:
        Callable returning a cumulative energy counter in joules. The counter
        must not decrease during one measurement window. ``None`` represents an
        unavailable energy source and is useful before a hardware backend has
        been selected.
    clock:
        Monotonic clock used for the measurement duration. ``time.perf_counter``
        is used by default; tests can inject a deterministic fake clock.
    """

    def __init__(
        self,
        energy_reader: Optional[EnergyReader] = None,
        clock: Clock = time.perf_counter,
        backend: Optional[EnergyBackend] = None,
    ) -> None:
        if energy_reader is not None and backend is not None:
            raise ValueError("Configure either energy_reader or backend, not both")

        self._energy_reader = energy_reader
        self._clock = clock
        self._backend = backend

    def available(self) -> bool:
        """Return whether the configured energy source is available."""
        if self._backend is not None:
            try:
                return bool(self._backend.available())
            except (EnergyBackendError, OSError):
                return False
        return callable(self._energy_reader)

    def measure(
        self,
        fn: Callable[[], Any],
        repeats: int,
        batch_size: int = 1,
        signal_duration_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Measure energy for a dedicated repeated-inference window.

        The energy reader is sampled once immediately before and once
        immediately after the inference loop. The returned counter values must
        be cumulative joules. The actual wall-clock duration is measured with a
        monotonic clock around the same window.

        Parameters
        ----------
        fn:
            Callable that performs exactly one model inference.
        repeats:
            Number of inference calls in the energy window. Must be an integer
            greater than zero.
        batch_size:
            Number of samples processed by one inference call. Must be an
            integer greater than zero.
        signal_duration_s:
            Optional duration represented by one sample, in seconds. When
            provided, ``energy_per_signal_s_j`` is reported.

        Returns
        -------
        Dict[str, Any]
            Hardware-independent energy metrics in joules, watts, and seconds.
        """
        self._validate_inputs(
            repeats=repeats,
            batch_size=batch_size,
            signal_duration_s=signal_duration_s,
        )

        if self._backend is not None:
            try:
                backend_available = bool(self._backend.available())
            except (EnergyBackendError, OSError) as exc:
                return self._unavailable_result(
                    f"Energy backend availability check failed: {exc}"
                )

            if not backend_available:
                return self._unavailable_result(self._backend.unavailable_reason())
        elif not self.available():
            raise RuntimeError("No energy reader is configured")

        start_time_s = self._read_clock("start")
        try:
            start_energy_j = self._read_energy("start")
        except (EnergyBackendError, OSError) as exc:
            return self._unavailable_result(f"Energy counter could not be read: {exc}")
        backend_start_time_s = self._backend_sample_time_s()

        for _ in range(repeats):
            fn()

        try:
            end_energy_j = self._read_energy("end")
        except (EnergyBackendError, OSError) as exc:
            return self._unavailable_result(f"Energy counter could not be read: {exc}")
        backend_end_time_s = self._backend_sample_time_s()
        end_time_s = self._read_clock("end")

        total_energy_j = end_energy_j - start_energy_j
        if total_energy_j < 0.0:
            raise ValueError(
                "Energy counter decreased during measurement; "
                "counter wraparound must be handled by the hardware backend"
            )

        wall_clock_duration_s = end_time_s - start_time_s
        if not math.isfinite(wall_clock_duration_s) or wall_clock_duration_s <= 0.0:
            raise ValueError("Measured duration must be a finite number > 0")

        duration_s = wall_clock_duration_s
        duration_source = "wall_clock"
        if backend_start_time_s is not None and backend_end_time_s is not None:
            backend_duration_s = backend_end_time_s - backend_start_time_s
            if math.isfinite(backend_duration_s) and backend_duration_s > 0.0:
                duration_s = backend_duration_s
                duration_source = "backend_counter"

        energy_per_inference_j = total_energy_j / repeats
        energy_per_sample_j = energy_per_inference_j / batch_size
        average_power_w = total_energy_j / duration_s

        result: Dict[str, Any] = {
            "total_energy_j": total_energy_j,
            "energy_per_inference_j": energy_per_inference_j,
            "energy_per_sample_j": energy_per_sample_j,
            "average_power_w": average_power_w,
            "duration_s": duration_s,
            "duration_source": duration_source,
            "wall_clock_duration_s": wall_clock_duration_s,
            "inference_count": repeats,
        }

        if signal_duration_s is not None:
            result["energy_per_signal_s_j"] = (
                energy_per_sample_j / signal_duration_s
            )

        if self._backend is not None:
            result = {
                "available": True,
                "backend": self._backend.name,
                "measurement_scope": "dedicated_inference_window",
                **result,
            }
            metadata = self._backend.metadata()
            if metadata:
                result["metadata"] = metadata
                primary_domain = metadata.get("primary_domain")
                if primary_domain:
                    result["primary_domain"] = str(primary_domain)

            domains = self._backend.domains()
            if domains:
                result["domains"] = domains

        return result


    def measure_idle(
        self,
        duration_s: float,
        sleep_fn: Callable[[float], Any] = time.sleep,
    ) -> Dict[str, Any]:
        """Measure energy during an inference-free idle interval.

        This method is used only by the optional Phase 13 baseline correction.
        It samples the same cumulative energy backend before and after a sleep
        interval, so the baseline uses exactly the same backend and domain as
        the gross energy measurement.
        """
        if isinstance(duration_s, bool):
            raise ValueError("Idle baseline duration_s must be a finite number > 0")
        duration_value = float(duration_s)
        if not math.isfinite(duration_value) or duration_value <= 0.0:
            raise ValueError("Idle baseline duration_s must be a finite number > 0")

        if self._backend is not None:
            try:
                backend_available = bool(self._backend.available())
            except (EnergyBackendError, OSError) as exc:
                return self._unavailable_result(
                    f"Energy backend availability check failed: {exc}"
                )
            if not backend_available:
                return self._unavailable_result(self._backend.unavailable_reason())
        elif not self.available():
            raise RuntimeError("No energy reader is configured")

        start_time_s = self._read_clock("start")
        try:
            start_energy_j = self._read_energy("start")
        except (EnergyBackendError, OSError) as exc:
            return self._unavailable_result(f"Energy counter could not be read: {exc}")
        backend_start_time_s = self._backend_sample_time_s()

        sleep_fn(duration_value)

        try:
            end_energy_j = self._read_energy("end")
        except (EnergyBackendError, OSError) as exc:
            return self._unavailable_result(f"Energy counter could not be read: {exc}")
        backend_end_time_s = self._backend_sample_time_s()
        end_time_s = self._read_clock("end")

        total_energy_j = end_energy_j - start_energy_j
        if total_energy_j < 0.0:
            raise ValueError(
                "Energy counter decreased during idle baseline measurement; "
                "counter wraparound must be handled by the hardware backend"
            )

        wall_clock_duration_s = end_time_s - start_time_s
        if not math.isfinite(wall_clock_duration_s) or wall_clock_duration_s <= 0.0:
            raise ValueError("Measured idle baseline duration must be a finite number > 0")

        measured_duration_s = wall_clock_duration_s
        duration_source = "wall_clock"
        if backend_start_time_s is not None and backend_end_time_s is not None:
            backend_duration_s = backend_end_time_s - backend_start_time_s
            if math.isfinite(backend_duration_s) and backend_duration_s > 0.0:
                measured_duration_s = backend_duration_s
                duration_source = "backend_counter"

        result: Dict[str, Any] = {
            "total_energy_j": total_energy_j,
            "average_power_w": total_energy_j / measured_duration_s,
            "duration_s": measured_duration_s,
            "duration_source": duration_source,
            "wall_clock_duration_s": wall_clock_duration_s,
        }

        if self._backend is not None:
            result = {
                "available": True,
                "backend": self._backend.name,
                "measurement_scope": "idle_baseline_window",
                **result,
            }
            metadata = self._backend.metadata()
            if metadata:
                result["metadata"] = metadata
                primary_domain = metadata.get("primary_domain")
                if primary_domain:
                    result["primary_domain"] = str(primary_domain)
            domains = self._backend.domains()
            if domains:
                result["domains"] = domains

        return result

    def _unavailable_result(self, reason: str) -> Dict[str, Any]:
        """Build a non-fatal result for unavailable hardware measurement."""
        backend_name = self._backend.name if self._backend is not None else "unknown"
        return {
            "available": False,
            "backend": backend_name,
            "reason": reason or "Energy measurement is unavailable.",
        }

    def _backend_sample_time_s(self) -> Optional[float]:
        """Return a validated timestamp tied to the last backend energy sample."""
        if self._backend is None:
            return None
        try:
            value = self._backend.last_sample_time_s()
        except (EnergyBackendError, OSError):
            return None
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            return None
        return numeric

    def _read_energy(self, position: str) -> float:
        """Read and validate one cumulative energy-counter value in joules."""
        if self._backend is not None:
            value = float(self._backend.read_energy())
        else:
            if self._energy_reader is None:
                raise RuntimeError("No energy reader is configured")
            value = float(self._energy_reader())
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{position} energy counter must be a finite number >= 0"
            )
        return value

    def _read_clock(self, position: str) -> float:
        """Read and validate one monotonic-clock value."""
        value = float(self._clock())
        if not math.isfinite(value):
            raise ValueError(f"{position} clock value must be finite")
        return value

    @staticmethod
    def _validate_inputs(
        repeats: int,
        batch_size: int,
        signal_duration_s: Optional[float],
    ) -> None:
        """Validate measurement parameters before invoking model inference."""
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be an integer >= 1")

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be an integer >= 1")

        if signal_duration_s is not None:
            duration = float(signal_duration_s)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("signal_duration_s must be a finite number > 0")

"""Common interface for hardware-specific energy measurement backends.

Backends are responsible only for hardware access and unit normalization. They
must expose cumulative energy in joules so that :class:`EnergyMeter` can remain
independent of RAPL, NVML, Jetson power rails, or any future platform API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class EnergyBackendError(RuntimeError):
    """Base exception for recoverable energy-backend access errors."""


class EnergyBackend(ABC):
    """Abstract contract implemented by all energy measurement backends.

    Concrete backends must normalize their native counter unit to cumulative
    joules before returning from :meth:`read_energy`. Counter wraparound, when
    relevant, is also a backend responsibility because only the backend knows
    the hardware-specific counter range.
    """

    name = "unknown"

    @abstractmethod
    def available(self) -> bool:
        """Return whether the backend can measure energy on this system."""

    @abstractmethod
    def read_energy(self) -> float:
        """Return a cumulative energy counter in joules."""

    def metadata(self) -> Dict[str, Any]:
        """Return optional backend-specific metadata."""
        return {}

    def domains(self) -> Dict[str, Any]:
        """Return optional energy-domain metadata or measurements."""
        return {}

    def last_sample_time_s(self) -> float | None:
        """Return the backend timestamp of the most recent energy sample.

        Backends that expose a timestamp tied to the energy counter can
        override this method. ``EnergyMeter`` prefers the resulting interval
        for average-power calculation while retaining wall-clock duration for
        auditability. Backends without such a timestamp keep returning ``None``.
        """
        return None

    def unavailable_reason(self) -> str:
        """Return a human-readable explanation when the backend is unavailable."""
        return "Energy backend is not available on this system."

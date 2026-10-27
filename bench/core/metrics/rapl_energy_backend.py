"""Intel RAPL energy backend using the Linux powercap sysfs interface.

The backend reads top-level Intel RAPL package counters from
``/sys/class/powercap``. Native microjoule counters are normalized to
cumulative joules before they are exposed through the common EnergyBackend
interface.

Counter wraparound is handled inside this backend because the RAPL-specific
``max_energy_range_uj`` value is required to reconstruct a monotonic counter.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .energy_backend import EnergyBackend, EnergyBackendError


_TOP_LEVEL_RAPL_RE = re.compile(r"^intel-rapl:\d+$")
_MICROJOULES_PER_JOULE = 1_000_000.0


@dataclass(frozen=True)
class _RaplDomain:
    """Files required to read one top-level Intel RAPL package domain."""

    key: str
    path: Path
    energy_path: Path
    max_range_path: Path
    label: str


class RaplEnergyBackend(EnergyBackend):
    """Read cumulative Intel package energy from Linux powercap sysfs.

    Parameters
    ----------
    powercap_root:
        Root directory containing ``intel-rapl:*`` package domains. The default
        is the standard Linux powercap sysfs location. Tests may inject a
        temporary directory.
    system_name:
        Optional operating-system name override. By default ``platform.system``
        is used. This keeps platform detection testable without patching global
        state.
    """

    name = "rapl"

    def __init__(
        self,
        powercap_root: str | Path = "/sys/class/powercap",
        *,
        system_name: Optional[str] = None,
    ) -> None:
        self._root = Path(powercap_root)
        self._system_name = system_name if system_name is not None else platform.system()
        self._domains_cache: Optional[List[_RaplDomain]] = None
        self._previous_raw_uj: Dict[str, int] = {}
        self._wrap_offset_uj: Dict[str, int] = {}

    def available(self) -> bool:
        """Return whether at least one top-level Intel RAPL package is present."""
        if self._system_name != "Linux":
            return False
        try:
            return bool(self._domains())
        except OSError:
            return False

    def unavailable_reason(self) -> str:
        """Explain why RAPL cannot be used on the current system."""
        if self._system_name != "Linux":
            return "Intel RAPL sysfs energy counters are only supported on Linux."
        if not self._root.exists():
            return f"Linux powercap path does not exist: {self._root}"
        return "No top-level Intel RAPL package energy counter was found."

    def read_energy(self) -> float:
        """Return a monotonic cumulative package-energy counter in joules.

        Multiple CPU package domains are summed. Wraparound is corrected per
        package before summation so one wrapped package cannot make the total
        counter decrease.
        """
        if self._system_name != "Linux":
            raise EnergyBackendError(
                "Intel RAPL sysfs energy counters are only supported on Linux."
            )

        try:
            domains = self._domains()
        except OSError as exc:
            raise EnergyBackendError(f"Could not discover Intel RAPL domains: {exc}") from exc

        if not domains:
            raise EnergyBackendError(self.unavailable_reason())

        total_effective_uj = 0
        for domain in domains:
            try:
                raw_uj = self._read_nonnegative_integer(domain.energy_path)
                max_range_uj = self._read_positive_integer(domain.max_range_path)
            except (OSError, ValueError) as exc:
                raise EnergyBackendError(
                    f"Could not read RAPL domain {domain.key}: {exc}"
                ) from exc

            previous = self._previous_raw_uj.get(domain.key)
            offset = self._wrap_offset_uj.get(domain.key, 0)

            if previous is not None and raw_uj < previous:
                offset += max_range_uj
                self._wrap_offset_uj[domain.key] = offset

            self._previous_raw_uj[domain.key] = raw_uj
            total_effective_uj += raw_uj + offset

        return total_effective_uj / _MICROJOULES_PER_JOULE

    def metadata(self) -> Dict[str, object]:
        """Return RAPL source and discovered package-domain information."""
        domains = self._safe_domains()
        return {
            "source": "linux_powercap_sysfs",
            "native_unit": "microjoule",
            "normalized_unit": "joule",
            "package_count": len(domains),
        }

    def domains(self) -> Dict[str, object]:
        """Describe discovered package domains without reading energy counters."""
        result: Dict[str, object] = {}
        for domain in self._safe_domains():
            result[domain.key] = {
                "label": domain.label,
                "path": str(domain.path),
                "native_unit": "microjoule",
            }
        return result

    def _domains(self) -> List[_RaplDomain]:
        """Discover and cache top-level Intel RAPL package domains."""
        if self._domains_cache is not None:
            return self._domains_cache

        if not self._root.exists() or not self._root.is_dir():
            self._domains_cache = []
            return self._domains_cache

        domains: List[_RaplDomain] = []
        for path in sorted(self._root.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or _TOP_LEVEL_RAPL_RE.fullmatch(path.name) is None:
                continue

            energy_path = path / "energy_uj"
            max_range_path = path / "max_energy_range_uj"
            if not energy_path.is_file() or not max_range_path.is_file():
                continue

            label = self._read_optional_label(path / "name", fallback=path.name)
            domains.append(
                _RaplDomain(
                    key=path.name,
                    path=path,
                    energy_path=energy_path,
                    max_range_path=max_range_path,
                    label=label,
                )
            )

        self._domains_cache = domains
        return domains

    def _safe_domains(self) -> List[_RaplDomain]:
        """Return domains for metadata without propagating discovery failures."""
        try:
            return self._domains()
        except OSError:
            return []

    @staticmethod
    def _read_optional_label(path: Path, *, fallback: str) -> str:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            return fallback
        return value or fallback

    @staticmethod
    def _read_nonnegative_integer(path: Path) -> int:
        value = int(path.read_text(encoding="utf-8").strip())
        if value < 0:
            raise ValueError(f"negative counter value in {path}")
        return value

    @staticmethod
    def _read_positive_integer(path: Path) -> int:
        value = int(path.read_text(encoding="utf-8").strip())
        if value <= 0:
            raise ValueError(f"non-positive max energy range in {path}")
        return value

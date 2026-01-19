"""
Schema for detailed hardware inventory reporting.

This schema captures *descriptive* system information rather than
capability flags. It is intended for:
    - reproducibility and audit trails in benchmark reports
    - appendix tables in papers or technical documentation
    - detailed environment snapshots (CPU, RAM, storage, accelerators)

In contrast to HardwareCapabilitiesSchema, this schema does not attempt
to infer suitability or routing decisions.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HardwareDetailSchema(BaseModel):
    """
    Detailed hardware description of the host system.

    Fields in this schema are intentionally flexible (Dict[str, Any] / Any)
    because hardware probing is platform-dependent and often produces
    heterogeneous metadata structures.
    """

    # Host / operating system
    hostname: Optional[str] = None           # System hostname
    os: Optional[str] = None                 # Operating system and version
    architecture: Optional[str] = None       # CPU architecture (e.g. x86_64, aarch64)

    # CPU and memory
    cpu: Optional[Dict[str, Any]] = None     # Detailed CPU information (model, cores, flags, etc.)
    ram_total_gb: Optional[float] = None     # Total system RAM in gigabytes
    ram_modules: Optional[Any] = None        # Per-DIMM/module information (if available)

    # Storage
    storage: Optional[Dict[str, Any]] = None         # Aggregated storage information
    storage_modules: Optional[Any] = None            # Per-device storage details (NVMe, SATA, etc.)

    # Accelerators
    gpus: Optional[List[Dict[str, Any]]] = None      # List of GPU devices with detailed metadata
    npus: Optional[List[Dict[str, Any]]] = None      # List of NPU / AI accelerator devices

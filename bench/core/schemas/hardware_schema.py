# bench/core/schemas/hardware_schema.py
"""
Top-level hardware schema.

This schema aggregates all hardware-related sub-schemas into a single,
stable structure that can be:
    - serialized into JSON for benchmark artifacts,
    - embedded into result tables and reports,
    - used as a reproducible hardware fingerprint reference.

It intentionally separates:
    - tabular summaries (HardwareTableSchema),
    - detailed inventory information (HardwareDetailSchema),
    - capability and routing-relevant flags (HardwareCapabilitiesSchema).
"""

from typing import Optional

from pydantic import BaseModel

from .hardware_table_schema import HardwareTableSchema
from .hardware_detail_schema import HardwareDetailSchema
from .hardware_capabilities_schema import HardwareCapabilitiesSchema


class HardwareSchema(BaseModel):
    """
    Aggregated hardware description.

    Fields:
        table:
            Compact, human-readable summary intended for tables and quick inspection.
        detail:
            Detailed hardware inventory (CPU, RAM, storage, GPUs, NPUs).
        capabilities:
            Capability and feature flags used for backend routing and reporting.
        fingerprint:
            Stable hardware fingerprint string used to uniquely identify a hardware setup
            across benchmark runs.
    """

    table: HardwareTableSchema
    detail: HardwareDetailSchema
    capabilities: HardwareCapabilitiesSchema
    fingerprint: str

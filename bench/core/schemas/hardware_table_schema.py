# bench/core/schemas/hardware_table_schema.py
"""
Schema for compact hardware table summaries.

This schema is intended for:
    - concise benchmark tables (e.g. in CSV/Markdown/LaTeX),
    - quick human-readable overviews of the test system,
    - comparison across multiple benchmark runs.

It deliberately contains only high-level, aggregated fields and avoids
fine-grained technical detail.
"""

from typing import Optional

from pydantic import BaseModel


class HardwareTableSchema(BaseModel):
    """
    Compact, table-oriented hardware summary.

    All fields are optional to allow partial population when certain
    information is unavailable or intentionally omitted.
    """

    cpu: Optional[str] = None          # CPU model / short description
    gpu: Optional[str] = None          # Primary GPU model (or summary string)
    npu: Optional[str] = None          # Primary NPU / accelerator (if present)
    ram_gb: Optional[float] = None     # Total system RAM in gigabytes
    storage_gb: Optional[float] = None # Total usable storage in gigabytes
    os: Optional[str] = None           # Operating system (short form)

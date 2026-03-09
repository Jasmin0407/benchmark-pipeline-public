"""Pydantic schemas for a benchmark run result.

The run schema is the persisted, user-facing contract for benchmark output.
It validates metadata, model information, metrics, and optional hardware/env
reports.
"""

from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from bench.core.schemas.env_schema import EnvSchema
from bench.core.schemas.hardware_schema import HardwareSchema


class MetaSchema(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    framework: str = "torch"
    device_target: str = "cpu"


class ModelSchema(BaseModel):
    path: Optional[str]
    input_shape: List[int]
    dtype: str
    parameters: Optional[int] = None
    size_on_disk_bytes: Optional[int] = None
    dtype_breakdown: Optional[Dict[str, int]] = None
    fs_hz: Optional[float] = None
    input_num_samples: Optional[int] = None
    input_duration_s: Optional[float] = None


class MemorySchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    rss_start_bytes: Optional[float] = None
    rss_end_bytes: Optional[float] = None
    rss_delta_bytes: Optional[float] = None
    rss_peak_bytes: Optional[float] = None

    duration_s: Optional[float] = None
    pre_roll_s: Optional[float] = None
    inference_duration_s: Optional[float] = None
    post_delay_s: Optional[float] = None

    infer_start_idx: Optional[int] = None
    infer_end_idx: Optional[int] = None
    sample_hz: Optional[float] = None

    rss_samples: Optional[List[int]] = None
    timestamps_s: Optional[List[float]] = None
    weights_bytes: Optional[float] = None
    mode: Optional[str] = None


class ThreadConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    requested_threads: Optional[int] = None
    backend_thread_control_supported: Optional[bool] = None
    applied_intra_op_threads: Optional[int] = None
    applied_inter_op_threads: Optional[int] = None
    execution_mode: Optional[str] = None
    thread_env: Optional[Dict[str, Optional[str]]] = None
    note: Optional[str] = None


class MetricsSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    macs: Optional[Dict[str, Optional[float]]] = None
    inference_time_ms: Optional[Dict[str, Any]] = None
    throughput_sps: Optional[float] = None
    cpu_utilization_pct: Optional[Dict[str, Any]] = None
    memory: Optional[MemorySchema] = None
    ms_per_signal_s: Optional[float] = None


class RunSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    meta: MetaSchema
    model: ModelSchema
    metrics: MetricsSchema
    config: Dict[str, Any]

    hardware: Optional[HardwareSchema] = None
    env: Optional[EnvSchema] = None
    thread_config: Optional[ThreadConfigSchema] = None

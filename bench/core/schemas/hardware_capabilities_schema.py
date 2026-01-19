# bench/core/schemas/hardware_capabilities_schema.py
"""
Schemas for hardware capability reporting.

Important notes:
    - gpus: list of per-GPU capability entries (required for hybrid / multi-GPU setups).
    - gpu (legacy): optional aggregated GPU block kept for backward compatibility;
      may be removed in a future schema version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class GpuCapabilitySchema(BaseModel):
    """
    Capability description for a single GPU device.

    This schema is intentionally lightweight and focuses on:
        - identity / inventory information
        - coarse capability flags that are useful for routing and reporting

    Detailed performance characteristics are out of scope and should be
    measured empirically by the benchmark pipeline.
    """

    # Identity / inventory
    device: Optional[str] = None          # Logical device identifier (e.g. "cuda:0")
    name: Optional[str] = None            # Human-readable GPU name
    vendor: Optional[str] = None          # GPU vendor (e.g. "NVIDIA", "Intel", "AMD")

    # Heuristics (routing / reporting hints)
    fp16: Optional[bool] = False           # Indicates FP16 support (heuristic)
    int8: Optional[bool] = False           # Indicates INT8 support (heuristic)
    tensor_cores: Optional[bool] = False   # Indicates presence of tensor cores (if applicable)

    # NVIDIA-specific metadata (if available)
    compute_capability: Optional[str] = None


class HardwareCapabilitiesSchema(BaseModel):
    """
    Aggregated hardware capability report.

    This schema summarizes which execution backends and instruction sets
    are available on the current system. It is primarily intended for:
        - reporting and audit purposes
        - backend routing decisions at a coarse level

    Fine-grained suitability (e.g. actual performance) must be determined
    by running benchmarks, not by this schema alone.
    """

    # Feature flags (backend / routing relevant)
    supports_cuda: Optional[bool] = False
    supports_tensorrt: Optional[bool] = False
    supports_openvino: Optional[bool] = False
    supports_onnx: Optional[bool] = False

    # CPU instruction set architecture (ISA) flags (e.g. AVX, AVX2, AVX-512)
    cpu_isa: Optional[Dict[str, bool]] = None

    # Legacy: aggregated GPU capabilities (kept for backward compatibility)
    gpu: Optional[Dict[str, Any]] = None

    # Preferred: per-GPU capability list (supports hybrid / multi-GPU systems)
    gpus: Optional[List[GpuCapabilitySchema]] = None

    # NPU capabilities (typically reported via OpenVINO)
    npu: Optional[Dict[str, Any]] = None

    # ONNX Runtime providers reported as available on this system
    # (Note: routing logic should rely on the *active* provider at runtime)
    onnx_providers: Optional[List[str]] = None

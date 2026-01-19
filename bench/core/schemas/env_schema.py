"""Environment metadata schema.

The schema is designed for *reporting* and *reproducibility*:
- All fields are optional (best-effort collectors must not break a run).
- Nested fields may be included for richer diagnostics (e.g. NumPy build config).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class EnvSchema(BaseModel):
    """Normalized environment report for a benchmark run."""

    # Allow forward/backward compatible collectors (new keys should not crash old runs).
    model_config = ConfigDict(extra="allow")

    # Python / Compiler
    python: Optional[str] = None
    python_compiler: Optional[str] = None

    # Core ML frameworks
    torch: Optional[str] = None
    torch_cuda: Optional[str] = None
    torch_cudnn: Optional[int] = None

    onnx: Optional[str] = None
    onnx_opset: Optional[int] = None

    onnxruntime: Optional[str] = None
    onnxruntime_providers: Optional[List[str]] = None
    onnxruntime_providers_active: Optional[List[str]] = None
    onnxruntime_cuda_available: Optional[bool] = None

    # OpenVINO
    openvino: Optional[str] = None
    openvino_devices: Optional[List[str]] = None
    openvino_capabilities: Optional[Dict[str, Optional[List[str]]]] = None
    openvino_ir_version: Optional[int] = None

    # NVIDIA / CUDA / TensorRT
    tensorrt: Optional[str] = None
    cuda_toolkit: Optional[str] = None
    cudnn: Optional[Any] = None  # int or str depending on source

    # Numerical stack
    mkl: Optional[str] = None
    numpy: Optional[str] = None
    numpy_blas: Optional[str] = None
    numpy_lapack: Optional[str] = None
    numpy_simd: Optional[Dict[str, Any]] = None
    numpy_env: Optional[Dict[str, Any]] = None
    numba: Optional[str] = None

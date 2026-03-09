"""
OnnxRunner – Executes ONNX models via ONNX Runtime (onnxruntime).

Supported device strings (high-level intent):
    - cpu / ov_cpu / openvino_cpu
        Uses OpenVINOExecutionProvider with device_type=CPU (if available).
    - gpu / ov_gpu / openvino_gpu
        Uses OpenVINOExecutionProvider with device_type=GPU (if available).
    - npu / ov_npu / openvino_npu
        Uses OpenVINOExecutionProvider with device_type=NPU (if available).
    - ort_cpu / onnx_cpu / ort:cpu
        Uses the classic ONNX Runtime CPUExecutionProvider.
    - ort:cuda / cuda / ort_cuda
        Uses CUDAExecutionProvider (NVIDIA, Windows/Linux).
    - ort:trt / trt / tensorrt / ort_trt
        Uses TensorrtExecutionProvider (primarily Linux/Jetson); may fall back to CUDA/CPU.

Provider selection is finalized by the central provider policy:
    bench/core/runner/provider_policy.py -> resolve_onnx_providers(...)

Design goals:
    - Best-effort operation with optional fallback, unless fail-fast is required.
    - Clear audit/reporting fields for downstream JSON/plots.
    - Dummy inputs are derived from the model signature by default.
"""

from __future__ import annotations

import platform
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort

from .base_runner import BaseRunner
from .provider_policy import resolve_onnx_providers

ProviderType = Union[str, Tuple[str, Dict[str, Any]]]


class OnnxRunner(BaseRunner):
    """Runner for ONNX Runtime with explicit provider and thread audit metadata."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        provider: Optional[ProviderType] = None,
        allow_fallback: bool = False,
        providers_override: Optional[List[ProviderType]] = None,
        threads: Optional[int] = None,
        inter_op_threads: Optional[int] = None,
        execution_mode: Optional[str] = None,
    ) -> None:
        super().__init__(model_path, device)

        self.session: Optional[ort.InferenceSession] = None
        self.provider: ProviderType = provider or self._map_device(device)
        self.allow_fallback: bool = bool(allow_fallback)
        self.providers_override: Optional[List[ProviderType]] = providers_override

        self.platform_system: str = platform.system()
        self.provider_chain: List[ProviderType] = []
        self.active_provider: Optional[str] = None
        self.fallback_occurred: bool = False

        requested_threads = None if threads is None else int(threads)
        requested_inter_op = None if inter_op_threads is None else int(inter_op_threads)
        self.requested_intra_op_threads: Optional[int] = requested_threads if requested_threads and requested_threads > 0 else None
        self.requested_inter_op_threads: Optional[int] = requested_inter_op if requested_inter_op and requested_inter_op > 0 else None
        self.requested_execution_mode: Optional[str] = str(execution_mode).lower() if execution_mode else None

        self.applied_intra_op_threads: Optional[int] = None
        self.applied_inter_op_threads: Optional[int] = None
        self.applied_execution_mode: Optional[str] = None
        self.backend_thread_control_supported: bool = True

    def _map_device(self, device: str) -> ProviderType:
        dev = (device or "").lower()
        if dev in ("cpu", "ov_cpu", "openvino_cpu"):
            return ("OpenVINOExecutionProvider", {"device_type": "CPU"})
        if dev in ("gpu", "ov_gpu", "openvino_gpu"):
            return ("OpenVINOExecutionProvider", {"device_type": "GPU"})
        if dev in ("npu", "ov_npu", "openvino_npu"):
            return ("OpenVINOExecutionProvider", {"device_type": "NPU"})
        if dev in ("ort_cpu", "onnx_cpu", "ort:cpu"):
            return "CPUExecutionProvider"
        if dev in ("ort_cuda", "onnx_cuda", "cuda", "ort:cuda"):
            return "CUDAExecutionProvider"
        if dev in ("ort_trt", "trt", "tensorrt", "ort:trt"):
            return "TensorrtExecutionProvider"
        return "CPUExecutionProvider"

    @staticmethod
    def _provider_name(p: ProviderType) -> str:
        if isinstance(p, tuple):
            return p[0]
        return p

    @staticmethod
    def _resolve_execution_mode(mode: Optional[str]) -> Optional[Any]:
        if mode is None:
            return None
        mode_norm = str(mode).strip().lower()
        if mode_norm in ("sequential", "seq"):
            return getattr(ort.ExecutionMode, "ORT_SEQUENTIAL", None)
        if mode_norm in ("parallel", "par"):
            return getattr(ort.ExecutionMode, "ORT_PARALLEL", None)
        return None

    def _build_session_options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()

        if self.requested_intra_op_threads is not None:
            options.intra_op_num_threads = int(self.requested_intra_op_threads)
            self.applied_intra_op_threads = int(self.requested_intra_op_threads)

        if self.requested_inter_op_threads is not None:
            options.inter_op_num_threads = int(self.requested_inter_op_threads)
            self.applied_inter_op_threads = int(self.requested_inter_op_threads)

        execution_mode = self._resolve_execution_mode(self.requested_execution_mode)
        if execution_mode is not None:
            options.execution_mode = execution_mode
            self.applied_execution_mode = str(self.requested_execution_mode)

        return options

    def get_thread_audit(self) -> Dict[str, Any]:
        return {
            "backend_thread_control_supported": bool(self.backend_thread_control_supported),
            "applied_intra_op_threads": self.applied_intra_op_threads,
            "applied_inter_op_threads": self.applied_inter_op_threads,
            "execution_mode": self.applied_execution_mode,
            "provider_chain": [self._provider_name(p) for p in self.provider_chain],
            "active_provider": self.active_provider,
            "fallback_occurred": self.fallback_occurred,
        }

    def load(self) -> None:
        try:
            available = ort.get_available_providers()
            decision = resolve_onnx_providers(
                device=self.device,
                platform_system=self.platform_system,
                allow_fallback=self.allow_fallback,
                yaml_override=self.providers_override,
            )

            self.provider_chain = list(decision.provider_chain)
            if not self.provider_chain:
                self.provider_chain = ["CPUExecutionProvider"]
                self.fallback_occurred = True

            first_requested = self._provider_name(self.provider_chain[0])
            if decision.fail_fast and first_requested not in available:
                raise RuntimeError(
                    f"Requested provider '{first_requested}' not available. "
                    f"Available providers: {available}. "
                    f"Device={self.device}, platform={self.platform_system}"
                )

            providers_filtered: List[ProviderType] = []
            for provider in self.provider_chain:
                provider_name = self._provider_name(provider)
                if provider_name in available:
                    providers_filtered.append(provider)

            if not providers_filtered:
                providers_filtered = ["CPUExecutionProvider"]
                self.fallback_occurred = True

            if self._provider_name(providers_filtered[0]) != first_requested:
                self.fallback_occurred = True

            session_options = self._build_session_options()
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers_filtered,
            )

            active = self.session.get_providers()
            self.active_provider = active[0] if active else None

            input_names = [i.name for i in self.session.get_inputs()]
            print(f"[OnnxRunner] Session Providers: {active}")
            print(
                f"[OnnxRunner] ONNX model loaded. Inputs={input_names}, "
                f"ActiveProvider={self.active_provider}, "
                f"IntraOpThreads={self.applied_intra_op_threads}, "
                f"InterOpThreads={self.applied_inter_op_threads}, "
                f"ExecutionMode={self.applied_execution_mode}"
            )

            if decision.fail_fast and first_requested == "CUDAExecutionProvider":
                if self.active_provider != "CUDAExecutionProvider":
                    raise RuntimeError(
                        "Fail-fast: Requested CUDAExecutionProvider but active provider is "
                        f"{self.active_provider}. Check CUDA/cuDNN/PATH and dependencies."
                    )

        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime failed to initialize session: {exc}") from exc

    def teardown(self) -> None:
        self.session = None
        self.active_provider = None
        self.provider_chain = []
        self.fallback_occurred = False
        self.applied_intra_op_threads = None
        self.applied_inter_op_threads = None
        self.applied_execution_mode = None

    def prepare(self, input_spec: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        if self.session is None:
            raise RuntimeError("ONNX: Session not loaded. Call load() first.")

        feed: Dict[str, np.ndarray] = {}
        shape_map: Dict[str, Any] = {}
        global_shape = None
        if input_spec is not None:
            shape_map = input_spec.get("shape_map", {}) or {}
            global_shape = input_spec.get("shape")

        for input_info in self.session.get_inputs():
            name = input_info.name
            onnx_type = input_info.type
            dtype = np.float32 if "float" in str(onnx_type).lower() else np.float32
            model_shape = list(input_info.shape)

            for i, dim in enumerate(model_shape):
                if dim is None or dim == 0 or dim == -1 or str(dim).lower() == "none":
                    model_shape[i] = 1

            if name in shape_map:
                shape_raw = shape_map[name]
            elif global_shape is not None:
                shape_raw = global_shape
            else:
                shape_raw = model_shape

            shape = [
                1 if (dim is None or dim == 0 or dim == -1 or str(dim).lower() == "none") else int(dim)
                for dim in shape_raw
            ]

            if global_shape is not None and name not in shape_map:
                if len(shape) != len(model_shape):
                    raise ValueError(
                        f"[OnnxRunner] YAML shape rank mismatch for '{name}': YAML={shape} vs Model={model_shape}"
                    )
                for dim_index in range(1, len(shape)):
                    if int(shape[dim_index]) != int(model_shape[dim_index]):
                        raise ValueError(
                            f"[OnnxRunner] YAML shape mismatch for '{name}' at dim {dim_index}: "
                            f"YAML={shape} vs Model={model_shape}. "
                            "Expected layout is typically [B,T,C] for ECG ONNX models."
                        )

            feed[name] = np.random.randn(*shape).astype(dtype)

        print("[OnnxRunner] Dummy input created: " + ", ".join(f"{k}: {v.shape}" for k, v in feed.items()))
        return feed

    def warmup(self, n: int = 10, input_spec: Optional[Dict[str, Any]] = None) -> None:
        if self.session is None:
            raise RuntimeError("ONNX: Session not loaded. Call load() first.")

        inputs = self.prepare(input_spec)
        for _ in range(n):
            _ = self.session.run(None, inputs)

        print(f"[OnnxRunner] Warmup completed ({n} iterations)")

    def infer(self, dummy_input: Dict[str, np.ndarray]) -> Any:
        if self.session is None:
            raise RuntimeError("ONNX: Session not loaded. Call load() first.")
        return self.session.run(None, dummy_input)

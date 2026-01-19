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
    """
    Runner for ONNX Runtime.

    Notes:
        - This runner intentionally does not hard-code provider chains.
          The provider policy decides the final chain depending on OS, device string,
          availability, and allow_fallback.
        - The `provider` argument is kept for backward compatibility, but provider selection
          should primarily be driven by the policy layer and/or config overrides.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        provider: Optional[ProviderType] = None,
        allow_fallback: bool = False,
        providers_override: Optional[List[ProviderType]] = None,
    ) -> None:
        super().__init__(model_path, device)

        self.session: Optional[ort.InferenceSession] = None

        # Device -> provider mapping is only a convenience default.
        # The provider policy layer decides the actual provider chain.
        self.provider: ProviderType = provider or self._map_device(device)

        # Runtime / policy flags
        self.allow_fallback: bool = bool(allow_fallback)
        self.providers_override: Optional[List[ProviderType]] = providers_override

        # Reporting / audit (persist into result JSON)
        self.platform_system: str = platform.system()
        self.provider_chain: List[ProviderType] = []
        self.active_provider: Optional[str] = None
        self.fallback_occurred: bool = False

    # ------------------------------------------------------------------
    # Provider mapping (baseline mapping; policy may override)
    # ------------------------------------------------------------------
    def _map_device(self, device: str) -> ProviderType:
        """
        Map the user-facing device string to an initial provider preference.

        This is a baseline mapping only. The provider policy is responsible for:
            - OS-specific constraints (e.g., Windows TRT limitations)
            - enforcing fail-fast requirements (if requested)
            - applying YAML overrides (providers_override)
            - filtering against onnxruntime.get_available_providers()
        """
        dev = (device or "").lower()

        # OpenVINO-based routes (preferred if OpenVINO EP is available)
        if dev in ("cpu", "ov_cpu", "openvino_cpu"):
            return ("OpenVINOExecutionProvider", {"device_type": "CPU"})
        if dev in ("gpu", "ov_gpu", "openvino_gpu"):
            return ("OpenVINOExecutionProvider", {"device_type": "GPU"})
        if dev in ("npu", "ov_npu", "openvino_npu"):
            return ("OpenVINOExecutionProvider", {"device_type": "NPU"})

        # Classic ONNX Runtime CPU
        if dev in ("ort_cpu", "onnx_cpu", "ort:cpu"):
            return "CPUExecutionProvider"

        # Classic ONNX Runtime CUDA (NVIDIA)
        if dev in ("ort_cuda", "onnx_cuda", "cuda", "ort:cuda"):
            return "CUDAExecutionProvider"

        # Optional: ORT TensorRT (primarily meaningful on Linux/Jetson)
        if dev in ("ort_trt", "trt", "tensorrt", "ort:trt"):
            return "TensorrtExecutionProvider"

        # Final fallback
        return "CPUExecutionProvider"

    @staticmethod
    def _provider_name(p: ProviderType) -> str:
        """Return the provider name for both string and (name, options) formats."""
        if isinstance(p, tuple):
            return p[0]
        return p

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """
        Create an ONNX Runtime session using the central provider policy.

        Policy behavior examples (depends on your resolve_onnx_providers implementation):
            - Windows + ort:cuda -> CUDA + CPU only (no TRT/OV if not supported)
            - Linux/Jetson + ort:trt -> TRT -> CUDA -> CPU (optional)
            - Fail-fast for requested provider if required

        Raises:
            RuntimeError: If session initialization fails, or fail-fast conditions are violated.
        """
        try:
            available = ort.get_available_providers()

            decision = resolve_onnx_providers(
                device=self.device,
                platform_system=self.platform_system,
                allow_fallback=self.allow_fallback,
                yaml_override=self.providers_override,
            )

            # For reporting / JSON
            self.provider_chain = list(decision.provider_chain)

            if not self.provider_chain:
                # Defensive: policy should never return an empty chain, but guard anyway.
                self.provider_chain = ["CPUExecutionProvider"]
                self.fallback_occurred = True

            # The "first" provider in the chain is considered the primary target.
            first_requested = self._provider_name(self.provider_chain[0])

            # Fail-fast: if explicitly required and the primary provider is not even available.
            if decision.fail_fast and first_requested not in available:
                raise RuntimeError(
                    f"Requested provider '{first_requested}' not available. "
                    f"Available providers: {available}. "
                    f"Device={self.device}, platform={self.platform_system}"
                )

            # If fallback is allowed, filter providers that are not available in this ORT build.
            providers_filtered: List[ProviderType] = []
            for p in self.provider_chain:
                name = self._provider_name(p)
                if name in available:
                    providers_filtered.append(p)

            if not providers_filtered:
                providers_filtered = ["CPUExecutionProvider"]
                self.fallback_occurred = True

            if self._provider_name(providers_filtered[0]) != first_requested:
                self.fallback_occurred = True

            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers_filtered,
            )

            active = self.session.get_providers()
            print(f"[OnnxRunner] Session Providers: {active}")

            self.active_provider = active[0] if active else None

            input_names = [i.name for i in self.session.get_inputs()]
            print(
                f"[OnnxRunner] ONNX model loaded. Inputs={input_names}, "
                f"ActiveProvider={self.active_provider}"
            )

            # Hard gate: if fail-fast requested CUDA as primary, it must be active.
            if decision.fail_fast and first_requested == "CUDAExecutionProvider":
                if self.active_provider != "CUDAExecutionProvider":
                    raise RuntimeError(
                        "Fail-fast: Requested CUDAExecutionProvider but active provider is "
                        f"{self.active_provider}. Check CUDA/cuDNN/PATH and dependencies."
                    )

        except Exception as exc:
            raise RuntimeError(
                f"ONNX Runtime failed to initialize session: {exc}"
            ) from exc

    def teardown(self) -> None:
        """Release session resources and reset audit fields."""
        self.session = None
        self.active_provider = None
        self.provider_chain = []
        self.fallback_occurred = False

    # ------------------------------------------------------------------
    # Input preparation (model-driven by default)
    # ------------------------------------------------------------------
    def prepare(self, input_spec: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Create a dummy input feed dict for session.run().

        Default behavior:
            - Derive shape from model signature: session.get_inputs()[i].shape
            - Replace dynamic dimensions (None, 0, -1, "None") with 1

        Optional override via input_spec:
            input_spec = {
                "shape": [ ... ],              # global shape for all inputs
                "shape_map": {
                    "input_name_1": [ ... ],   # per-input shape override
                }
            }

        Guardrails:
            - If a global YAML shape is used (and no per-input shape_map), the code checks
              rank and non-batch dimensions against the model signature to prevent
              accidental mismatches (except batch dim 0).
        """
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

            # Minimal dtype handling: extend if you support integer/bool inputs.
            dtype = np.float32 if "float" in str(onnx_type).lower() else np.float32

            # 1) Read model shape
            model_shape = list(input_info.shape)

            # 2) Dynamic dimensions -> 1
            for i, d in enumerate(model_shape):
                if d is None or d == 0 or d == -1 or str(d).lower() == "none":
                    model_shape[i] = 1

            # 3) Apply optional override (shape_map takes precedence over global shape)
            if name in shape_map:
                shp_raw = shape_map[name]
            elif global_shape is not None:
                shp_raw = global_shape
            else:
                shp_raw = model_shape

            # Normalize shape values
            shp = [
                1 if (d is None or d == 0 or d == -1 or str(d).lower() == "none") else int(d)
                for d in shp_raw
            ]

            # Guardrail: validate YAML-provided global shape against model signature (except batch).
            # Only enforced when a global_shape is used and no per-input override exists.
            if global_shape is not None and name not in shape_map:
                if len(shp) != len(model_shape):
                    raise ValueError(
                        f"[OnnxRunner] YAML shape rank mismatch for '{name}': YAML={shp} vs Model={model_shape}"
                    )

                # Do not check batch dimension (dim 0); batch is allowed to vary.
                for di in range(1, len(shp)):
                    if int(shp[di]) != int(model_shape[di]):
                        raise ValueError(
                            f"[OnnxRunner] YAML shape mismatch for '{name}' at dim {di}: "
                            f"YAML={shp} vs Model={model_shape}. "
                            "Expected layout is typically [B,T,C] for ECG ONNX models."
                        )

            arr = np.random.randn(*shp).astype(dtype)
            feed[name] = arr

        print(
            "[OnnxRunner] Dummy input created: "
            + ", ".join(f"{k}: {v.shape}" for k, v in feed.items())
        )
        return feed

    # ------------------------------------------------------------------
    # Warmup / inference
    # ------------------------------------------------------------------
    def warmup(self, n: int = 10, input_spec: Optional[Dict[str, Any]] = None) -> None:
        """
        Run warmup iterations.

        If input_spec is None, shapes are derived strictly from the model signature.
        """
        if self.session is None:
            raise RuntimeError("ONNX: Session not loaded. Call load() first.")

        inputs = self.prepare(input_spec)
        for _ in range(n):
            _ = self.session.run(None, inputs)

        print(f"[OnnxRunner] Warmup completed ({n} iterations)")

    def infer(self, dummy_input: Dict[str, np.ndarray]) -> Any:
        """Run a single inference call."""
        if self.session is None:
            raise RuntimeError("ONNX: Session not loaded. Call load() first.")
        return self.session.run(None, dummy_input)

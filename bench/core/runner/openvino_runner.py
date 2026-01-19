"""
OpenVinoRunner – OpenVINO backend runner for multi-device benchmarking.

Key behavior:
    - Compiles an ONNX/IR model using OpenVINO Runtime (ov.Core).
    - Automatically blocks NPU execution for models with dynamic input shapes,
      because OpenVINO NPU backends typically require static shapes for reliable compilation.
    - Uses a compilation timeout as a safety net (NPU compilation can take long or stall).

Design goals:
    - Deterministic and auditable behavior (clear errors, clear prints).
    - Model-driven dummy input by default, with optional YAML shape guardrails.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import numpy as np
import openvino.runtime as ov

from .base_runner import BaseRunner


class OpenVinoRunner(BaseRunner):
    """
    OpenVINO runner that compiles a model for a target device and executes inference.

    Notes:
        - `device` is expected in OpenVINO format ("CPU", "GPU", "NPU", "AUTO", "HETERO", ...).
        - For NPU, this runner enforces static input shapes by default.
    """

    COMPILE_TIMEOUT = 30  # seconds; compilation can be slow (especially on NPU)

    def __init__(self, model_path: str, device: str = "CPU") -> None:
        super().__init__(model_path, device)
        self.core: ov.Core = ov.Core()
        self.compiled_model: Optional[ov.CompiledModel] = None
        self.input_layer = None

        # Internal compilation error capture (threaded compile)
        self._compile_error: Optional[BaseException] = None

    # -------------------------------------------------------------
    # Device mapping
    # -------------------------------------------------------------
    def _map_device(self, device: str) -> str:
        """Normalize device string to OpenVINO's expected format."""
        return (device or "CPU").upper()

    # -------------------------------------------------------------
    # Detect dynamic shapes in an OpenVINO model
    # -------------------------------------------------------------
    def _has_dynamic_shape(self, model: ov.Model) -> bool:
        """
        Return True if any input has dynamic dimensions.

        OpenVINO represents dynamic dimensions as "?" or via Dimension.is_dynamic.
        """
        for inp in model.inputs:
            for dim in inp.partial_shape:
                if str(dim) == "?" or (hasattr(dim, "is_dynamic") and dim.is_dynamic):
                    return True
        return False

    # -------------------------------------------------------------
    # Load & compile
    # -------------------------------------------------------------
    def load(self) -> None:
        """
        Read and compile the model for the selected device.

        Safety/robustness:
            - Uses a separate thread and join timeout as a guardrail against stalls.
            - Captures compilation exceptions and rethrows with context.
            - Blocks NPU compilation for models with dynamic input dimensions.
        """
        dev = self._map_device(self.device)

        # Read model
        model = self.core.read_model(self.model_path)

        # Block NPU for dynamic shapes (reliability/compatibility guardrail)
        if dev == "NPU" and self._has_dynamic_shape(model):
            raise RuntimeError(
                "NPU disabled: model contains dynamic dimensions. "
                "Please export a fully static ONNX/IR model for NPU execution."
            )

        # Compile with timeout (safety net)
        def _compile() -> None:
            try:
                self.compiled_model = self.core.compile_model(model, dev)
            except Exception as exc:
                self._compile_error = exc

        self._compile_error = None
        thread = threading.Thread(target=_compile, daemon=True)
        thread.start()
        thread.join(timeout=self.COMPILE_TIMEOUT)

        if thread.is_alive():
            raise RuntimeError(f"OpenVINO compile_model() timed out (Device={dev})")

        if self._compile_error:
            raise RuntimeError(
                f"OpenVINO failed to compile model for Device={dev}: {self._compile_error}"
            ) from self._compile_error

        if self.compiled_model is None:
            raise RuntimeError(f"OpenVINO compilation produced no compiled_model (Device={dev})")

        self.input_layer = self.compiled_model.input(0)

        print(
            f"[OpenVinoRunner] Model loaded. Device={dev}, "
            f"PartialShape={self.input_layer.partial_shape}"
        )

    def teardown(self) -> None:
        """Release compiled model resources."""
        self.compiled_model = None
        self.input_layer = None
        self._compile_error = None

    # -------------------------------------------------------------
    # OpenVINO dimension conversion helpers
    # -------------------------------------------------------------
    def _dim_to_int(self, d) -> int:
        """
        Convert an OpenVINO Dimension to an integer.

        Rules:
            - Dynamic -> 1
            - Fixed -> length
            - Ranged -> min_length if positive, else 1
        """
        # Dynamic dimension
        if str(d) == "?" or getattr(d, "is_dynamic", False):
            return 1

        # Common OpenVINO dimension variants
        if hasattr(d, "get_length"):
            return int(d.get_length())

        if hasattr(d, "length"):
            return int(d.length)

        if hasattr(d, "min_length") and hasattr(d, "max_length"):
            mn = int(d.min_length)
            mx = int(d.max_length)
            if mn == mx:
                return mn
            return mn if mn > 0 else 1

        # Fallback: parse as string if possible
        s = str(d)
        if s.isdigit():
            return int(s)

        raise TypeError(f"Cannot convert OpenVINO Dimension to int: {d} (type={type(d)})")

    # -------------------------------------------------------------
    # Guardrail: YAML shape must match model signature (except batch)
    # -------------------------------------------------------------
    def _guardrail_shape_ov(self, yaml_shape, partial_shape, name: str) -> None:
        """
        Validate a user-provided shape against the model's (partial) signature.

        Policy:
            - Rank must match.
            - All dimensions except batch (dim 0) must match exactly.
        """
        model_shape = [self._dim_to_int(d) for d in list(partial_shape)]

        if len(yaml_shape) != len(model_shape):
            raise ValueError(
                f"[OpenVinoRunner] YAML shape rank mismatch for '{name}': "
                f"YAML={yaml_shape} vs Model={model_shape}"
            )

        # Do not validate batch dimension (dim 0); batch size may vary for performance runs.
        for di in range(1, len(model_shape)):
            if int(yaml_shape[di]) != int(model_shape[di]):
                raise ValueError(
                    f"[OpenVinoRunner] YAML shape mismatch for '{name}' at dim {di}: "
                    f"YAML={yaml_shape} vs Model={model_shape}."
                )

    # -------------------------------------------------------------
    # Dummy input creation
    # -------------------------------------------------------------
    def prepare(self, input_spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Create a dummy input dict for OpenVINO inference.

        input_spec:
            - shape: optional, global input shape list
              If omitted, the shape is derived from the compiled model input partial shape.
        """
        if self.compiled_model is None:
            raise RuntimeError("OpenVinoRunner: compiled_model not loaded. Call load() first.")
        if self.input_layer is None:
            raise RuntimeError("OpenVinoRunner: input_layer not initialized. Call load() first.")

        shape = input_spec.get("shape")

        if shape is None:
            partial = list(self.input_layer.partial_shape)
            shape = [self._dim_to_int(d) for d in partial]
        else:
            self._guardrail_shape_ov(shape, self.input_layer.partial_shape, name=self.input_layer.any_name)

        dummy = np.random.randn(*shape).astype(np.float32)
        return {self.input_layer.any_name: dummy}

    # -------------------------------------------------------------
    # Warmup / inference
    # -------------------------------------------------------------
    def warmup(self, n: int = 10, input_spec=None) -> None:
        """
        Run warmup iterations.

        If input_spec is None, the input shape is derived from the model signature.
        """
        if self.compiled_model is None:
            raise RuntimeError("OpenVinoRunner: compiled_model not loaded. Call load() first.")

        inputs = self.prepare(input_spec or {})
        req = self.compiled_model.create_infer_request()

        for _ in range(n):
            req.infer(inputs)

        print(f"[OpenVinoRunner] Warmup completed ({n} iterations)")

    def infer(self, dummy_input: Dict[str, np.ndarray]) -> Any:
        """Run a single inference call and return outputs as numpy arrays."""
        if self.compiled_model is None:
            raise RuntimeError("OpenVinoRunner: compiled_model not loaded. Call load() first.")

        req = self.compiled_model.create_infer_request()
        req.infer(dummy_input)
        return [req.get_tensor(out).data for out in self.compiled_model.outputs]

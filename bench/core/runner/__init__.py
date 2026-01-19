"""Runner package.

This package contains backend-specific runners (Torch, ONNX Runtime, OpenVINO,
TensorRT, ...).

"""

from .base_runner import BaseRunner

# Optional runners are imported defensively so that environments without optional
# dependencies (e.g., onnxruntime, torch, openvino, tensorrt) can still import the
# package and use the parts they have installed.

__all__ = ["BaseRunner"]

try:  # pragma: no cover
    from .torch_runner import TorchRunner

    __all__.append("TorchRunner")
except Exception:
    TorchRunner = None  # type: ignore

try:  # pragma: no cover
    from .onnx_runner import OnnxRunner

    __all__.append("OnnxRunner")
except Exception:
    OnnxRunner = None  # type: ignore

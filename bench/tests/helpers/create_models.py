"""Test model helpers.

This module provides tiny, deterministic models used by unit and integration tests.
Keep these models intentionally small to ensure fast runtime in CI environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class TinyModel(torch.nn.Module if torch is not None else object):
    """A minimal MLP used for smoke tests (Torch / ONNX export).

    The goal is not accuracy but to exercise the full runner/measurement stack with
    predictable shapes and minimal compute.
    """

    def __init__(self, in_features: int = 8, hidden: int = 16, out_features: int = 2):
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required to construct TinyModel.")
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_features),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        return self.fc(x)


def create_tiny_torch_model(path: Path) -> Path:
    """Create and persist a TinyModel for TorchRunner tests.

    Note: we intentionally store the full module (not only a state_dict) so that the
    TorchRunner can load it without requiring custom glue code.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to create test models.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = TinyModel()
    model.eval()
    torch.save(model, path)
    return path


def create_tiny_onnx_model(path: Path, input_shape: tuple[int, int] = (1, 8), opset: int = 13) -> Path:
    """Export a TinyModel to ONNX.

    This is primarily used by TensorRT / ONNX integration tests to avoid any
    hardcoded, machine-local model paths.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required to export ONNX models.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = TinyModel(in_features=input_shape[1])
    model.eval()

    dummy = torch.randn(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # keep static for TensorRT friendliness
    )
    return path

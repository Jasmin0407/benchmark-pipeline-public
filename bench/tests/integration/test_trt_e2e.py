"""TensorRT end-to-end smoke test.

This is a best-effort integration test:
- Skips automatically on Windows (TensorRT support is typically Linux-only in this project).
- Skips if TensorRT is not importable.
- Generates a tiny ONNX model in a temporary directory (no machine-local paths).

The goal is to validate that the TensorRTRunner can load, prepare, warm up, and
run a single inference without crashing.
"""

from __future__ import annotations

import platform

import pytest

# TensorRT is only supported for our test setup on Linux.
if platform.system().lower() == "windows":
    pytest.skip("TensorRT is not supported on Windows in this test suite.", allow_module_level=True)

pytest.importorskip("tensorrt")

from bench.core.runner.tensorrt_runner import TensorRTRunner
from bench.tests.helpers.create_models import create_tiny_onnx_model


def test_trt_e2e(tmp_path):
    onnx_path = create_tiny_onnx_model(tmp_path / "tiny_model.onnx", input_shape=(1, 8))

    runner = TensorRTRunner(onnx_path.as_posix(), device="cuda")
    runner.load()

    dummy = runner.prepare({})
    runner.warmup(2, {})
    out = runner.infer(dummy)

    assert out is not None

    runner.teardown()

"""Basic runner smoke tests.

These tests validate that the TorchRunner and OnnxRunner can execute a minimal model.
They are dependency-gated to avoid failing on setups without ONNX/ONNX Runtime.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bench.core.measure.measure_controller import MeasureController
from bench.core.runner.onnx_runner import OnnxRunner
from bench.core.runner.torch_runner import TorchRunner
from bench.tests.helpers.create_models import create_tiny_onnx_model, create_tiny_torch_model


def test_runners_basic(tmp_path):
    # TorchRunner
    pt_path = create_tiny_torch_model(tmp_path / "tiny_model.pt")
    trunner = TorchRunner(model_path=pt_path.as_posix(), device="cpu")
    trunner.load()

    cfg = {
        "run": {"warmups": 2, "repeats": 5},
        "metrics": {"sampler_hz": 10},
        "model": {"backend": "torch", "path": pt_path.as_posix()},
    }

    controller = MeasureController(cfg)
    dummy = trunner.prepare({"shape": [1, 8], "dtype": "float32"})
    result_torch = controller.run_benchmark(trunner, dummy_input=dummy)
    trunner.teardown()
    assert "memory" in result_torch

    # OnnxRunner
    onnx_path = create_tiny_onnx_model(tmp_path / "tiny_model.onnx", input_shape=(1, 8))
    orunner = OnnxRunner(model_path=onnx_path.as_posix(), device="cpu")
    orunner.load()

    cfg["model"]["path"] = onnx_path.as_posix()
    cfg["model"]["backend"] = "onnx"

    controller = MeasureController(cfg)
    result_onnx = controller.run_benchmark(orunner)
    orunner.teardown()
    assert "memory" in result_onnx

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

ov = pytest.importorskip("openvino")

from bench.core.measure.measure_controller import MeasureController
from bench.core.runner.openvino_runner import OpenVinoRunner
from bench.tests.helpers.create_models import create_tiny_onnx_model


def test_openvino_e2e_tmp_model(tmp_path: Path):
    """
    OpenVINO end-to-end integration test.

    This test creates a tiny ONNX model in a temporary directory, converts/loads it
    via the OpenVinoRunner, and runs a small benchmark to ensure the full path works.

    The test is intentionally lightweight and should run on any machine where
    OpenVINO is installed.
    """
    # 1) Create a tiny ONNX model in tmp_path (no absolute/local user paths).
    onnx_path = create_tiny_onnx_model(tmp_path / "tiny_model.onnx", input_shape=(1, 8))

    # 2) Configure and run the OpenVINO runner.
    runner = OpenVinoRunner(model_path=onnx_path.as_posix(), device="CPU")
    runner.load()
    dummy = runner.prepare({"shape": [1, 8], "dtype": "float32"})

    cfg = {
        "run": {"warmups": 2, "repeats": 5},
        "metrics": {"sampler_hz": 10},
        "model": {"backend": "openvino", "path": onnx_path.as_posix()},
    }

    controller = MeasureController(cfg)
    result = controller.run_benchmark(runner, dummy_input=dummy)

    runner.teardown()

    # 3) Minimal sanity assertions: ensure we got metrics back.
    metrics = result.get("metrics", result)
    assert "memory" in metrics
    assert "cpu_utilization_pct" in metrics

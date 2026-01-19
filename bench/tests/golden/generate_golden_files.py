"""Golden file generator (manual).

This script generates baseline JSON artifacts used by the golden regression tests.
Run it manually from the repository root, ideally in a controlled environment.

Notes
-----
- The generated files are committed to the repository.
- Keep the model and measurement settings stable to avoid noisy diffs.
- The script is dependency-gated and will exit early if ONNX/ORT are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from bench.core.measure.measure_controller import MeasureController
from bench.core.runner.onnx_runner import OnnxRunner
from bench.core.runner.torch_runner import TorchRunner
from bench.core.schemas.run_schema import MetaSchema, MetricsSchema, ModelSchema, RunSchema
from bench.tests.helpers.create_models import create_tiny_onnx_model, create_tiny_torch_model
from bench.tests.helpers.golden_sanitize import sanitize_golden_run


def _run_torch(tmp_dir: Path) -> dict:
    model_path = create_tiny_torch_model(tmp_dir / "tiny_model.pt")
    runner = TorchRunner(model_path.as_posix(), device="cpu")
    runner.load()
    dummy = runner.prepare({"shape": [1, 8]})

    cfg = {
        "run": {"warmups": 2, "repeats": 10},
        "metrics": {"sampler_hz": 20},
        "model": {"backend": "torch", "path": model_path.as_posix()},
    }

    controller = MeasureController(cfg)
    result = controller.run_benchmark(runner, dummy_input=dummy)
    metrics_dict = result.get("metrics", result)

    run = RunSchema(
        meta=MetaSchema(framework="torch", device_target="cpu"),
        model=ModelSchema(path=model_path.as_posix(), input_shape=[1, 8], dtype="float32"),
        metrics=MetricsSchema(**metrics_dict),
        config=cfg,
    )

    runner.teardown()
    return run.model_dump()


def _run_onnx(tmp_dir: Path) -> dict:
    onnx_path = create_tiny_onnx_model(tmp_dir / "tiny_model.onnx", input_shape=(1, 8))
    runner = OnnxRunner(onnx_path.as_posix(), device="cpu")
    runner.load()
    dummy = runner.prepare({"shape": [1, 8]})

    cfg = {
        "run": {"warmups": 2, "repeats": 10},
        "metrics": {"sampler_hz": 20},
        "model": {"backend": "onnx", "path": onnx_path.as_posix()},
    }

    controller = MeasureController(cfg)
    result = controller.run_benchmark(runner, dummy_input=dummy)
    metrics_dict = result.get("metrics", result)

    run = RunSchema(
        meta=MetaSchema(framework="onnx", device_target="cpu"),
        model=ModelSchema(path=onnx_path.as_posix(), input_shape=[1, 8], dtype="float32"),
        metrics=MetricsSchema(**metrics_dict),
        config=cfg,
    )

    runner.teardown()
    return run.model_dump()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    golden_dir = repo_root / "bench" / "tests" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = golden_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    torch_raw = _run_torch(tmp_dir)
    onnx_raw = _run_onnx(tmp_dir)

    torch_run = sanitize_golden_run(torch_raw)
    onnx_run = sanitize_golden_run(onnx_raw)

    (golden_dir / "torch_cpu.json").write_text(json.dumps(torch_run, indent=2), encoding="utf-8")
    (golden_dir / "onnx_cpu.json").write_text(json.dumps(onnx_run, indent=2), encoding="utf-8")

    print("Golden files created:")
    print(f"- {golden_dir / 'torch_cpu.json'}")
    print(f"- {golden_dir / 'onnx_cpu.json'}")


if __name__ == "__main__":
    main()

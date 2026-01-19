"""Golden regression tests.

These tests are intentionally conservative: they validate that the *structure* of
the produced benchmark artifacts does not regress (missing keys, wrong nesting, etc.).
They do not assert exact numeric values, which can be unstable across machines.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_golden_regression_torch_cpu():
    base = Path(__file__).resolve().parent
    golden = _load_json(base / "torch_cpu.json")

    assert "metrics" in golden
    assert "inference_time_ms" in golden["metrics"]
    assert "macs" in golden["metrics"]


def test_golden_regression_onnx_cpu():
    base = Path(__file__).resolve().parent
    golden = _load_json(base / "onnx_cpu.json")

    assert "metrics" in golden
    assert "inference_time_ms" in golden["metrics"]
    assert "macs" in golden["metrics"]

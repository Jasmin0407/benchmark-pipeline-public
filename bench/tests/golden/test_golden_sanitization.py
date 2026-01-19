from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_golden_files_are_sanitized():
    base = Path(__file__).resolve().parent
    for name in ["torch_cpu.json", "onnx_cpu.json"]:
        d = _load(base / name)

        # No local file paths
        assert d["model"]["path"] in ("inline", "_inline")

        # No identifying values
        assert d["meta"]["run_id"] == "redacted"
        assert d["hardware"]["fingerprint"] == "redacted"

        # No hostname should be present
        detail = d.get("hardware", {}).get("detail", {}) or {}
        assert "hostname" not in detail

        # No verbose build logs
        env = d.get("env", {}) or {}
        assert env.get("numpy_blas") in (None, "redacted")
        assert env.get("cuda_toolkit") in (None, "redacted")
        assert env.get("mkl") in (None, "redacted")

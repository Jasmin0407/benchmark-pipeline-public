"""TensorRT worker bridge.

The TensorRT runner is utilized when executing in a dedicated virtual environment (e.g. on
systems where TensorRT bindings are only available for a specific Python
version). This helper launches an external worker process and returns its JSON
result.

Security notes:
 - Uses `subprocess.run` with a list of arguments (no shell) to avoid command
   injection.
 - Ensures the output directory exists before execution.
"""
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def run_trt_worker(
    venv_python: str,
    worker_py: str,
    model_path: str,
    iters: int,
    warmup: int,
    shape: Optional[str] = None,
    dla: int = -1,
    out_json: str = "bench_out/trt_result.json",
) -> Dict[str, Any]:
    """Run the TensorRT worker as a subprocess and parse its JSON output.

    Args:
        venv_python: Path to the Python executable inside the TensorRT environment.
        worker_py: Path to the worker script.
        model_path: Path to the ONNX model.
        iters: Number of measured iterations.
        warmup: Number of warm-up iterations.
        shape: Optional static input shape override (backend dependent format).
        dla: DLA index (-1 means GPU).
        out_json: Output JSON path written by the worker.

    Returns:
        Parsed JSON dict.

    Raises:
        RuntimeError: If the worker fails or produces invalid output.
    """

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        venv_python,
        worker_py,
        "--model", model_path,
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--dla", str(dla),
        "--out", out_json,
    ]
    if shape:
        cmd += ["--shape", shape]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "TensorRT worker failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )

    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"TensorRT worker produced invalid JSON output at {out_path!s}: {e}")
    return data


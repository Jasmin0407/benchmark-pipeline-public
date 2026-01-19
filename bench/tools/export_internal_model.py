from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    # bench/tools/export_internal_model.py -> repo root is 3 levels up
    return Path(__file__).resolve().parents[2]


def _try_import_internal_model():
    """
    Import the proprietary/internal model definition from ./internal.

    The internal module is intentionally excluded from version control.
    This helper makes the export pipeline optional and fail-safe.
    """
    repo_root = _repo_root()
    internal_dir = repo_root / "internal"
    module_path = internal_dir / "dual_transformer_exportable.py"

    if not module_path.exists():
        return None, f"Missing internal model file: {module_path}"

    # Import by path to avoid packaging/internal install requirements.
    import importlib.util

    spec = importlib.util.spec_from_file_location("dual_transformer_exportable", module_path)
    if spec is None or spec.loader is None:
        return None, f"Could not load module spec from: {module_path}"

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export proprietary internal model to ONNX without committing model source files."
    )
    parser.add_argument("--out", type=str, default="models/dual_transformer.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for dummy input")
    parser.add_argument("--require", action="store_true", help="Exit with code 2 if internal model file is missing")
    args = parser.parse_args()

    mod, err = _try_import_internal_model()
    if mod is None:
        msg = f"[export_internal_model] {err}"
        if args.require:
            print(msg, file=sys.stderr)
            return 2
        print(msg)
        print("[export_internal_model] Skipping export (internal source is not present).")
        return 0

    # Dependency-gate: only import torch if we actually export.
    try:
        import torch
    except Exception as e:
        print(f"[export_internal_model] torch not available: {e}", file=sys.stderr)
        return 1

    if not hasattr(mod, "DualTransformer"):
        print("[export_internal_model] Internal module does not define DualTransformer.", file=sys.stderr)
        return 1

    model = mod.DualTransformer()
    model.eval()

    out_path = _repo_root() / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input must match your model's expected input.
    # Based on your internal model forward: [B, 1, 2, 7680]
    dummy = torch.randn(args.batch, 1, 2, 7680, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"[export_internal_model] Exported ONNX to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

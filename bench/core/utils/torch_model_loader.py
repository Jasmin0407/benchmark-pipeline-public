"""
bench.core.utils.torch_model_loader

Centralized, configuration-driven loading for PyTorch model artifacts.

Supported torch save formats
---------------------------
1) Full module:
      torch.save(model, path)
2) Weights/checkpoint only:
      torch.save(state_dict_or_checkpoint, path)

For (2), the benchmark pipeline must reconstruct a nn.Module via a model factory.

Configuration contract (YAML)
-----------------------------
model:
  backend: torch
  path: /path/to/model.pt
  factory: "internal.dual_transformer_exportable:DualTransformer"
  factory_kwargs: { ... }              # optional
  factory_state_dict_key: "state_dict" # optional
  factory_strict: true                # optional
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple

import torch


def _safe_torch_load(path: str, map_location: str) -> Any:
    """
    Prefer safe loads to reduce pickle attack surface (PyTorch >= 2.0 supports weights_only).
    Falls back to classic torch.load for older versions or non-weights-only artifacts.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        # Older torch versions: no weights_only argument.
        return torch.load(path, map_location=map_location)
    except Exception:
        # Some artifacts may not be compatible with weights_only; fall back.
        return torch.load(path, map_location=map_location)


def _resolve_factory(factory: str):
    """
    Resolve a factory string to a callable/class.

    Supported formats:
      - "pkg.module:ClassName"
      - "pkg.module.ClassName"
    """
    if ":" in factory:
        mod_name, attr = factory.split(":", 1)
    else:
        parts = factory.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                "Invalid model.factory. Use 'module:Class' or 'module.Class'. "
                f"Got: {factory!r}"
            )
        mod_name, attr = parts[0], parts[1]

    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise ImportError(f"Factory attribute not found: {factory!r}") from exc


def _extract_state_dict(obj: Any, state_dict_key: str = "state_dict") -> Dict[str, torch.Tensor]:
    """
    Extract a state_dict from common checkpoint structures.

    Accepts:
      - state_dict dict itself
      - checkpoint dict containing state_dict_key
    """
    if isinstance(obj, dict):
        if state_dict_key in obj and isinstance(obj[state_dict_key], dict):
            return obj[state_dict_key]
        # Heuristic: treat dict as direct state_dict if it looks like one.
        if all(isinstance(k, str) for k in obj.keys()):
            return obj  # type: ignore[return-value]
    raise TypeError("Loaded torch object is not a state_dict/checkpoint dict.")


def load_torch_model(
    model_path: str,
    cfg: Optional[Dict[str, Any]],
    device: str,
) -> torch.nn.Module:
    """
    Load a torch model as nn.Module.

    Behavior:
      - If model file contains a nn.Module, returns it.
      - If model file contains a state_dict/checkpoint dict:
          - Requires cfg['model']['factory'] to reconstruct the module.
    """
    obj = _safe_torch_load(model_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        model = obj.to(device)
        model.eval()
        return model

    model_cfg = (cfg or {}).get("model") or {}
    factory = model_cfg.get("factory")
    if not factory:
        raise RuntimeError(
            "[TorchModelLoader] Torch artifact is not a torch.nn.Module. "
            "To load state-dict-only checkpoints, set model.factory in YAML, e.g.:\n"
            "  model:\n"
            "    factory: internal.dual_transformer_exportable:DualTransformer\n"
        )

    factory_kwargs = model_cfg.get("factory_kwargs") or {}
    state_dict_key = str(model_cfg.get("factory_state_dict_key", "state_dict"))
    strict = bool(model_cfg.get("factory_strict", True))

    cls_or_fn = _resolve_factory(str(factory))
    model = cls_or_fn(**factory_kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Resolved factory did not return a torch.nn.Module: {factory!r}")

    sd = _extract_state_dict(obj, state_dict_key=state_dict_key)
    missing, unexpected = model.load_state_dict(sd, strict=strict)

    # Keep output predictable (no hard fail here; strict already enforces if desired).
    if missing or unexpected:
        print(
            "[TorchModelLoader] State dict load mismatch:\n"
            f"  missing keys: {len(missing)}\n"
            f"  unexpected keys: {len(unexpected)}"
        )

    model = model.to(device)
    model.eval()
    return model

"""bench.core.system.system_probe

Compatibility wrapper for environment probing.

Historically the project exposed a :func:`collect_env_info` helper from this module.
The canonical implementation lives in :mod:`bench.core.system.env_probe`.

This wrapper deliberately keeps a stable import path for downstream code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from bench.core.schemas.env_schema import EnvSchema
from bench.core.system.env_probe import collect_env_info as _collect_env_info


def collect_env_info(model_path: Optional[str] = None, *, hardware: Optional[Dict[str, Any]] = None) -> EnvSchema:
    """Collect environment and capability information.

    Args:
        model_path: Optional model path. Used by OpenVINO to detect IR version for `.xml` models.
        hardware: Optional hardware inventory (e.g., GPUs list). When provided, it is used for
            conservative provider probing decisions (best-effort; never fails the run).

    Returns:
        An :class:`~bench.core.schemas.env_schema.EnvSchema` instance.
    """

    return _collect_env_info(model_path=model_path, hardware=hardware)

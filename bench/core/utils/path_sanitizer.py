# bench/core/utils/path_sanitizer.py
from __future__ import annotations

import re


def sanitize_component(value: str) -> str:
    """
    Sanitize a generic string so it can be safely used as a single path component.

    This is intentionally conservative and removes anything beyond:
    letters, digits, underscore, dash, dot.
    """
    if value is None:
        return "run"

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    safe = safe.strip("._-")
    return safe or "run"


def sanitize_device_key(value: str) -> str:
    """
    Sanitize a device identifier so it can be safely used in file/folder names.

    Unit-test contract:
    - "cuda:0"    -> "cuda_0"
    - "cuda::0"   -> "cuda_0"
    - "a/b:c"     -> "a_b_c"
    - "../secret" -> "_secret"
    - "///" / ""  -> "_"
    """
    if value is None:
        return "_"

    s = str(value)

    # Replace common separators explicitly.
    s = s.replace(":", "_").replace("/", "_").replace("\\", "_")

    # Replace everything else that is not safe (keep '.' for normal tokens).
    s = re.sub(r"[^A-Za-z0-9\-\_\.]+", "_", s)

    # Collapse consecutive underscores.
    s = re.sub(r"_+", "_", s)

    # If the sanitized string starts with dots (traversal-/dotfile-like),
    # drop the dots and ensure the result is clearly sanitized.
    if s.startswith("."):
        s = s.lstrip(".")
        # After stripping dots, a leading underscore may already exist (e.g. "../x" -> ".._x" -> "_x").
        if not s.startswith("_"):
            s = "_" + s if s else "_"

    # Avoid trailing underscores.
    s = s.rstrip("_")

    return s or "_"

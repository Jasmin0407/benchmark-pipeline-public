"""Unit tests for device key sanitization.

This is a pure unit test and must not import heavy CLI modules or runtime backends.
"""

from __future__ import annotations

from bench.core.utils.path_sanitizer import sanitize_device_key


def test_sanitize_device_key_basic():
    assert sanitize_device_key("cuda:0") == "cuda_0"
    assert sanitize_device_key("AUTO:CPU") == "AUTO_CPU"
    assert sanitize_device_key("npu/0") == "npu_0"


def test_sanitize_device_key_already_safe():
    assert sanitize_device_key("cpu") == "cpu"
    assert sanitize_device_key("cuda_1") == "cuda_1"
    assert sanitize_device_key("GPU-0") == "GPU-0"


def test_sanitize_device_key_edge_cases():
    # Multiple invalid separators collapse to underscores.
    assert sanitize_device_key("cuda::0") == "cuda_0"
    assert sanitize_device_key("a/b:c") == "a_b_c"

    # Path traversal-like strings are neutralized.
    assert sanitize_device_key("../secret") == "_secret"

    # All-invalid strings should not become empty.
    assert sanitize_device_key("///") == "_"
    assert sanitize_device_key("") == "_"

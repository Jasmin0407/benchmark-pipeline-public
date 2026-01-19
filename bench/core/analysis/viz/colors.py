"""Central color palette for all visualizations.

This module defines stable color constants used across plots.
"""

from __future__ import annotations

FRAMEWORK_COLORS = {
    "torch": "#1f77b4",
    "onnx": "#2ca02c",
    "openvino": "#ff7f0e",
    "tensorrt": "#d62728",
}

DEVICE_COLORS = {
    "cpu": "#1f77b4",
    "gpu": "#2ca02c",
    "npu": "#ffd30e",
    "auto": "#7f7f7f",
    "hetero": "#7f7f7f",
}

PHASE_COLORS = {
    "preroll": "#7f7f7f",
    "inference": "#2ca02c",
    "post": "#ff7f0e",
}

# Metric-specific convenience colors
COLOR_CPU_MEAN = "#1f77b4"
COLOR_CPU_P95 = "#ff7f0e"

COLOR_RAM_DELTA = "#1f77b4"

COLOR_SAMPLING_INTERVAL = "#1f77b4"
COLOR_SAMPLING_TARGET = "#ff7f0e"


def color_for_framework(name: str) -> str:
    return FRAMEWORK_COLORS.get(str(name).lower(), "#888888")


def color_for_device(dev: str) -> str:
    d = dev.lower()
    if d.startswith("hetero"):
        return DEVICE_COLORS["hetero"]
    return DEVICE_COLORS.get(d, "#888888")


def phase_color(phase: str) -> str:
    return PHASE_COLORS.get(phase, "#cccccc")

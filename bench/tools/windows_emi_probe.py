"""Read-only Windows EMI probe for the configured package-energy channel."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Sequence

from bench.core.metrics.windows_emi_energy_backend import (
    DEFAULT_PACKAGE_CHANNEL,
    WindowsEmiEnergyBackend,
)


def run_probe(channel: str, interval_s: float) -> dict:
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("interval_s must be a finite number > 0")

    backend = WindowsEmiEnergyBackend(channel_name=channel)
    if not backend.available():
        return {
            "available": False,
            "backend": backend.name,
            "channel": channel,
            "reason": backend.unavailable_reason(),
            "go_for_phase_11g": False,
        }

    first_j = backend.read_energy()
    first_time_s = backend.last_sample_time_s()
    time.sleep(interval_s)
    second_j = backend.read_energy()
    second_time_s = backend.last_sample_time_s()

    delta_j = second_j - first_j
    sensor_duration_s = None
    average_power_w = None
    if first_time_s is not None and second_time_s is not None:
        sensor_duration_s = second_time_s - first_time_s
        if sensor_duration_s > 0.0:
            average_power_w = delta_j / sensor_duration_s

    valid = (
        math.isfinite(first_j)
        and math.isfinite(second_j)
        and second_j >= first_j >= 0.0
        and delta_j > 0.0
        and sensor_duration_s is not None
        and sensor_duration_s > 0.0
        and average_power_w is not None
        and math.isfinite(average_power_w)
        and average_power_w >= 0.0
    )

    return {
        "available": True,
        "backend": backend.name,
        "channel": channel,
        "first_energy_j": first_j,
        "second_energy_j": second_j,
        "delta_energy_j": delta_j,
        "sensor_duration_s": sensor_duration_s,
        "average_power_w": average_power_w,
        "metadata": backend.metadata(),
        "domains": backend.domains(),
        "go_for_phase_11g": bool(valid),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default=DEFAULT_PACKAGE_CHANNEL)
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    result = run_probe(args.channel, args.interval_s)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")

    return 0 if result.get("go_for_phase_11g") else 1


if __name__ == "__main__":
    raise SystemExit(main())

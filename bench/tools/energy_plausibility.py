"""Phase 11 hardware plausibility utilities for energy measurements.

The module deliberately separates two concerns:

1. Preparing controlled benchmark configurations with exact energy inference
   counts (for example 100, 500, and 1000).
2. Analysing persisted ``bench.json`` files after they were measured on real
   hardware.

No hardware is emulated here. Real Phase 11 acceptance still requires running
these configurations on the target machine and reviewing the resulting report.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


DEFAULT_COUNTS: Tuple[int, ...] = (100, 500, 1000)


@dataclass(frozen=True)
class EnergyPlausibilityPoint:
    """One measured point used for Phase 11 plausibility analysis."""

    inference_count: int
    total_energy_j: float
    energy_per_inference_j: float
    duration_s: float
    average_power_w: float
    source_file: str


@dataclass(frozen=True)
class EnergyPlausibilityReport:
    """Machine-readable summary of the Phase 11 plausibility checks."""

    comparable: bool
    fingerprint: Dict[str, Any]
    points: List[Dict[str, Any]]
    total_energy_monotonic: bool
    normalized_relative_spread_pct: float
    max_relative_spread_pct: Optional[float]
    normalized_stability_passed: Optional[bool]
    overall_status: str


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1")
    parsed = int(value)
    if float(value) != float(parsed) or parsed < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return parsed


def prepare_plausibility_config(
    base_cfg: Mapping[str, Any],
    inference_count: int,
) -> Dict[str, Any]:
    """Return a benchmark config with an exact Energy-pass inference count.

    ``determine_energy_repeats`` uses ``run.repeats`` as a lower bound and
    ``max_repeats`` as a hard upper bound. Setting both to the same value makes
    the dedicated Energy pass deterministic, independent of model latency.

    A tiny positive ``min_duration_s`` keeps the regular configuration contract
    valid while allowing the explicit repeat count to control Phase 11.
    """

    count = _require_positive_int(inference_count, "inference_count")
    cfg = deepcopy(dict(base_cfg))

    run_cfg = cfg.setdefault("run", {})
    run_cfg["repeats"] = count

    metrics_cfg = cfg.setdefault("metrics", {})
    energy_cfg = metrics_cfg.setdefault("energy", {})
    energy_cfg.update(
        {
            "enabled": True,
            "min_duration_s": 1e-9,
            "max_repeats": count,
        }
    )

    output_cfg = cfg.setdefault("output", {})
    base_name = str(output_cfg.get("run_name") or "energy_plausibility")
    output_cfg["run_name"] = f"{base_name}_n{count}"

    return cfg


def write_plausibility_configs(
    base_cfg: Mapping[str, Any],
    output_dir: Path,
    counts: Iterable[int] = DEFAULT_COUNTS,
) -> List[Path]:
    """Write controlled Phase 11 YAML configurations and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for raw_count in counts:
        count = _require_positive_int(raw_count, "inference_count")
        cfg = prepare_plausibility_config(base_cfg, count)
        path = output_dir / f"energy_plausibility_n{count}.yaml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        paths.append(path)

    return paths


def load_benchmark_result(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark result must be a JSON object: {path}")
    return payload


def build_comparability_fingerprint(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fields that must remain identical across plausibility runs."""

    meta = payload.get("meta") or {}
    model = payload.get("model") or {}
    metrics = payload.get("metrics") or {}
    energy = metrics.get("energy") or {}
    timing = metrics.get("inference_time_ms") or {}
    thread_config = payload.get("thread_config") or {}
    context = payload.get("experiment_context") or {}
    power = context.get("power") or {}
    manual = context.get("manual") or {}

    return {
        "framework": meta.get("framework"),
        "device_target": meta.get("device_target"),
        "model_path": model.get("path"),
        "input_shape": model.get("input_shape"),
        "dtype": model.get("dtype"),
        "batch_size": timing.get("batch_size"),
        "requested_threads": thread_config.get("requested_threads"),
        "applied_intra_op_threads": thread_config.get("applied_intra_op_threads"),
        "active_provider": thread_config.get("active_provider"),
        "energy_backend": energy.get("backend"),
        "primary_domain": energy.get("primary_domain"),
        "power_scheme_guid": power.get("active_scheme_guid"),
        "power_scheme_name": power.get("active_scheme_name"),
        "ac_connected": power.get("ac_connected"),
        "battery_saver": power.get("battery_saver"),
        "dell_thermal_mode": manual.get("dell_thermal_mode"),
        "cooling_setup": manual.get("cooling_setup"),
    }


def extract_energy_point(payload: Mapping[str, Any], source_file: Path) -> EnergyPlausibilityPoint:
    metrics = payload.get("metrics") or {}
    energy = metrics.get("energy")
    if not isinstance(energy, Mapping):
        raise ValueError(f"Missing metrics.energy block: {source_file}")
    if energy.get("available") is not True:
        reason = energy.get("reason") or "unknown reason"
        raise ValueError(f"Energy measurement unavailable in {source_file}: {reason}")

    count = _require_positive_int(energy.get("inference_count"), "inference_count")
    total = float(energy.get("total_energy_j"))
    duration = float(energy.get("duration_s"))
    average_power = float(energy.get("average_power_w"))

    if total < 0.0:
        raise ValueError(f"total_energy_j must be >= 0 in {source_file}")
    if duration <= 0.0:
        raise ValueError(f"duration_s must be > 0 in {source_file}")
    if average_power < 0.0:
        raise ValueError(f"average_power_w must be >= 0 in {source_file}")

    reported_per_inference = energy.get("energy_per_inference_j")
    calculated_per_inference = total / count
    if reported_per_inference is None:
        per_inference = calculated_per_inference
    else:
        per_inference = float(reported_per_inference)
        tolerance = max(1e-12, abs(calculated_per_inference) * 1e-9)
        if abs(per_inference - calculated_per_inference) > tolerance:
            raise ValueError(
                "energy_per_inference_j is inconsistent with total_energy_j / "
                f"inference_count in {source_file}"
            )

    return EnergyPlausibilityPoint(
        inference_count=count,
        total_energy_j=total,
        energy_per_inference_j=per_inference,
        duration_s=duration,
        average_power_w=average_power,
        source_file=str(source_file),
    )


def validate_energy_result_context(
    payload: Mapping[str, Any],
    source_file: Path,
) -> None:
    """Validate platform/domain and experiment-context invariants.

    This helper is shared by Phase 11 plausibility and Phase 12 repeatability
    analysis so both phases apply the same comparability rules.
    """

    metrics = payload.get("metrics") or {}
    energy = metrics.get("energy") or {}
    if energy.get("backend") == "windows_emi" and not energy.get("primary_domain"):
        raise ValueError(
            f"Windows EMI result is missing primary_domain in {source_file}"
        )

    context = payload.get("experiment_context") or {}
    finalization = context.get("finalization") or {}
    if finalization.get("power_state_consistent") is False:
        warnings = finalization.get("warnings") or []
        detail = "; ".join(str(item) for item in warnings) or "power state changed"
        raise ValueError(
            f"Energy result has inconsistent power context in {source_file}: {detail}"
        )


# Backwards-compatible private aliases for any external/internal callers that may
# have used the earlier helper names. New code should use the public functions.
_load_json = load_benchmark_result
_result_fingerprint = build_comparability_fingerprint
_extract_point = extract_energy_point


def analyze_plausibility_results(
    result_files: Sequence[Path],
    max_relative_spread_pct: Optional[float] = None,
) -> EnergyPlausibilityReport:
    """Analyse real benchmark results for Phase 11 physical plausibility.

    The requirements define two expectations:
    - total energy increases with inference count;
    - energy per inference is approximately stable.

    The first is evaluated strictly. The second is reported as relative spread
    ``(max - min) / mean * 100``. No universal tolerance is hidden in the code:
    callers may supply ``max_relative_spread_pct`` for an automated decision,
    otherwise the normalized stability remains a documented manual review item.
    """

    if len(result_files) < 2:
        raise ValueError("At least two benchmark result files are required")

    if max_relative_spread_pct is not None:
        max_relative_spread_pct = float(max_relative_spread_pct)
        if max_relative_spread_pct < 0.0:
            raise ValueError("max_relative_spread_pct must be >= 0")

    payloads = [load_benchmark_result(Path(path)) for path in result_files]

    for path, payload in zip(result_files, payloads):
        validate_energy_result_context(payload, Path(path))

    fingerprints = [build_comparability_fingerprint(payload) for payload in payloads]
    reference = fingerprints[0]
    if any(fingerprint != reference for fingerprint in fingerprints[1:]):
        raise ValueError(
            "Phase 11 results are not comparable: model/device/backend fingerprint differs"
        )

    points = sorted(
        (
            extract_energy_point(payload, Path(path))
            for payload, path in zip(payloads, result_files)
        ),
        key=lambda point: point.inference_count,
    )

    counts = [point.inference_count for point in points]
    if len(set(counts)) != len(counts):
        raise ValueError("Inference counts must be unique across Phase 11 result files")

    total_energy_monotonic = all(
        right.total_energy_j > left.total_energy_j
        for left, right in zip(points, points[1:])
    )

    normalized = [point.energy_per_inference_j for point in points]
    normalized_mean = mean(normalized)
    if normalized_mean == 0.0:
        normalized_relative_spread_pct = 0.0 if max(normalized) == 0.0 else float("inf")
    else:
        normalized_relative_spread_pct = (
            (max(normalized) - min(normalized)) / normalized_mean * 100.0
        )

    normalized_stability_passed: Optional[bool]
    if max_relative_spread_pct is None:
        normalized_stability_passed = None
        overall_status = "manual_review_required" if total_energy_monotonic else "failed"
    else:
        normalized_stability_passed = (
            normalized_relative_spread_pct <= max_relative_spread_pct
        )
        overall_status = (
            "passed"
            if total_energy_monotonic and normalized_stability_passed
            else "failed"
        )

    return EnergyPlausibilityReport(
        comparable=True,
        fingerprint=reference,
        points=[asdict(point) for point in points],
        total_energy_monotonic=total_energy_monotonic,
        normalized_relative_spread_pct=normalized_relative_spread_pct,
        max_relative_spread_pct=max_relative_spread_pct,
        normalized_stability_passed=normalized_stability_passed,
        overall_status=overall_status,
    )


def write_report(report: EnergyPlausibilityReport, output_path: Path) -> Path:
    """Persist a Phase 11 plausibility report as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _parse_counts(values: Sequence[str]) -> Tuple[int, ...]:
    return tuple(_require_positive_int(value, "inference_count") for value in values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or analyse Phase 11 energy plausibility measurements"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create controlled YAML configs for real hardware measurements",
    )
    prepare.add_argument("config", type=Path, help="Base benchmark YAML configuration")
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase11_configs"),
        help="Directory for generated YAML files",
    )
    prepare.add_argument(
        "--counts",
        nargs="+",
        default=[str(value) for value in DEFAULT_COUNTS],
        help="Energy inference counts, default: 100 500 1000",
    )

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyse real bench.json files from the same model/device/backend",
    )
    analyze.add_argument("results", nargs="+", type=Path, help="bench.json files")
    analyze.add_argument(
        "--output",
        type=Path,
        default=Path("phase11_energy_plausibility_report.json"),
        help="Output JSON report",
    )
    analyze.add_argument(
        "--max-relative-spread-pct",
        type=float,
        default=None,
        help=(
            "Optional acceptance tolerance for energy-per-inference spread. "
            "If omitted, normalized stability requires manual review."
        ),
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "prepare":
        base_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(base_cfg, dict):
            raise ValueError("Base configuration must be a YAML mapping")
        paths = write_plausibility_configs(
            base_cfg,
            output_dir=args.output_dir,
            counts=_parse_counts(args.counts),
        )
        for path in paths:
            print(path)
        return 0

    if args.command == "analyze":
        report = analyze_plausibility_results(
            args.results,
            max_relative_spread_pct=args.max_relative_spread_pct,
        )
        write_report(report, args.output)
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        return 0 if report.overall_status != "failed" else 1

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 13 validation for optional idle-baseline correction.

The tool analyses persisted benchmark results after baseline correction was
explicitly enabled. Gross energy remains the primary measurement; corrected
values are evaluated only as an additional diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from bench.tools.energy_plausibility import (
    build_comparability_fingerprint,
    load_benchmark_result,
    validate_energy_result_context,
)


DEFAULT_COUNTS: Tuple[int, ...] = (100, 500, 1000)
RUN_NAME_PREFIX = "energy_baseline_phase13_n"


@dataclass(frozen=True)
class BaselineValidationPoint:
    inference_count: int
    gross_energy_j: float
    baseline_power_mean_w: float
    baseline_power_cv_pct: Optional[float]
    estimated_idle_energy_j: float
    baseline_fraction_of_gross_pct: Optional[float]
    baseline_adjusted_energy_j: float
    baseline_adjusted_energy_per_inference_j: float
    correction_valid: bool
    source_file: str


@dataclass(frozen=True)
class BaselineValidationReport:
    comparable: bool
    fingerprint: Dict[str, Any]
    points: List[Dict[str, Any]]
    gross_total_energy_monotonic: bool
    adjusted_total_energy_monotonic: bool
    adjusted_energy_per_inference_relative_spread_pct: float
    baseline_power_relative_spread_pct: float
    overall_status: str
    notes: List[str]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1")
    parsed = int(value)
    if float(value) != float(parsed) or parsed < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return parsed


def discover_baseline_results(results_root: Path) -> List[Path]:
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"Phase 13 result root does not exist: {root}")

    matches: List[Path] = []
    for path in sorted(root.rglob("bench.json")):
        try:
            payload = load_benchmark_result(path)
        except Exception:
            continue
        output = (payload.get("config") or {}).get("output") or {}
        run_name = str(output.get("run_name") or "")
        if run_name.startswith(RUN_NAME_PREFIX):
            matches.append(path)
    return matches


def _extract_point(payload: Mapping[str, Any], source: Path) -> BaselineValidationPoint:
    metrics = payload.get("metrics") or {}
    energy = metrics.get("energy") or {}
    if energy.get("available") is not True:
        raise ValueError(f"Gross energy unavailable in {source}")

    baseline = energy.get("baseline_correction")
    if not isinstance(baseline, Mapping):
        raise ValueError(f"Missing metrics.energy.baseline_correction in {source}")
    if baseline.get("enabled") is not True or baseline.get("available") is not True:
        raise ValueError(f"Baseline correction unavailable in {source}")
    if baseline.get("method") != "post_energy_idle":
        raise ValueError(f"Unexpected Phase 13 baseline method in {source}")
    if baseline.get("correction_valid") is not True:
        raise ValueError(f"Baseline correction is marked invalid in {source}")

    backend = energy.get("backend")
    domain = energy.get("primary_domain")
    if baseline.get("backend") != backend:
        raise ValueError(f"Baseline backend differs from gross backend in {source}")
    if baseline.get("primary_domain") != domain:
        raise ValueError(f"Baseline domain differs from gross domain in {source}")

    gross = float(energy.get("total_energy_j"))
    gross_alias = float(energy.get("gross_energy_j"))
    tolerance = max(1e-12, abs(gross) * 1e-9)
    if abs(gross - gross_alias) > tolerance:
        raise ValueError(f"gross_energy_j differs from total_energy_j in {source}")

    count = _positive_int(energy.get("inference_count"), "inference_count")
    adjusted = float(energy.get("baseline_adjusted_energy_j"))
    adjusted_per_inference = float(
        energy.get("baseline_adjusted_energy_per_inference_j")
    )
    expected_per_inference = adjusted / count
    tolerance = max(1e-12, abs(expected_per_inference) * 1e-9)
    if abs(adjusted_per_inference - expected_per_inference) > tolerance:
        raise ValueError(
            f"baseline_adjusted_energy_per_inference_j is inconsistent in {source}"
        )

    power = baseline.get("baseline_power_w") or {}
    power_mean = float(power.get("mean"))
    power_cv = power.get("cv_pct")
    power_cv = None if power_cv is None else float(power_cv)

    return BaselineValidationPoint(
        inference_count=count,
        gross_energy_j=gross,
        baseline_power_mean_w=power_mean,
        baseline_power_cv_pct=power_cv,
        estimated_idle_energy_j=float(baseline.get("estimated_idle_energy_j")),
        baseline_fraction_of_gross_pct=(
            None
            if baseline.get("baseline_fraction_of_gross_pct") is None
            else float(baseline.get("baseline_fraction_of_gross_pct"))
        ),
        baseline_adjusted_energy_j=adjusted,
        baseline_adjusted_energy_per_inference_j=adjusted_per_inference,
        correction_valid=True,
        source_file=str(source),
    )


def _relative_spread(values: Sequence[float]) -> float:
    parsed = [float(value) for value in values]
    mu = mean(parsed)
    if mu == 0.0:
        return 0.0 if max(parsed) == 0.0 else float("inf")
    return (max(parsed) - min(parsed)) / abs(mu) * 100.0


def analyze_baseline_results(
    result_files: Sequence[Path],
    *,
    expected_counts: Sequence[int] = DEFAULT_COUNTS,
) -> BaselineValidationReport:
    counts = tuple(_positive_int(value, "expected_count") for value in expected_counts)
    if len(result_files) != len(counts):
        raise ValueError(
            f"Expected exactly {len(counts)} Phase 13 result files, got {len(result_files)}"
        )

    paths = [Path(path) for path in result_files]
    payloads = [load_benchmark_result(path) for path in paths]
    for path, payload in zip(paths, payloads):
        validate_energy_result_context(payload, path)

    fingerprints = [build_comparability_fingerprint(payload) for payload in payloads]
    reference = fingerprints[0]
    if any(fingerprint != reference for fingerprint in fingerprints[1:]):
        raise ValueError("Phase 13 results are not comparable")

    points = sorted(
        (_extract_point(payload, path) for payload, path in zip(payloads, paths)),
        key=lambda point: point.inference_count,
    )
    actual_counts = tuple(point.inference_count for point in points)
    if actual_counts != tuple(sorted(counts)):
        raise ValueError(
            f"Phase 13 inference counts must be {tuple(sorted(counts))}, got {actual_counts}"
        )

    gross_monotonic = all(
        right.gross_energy_j > left.gross_energy_j
        for left, right in zip(points, points[1:])
    )
    adjusted_monotonic = all(
        right.baseline_adjusted_energy_j > left.baseline_adjusted_energy_j
        for left, right in zip(points, points[1:])
    )

    adjusted_spread = _relative_spread(
        [point.baseline_adjusted_energy_per_inference_j for point in points]
    )
    baseline_power_spread = _relative_spread(
        [point.baseline_power_mean_w for point in points]
    )

    status = (
        "complete_manual_review_required"
        if gross_monotonic and adjusted_monotonic
        else "failed"
    )
    notes = [
        "Gross energy remains the primary measurement.",
        "No universal acceptance threshold is applied to baseline CV or adjusted-energy spread.",
        "Baseline correction is valid only for the same backend/domain and comparable system context.",
    ]

    return BaselineValidationReport(
        comparable=True,
        fingerprint=reference,
        points=[asdict(point) for point in points],
        gross_total_energy_monotonic=gross_monotonic,
        adjusted_total_energy_monotonic=adjusted_monotonic,
        adjusted_energy_per_inference_relative_spread_pct=adjusted_spread,
        baseline_power_relative_spread_pct=baseline_power_spread,
        overall_status=status,
        notes=notes,
    )


def write_report(report: BaselineValidationReport, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 13 baseline validation")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyse Phase 13 benchmark results")
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    analyze.add_argument(
        "--output",
        default="phase13_baseline_validation_report.json",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "analyze":
        files = discover_baseline_results(Path(args.results_root))
        report = analyze_baseline_results(files, expected_counts=args.counts)
        write_report(report, Path(args.output))
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        return 0
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

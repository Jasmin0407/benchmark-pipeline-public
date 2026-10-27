"""Phase 12 repeatability preparation and analysis for energy benchmarks.

Phase 12 quantifies measurement stability after the Phase 11 hardware plausibility
check. The module deliberately does not modify benchmark-core measurement logic.
It prepares a counterbalanced run schedule and analyses persisted ``bench.json``
files from the real target hardware.

For three workload conditions and six repetitions, all six possible condition
orders are used exactly once. This gives perfect first-order position balance:
each workload appears twice in position 1, twice in position 2, and twice in
position 3.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from bench.tools.energy_plausibility import (
    build_comparability_fingerprint,
    extract_energy_point,
    load_benchmark_result,
    validate_energy_result_context,
)


DEFAULT_COUNTS: Tuple[int, ...] = (100, 500, 1000)
DEFAULT_REPETITIONS = 6
DEFAULT_SEED = 20261027
DEFAULT_SCHEDULE_ID = "phase12_emi_20261027"
RUN_NAME_PREFIX = "energy_repeatability"

_RUN_NAME_RE = re.compile(
    r"^energy_repeatability_(?P<schedule>.+)_r(?P<round>\d+)_p(?P<position>\d+)_n(?P<count>\d+)$"
)


@dataclass(frozen=True)
class MetricSummary:
    """Descriptive statistics for one repeated metric."""

    n: int
    mean: float
    sample_stddev: float
    cv_pct: Optional[float]
    median: float
    minimum: float
    maximum: float
    relative_range_pct: Optional[float]


@dataclass(frozen=True)
class RepeatabilityObservation:
    """One Phase 12 benchmark observation."""

    round_index: int
    position: int
    inference_count: int
    total_energy_j: float
    energy_per_inference_j: float
    duration_s: float
    average_power_w: float
    background_cpu_mean_pct: Optional[float]
    background_cpu_max_pct: Optional[float]
    preflight_passed: Optional[bool]
    run_name: str
    source_file: str


@dataclass(frozen=True)
class RepeatabilityConditionSummary:
    """Aggregate repeatability statistics for one inference-count condition."""

    inference_count: int
    run_count: int
    total_energy_j: Dict[str, Any]
    energy_per_inference_j: Dict[str, Any]
    duration_s: Dict[str, Any]
    average_power_w: Dict[str, Any]
    background_cpu_mean_pct: Optional[Dict[str, Any]]
    background_cpu_max_pct: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class RepeatabilityReport:
    """Machine-readable Phase 12 repeatability report."""

    comparable: bool
    schedule_id: str
    expected_repetitions: int
    expected_counts: List[int]
    fingerprint: Dict[str, Any]
    observations: List[Dict[str, Any]]
    condition_summaries: List[Dict[str, Any]]
    position_balance: Dict[str, Dict[str, int]]
    max_cv_pct: Optional[float]
    cv_acceptance_passed: Optional[bool]
    overall_status: str
    notes: List[str]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1")
    parsed = int(value)
    if float(value) != float(parsed) or parsed < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return parsed


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark configuration must be a YAML mapping: {path}")
    return payload


def _config_inference_count(cfg: Mapping[str, Any], source: Path) -> int:
    run = cfg.get("run") or {}
    metrics = cfg.get("metrics") or {}
    energy = metrics.get("energy") or {}

    count = _positive_int(run.get("repeats"), f"run.repeats in {source}")
    max_repeats = _positive_int(
        energy.get("max_repeats"), f"metrics.energy.max_repeats in {source}"
    )
    if count != max_repeats:
        raise ValueError(
            f"Phase 12 base configuration must pin run.repeats and "
            f"metrics.energy.max_repeats to the same value in {source}"
        )
    if energy.get("enabled") is not True:
        raise ValueError(f"Energy must be enabled in Phase 12 base configuration: {source}")
    return count


def build_counterbalanced_schedule(
    counts: Sequence[int] = DEFAULT_COUNTS,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> List[Tuple[int, ...]]:
    """Return a deterministic, counterbalanced condition order per round.

    With three conditions and six repetitions this uses every permutation once,
    which gives exact position balance. For other repetition counts, shuffled
    permutations are cycled deterministically. Exact balance may then be
    impossible; the generated manifest records the actual position counts.
    """

    parsed_counts = tuple(_positive_int(value, "inference_count") for value in counts)
    if len(parsed_counts) < 2:
        raise ValueError("At least two workload conditions are required")
    if len(set(parsed_counts)) != len(parsed_counts):
        raise ValueError("Workload conditions must be unique")

    repetitions = _positive_int(repetitions, "repetitions")
    permutations = list(itertools.permutations(parsed_counts))
    rng = random.Random(int(seed))
    rng.shuffle(permutations)

    schedule: List[Tuple[int, ...]] = []
    while len(schedule) < repetitions:
        for permutation in permutations:
            schedule.append(permutation)
            if len(schedule) == repetitions:
                break
    return schedule


def _position_balance(schedule: Sequence[Sequence[int]]) -> Dict[str, Dict[str, int]]:
    balance: Dict[str, Dict[str, int]] = {}
    for order in schedule:
        for position, count in enumerate(order, start=1):
            count_key = str(int(count))
            position_key = str(position)
            per_count = balance.setdefault(count_key, {})
            per_count[position_key] = per_count.get(position_key, 0) + 1
    return balance


def _append_phase12_note(
    existing: Optional[str],
    *,
    schedule_id: str,
    round_index: int,
    position: int,
    count: int,
    seed: int,
) -> str:
    tag = (
        f"Phase 12 repeatability; schedule={schedule_id}; round={round_index}; "
        f"position={position}; inference_count={count}; seed={seed}"
    )
    if existing:
        return f"{existing} | {tag}"
    return tag


def prepare_repeatability_experiment(
    base_config_paths: Sequence[Path],
    output_dir: Path,
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    schedule_id: str = DEFAULT_SCHEDULE_ID,
    results_dir: str = "runs/phase12_windows_emi",
) -> Dict[str, Any]:
    """Create Phase 12 configs, a balanced schedule manifest and PowerShell runner."""

    if not schedule_id or not str(schedule_id).strip():
        raise ValueError("schedule_id must not be empty")
    if not base_config_paths:
        raise ValueError("At least one base configuration is required")

    configs_by_count: Dict[int, Dict[str, Any]] = {}
    for raw_path in base_config_paths:
        path = Path(raw_path)
        cfg = _load_yaml(path)
        count = _config_inference_count(cfg, path)
        if count in configs_by_count:
            raise ValueError(f"Duplicate Phase 12 base condition for inference_count={count}")
        configs_by_count[count] = cfg

    counts = tuple(sorted(configs_by_count))
    schedule = build_counterbalanced_schedule(counts, repetitions=repetitions, seed=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []
    config_paths: List[Path] = []

    for round_index, order in enumerate(schedule, start=1):
        for position, count in enumerate(order, start=1):
            cfg = deepcopy(configs_by_count[count])
            output_cfg = cfg.setdefault("output", {})
            output_cfg["base_dir"] = str(results_dir)
            output_cfg.pop("dir", None)
            run_name = (
                f"{RUN_NAME_PREFIX}_{schedule_id}_r{round_index:02d}_"
                f"p{position:02d}_n{count}"
            )
            output_cfg["run_name"] = run_name

            context_cfg = cfg.setdefault("experiment", {}).setdefault("context", {})
            manual_cfg = context_cfg.setdefault("manual", {})
            manual_cfg["notes"] = _append_phase12_note(
                manual_cfg.get("notes"),
                schedule_id=schedule_id,
                round_index=round_index,
                position=position,
                count=count,
                seed=seed,
            )

            filename = f"r{round_index:02d}_p{position:02d}_n{count}.yaml"
            config_path = output_dir / filename
            config_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            config_paths.append(config_path)
            manifest_entries.append(
                {
                    "round": round_index,
                    "position": position,
                    "inference_count": count,
                    "run_name": run_name,
                    "config": filename,
                }
            )

    manifest = {
        "phase": 12,
        "purpose": "energy_repeatability",
        "schedule_id": str(schedule_id),
        "seed": int(seed),
        "repetitions": int(repetitions),
        "counts": list(counts),
        "total_runs": len(manifest_entries),
        "position_balance": _position_balance(schedule),
        "results_dir": str(results_dir),
        "runs": manifest_entries,
    }
    manifest_path = output_dir / "phase12_schedule.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ps_lines = [
        "$ErrorActionPreference = \"Stop\"",
        "",
        "# Phase 12 counterbalanced repeatability run order.",
        "# Run this script from the repository root in the benchmark environment.",
        "",
    ]
    for entry in manifest_entries:
        relative = (output_dir / entry["config"]).as_posix()
        ps_lines.append(
            f'Write-Host "Round {entry["round"]}, position {entry["position"]}, '
            f'N={entry["inference_count"]}"'
        )
        ps_lines.append(f"python -m bench.cli.bench \"{relative}\"")
        ps_lines.append("if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }")
        ps_lines.append("")

    runner_path = output_dir / "run_phase12.ps1"
    runner_path.write_text("\n".join(ps_lines), encoding="utf-8-sig")

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "runner_path": str(runner_path),
        "config_paths": [str(path) for path in config_paths],
    }


def _metric_summary(values: Sequence[float]) -> MetricSummary:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence")
    parsed = [float(value) for value in values]
    n = len(parsed)
    mu = mean(parsed)
    sigma = stdev(parsed) if n >= 2 else 0.0
    cv = None if mu == 0.0 else abs(sigma / mu) * 100.0
    minimum = min(parsed)
    maximum = max(parsed)
    relative_range = None if mu == 0.0 else (maximum - minimum) / abs(mu) * 100.0
    return MetricSummary(
        n=n,
        mean=mu,
        sample_stddev=sigma,
        cv_pct=cv,
        median=median(parsed),
        minimum=minimum,
        maximum=maximum,
        relative_range_pct=relative_range,
    )


def _optional_metric_summary(values: Sequence[Optional[float]]) -> Optional[MetricSummary]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return _metric_summary(present)


def _parse_run_identity(run_name: str) -> Tuple[str, int, int, int]:
    match = _RUN_NAME_RE.match(str(run_name))
    if not match:
        raise ValueError(
            "Phase 12 run_name does not follow the generated schedule format: "
            f"{run_name!r}"
        )
    return (
        match.group("schedule"),
        int(match.group("round")),
        int(match.group("position")),
        int(match.group("count")),
    )


def _extract_observation(payload: Mapping[str, Any], source_file: Path) -> RepeatabilityObservation:
    point = extract_energy_point(payload, source_file)
    config = payload.get("config") or {}
    output = config.get("output") or {}
    run_name = str(output.get("run_name") or "")
    _, round_index, position, count_from_name = _parse_run_identity(run_name)
    if count_from_name != point.inference_count:
        raise ValueError(
            f"Run name count n{count_from_name} does not match metrics.energy.inference_count "
            f"{point.inference_count} in {source_file}"
        )

    context = payload.get("experiment_context") or {}
    background = context.get("background") or {}
    preflight = context.get("preflight") or {}

    def optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    return RepeatabilityObservation(
        round_index=round_index,
        position=position,
        inference_count=point.inference_count,
        total_energy_j=point.total_energy_j,
        energy_per_inference_j=point.energy_per_inference_j,
        duration_s=point.duration_s,
        average_power_w=point.average_power_w,
        background_cpu_mean_pct=optional_float(background.get("cpu_mean_pct")),
        background_cpu_max_pct=optional_float(background.get("cpu_max_pct")),
        preflight_passed=(
            bool(preflight.get("passed")) if preflight.get("passed") is not None else None
        ),
        run_name=run_name,
        source_file=str(source_file),
    )


def discover_repeatability_results(
    results_root: Path,
    *,
    schedule_id: str,
) -> List[Path]:
    """Find Phase 12 bench.json files for one schedule under a result root."""

    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"Phase 12 result root does not exist: {root}")

    prefix = f"{RUN_NAME_PREFIX}_{schedule_id}_"
    matches: List[Path] = []
    for path in sorted(root.rglob("bench.json")):
        try:
            payload = load_benchmark_result(path)
        except Exception:
            continue
        config = payload.get("config") or {}
        output = config.get("output") or {}
        run_name = str(output.get("run_name") or "")
        if run_name.startswith(prefix):
            matches.append(path)
    return matches


def analyze_repeatability_results(
    result_files: Sequence[Path],
    *,
    expected_repetitions: int = DEFAULT_REPETITIONS,
    expected_counts: Sequence[int] = DEFAULT_COUNTS,
    max_cv_pct: Optional[float] = None,
) -> RepeatabilityReport:
    """Analyse repeated hardware measurements from one Phase 12 schedule.

    The acceptance criterion is intentionally not hidden in the implementation.
    Without ``max_cv_pct`` the report is complete but remains a manual scientific
    review item. If a project-specific CV threshold is supplied, it is applied to
    ``energy_per_inference_j`` for every workload condition.
    """

    expected_repetitions = _positive_int(expected_repetitions, "expected_repetitions")
    counts = tuple(_positive_int(value, "expected_count") for value in expected_counts)
    if len(set(counts)) != len(counts):
        raise ValueError("expected_counts must be unique")
    if max_cv_pct is not None and float(max_cv_pct) < 0.0:
        raise ValueError("max_cv_pct must be >= 0")
    max_cv_pct = None if max_cv_pct is None else float(max_cv_pct)

    expected_total = expected_repetitions * len(counts)
    if len(result_files) != expected_total:
        raise ValueError(
            f"Expected exactly {expected_total} Phase 12 result files "
            f"({expected_repetitions} repetitions x {len(counts)} conditions), "
            f"got {len(result_files)}"
        )

    paths = [Path(path) for path in result_files]
    payloads = [load_benchmark_result(path) for path in paths]
    for path, payload in zip(paths, payloads):
        validate_energy_result_context(payload, path)

    fingerprints = [build_comparability_fingerprint(payload) for payload in payloads]
    reference = fingerprints[0]
    if any(fingerprint != reference for fingerprint in fingerprints[1:]):
        raise ValueError(
            "Phase 12 results are not comparable: model/device/backend/context fingerprint differs"
        )

    observations = [
        _extract_observation(payload, path) for payload, path in zip(payloads, paths)
    ]

    schedule_ids = {_parse_run_identity(obs.run_name)[0] for obs in observations}
    if len(schedule_ids) != 1:
        raise ValueError("Phase 12 results contain more than one schedule_id")
    schedule_id = next(iter(schedule_ids))

    identities = [(obs.round_index, obs.position) for obs in observations]
    if len(set(identities)) != len(identities):
        raise ValueError("Phase 12 results contain duplicate round/position entries")

    by_round: Dict[int, List[RepeatabilityObservation]] = {}
    for obs in observations:
        by_round.setdefault(obs.round_index, []).append(obs)

    expected_rounds = set(range(1, expected_repetitions + 1))
    if set(by_round) != expected_rounds:
        raise ValueError(
            f"Phase 12 rounds must be exactly {sorted(expected_rounds)}, "
            f"got {sorted(by_round)}"
        )

    schedule_orders: List[Tuple[int, ...]] = []
    for round_index in sorted(by_round):
        round_obs = sorted(by_round[round_index], key=lambda obs: obs.position)
        if [obs.position for obs in round_obs] != list(range(1, len(counts) + 1)):
            raise ValueError(f"Round {round_index} does not contain all expected positions")
        order = tuple(obs.inference_count for obs in round_obs)
        if set(order) != set(counts) or len(order) != len(counts):
            raise ValueError(
                f"Round {round_index} must contain each expected inference count exactly once"
            )
        schedule_orders.append(order)

    condition_summaries: List[RepeatabilityConditionSummary] = []
    all_cv_pass = True
    notes: List[str] = []

    for count in sorted(counts):
        group = [obs for obs in observations if obs.inference_count == count]
        if len(group) != expected_repetitions:
            raise ValueError(
                f"Expected {expected_repetitions} results for inference_count={count}, "
                f"got {len(group)}"
            )
        if any(obs.preflight_passed is False for obs in group):
            raise ValueError(f"At least one Phase 12 preflight failed for inference_count={count}")

        total_summary = _metric_summary([obs.total_energy_j for obs in group])
        normalized_summary = _metric_summary([obs.energy_per_inference_j for obs in group])
        duration_summary = _metric_summary([obs.duration_s for obs in group])
        power_summary = _metric_summary([obs.average_power_w for obs in group])
        bg_mean = _optional_metric_summary([obs.background_cpu_mean_pct for obs in group])
        bg_max = _optional_metric_summary([obs.background_cpu_max_pct for obs in group])

        if max_cv_pct is not None:
            cv = normalized_summary.cv_pct
            passed = cv is not None and cv <= max_cv_pct
            all_cv_pass = all_cv_pass and passed

        condition_summaries.append(
            RepeatabilityConditionSummary(
                inference_count=count,
                run_count=len(group),
                total_energy_j=asdict(total_summary),
                energy_per_inference_j=asdict(normalized_summary),
                duration_s=asdict(duration_summary),
                average_power_w=asdict(power_summary),
                background_cpu_mean_pct=None if bg_mean is None else asdict(bg_mean),
                background_cpu_max_pct=None if bg_max is None else asdict(bg_max),
            )
        )

    position_balance = _position_balance(schedule_orders)
    if len(counts) == 3 and expected_repetitions == 6:
        ideal = {"1": 2, "2": 2, "3": 2}
        if any(position_balance.get(str(count)) != ideal for count in counts):
            raise ValueError("Six-round Phase 12 schedule is not perfectly position-balanced")
        notes.append(
            "All six permutations were represented; each workload appeared twice in each position."
        )

    # Windows EMI should remain pinned to the validated package domain on the Dell.
    if reference.get("energy_backend") == "windows_emi":
        if reference.get("primary_domain") != "RAPL_Package0_PKG":
            raise ValueError(
                "Dell Windows Phase 12 requires primary_domain=RAPL_Package0_PKG"
            )

    cv_acceptance_passed: Optional[bool]
    if max_cv_pct is None:
        cv_acceptance_passed = None
        overall_status = "complete_manual_review_required"
        notes.append(
            "No universal CV threshold was applied; interpret the measured CVs in project context."
        )
    else:
        cv_acceptance_passed = all_cv_pass
        overall_status = "passed" if all_cv_pass else "failed"

    observations_sorted = sorted(
        observations, key=lambda obs: (obs.round_index, obs.position)
    )

    return RepeatabilityReport(
        comparable=True,
        schedule_id=schedule_id,
        expected_repetitions=expected_repetitions,
        expected_counts=list(counts),
        fingerprint=reference,
        observations=[asdict(obs) for obs in observations_sorted],
        condition_summaries=[asdict(summary) for summary in condition_summaries],
        position_balance=position_balance,
        max_cv_pct=max_cv_pct,
        cv_acceptance_passed=cv_acceptance_passed,
        overall_status=overall_status,
        notes=notes,
    )


def write_repeatability_report(report: RepeatabilityReport, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _parse_counts(values: Sequence[str]) -> Tuple[int, ...]:
    return tuple(_positive_int(value, "inference_count") for value in values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or analyse Phase 12 energy repeatability measurements"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create counterbalanced Phase 12 configs and a PowerShell runner",
    )
    prepare.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Base YAML configs, one per workload condition",
    )
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase12_configs"),
    )
    prepare.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--schedule-id", default=DEFAULT_SCHEDULE_ID)
    prepare.add_argument("--results-dir", default="runs/phase12_windows_emi")

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyse repeated Phase 12 bench.json files",
    )
    analyze_source = analyze.add_mutually_exclusive_group(required=True)
    analyze_source.add_argument(
        "--results",
        nargs="+",
        type=Path,
        help="Explicit bench.json files",
    )
    analyze_source.add_argument(
        "--results-root",
        type=Path,
        help="Root directory to scan recursively for one schedule",
    )
    analyze.add_argument("--schedule-id", default=DEFAULT_SCHEDULE_ID)
    analyze.add_argument("--expected-repetitions", type=int, default=DEFAULT_REPETITIONS)
    analyze.add_argument(
        "--counts",
        nargs="+",
        default=[str(value) for value in DEFAULT_COUNTS],
    )
    analyze.add_argument(
        "--max-cv-pct",
        type=float,
        default=None,
        help=(
            "Optional project-specific acceptance threshold applied to the sample CV "
            "of energy_per_inference_j for every workload."
        ),
    )
    analyze.add_argument(
        "--output",
        type=Path,
        default=Path("phase12_repeatability_report.json"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "prepare":
        result = prepare_repeatability_experiment(
            args.configs,
            args.output_dir,
            repetitions=args.repetitions,
            seed=args.seed,
            schedule_id=args.schedule_id,
            results_dir=args.results_dir,
        )
        print(json.dumps(result["manifest"], indent=2, ensure_ascii=False))
        print(f"Schedule: {result['manifest_path']}")
        print(f"Runner:   {result['runner_path']}")
        return 0

    if args.command == "analyze":
        if args.results is not None:
            result_files = args.results
        else:
            result_files = discover_repeatability_results(
                args.results_root,
                schedule_id=args.schedule_id,
            )

        report = analyze_repeatability_results(
            result_files,
            expected_repetitions=args.expected_repetitions,
            expected_counts=_parse_counts(args.counts),
            max_cv_pct=args.max_cv_pct,
        )
        write_repeatability_report(report, args.output)
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        return 0 if report.overall_status != "failed" else 1

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

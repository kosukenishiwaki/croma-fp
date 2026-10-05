#!/usr/bin/env python3
"""Analyze bucketstats output from tracer FP runs.

This script reads per-rank bucket statistics written by
`file_output_mode=bucketstats`:

- bucketstats_rankNNN.tsv

It can also join optional companion files from the same or another run:

- load_balance_rank_totals.txt
- timing_coreNNN.tsv

The main goal is to answer questions such as:

- Are some ranks split into many more buckets than others?
- Is residual cost better explained by `count`, `bucket calls`, or
  `bucket_size * nsub`?
- Do small buckets appear frequently enough to matter?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from analyze_tracer_load_balance import read_load_balance_rows, read_timing_rows


BUCKET_RE = re.compile(r"^bucketstats_rank\d+\.tsv$")


@dataclass
class BucketRow:
    rank: int
    snapshot_index: int
    physical_snapshot: int
    z: float
    bucket_index: int
    nbuckets: int
    bucket_size: int
    nsub: int
    n_on: int
    base_nsub: int
    fp_cadence: int
    coeff_segments: int
    target_min: int
    target_max: int
    target_avg: float
    inflate_avg: float
    inflate_max: float


@dataclass
class RankBucketReport:
    rank: int
    snapshots: int
    bucket_calls: int
    tracer_snapshots: int
    mean_tracers_per_snapshot: float
    mean_buckets_per_snapshot: float
    max_buckets_per_snapshot: int
    mean_bucket_size: float
    p50_bucket_size: float
    p95_bucket_size: float
    min_bucket_size: int
    max_bucket_size: int
    mean_nsub: float
    p95_nsub: float
    max_nsub: int
    mean_base_nsub: float
    mean_coeff_segments: float
    total_bucket_nsub: float
    total_bucket_base_nsub: float
    total_bucket_coeff_segments: float
    total_work_nsub: float
    total_work_base_nsub: float
    total_work_coeff_segments: float
    total_target_work: float
    work_nsub_inflation: float
    work_coeff_inflation: float
    bucket_nsub_inflation: float
    mean_inflate_avg: float
    max_inflate_max: float
    small_bucket_fraction: float
    small_bucket_tracer_fraction: float
    small_bucket_work_fraction: float
    base_count: int | None = None
    final_count: int | None = None
    base_sum_nsub: float | None = None
    final_sum_nsub: float | None = None
    wall_ms: float | None = None
    solve_ms: float | None = None
    solve_rhs_ms: float | None = None
    solve_tridiag_ms: float | None = None
    coeff_ms: float | None = None


@dataclass
class SnapshotBucketReport:
    snapshot_index: int
    physical_snapshot: int
    z: float
    ranks_reporting: int
    bucket_calls: int
    tracer_count: int
    mean_bucket_size: float
    p95_bucket_size: float
    max_bucket_size: int
    mean_nsub: float
    max_nsub: int
    total_work_nsub: float
    total_work_coeff_segments: float
    small_bucket_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tracer bucketstats output in a run directory."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing bucketstats_rank*.tsv.",
    )
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=None,
        help="Optional directory containing timing_core*.tsv for runtime correlation.",
    )
    parser.add_argument(
        "--small-bucket-threshold",
        type=int,
        default=512,
        help="Bucket-size threshold used to define 'small buckets'. Default: 512.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of ranks to show in the detailed table.",
    )
    parser.add_argument(
        "--sort-by",
        choices=(
            "bucket_calls",
            "work_nsub",
            "inflation",
            "work_coeff",
            "small_frac",
            "mean_bucket",
            "solve_tridiag",
            "wall",
        ),
        default="bucket_calls",
        help="Sort key for the detailed rank table.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    return parser.parse_args()


def parse_int(raw: str) -> int:
    return int(raw.strip())


def parse_float(raw: str) -> float:
    return float(raw.strip())


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("nan")
    if q <= 0.0:
        return vals[0]
    if q >= 1.0:
        return vals[-1]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sxx = sum(v * v for v in dx)
    syy = sum(v * v for v in dy)
    if sxx == 0.0 or syy == 0.0:
        return float("nan")
    sxy = sum(a * b for a, b in zip(dx, dy))
    return sxy / math.sqrt(sxx * syy)


def imbalance_ratio(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    avg = mean(vals)
    if avg == 0.0:
        return float("nan")
    return max(vals) / avg


def format_float(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def finite_values(values: Iterable[float]) -> list[float]:
    vals: list[float] = []
    for value in values:
        fvalue = float(value)
        if math.isnan(fvalue) or not math.isfinite(fvalue):
            continue
        vals.append(fvalue)
    return vals


def read_bucket_rows(run_dir: Path) -> list[BucketRow]:
    rows: list[BucketRow] = []
    for path in sorted(p for p in run_dir.iterdir() if BUCKET_RE.match(p.name)):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for rec in reader:
                rows.append(
                    BucketRow(
                        rank=parse_int(rec["rank"]),
                        snapshot_index=parse_int(rec["snapshot_index"]),
                        physical_snapshot=parse_int(rec["physical_snapshot"]),
                        z=parse_float(rec["z"]),
                        bucket_index=parse_int(rec["bucket_index"]),
                        nbuckets=parse_int(rec["nbuckets"]),
                        bucket_size=parse_int(rec["bucket_size"]),
                        nsub=parse_int(rec["nsub"]),
                        n_on=parse_int(rec["n_on"]),
                        base_nsub=parse_int(rec["base_nsub"]),
                        fp_cadence=parse_int(rec["fp_cadence"]),
                        coeff_segments=parse_int(rec["coeff_segments"]),
                        target_min=parse_int(rec["target_min"]),
                        target_max=parse_int(rec["target_max"]),
                        target_avg=parse_float(rec["target_avg"]),
                        inflate_avg=parse_float(rec["inflate_avg"]),
                        inflate_max=parse_float(rec["inflate_max"]),
                    )
                )
    rows.sort(key=lambda row: (row.rank, row.snapshot_index, row.bucket_index))
    return rows


def build_rank_reports(
    bucket_rows: list[BucketRow],
    small_bucket_threshold: int,
    lb_rows,
    timing_rows,
) -> list[RankBucketReport]:
    rows_by_rank: dict[int, list[BucketRow]] = {}
    for row in bucket_rows:
        rows_by_rank.setdefault(row.rank, []).append(row)

    lb_by_rank = {row.rank: row for row in lb_rows}
    timing_by_rank = {row.rank: row for row in timing_rows}
    reports: list[RankBucketReport] = []

    for rank, rows in sorted(rows_by_rank.items()):
        bucket_sizes = [row.bucket_size for row in rows]
        nsubs = [row.nsub for row in rows]
        base_nsubs = [row.base_nsub for row in rows]
        coeffs = [row.coeff_segments for row in rows]
        tracer_snapshots = sum(bucket_sizes)
        total_work_nsub = sum(row.bucket_size * row.nsub for row in rows)
        total_work_base_nsub = sum(row.bucket_size * row.base_nsub for row in rows)
        total_work_coeff = sum(row.bucket_size * row.coeff_segments for row in rows)
        total_target_work = sum(row.bucket_size * row.target_avg for row in rows)
        snapshots = len({row.snapshot_index for row in rows})
        snapshot_bucket_counts: dict[int, int] = {}
        for row in rows:
            snapshot_bucket_counts[row.snapshot_index] = row.nbuckets

        small_rows = [row for row in rows if row.bucket_size < small_bucket_threshold]
        small_bucket_calls = len(small_rows)
        small_tracer_snapshots = sum(row.bucket_size for row in small_rows)
        small_work_nsub = sum(row.bucket_size * row.nsub for row in small_rows)
        lb = lb_by_rank.get(rank)
        timing = timing_by_rank.get(rank)
        final_sum_nsub = lb.final_sum_nsub if lb else None
        work_nsub_inflation = (
            float(total_work_nsub) / float(final_sum_nsub)
            if final_sum_nsub not in (None, 0.0)
            else float("nan")
        )
        work_coeff_inflation = (
            float(total_work_coeff) / float(final_sum_nsub)
            if final_sum_nsub not in (None, 0.0)
            else float("nan")
        )
        bucket_nsub_inflation = (
            sum(float(v) for v in nsubs) / float(final_sum_nsub)
            if final_sum_nsub not in (None, 0.0)
            else float("nan")
        )

        reports.append(
            RankBucketReport(
                rank=rank,
                snapshots=snapshots,
                bucket_calls=len(rows),
                tracer_snapshots=tracer_snapshots,
                mean_tracers_per_snapshot=(tracer_snapshots / snapshots) if snapshots else float("nan"),
                mean_buckets_per_snapshot=(len(rows) / snapshots) if snapshots else float("nan"),
                max_buckets_per_snapshot=max(snapshot_bucket_counts.values()) if snapshot_bucket_counts else 0,
                mean_bucket_size=mean(bucket_sizes),
                p50_bucket_size=percentile(bucket_sizes, 0.50),
                p95_bucket_size=percentile(bucket_sizes, 0.95),
                min_bucket_size=min(bucket_sizes),
                max_bucket_size=max(bucket_sizes),
                mean_nsub=mean(nsubs),
                p95_nsub=percentile(nsubs, 0.95),
                max_nsub=max(nsubs),
                mean_base_nsub=mean(base_nsubs),
                mean_coeff_segments=mean(coeffs),
                total_bucket_nsub=sum(float(v) for v in nsubs),
                total_bucket_base_nsub=sum(float(v) for v in base_nsubs),
                total_bucket_coeff_segments=sum(float(v) for v in coeffs),
                total_work_nsub=float(total_work_nsub),
                total_work_base_nsub=float(total_work_base_nsub),
                total_work_coeff_segments=float(total_work_coeff),
                total_target_work=float(total_target_work),
                work_nsub_inflation=work_nsub_inflation,
                work_coeff_inflation=work_coeff_inflation,
                bucket_nsub_inflation=bucket_nsub_inflation,
                mean_inflate_avg=mean(row.inflate_avg for row in rows),
                max_inflate_max=max(row.inflate_max for row in rows),
                small_bucket_fraction=(small_bucket_calls / len(rows)) if rows else float("nan"),
                small_bucket_tracer_fraction=(small_tracer_snapshots / tracer_snapshots) if tracer_snapshots else float("nan"),
                small_bucket_work_fraction=(small_work_nsub / total_work_nsub) if total_work_nsub else float("nan"),
                base_count=(lb.base_count if lb else None),
                final_count=(lb.final_count if lb else None),
                base_sum_nsub=(lb.base_sum_nsub if lb else None),
                final_sum_nsub=(lb.final_sum_nsub if lb else None),
                wall_ms=(timing.wall_ms if timing else None),
                solve_ms=(timing.solve_ms if timing else None),
                solve_rhs_ms=(timing.solve_rhs_ms if timing else None),
                solve_tridiag_ms=(timing.solve_tridiag_ms if timing else None),
                coeff_ms=(timing.coeff_ms if timing else None),
            )
        )

    return reports


def build_snapshot_reports(
    bucket_rows: list[BucketRow],
    small_bucket_threshold: int,
) -> list[SnapshotBucketReport]:
    rows_by_snapshot: dict[int, list[BucketRow]] = {}
    for row in bucket_rows:
        rows_by_snapshot.setdefault(row.snapshot_index, []).append(row)

    reports: list[SnapshotBucketReport] = []
    for snapshot_index, rows in sorted(rows_by_snapshot.items()):
        bucket_sizes = [row.bucket_size for row in rows]
        nsubs = [row.nsub for row in rows]
        small_rows = [row for row in rows if row.bucket_size < small_bucket_threshold]
        reports.append(
            SnapshotBucketReport(
                snapshot_index=snapshot_index,
                physical_snapshot=rows[0].physical_snapshot,
                z=rows[0].z,
                ranks_reporting=len({row.rank for row in rows}),
                bucket_calls=len(rows),
                tracer_count=sum(bucket_sizes),
                mean_bucket_size=mean(bucket_sizes),
                p95_bucket_size=percentile(bucket_sizes, 0.95),
                max_bucket_size=max(bucket_sizes),
                mean_nsub=mean(nsubs),
                max_nsub=max(nsubs),
                total_work_nsub=float(sum(row.bucket_size * row.nsub for row in rows)),
                total_work_coeff_segments=float(
                    sum(row.bucket_size * row.coeff_segments for row in rows)
                ),
                small_bucket_fraction=(len(small_rows) / len(rows)) if rows else float("nan"),
            )
        )
    return reports


def sort_reports(rows: list[RankBucketReport], sort_by: str) -> list[RankBucketReport]:
    def key(row: RankBucketReport) -> float:
        if sort_by == "bucket_calls":
            return float(row.bucket_calls)
        if sort_by == "work_nsub":
            return row.total_work_nsub
        if sort_by == "inflation":
            return row.work_nsub_inflation
        if sort_by == "work_coeff":
            return row.total_work_coeff_segments
        if sort_by == "small_frac":
            return row.small_bucket_fraction
        if sort_by == "mean_bucket":
            return row.mean_bucket_size
        if sort_by == "solve_tridiag":
            return row.solve_tridiag_ms if row.solve_tridiag_ms is not None else float("-inf")
        if sort_by == "wall":
            return row.wall_ms if row.wall_ms is not None else float("-inf")
        raise ValueError(f"unsupported sort key: {sort_by}")

    reverse = sort_by != "mean_bucket"
    return sorted(rows, key=key, reverse=reverse)


def print_summary(reports: list[RankBucketReport], small_bucket_threshold: int) -> dict[str, object]:
    bucket_sizes = [row.mean_bucket_size for row in reports]
    per_rank_bucket_calls = [float(row.bucket_calls) for row in reports]
    per_rank_work_nsub = [row.total_work_nsub for row in reports]
    per_rank_work_coeff = [row.total_work_coeff_segments for row in reports]
    per_rank_small_frac = [row.small_bucket_fraction for row in reports]
    per_rank_inflation = [
        row.work_nsub_inflation for row in reports if not math.isnan(row.work_nsub_inflation)
    ]
    payload: dict[str, object] = {
        "ranks": len(reports),
        "snapshots_mean": mean(float(row.snapshots) for row in reports),
        "bucket_calls_mean": mean(per_rank_bucket_calls),
        "bucket_calls_max_over_mean": imbalance_ratio(per_rank_bucket_calls),
        "tracer_snapshots_mean": mean(float(row.tracer_snapshots) for row in reports),
        "work_nsub_mean": mean(per_rank_work_nsub),
        "work_nsub_max_over_mean": imbalance_ratio(per_rank_work_nsub),
        "work_coeff_mean": mean(per_rank_work_coeff),
        "work_coeff_max_over_mean": imbalance_ratio(per_rank_work_coeff),
        "work_nsub_inflation_mean": mean(per_rank_inflation),
        "work_nsub_inflation_max_over_mean": imbalance_ratio(per_rank_inflation),
        "mean_bucket_size_mean": mean(bucket_sizes),
        "mean_bucket_size_p50": percentile(bucket_sizes, 0.50),
        "mean_bucket_size_p95": percentile(bucket_sizes, 0.95),
        "small_bucket_fraction_mean": mean(per_rank_small_frac),
        "small_bucket_fraction_max": max(per_rank_small_frac) if per_rank_small_frac else float("nan"),
        "small_bucket_threshold": small_bucket_threshold,
    }

    print("Summary")
    print(
        "  ranks={} snapshots/rank(mean)={} bucket_calls/rank(mean,max/mean)={}/{}".format(
            payload["ranks"],
            format_float(payload["snapshots_mean"]),
            format_float(payload["bucket_calls_mean"]),
            format_float(payload["bucket_calls_max_over_mean"]),
        )
    )
    print(
        "  tracer_snapshots/rank(mean)={} work_nsub/rank(mean,max/mean)={}/{}".format(
            format_float(payload["tracer_snapshots_mean"]),
            format_float(payload["work_nsub_mean"]),
            format_float(payload["work_nsub_max_over_mean"]),
        )
    )
    print(
        "  work_coeff/rank(mean,max/mean)={}/{} inflation(work_nsub/final_sum, mean,max/mean)={}/{}".format(
            format_float(payload["work_coeff_mean"]),
            format_float(payload["work_coeff_max_over_mean"]),
            format_float(payload["work_nsub_inflation_mean"]),
            format_float(payload["work_nsub_inflation_max_over_mean"]),
        )
    )
    print(
        "  mean_bucket_size(rank mean/p50/p95)={}/{}/{}".format(
            format_float(payload["mean_bucket_size_mean"]),
            format_float(payload["mean_bucket_size_p50"]),
            format_float(payload["mean_bucket_size_p95"]),
        )
    )
    print(
        "  small_bucket_fraction(size<{}, rank mean/max)={}/{}".format(
            small_bucket_threshold,
            format_float(payload["small_bucket_fraction_mean"]),
            format_float(payload["small_bucket_fraction_max"]),
        )
    )
    return payload


def print_consistency(reports: list[RankBucketReport]) -> dict[str, float]:
    final_sum_rows = [
        row for row in reports if row.final_sum_nsub is not None and row.final_sum_nsub != 0.0
    ]
    if not final_sum_rows:
        return {}

    global_final_sum = sum(
        float(row.final_sum_nsub) for row in final_sum_rows if row.final_sum_nsub is not None
    )
    global_work_nsub = sum(row.total_work_nsub for row in final_sum_rows)
    global_target_work = sum(row.total_target_work for row in final_sum_rows)
    global_bucket_nsub = sum(row.total_bucket_nsub for row in final_sum_rows)

    work_ratios = finite_values(
        row.total_work_nsub / float(row.final_sum_nsub)
        for row in final_sum_rows
        if row.final_sum_nsub not in (None, 0.0)
    )
    target_ratios = finite_values(
        row.total_target_work / float(row.final_sum_nsub)
        for row in final_sum_rows
        if row.final_sum_nsub not in (None, 0.0)
    )

    payload = {
        "global_final_sum_nsub": global_final_sum,
        "global_work_nsub": global_work_nsub,
        "global_target_work": global_target_work,
        "global_bucket_nsub": global_bucket_nsub,
        "global_work_over_final": (
            global_work_nsub / global_final_sum if global_final_sum else float("nan")
        ),
        "global_target_over_final": (
            global_target_work / global_final_sum if global_final_sum else float("nan")
        ),
        "global_bucket_over_final": (
            global_bucket_nsub / global_final_sum if global_final_sum else float("nan")
        ),
        "work_ratio_mean": mean(work_ratios),
        "work_ratio_p50": percentile(work_ratios, 0.50),
        "work_ratio_p95": percentile(work_ratios, 0.95),
        "work_ratio_min": min(work_ratios) if work_ratios else float("nan"),
        "work_ratio_max": max(work_ratios) if work_ratios else float("nan"),
        "target_ratio_mean": mean(target_ratios),
        "target_ratio_p50": percentile(target_ratios, 0.50),
        "target_ratio_p95": percentile(target_ratios, 0.95),
    }

    print("Consistency")
    print(
        "  global final_sum/work_nsub/target_work/bucket_nsub = {}/{}/{}/{}".format(
            format_float(payload["global_final_sum_nsub"]),
            format_float(payload["global_work_nsub"]),
            format_float(payload["global_target_work"]),
            format_float(payload["global_bucket_nsub"]),
        )
    )
    print(
        "  global work/final = {}  target/final = {}  bucket/final = {}".format(
            format_float(payload["global_work_over_final"]),
            format_float(payload["global_target_over_final"]),
            format_float(payload["global_bucket_over_final"]),
        )
    )
    print(
        "  per-rank work/final ratio mean/p50/p95/min/max = {}/{}/{}/{}/{}".format(
            format_float(payload["work_ratio_mean"]),
            format_float(payload["work_ratio_p50"]),
            format_float(payload["work_ratio_p95"]),
            format_float(payload["work_ratio_min"]),
            format_float(payload["work_ratio_max"]),
        )
    )
    print(
        "  per-rank target/final ratio mean/p50/p95 = {}/{}/{}".format(
            format_float(payload["target_ratio_mean"]),
            format_float(payload["target_ratio_p50"]),
            format_float(payload["target_ratio_p95"]),
        )
    )
    return payload


def print_correlations(reports: list[RankBucketReport]) -> dict[str, float]:
    corrs: dict[str, float] = {}
    if not reports:
        return corrs

    final_count_rows = [row for row in reports if row.final_count is not None]
    final_sum_rows = [row for row in reports if row.final_sum_nsub is not None]
    timing_rows = [row for row in reports if row.wall_ms is not None and row.solve_tridiag_ms is not None]

    if final_count_rows:
        counts = [float(row.final_count) for row in final_count_rows if row.final_count is not None]
        corrs["corr_final_count_bucket_calls"] = pearson(
            counts, [float(row.bucket_calls) for row in final_count_rows]
        )
        corrs["corr_final_count_mean_bucket_size"] = pearson(
            counts, [row.mean_bucket_size for row in final_count_rows]
        )
        corrs["corr_final_count_small_bucket_fraction"] = pearson(
            counts, [row.small_bucket_fraction for row in final_count_rows]
        )
        corrs["corr_final_count_work_nsub"] = pearson(
            counts, [row.total_work_nsub for row in final_count_rows]
        )
        corrs["corr_final_count_bucket_nsub"] = pearson(
            counts, [row.total_bucket_nsub for row in final_count_rows]
        )
        corrs["corr_final_count_work_coeff"] = pearson(
            counts, [row.total_work_coeff_segments for row in final_count_rows]
        )
        corrs["corr_final_count_work_nsub_inflation"] = pearson(
            counts, [row.work_nsub_inflation for row in final_count_rows]
        )
        corrs["corr_final_count_bucket_nsub_inflation"] = pearson(
            counts, [row.bucket_nsub_inflation for row in final_count_rows]
        )

    if final_sum_rows:
        sums = [float(row.final_sum_nsub) for row in final_sum_rows if row.final_sum_nsub is not None]
        corrs["corr_final_sum_work_nsub"] = pearson(
            sums, [row.total_work_nsub for row in final_sum_rows]
        )
        corrs["corr_final_sum_bucket_calls"] = pearson(
            sums, [float(row.bucket_calls) for row in final_sum_rows]
        )
        corrs["corr_final_sum_small_bucket_fraction"] = pearson(
            sums, [row.small_bucket_fraction for row in final_sum_rows]
        )
        inflation_rows = [
            row
            for row in final_sum_rows
            if not math.isnan(row.work_nsub_inflation) and row.final_count is not None
        ]
        if inflation_rows:
            corrs["corr_inflation_final_count"] = pearson(
                [row.work_nsub_inflation for row in inflation_rows],
                [float(row.final_count) for row in inflation_rows if row.final_count is not None],
            )
            corrs["corr_inflation_bucket_calls"] = pearson(
                [row.work_nsub_inflation for row in inflation_rows],
                [float(row.bucket_calls) for row in inflation_rows],
            )
            corrs["corr_inflation_small_bucket_fraction"] = pearson(
                [row.work_nsub_inflation for row in inflation_rows],
                [row.small_bucket_fraction for row in inflation_rows],
            )
            corrs["corr_inflation_mean_bucket_size"] = pearson(
                [row.work_nsub_inflation for row in inflation_rows],
                [row.mean_bucket_size for row in inflation_rows],
            )

    if timing_rows:
        wall = [float(row.wall_ms) for row in timing_rows if row.wall_ms is not None]
        solve = [float(row.solve_ms) for row in timing_rows if row.solve_ms is not None]
        solve_rhs = [float(row.solve_rhs_ms) for row in timing_rows if row.solve_rhs_ms is not None]
        solve_tridiag = [
            float(row.solve_tridiag_ms) for row in timing_rows if row.solve_tridiag_ms is not None
        ]
        coeff = [float(row.coeff_ms) for row in timing_rows if row.coeff_ms is not None]
        bucket_calls = [float(row.bucket_calls) for row in timing_rows]
        work_nsub = [row.total_work_nsub for row in timing_rows]
        bucket_nsub = [row.total_bucket_nsub for row in timing_rows]
        work_coeff = [row.total_work_coeff_segments for row in timing_rows]
        mean_bucket = [row.mean_bucket_size for row in timing_rows]
        small_frac = [row.small_bucket_fraction for row in timing_rows]

        corrs["corr_bucket_calls_wall"] = pearson(bucket_calls, wall)
        corrs["corr_bucket_calls_solve"] = pearson(bucket_calls, solve)
        corrs["corr_bucket_calls_solve_rhs"] = pearson(bucket_calls, solve_rhs)
        corrs["corr_bucket_calls_solve_tridiag"] = pearson(bucket_calls, solve_tridiag)
        corrs["corr_bucket_calls_coeff"] = pearson(bucket_calls, coeff)
        corrs["corr_work_nsub_wall"] = pearson(work_nsub, wall)
        corrs["corr_work_nsub_solve_tridiag"] = pearson(work_nsub, solve_tridiag)
        corrs["corr_bucket_nsub_solve_tridiag"] = pearson(bucket_nsub, solve_tridiag)
        corrs["corr_work_coeff_coeff"] = pearson(work_coeff, coeff)
        corrs["corr_mean_bucket_solve_tridiag"] = pearson(mean_bucket, solve_tridiag)
        corrs["corr_small_frac_solve_tridiag"] = pearson(small_frac, solve_tridiag)

    if corrs:
        print("Correlations")
        if "corr_final_count_bucket_calls" in corrs:
            print(
                "  final_count vs bucket_calls/mean_bucket/small_frac/work_nsub/bucket_nsub/work_coeff/work_inflation = {}/{}/{}/{}/{}/{}/{}".format(
                    format_float(corrs["corr_final_count_bucket_calls"]),
                    format_float(corrs["corr_final_count_mean_bucket_size"]),
                    format_float(corrs["corr_final_count_small_bucket_fraction"]),
                    format_float(corrs["corr_final_count_work_nsub"]),
                    format_float(corrs["corr_final_count_bucket_nsub"]),
                    format_float(corrs["corr_final_count_work_coeff"]),
                    format_float(corrs["corr_final_count_work_nsub_inflation"]),
                )
            )
        if "corr_final_sum_work_nsub" in corrs:
            print(
                "  final_sum_nsub vs work_nsub/bucket_calls/small_frac = {}/{}/{}".format(
                    format_float(corrs["corr_final_sum_work_nsub"]),
                    format_float(corrs["corr_final_sum_bucket_calls"]),
                    format_float(corrs["corr_final_sum_small_bucket_fraction"]),
                )
            )
        if "corr_inflation_final_count" in corrs:
            print(
                "  inflation(work_nsub/final_sum) vs final_count/bucket_calls/small_frac/mean_bucket = {}/{}/{}/{}".format(
                    format_float(corrs["corr_inflation_final_count"]),
                    format_float(corrs["corr_inflation_bucket_calls"]),
                    format_float(corrs["corr_inflation_small_bucket_fraction"]),
                    format_float(corrs["corr_inflation_mean_bucket_size"]),
                )
            )
        if "corr_bucket_calls_wall" in corrs:
            print(
                "  bucket_calls vs wall/solve/solve_rhs/solve_tridiag/coeff = {}/{}/{}/{}/{}".format(
                    format_float(corrs["corr_bucket_calls_wall"]),
                    format_float(corrs["corr_bucket_calls_solve"]),
                    format_float(corrs["corr_bucket_calls_solve_rhs"]),
                    format_float(corrs["corr_bucket_calls_solve_tridiag"]),
                    format_float(corrs["corr_bucket_calls_coeff"]),
                )
            )
            print(
                "  work_nsub vs wall/solve_tridiag = {}/{}  bucket_nsub vs solve_tridiag = {}".format(
                    format_float(corrs["corr_work_nsub_wall"]),
                    format_float(corrs["corr_work_nsub_solve_tridiag"]),
                    format_float(corrs["corr_bucket_nsub_solve_tridiag"]),
                )
            )
            print(
                "  mean_bucket/small_frac vs solve_tridiag = {}/{}  work_coeff vs coeff = {}".format(
                    format_float(corrs["corr_mean_bucket_solve_tridiag"]),
                    format_float(corrs["corr_small_frac_solve_tridiag"]),
                    format_float(corrs["corr_work_coeff_coeff"]),
                )
            )
    return corrs


def print_rank_table(reports: list[RankBucketReport], top: int) -> None:
    print("Ranks")
    print(
        "  rank snapshots bucket_calls tracers/snap mean_bucket p95_bucket "
        "small_frac work_nsub inflation bucket_nsub max_nsub final_count final_sum_nsub "
        "solve_tridiag_ms wall_ms"
    )
    for row in reports[:top]:
        final_count = str(row.final_count) if row.final_count is not None else "na"
        final_sum = (
            format_float(row.final_sum_nsub) if row.final_sum_nsub is not None else "na"
        )
        solve_tridiag = (
            format_float(row.solve_tridiag_ms) if row.solve_tridiag_ms is not None else "na"
        )
        wall = format_float(row.wall_ms) if row.wall_ms is not None else "na"
        print(
            "  {rank:4d} {snapshots:9d} {bucket_calls:12d} {tracers:12.1f} {mean_bucket:11.1f} "
            "{p95_bucket:10.1f} {small_frac:10.3f} {work_nsub:9.0f} {inflation:9.3f} "
            "{bucket_nsub:11.0f} {max_nsub:8d} {final_count:11s} {final_sum:14s} "
            "{solve_tridiag:16s} {wall:10s}".format(
                rank=row.rank,
                snapshots=row.snapshots,
                bucket_calls=row.bucket_calls,
                tracers=row.mean_tracers_per_snapshot,
                mean_bucket=row.mean_bucket_size,
                p95_bucket=row.p95_bucket_size,
                small_frac=row.small_bucket_fraction,
                work_nsub=row.total_work_nsub,
                inflation=row.work_nsub_inflation,
                bucket_nsub=row.total_bucket_nsub,
                max_nsub=row.max_nsub,
                final_count=final_count,
                final_sum=final_sum,
                solve_tridiag=solve_tridiag,
                wall=wall,
            )
        )


def print_snapshot_table(reports: list[SnapshotBucketReport], top: int) -> None:
    reports_sorted = sorted(reports, key=lambda row: row.total_work_nsub, reverse=True)
    print("Snapshots")
    print(
        "  snap physical z bucket_calls tracer_count mean_bucket p95_bucket "
        "small_frac mean_nsub max_nsub work_nsub work_coeff"
    )
    for row in reports_sorted[:top]:
        print(
            "  {snap:4d} {physical:8d} {z:8.4f} {bucket_calls:12d} {tracers:12d} "
            "{mean_bucket:11.1f} {p95_bucket:10.1f} {small_frac:10.3f} "
            "{mean_nsub:9.1f} {max_nsub:8d} {work_nsub:9.0f} {work_coeff:10.0f}".format(
                snap=row.snapshot_index,
                physical=row.physical_snapshot,
                z=row.z,
                bucket_calls=row.bucket_calls,
                tracers=row.tracer_count,
                mean_bucket=row.mean_bucket_size,
                p95_bucket=row.p95_bucket_size,
                small_frac=row.small_bucket_fraction,
                mean_nsub=row.mean_nsub,
                max_nsub=row.max_nsub,
                work_nsub=row.total_work_nsub,
                work_coeff=row.total_work_coeff_segments,
            )
        )


def build_findings(
    summary: dict[str, object],
    consistency: dict[str, float],
    corrs: dict[str, float],
) -> list[str]:
    findings: list[str] = []
    bucket_imb = float(summary["bucket_calls_max_over_mean"])
    work_imb = float(summary["work_nsub_max_over_mean"])
    if work_imb < 1.10 and bucket_imb > 1.20:
        findings.append(
            "nsub-weighted tracer work is fairly flat, but bucket-call count still varies materially across ranks"
        )
    if corrs.get("corr_final_count_bucket_calls", float("nan")) > 0.7:
        findings.append(
            "higher-count ranks are split into more buckets, so residual cost can still grow through per-bucket overhead"
        )
    global_work_over_final = consistency.get("global_work_over_final", float("nan"))
    if math.isfinite(global_work_over_final):
        if abs(global_work_over_final - 1.0) < 0.05:
            findings.append(
                "global work_nsub matches final_sum_nsub closely, so the LB estimate is consistent with runtime nsub totals"
            )
        elif global_work_over_final > 1.2:
            findings.append(
                "global work_nsub exceeds final_sum_nsub materially, so this comparison is mixing different quantities or mismatched run artifacts"
            )
    if corrs.get("corr_inflation_final_count", float("nan")) > 0.6:
        findings.append(
            "bucketized work inflation rises with final tracer count, so high-count ranks pay more work per unit final_sum_nsub"
        )
    if corrs.get("corr_inflation_small_bucket_fraction", float("nan")) > 0.6:
        findings.append(
            "bucketized work inflation rises with the fraction of small buckets, linking the sum_nsub mismatch to fragmentation"
        )
    if corrs.get("corr_final_count_mean_bucket_size", float("nan")) < -0.5:
        findings.append(
            "higher-count ranks tend to have smaller average buckets, which is consistent with stronger fragmentation"
        )
    if corrs.get("corr_bucket_calls_solve_tridiag", float("nan")) > 0.7:
        findings.append(
            "solve_tridiag tracks bucket-call count strongly, so fragmentation is a plausible source of residual solver spread"
        )
    if corrs.get("corr_work_nsub_solve_tridiag", float("nan")) > 0.8:
        findings.append(
            "solve_tridiag is still well explained by total bucket_size*nsub work, so most cost remains inside the linear solves"
        )
    if corrs.get("corr_bucket_nsub_solve_tridiag", float("nan")) > corrs.get("corr_work_nsub_solve_tridiag", -1.0) + 0.05:
        findings.append(
            "unweighted bucket nsub correlates with solve_tridiag at least as strongly as weighted work, suggesting per-bucket/per-call overhead matters"
        )
    if corrs.get("corr_small_frac_solve_tridiag", float("nan")) > 0.5:
        findings.append(
            "ranks with more small buckets tend to have slower tridiag time, so small-bucket inefficiency is worth checking"
        )
    if corrs.get("corr_work_coeff_coeff", float("nan")) > 0.8:
        findings.append(
            "coefficient-preparation cost follows bucketized tracer work closely, so bucket shape likely affects coeff time too"
        )
    if not findings:
        findings.append(
            "no strong heuristic finding was triggered; inspect bucket_calls, mean_bucket_size, and timing correlations directly"
        )
    return findings


def main() -> int:
    args = parse_args()
    bucket_rows = read_bucket_rows(args.run_dir)
    if not bucket_rows:
        raise SystemExit(f"{args.run_dir}: no bucketstats_rank*.tsv files were found")

    lb_rows = read_load_balance_rows(args.run_dir)
    timing_rows = read_timing_rows(args.timing_dir or args.run_dir)
    snapshot_reports = build_snapshot_reports(bucket_rows, args.small_bucket_threshold)
    reports = build_rank_reports(
        bucket_rows,
        args.small_bucket_threshold,
        lb_rows,
        timing_rows,
    )
    reports_sorted = sort_reports(reports, args.sort_by)
    summary = print_summary(reports, args.small_bucket_threshold)
    consistency = print_consistency(reports)
    corrs = print_correlations(reports)
    print_rank_table(reports_sorted, args.top)
    print_snapshot_table(snapshot_reports, args.top)
    findings = build_findings(summary, consistency, corrs)
    print("Findings")
    for finding in findings:
        print(f"  - {finding}")

    if args.json is not None:
        payload = {
            "summary": summary,
            "consistency": consistency,
            "correlations": corrs,
            "findings": findings,
            "ranks": [asdict(row) for row in reports_sorted],
            "snapshots": [asdict(row) for row in snapshot_reports],
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

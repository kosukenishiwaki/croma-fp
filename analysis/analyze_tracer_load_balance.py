#!/usr/bin/env python3
"""Analyze tracer load-balance output from a tracer FP run directory.

This script combines:

- timing_coreNNN.tsv
- run_summary.tsv
- load_balance_rank_totals.txt
- heavy_summary.txt

It is intended to answer questions such as:

- Does `final_sum_nsub` correlate with wall time?
- Is wall-time spread much larger than staged-time spread?
- How much time is outside `total_staged_ms`?
- Did the load balancer flatten `sum_nsub` across ranks?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TIMING_RE = re.compile(r"^timing_core\d+\.tsv$")


@dataclass
class TimingRow:
    rank: int
    local_ntracer: int
    wall_ms: float
    wall_ms_max: float
    wall_ms_mean: float
    wall_ms_min: float
    load_balance_ms: float
    bucket_build_ms: float
    pack_ms: float
    interp_ms: float
    coeff_ms: float
    secondary_ms: float
    solve_ms: float
    solve_alloc_ms: float
    solve_rhs_ms: float
    solve_tridiag_ms: float
    synch_ms: float
    gamma_ms: float
    cuda_setup_ms: float
    cuda_h2d_ms: float
    cuda_d2h_ms: float
    cuda_total_ms: float
    input_read_ms: float
    bg_prepare_ms: float
    tracer_mass_ms: float
    output_write_ms: float
    output_sync_ms: float
    checkpoint_ms: float
    restart_ms: float
    input_read_calls: int
    input_selected_runs: int
    output_write_calls: int
    output_sync_calls: int
    checkpoint_calls: int
    total_staged_ms: float


@dataclass
class LoadBalanceRow:
    rank: int
    base_count: int
    heavy_removed_count: int
    heavy_assigned_count: int
    final_count: int
    base_sum_nsub: float
    heavy_removed_sum_nsub: float
    heavy_assigned_sum_nsub: float
    final_sum_nsub: float


@dataclass
class RankReport:
    rank: int
    local_ntracer: int
    wall_ms: float
    total_staged_ms: float
    load_balance_ms: float
    bucket_build_ms: float
    solve_ms: float
    solve_alloc_ms: float
    solve_rhs_ms: float
    solve_tridiag_ms: float
    coeff_ms: float
    secondary_ms: float
    synch_ms: float
    input_read_ms: float
    bg_prepare_ms: float
    tracer_mass_ms: float
    output_write_ms: float
    output_sync_ms: float
    checkpoint_ms: float
    restart_ms: float
    input_selected_runs: int
    unaccounted_ms: float
    staged_fraction: float
    unaccounted_fraction: float
    base_count: int | None = None
    final_count: int | None = None
    heavy_removed_count: int | None = None
    heavy_assigned_count: int | None = None
    base_sum_nsub: float | None = None
    final_sum_nsub: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tracer load-balance output in a run directory."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing timing_core*.tsv and optional load-balance files.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of ranks to show in the detailed table.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("wall", "bucket_build", "unaccounted", "solve", "solve_alloc", "solve_rhs", "solve_tridiag", "final_sum", "coeff", "synch", "input"),
        default="wall",
        help="Sort key for the detailed rank table.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    return parser.parse_args()


def parse_float(raw: str) -> float:
    return float(raw.strip())


def parse_int(raw: str) -> int:
    return int(raw.strip())


def lookup_field(
    rec: dict[str, str],
    names: Iterable[str],
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    for name in names:
        if name in rec and rec[name] != "":
            return rec[name]
    if required:
        available = ", ".join(sorted(rec.keys()))
        wanted = ", ".join(names)
        raise KeyError(f"missing required field; wanted one of [{wanted}], available [{available}]")
    if default is None:
        available = ", ".join(sorted(rec.keys()))
        wanted = ", ".join(names)
        raise KeyError(f"missing optional field without default; wanted one of [{wanted}], available [{available}]")
    return default


def read_timing_rows(run_dir: Path) -> list[TimingRow]:
    rows: list[TimingRow] = []
    for path in sorted(p for p in run_dir.iterdir() if TIMING_RE.match(p.name)):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for rec in reader:
                rows.append(
                    TimingRow(
                        rank=parse_int(lookup_field(rec, ("rank",), required=True)),
                        local_ntracer=parse_int(
                            lookup_field(rec, ("local_ntracer", "ntracer"), required=True)
                        ),
                        wall_ms=parse_float(lookup_field(rec, ("wall_ms",), required=True)),
                        wall_ms_max=parse_float(
                            lookup_field(rec, ("wall_ms_max", "max_wall_ms"), required=True)
                        ),
                        wall_ms_mean=parse_float(
                            lookup_field(rec, ("wall_ms_mean", "mean_wall_ms"), required=True)
                        ),
                        wall_ms_min=parse_float(
                            lookup_field(rec, ("wall_ms_min", "min_wall_ms"), required=True)
                        ),
                        load_balance_ms=parse_float(
                            lookup_field(rec, ("load_balance_ms",), default="0")
                        ),
                        bucket_build_ms=parse_float(
                            lookup_field(rec, ("bucket_build_ms",), default="0")
                        ),
                        pack_ms=parse_float(
                            lookup_field(
                                rec,
                                ("pack_ms", "host_pack_ms", "host_pack_scatter_ms"),
                                default="0",
                            )
                        ),
                        interp_ms=parse_float(
                            lookup_field(rec, ("interp_ms", "coeff_interp_ms"), default="0")
                        ),
                        coeff_ms=parse_float(
                            lookup_field(rec, ("coeff_ms", "coeff_prep_ms"), default="0")
                        ),
                        secondary_ms=parse_float(
                            lookup_field(rec, ("secondary_ms",), default="0")
                        ),
                        solve_ms=parse_float(
                            lookup_field(rec, ("solve_ms", "fp_solve_ms"), required=True)
                        ),
                        solve_alloc_ms=parse_float(
                            lookup_field(rec, ("solve_alloc_ms",), default="0")
                        ),
                        solve_rhs_ms=parse_float(
                            lookup_field(rec, ("solve_rhs_ms",), default="0")
                        ),
                        solve_tridiag_ms=parse_float(
                            lookup_field(rec, ("solve_tridiag_ms",), default="0")
                        ),
                        synch_ms=parse_float(
                            lookup_field(rec, ("synch_ms", "synchrotron_ms"), default="0")
                        ),
                        gamma_ms=parse_float(
                            lookup_field(rec, ("gamma_ms",), default="0")
                        ),
                        cuda_setup_ms=parse_float(
                            lookup_field(rec, ("cuda_setup_ms",), default="0")
                        ),
                        cuda_h2d_ms=parse_float(
                            lookup_field(rec, ("cuda_h2d_ms",), default="0")
                        ),
                        cuda_d2h_ms=parse_float(
                            lookup_field(rec, ("cuda_d2h_ms",), default="0")
                        ),
                        cuda_total_ms=parse_float(
                            lookup_field(rec, ("cuda_total_ms", "backend_total_ms"), default="0")
                        ),
                        input_read_ms=parse_float(
                            lookup_field(rec, ("input_read_ms", "hdf5_read_ms"), default="0")
                        ),
                        bg_prepare_ms=parse_float(
                            lookup_field(rec, ("bg_prepare_ms", "background_prepare_ms"), default="0")
                        ),
                        tracer_mass_ms=parse_float(
                            lookup_field(rec, ("tracer_mass_ms", "mass_read_ms"), default="0")
                        ),
                        output_write_ms=parse_float(
                            lookup_field(rec, ("output_write_ms",), default="0")
                        ),
                        output_sync_ms=parse_float(
                            lookup_field(rec, ("output_sync_ms",), default="0")
                        ),
                        checkpoint_ms=parse_float(
                            lookup_field(rec, ("checkpoint_ms",), default="0")
                        ),
                        restart_ms=parse_float(
                            lookup_field(rec, ("restart_ms",), default="0")
                        ),
                        input_read_calls=parse_int(
                            lookup_field(rec, ("input_read_calls",), default="0")
                        ),
                        input_selected_runs=parse_int(
                            lookup_field(rec, ("input_selected_runs",), default="0")
                        ),
                        output_write_calls=parse_int(
                            lookup_field(rec, ("output_write_calls",), default="0")
                        ),
                        output_sync_calls=parse_int(
                            lookup_field(rec, ("output_sync_calls",), default="0")
                        ),
                        checkpoint_calls=parse_int(
                            lookup_field(rec, ("checkpoint_calls",), default="0")
                        ),
                        total_staged_ms=parse_float(
                            lookup_field(rec, ("total_staged_ms", "staged_ms"), required=True)
                        ),
                    )
                )
    rows.sort(key=lambda row: row.rank)
    return rows


def read_space_table(path: Path) -> tuple[list[str], list[list[str]]]:
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                maybe_header = stripped[1:].strip().split()
                if maybe_header and maybe_header[0] == "rank":
                    header = maybe_header
                continue
            rows.append(stripped.split())
    if header is None:
        raise ValueError(f"{path}: header line starting with '# rank' was not found")
    return header, rows


def read_load_balance_rows(run_dir: Path) -> list[LoadBalanceRow]:
    path = run_dir / "load_balance_rank_totals.txt"
    if not path.exists():
        return []
    header, raw_rows = read_space_table(path)
    expected = [
        "rank",
        "base_count",
        "heavy_removed_count",
        "heavy_assigned_count",
        "final_count",
        "base_sum_nsub",
        "heavy_removed_sum_nsub",
        "heavy_assigned_sum_nsub",
        "final_sum_nsub",
    ]
    if header != expected:
        raise ValueError(f"{path}: unexpected header: {' '.join(header)}")
    rows: list[LoadBalanceRow] = []
    for rec in raw_rows:
        rows.append(
            LoadBalanceRow(
                rank=int(rec[0]),
                base_count=int(rec[1]),
                heavy_removed_count=int(rec[2]),
                heavy_assigned_count=int(rec[3]),
                final_count=int(rec[4]),
                base_sum_nsub=float(rec[5]),
                heavy_removed_sum_nsub=float(rec[6]),
                heavy_assigned_sum_nsub=float(rec[7]),
                final_sum_nsub=float(rec[8]),
            )
        )
    rows.sort(key=lambda row: row.rank)
    return rows


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


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


def build_rank_reports(
    timing_rows: list[TimingRow], lb_rows: list[LoadBalanceRow]
) -> list[RankReport]:
    lb_by_rank = {row.rank: row for row in lb_rows}
    reports: list[RankReport] = []
    for timing in timing_rows:
        lb = lb_by_rank.get(timing.rank)
        unaccounted_ms = timing.wall_ms - timing.total_staged_ms - timing.load_balance_ms
        reports.append(
            RankReport(
                rank=timing.rank,
                local_ntracer=timing.local_ntracer,
                wall_ms=timing.wall_ms,
                total_staged_ms=timing.total_staged_ms,
                load_balance_ms=timing.load_balance_ms,
                bucket_build_ms=timing.bucket_build_ms,
                solve_ms=timing.solve_ms,
                solve_alloc_ms=timing.solve_alloc_ms,
                solve_rhs_ms=timing.solve_rhs_ms,
                solve_tridiag_ms=timing.solve_tridiag_ms,
                coeff_ms=timing.coeff_ms,
                secondary_ms=timing.secondary_ms,
                synch_ms=timing.synch_ms,
                input_read_ms=timing.input_read_ms,
                bg_prepare_ms=timing.bg_prepare_ms,
                tracer_mass_ms=timing.tracer_mass_ms,
                output_write_ms=timing.output_write_ms,
                output_sync_ms=timing.output_sync_ms,
                checkpoint_ms=timing.checkpoint_ms,
                restart_ms=timing.restart_ms,
                input_selected_runs=timing.input_selected_runs,
                unaccounted_ms=unaccounted_ms,
                staged_fraction=(timing.total_staged_ms / timing.wall_ms) if timing.wall_ms else float("nan"),
                unaccounted_fraction=(unaccounted_ms / timing.wall_ms) if timing.wall_ms else float("nan"),
                base_count=(lb.base_count if lb else None),
                final_count=(lb.final_count if lb else None),
                heavy_removed_count=(lb.heavy_removed_count if lb else None),
                heavy_assigned_count=(lb.heavy_assigned_count if lb else None),
                base_sum_nsub=(lb.base_sum_nsub if lb else None),
                final_sum_nsub=(lb.final_sum_nsub if lb else None),
            )
        )
    return reports


def sort_rank_reports(rows: list[RankReport], sort_by: str) -> list[RankReport]:
    def key(row: RankReport) -> float:
        if sort_by == "wall":
            return row.wall_ms
        if sort_by == "bucket_build":
            return row.bucket_build_ms
        if sort_by == "unaccounted":
            return row.unaccounted_ms
        if sort_by == "solve":
            return row.solve_ms
        if sort_by == "solve_alloc":
            return row.solve_alloc_ms
        if sort_by == "solve_rhs":
            return row.solve_rhs_ms
        if sort_by == "solve_tridiag":
            return row.solve_tridiag_ms
        if sort_by == "final_sum":
            return row.final_sum_nsub if row.final_sum_nsub is not None else float("-inf")
        if sort_by == "coeff":
            return row.coeff_ms
        if sort_by == "synch":
            return row.synch_ms
        if sort_by == "input":
            return row.input_read_ms
        raise ValueError(f"unsupported sort key: {sort_by}")

    return sorted(rows, key=key, reverse=True)


def print_summary(reports: list[RankReport], lb_rows: list[LoadBalanceRow]) -> dict[str, object]:
    wall_vals = [row.wall_ms for row in reports]
    staged_vals = [row.total_staged_ms for row in reports]
    solve_vals = [row.solve_ms for row in reports]
    bucket_build_vals = [row.bucket_build_ms for row in reports]
    solve_alloc_vals = [row.solve_alloc_ms for row in reports]
    solve_rhs_vals = [row.solve_rhs_ms for row in reports]
    solve_tridiag_vals = [row.solve_tridiag_ms for row in reports]
    unaccounted_vals = [row.unaccounted_ms for row in reports]
    payload: dict[str, object] = {
        "ranks": len(reports),
        "wall_ms_min": min(wall_vals),
        "wall_ms_mean": mean(wall_vals),
        "wall_ms_max": max(wall_vals),
        "wall_imbalance_max_over_mean": imbalance_ratio(wall_vals),
        "staged_ms_mean": mean(staged_vals),
        "bucket_build_ms_mean": mean(bucket_build_vals),
        "solve_ms_mean": mean(solve_vals),
        "solve_alloc_ms_mean": mean(solve_alloc_vals),
        "solve_rhs_ms_mean": mean(solve_rhs_vals),
        "solve_tridiag_ms_mean": mean(solve_tridiag_vals),
        "unaccounted_ms_mean": mean(unaccounted_vals),
        "input_read_ms_mean": mean(row.input_read_ms for row in reports),
        "bg_prepare_ms_mean": mean(row.bg_prepare_ms for row in reports),
        "tracer_mass_ms_mean": mean(row.tracer_mass_ms for row in reports),
        "output_write_ms_mean": mean(row.output_write_ms for row in reports),
        "output_sync_ms_mean": mean(row.output_sync_ms for row in reports),
        "staged_fraction_mean": mean(row.staged_fraction for row in reports),
        "unaccounted_fraction_mean": mean(row.unaccounted_fraction for row in reports),
    }

    print("Summary")
    print(
        "  ranks={} wall[min/mean/max]={}/{}/{} ms max/mean={}".format(
            payload["ranks"],
            format_float(payload["wall_ms_min"]),
            format_float(payload["wall_ms_mean"]),
            format_float(payload["wall_ms_max"]),
            format_float(payload["wall_imbalance_max_over_mean"]),
        )
    )
    print(
        "  staged_mean={} ms bucket_build_mean={} ms solve_mean={} ms unaccounted_mean={} ms".format(
            format_float(payload["staged_ms_mean"]),
            format_float(payload["bucket_build_ms_mean"]),
            format_float(payload["solve_ms_mean"]),
            format_float(payload["unaccounted_ms_mean"]),
        )
    )
    print(
        "  solve_alloc_mean={} ms solve_rhs_mean={} ms solve_tridiag_mean={} ms".format(
            format_float(payload["solve_alloc_ms_mean"]),
            format_float(payload["solve_rhs_ms_mean"]),
            format_float(payload["solve_tridiag_ms_mean"]),
        )
    )
    print(
        "  input_read_mean={} ms bg_prepare_mean={} ms tracer_mass_mean={} ms output_write_mean={} ms output_sync_mean={} ms".format(
            format_float(payload["input_read_ms_mean"]),
            format_float(payload["bg_prepare_ms_mean"]),
            format_float(payload["tracer_mass_ms_mean"]),
            format_float(payload["output_write_ms_mean"]),
            format_float(payload["output_sync_ms_mean"]),
        )
    )
    print(
        "  staged_fraction_mean={} unaccounted_fraction_mean={}".format(
            format_float(payload["staged_fraction_mean"]),
            format_float(payload["unaccounted_fraction_mean"]),
        )
    )

    if lb_rows:
        base_vals = [row.base_sum_nsub for row in lb_rows]
        final_vals = [row.final_sum_nsub for row in lb_rows]
        payload["base_sum_imbalance_max_over_mean"] = imbalance_ratio(base_vals)
        payload["final_sum_imbalance_max_over_mean"] = imbalance_ratio(final_vals)
        print(
            "  sum_nsub imbalance base max/mean={} final max/mean={}".format(
                format_float(payload["base_sum_imbalance_max_over_mean"]),
                format_float(payload["final_sum_imbalance_max_over_mean"]),
            )
        )

    return payload


def print_correlations(reports: list[RankReport], lb_rows: list[LoadBalanceRow]) -> dict[str, float]:
    if not lb_rows:
        return {}
    rows = [row for row in reports if row.final_sum_nsub is not None and row.base_sum_nsub is not None]
    final_sum = [float(row.final_sum_nsub) for row in rows]
    base_sum = [float(row.base_sum_nsub) for row in rows]
    wall = [row.wall_ms for row in rows]
    staged = [row.total_staged_ms for row in rows]
    solve = [row.solve_ms for row in rows]
    solve_alloc = [row.solve_alloc_ms for row in rows]
    solve_rhs = [row.solve_rhs_ms for row in rows]
    solve_tridiag = [row.solve_tridiag_ms for row in rows]
    coeff = [row.coeff_ms for row in rows]
    bucket_build = [row.bucket_build_ms for row in rows]
    secondary = [row.secondary_ms for row in rows]
    synch = [row.synch_ms for row in rows]
    input_read = [row.input_read_ms for row in rows]
    bg_prepare = [row.bg_prepare_ms for row in rows]
    tracer_mass = [row.tracer_mass_ms for row in rows]
    selected_runs = [float(row.input_selected_runs) for row in rows]
    unaccounted = [row.unaccounted_ms for row in rows]
    final_count = [float(row.final_count) for row in rows if row.final_count is not None]
    wall_for_count = [row.wall_ms for row in rows if row.final_count is not None]
    coeff_for_count = [row.coeff_ms for row in rows if row.final_count is not None]
    bucket_build_for_count = [row.bucket_build_ms for row in rows if row.final_count is not None]
    solve_for_count = [row.solve_ms for row in rows if row.final_count is not None]
    solve_alloc_for_count = [row.solve_alloc_ms for row in rows if row.final_count is not None]
    solve_rhs_for_count = [row.solve_rhs_ms for row in rows if row.final_count is not None]
    solve_tridiag_for_count = [row.solve_tridiag_ms for row in rows if row.final_count is not None]
    synch_for_count = [row.synch_ms for row in rows if row.final_count is not None]
    input_for_count = [row.input_read_ms for row in rows if row.final_count is not None]
    selected_runs_for_count = [float(row.input_selected_runs) for row in rows if row.final_count is not None]

    corrs = {
        "corr_final_sum_wall": pearson(final_sum, wall),
        "corr_final_sum_staged": pearson(final_sum, staged),
        "corr_final_sum_solve": pearson(final_sum, solve),
        "corr_final_sum_solve_alloc": pearson(final_sum, solve_alloc),
        "corr_final_sum_solve_rhs": pearson(final_sum, solve_rhs),
        "corr_final_sum_solve_tridiag": pearson(final_sum, solve_tridiag),
        "corr_final_sum_coeff": pearson(final_sum, coeff),
        "corr_final_sum_bucket_build": pearson(final_sum, bucket_build),
        "corr_final_sum_secondary": pearson(final_sum, secondary),
        "corr_final_sum_synch": pearson(final_sum, synch),
        "corr_final_sum_input_read": pearson(final_sum, input_read),
        "corr_final_sum_selected_runs": pearson(final_sum, selected_runs),
        "corr_final_sum_unaccounted": pearson(final_sum, unaccounted),
        "corr_base_sum_wall": pearson(base_sum, wall),
        "corr_final_count_wall": pearson(final_count, wall_for_count),
        "corr_final_count_coeff": pearson(final_count, coeff_for_count),
        "corr_final_count_bucket_build": pearson(final_count, bucket_build_for_count),
        "corr_final_count_solve": pearson(final_count, solve_for_count),
        "corr_final_count_solve_alloc": pearson(final_count, solve_alloc_for_count),
        "corr_final_count_solve_rhs": pearson(final_count, solve_rhs_for_count),
        "corr_final_count_solve_tridiag": pearson(final_count, solve_tridiag_for_count),
        "corr_final_count_synch": pearson(final_count, synch_for_count),
        "corr_final_count_input_read": pearson(final_count, input_for_count),
        "corr_final_count_selected_runs": pearson(final_count, selected_runs_for_count),
        "corr_selected_runs_input_read": pearson(selected_runs, input_read),
        "corr_selected_runs_wall": pearson(selected_runs, wall),
        "corr_coeff_wall": pearson(coeff, wall),
        "corr_bucket_build_wall": pearson(bucket_build, wall),
        "corr_solve_wall": pearson(solve, wall),
        "corr_solve_alloc_wall": pearson(solve_alloc, wall),
        "corr_solve_rhs_wall": pearson(solve_rhs, wall),
        "corr_solve_tridiag_wall": pearson(solve_tridiag, wall),
        "corr_synch_wall": pearson(synch, wall),
        "corr_input_read_wall": pearson(input_read, wall),
        "corr_bg_prepare_wall": pearson(bg_prepare, wall),
        "corr_tracer_mass_wall": pearson(tracer_mass, wall),
    }

    print("Correlations")
    print(
        "  final_sum_nsub vs wall/staged/solve/coeff/bucket_build/synch/input/unaccounted = {}/{}/{}/{}/{}/{}/{}/{}".format(
            format_float(corrs["corr_final_sum_wall"]),
            format_float(corrs["corr_final_sum_staged"]),
            format_float(corrs["corr_final_sum_solve"]),
            format_float(corrs["corr_final_sum_coeff"]),
            format_float(corrs["corr_final_sum_bucket_build"]),
            format_float(corrs["corr_final_sum_synch"]),
            format_float(corrs["corr_final_sum_input_read"]),
            format_float(corrs["corr_final_sum_unaccounted"]),
        )
    )
    print(
        "  base_sum_nsub vs wall = {}  final_count vs wall/coeff/bucket_build/solve/synch/input/selected_runs = {}/{}/{}/{}/{}/{}/{}".format(
            format_float(corrs["corr_base_sum_wall"]),
            format_float(corrs["corr_final_count_wall"]),
            format_float(corrs["corr_final_count_coeff"]),
            format_float(corrs["corr_final_count_bucket_build"]),
            format_float(corrs["corr_final_count_solve"]),
            format_float(corrs["corr_final_count_synch"]),
            format_float(corrs["corr_final_count_input_read"]),
            format_float(corrs["corr_final_count_selected_runs"]),
        )
    )
    print(
        "  selected_runs vs input_read/wall = {}/{}  coeff/bucket_build/solve/synch/input_read vs wall = {}/{}/{}/{}/{}".format(
            format_float(corrs["corr_selected_runs_input_read"]),
            format_float(corrs["corr_selected_runs_wall"]),
            format_float(corrs["corr_coeff_wall"]),
            format_float(corrs["corr_bucket_build_wall"]),
            format_float(corrs["corr_solve_wall"]),
            format_float(corrs["corr_synch_wall"]),
            format_float(corrs["corr_input_read_wall"]),
        )
    )
    print(
        "  solve_alloc/solve_rhs/solve_tridiag vs wall = {}/{}/{}  final_count vs solve_alloc/solve_rhs/solve_tridiag = {}/{}/{}".format(
            format_float(corrs["corr_solve_alloc_wall"]),
            format_float(corrs["corr_solve_rhs_wall"]),
            format_float(corrs["corr_solve_tridiag_wall"]),
            format_float(corrs["corr_final_count_solve_alloc"]),
            format_float(corrs["corr_final_count_solve_rhs"]),
            format_float(corrs["corr_final_count_solve_tridiag"]),
        )
    )
    return corrs


def print_rank_table(reports: list[RankReport], top: int) -> None:
    print("Ranks")
    header = (
        "  rank local_ntracer wall_ms staged_ms solve_ms solve_rhs_ms solve_tridiag_ms "
        "bucket_build_ms unaccounted_ms "
        "input_read_ms output_write_ms output_sync_ms selected_runs "
        "staged_frac unaccounted_frac final_sum_nsub base_sum_nsub heavy(-/+) count(final)"
    )
    print(header)
    for row in reports[:top]:
        final_sum = format_float(row.final_sum_nsub or float("nan"))
        base_sum = format_float(row.base_sum_nsub or float("nan"))
        heavy = (
            f"{row.heavy_removed_count}/{row.heavy_assigned_count}"
            if row.heavy_removed_count is not None and row.heavy_assigned_count is not None
            else "na"
        )
        final_count = str(row.final_count) if row.final_count is not None else "na"
        print(
            "  {rank:4d} {local:13d} {wall:10.3f} {staged:10.3f} {solve:10.3f} "
            "{solve_rhs:12.3f} {solve_tridiag:16.3f} {bucket_build:15.3f} {unacc:14.3f} "
            "{input_read:13.3f} {output_write:15.3f} {output_sync:14.3f} "
            "{selected_runs:13d} {staged_frac:11.3f} {unacc_frac:16.3f} {final_sum:14s} "
            "{base_sum:13s} {heavy:10s} {final_count:>11s}".format(
                rank=row.rank,
                local=row.local_ntracer,
                wall=row.wall_ms,
                staged=row.total_staged_ms,
                solve=row.solve_ms,
                solve_rhs=row.solve_rhs_ms,
                solve_tridiag=row.solve_tridiag_ms,
                bucket_build=row.bucket_build_ms,
                unacc=row.unaccounted_ms,
                input_read=row.input_read_ms,
                output_write=row.output_write_ms,
                output_sync=row.output_sync_ms,
                selected_runs=row.input_selected_runs,
                staged_frac=row.staged_fraction,
                unacc_frac=row.unaccounted_fraction,
                final_sum=final_sum,
                base_sum=base_sum,
                heavy=heavy,
                final_count=final_count,
            )
        )


def build_findings(reports: list[RankReport], summary: dict[str, object], corrs: dict[str, float]) -> list[str]:
    findings: list[str] = []
    wall_ratio = float(summary["wall_imbalance_max_over_mean"])
    unaccounted_frac = float(summary["unaccounted_fraction_mean"])
    if unaccounted_frac > 0.20:
        findings.append(
            "wall time has a large non-staged component on average; input/output or other uninstrumented overhead is significant"
        )
    if "final_sum_imbalance_max_over_mean" in summary:
        final_ratio = float(summary["final_sum_imbalance_max_over_mean"])
        base_ratio = float(summary["base_sum_imbalance_max_over_mean"])
        if final_ratio < base_ratio:
            findings.append("load balancing reduced the predicted sum_nsub imbalance across ranks")
        if final_ratio < 1.10 and wall_ratio > 1.30:
            findings.append(
                "predicted compute load is fairly flat after balancing, but wall time is still broad; the dominant spread is outside the current cost model"
            )
    corr_wall = corrs.get("corr_final_sum_wall")
    corr_solve = corrs.get("corr_final_sum_solve")
    corr_unacc = corrs.get("corr_final_sum_unaccounted")
    corr_count_wall = corrs.get("corr_final_count_wall")
    corr_count_coeff = corrs.get("corr_final_count_coeff")
    corr_count_solve = corrs.get("corr_final_count_solve")
    corr_count_solve_rhs = corrs.get("corr_final_count_solve_rhs")
    corr_count_solve_tridiag = corrs.get("corr_final_count_solve_tridiag")
    corr_count_synch = corrs.get("corr_final_count_synch")
    corr_count_input = corrs.get("corr_final_count_input_read")
    corr_sel_input = corrs.get("corr_selected_runs_input_read")
    if not math.isnan(corr_wall if corr_wall is not None else float("nan")):
        if corr_solve is not None and corr_wall is not None and corr_solve > 0.7 and corr_wall < 0.5:
            findings.append(
                "final_sum_nsub tracks solve time better than wall time; the load model is closer to compute cost than to end-to-end runtime"
            )
        if corr_unacc is not None and abs(corr_unacc) < 0.3:
            findings.append(
                "unaccounted time is weakly correlated with final_sum_nsub; the missing cost is likely filesystem, synchronization, or rank-local variability rather than FP work itself"
            )
    if corr_count_wall is not None and corr_count_wall > 0.7:
        findings.append(
            "wall time is strongly correlated with final tracer count, so O(ntracer) work is still a major source of spread after load balancing"
        )
    if corr_count_coeff is not None and corr_count_coeff > 0.7:
        findings.append(
            "coefficient preparation appears to scale with final tracer count and likely contributes materially to residual wall-time spread"
        )
    if corr_count_solve is not None and corr_count_solve > 0.7:
        findings.append(
            "solve time still scales strongly with final tracer count, suggesting the current sum_nsub-only model misses per-tracer fixed work"
        )
    if corr_count_solve_rhs is not None and corr_count_solve_rhs > 0.7:
        findings.append(
            "solve rhs construction scales strongly with final tracer count, so per-tracer source assembly is part of the residual imbalance"
        )
    if corr_count_solve_tridiag is not None and corr_count_solve_tridiag > 0.7:
        findings.append(
            "solve tridiagonal work scales strongly with final tracer count, indicating the residual spread is still dominated by per-tracer linear solves"
        )
    if corr_count_synch is not None and corr_count_synch > 0.7:
        findings.append(
            "synch time scales strongly with final tracer count, so emission work remains a residual load-balance term even in nowrite mode"
        )
    if corr_count_input is not None and corr_count_input > 0.7:
        findings.append(
            "input read time scales strongly with final tracer count, so HDF5 read volume is still rank-dependent after balancing"
        )
    if corr_sel_input is not None and corr_sel_input > 0.7:
        findings.append(
            "input selected-run fragmentation is strongly correlated with input read time, so non-contiguous HDF5 reads are a plausible secondary source of spread"
        )
    if not findings:
        findings.append("no strong heuristic finding was triggered; inspect the rank table and correlations directly")
    return findings


def maybe_write_json(
    path: Path | None,
    reports: list[RankReport],
    summary: dict[str, object],
    corrs: dict[str, float],
    findings: list[str],
) -> None:
    if path is None:
        return
    payload = {
        "summary": summary,
        "correlations": corrs,
        "findings": findings,
        "ranks": [asdict(row) for row in reports],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"error: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    timing_rows = read_timing_rows(run_dir)
    if not timing_rows:
        print(f"error: no timing_core*.tsv files found in {run_dir}", file=sys.stderr)
        return 1

    lb_rows = read_load_balance_rows(run_dir)
    reports = build_rank_reports(timing_rows, lb_rows)
    sorted_reports = sort_rank_reports(reports, args.sort_by)

    summary = print_summary(reports, lb_rows)
    corrs = print_correlations(reports, lb_rows)
    print_rank_table(sorted_reports, args.top)

    findings = build_findings(reports, summary, corrs)
    print("Findings")
    for finding in findings:
        print(f"  - {finding}")

    maybe_write_json(args.json, reports, summary, corrs, findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

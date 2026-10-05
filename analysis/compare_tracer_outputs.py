#!/usr/bin/env python3
"""Compare tracer-evolving CPU and CUDA output directories quantitatively.

The FP tracer path writes rank-local files:

  tracerid_coreNN.txt
  CRE_coreNN.bin, CRP_coreNN.bin, eSyn_coreNN.bin, optional eGamma_coreNN.bin

Binary files are double arrays packed as [tracer][flat output column].  This
script can match rows by tracer ID, so CPU and CUDA runs may use different MPI
rank counts.  It uses memmap + chunks where possible, so it does not need to
load full output files into memory.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


DATASETS = ("CRE", "CRP", "eSyn", "eGamma")
TRACER_RE = re.compile(r"^tracerid_core(?P<rank>\d+)\.txt$")
BIN_RE = re.compile(r"^(?P<name>CRE|CRP|eSyn|eGamma)_core(?P<rank>\d+)\.bin$")


@dataclass
class TopDiff:
    rel: float
    absdiff: float
    rank: str
    tracer_index: int
    tracer_id: int
    offset: int
    cpu: float
    cuda: float


@dataclass
class DatasetMetrics:
    name: str
    elements: int = 0
    finite_elements: int = 0
    nonfinite_elements: int = 0
    ignored_elements: int = 0
    fail_elements: int = 0
    ranks: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    sum_sq_abs: float = 0.0
    sum_sq_rel: float = 0.0
    sum_abs_diff: float = 0.0
    sum_abs_ref: float = 0.0
    top_heap: list[tuple[float, int, TopDiff]] = field(default_factory=list)
    top_abs_heap: list[tuple[float, int, TopDiff]] = field(default_factory=list)
    _top_counter: int = 0

    def add_top(self, item: TopDiff, limit: int) -> None:
        if limit <= 0:
            return
        entry = (item.rel, self._top_counter, item)
        self._top_counter += 1
        if len(self.top_heap) < limit:
            heapq.heappush(self.top_heap, entry)
        elif item.rel > self.top_heap[0][0]:
            heapq.heapreplace(self.top_heap, entry)

    def add_top_abs(self, item: TopDiff, limit: int) -> None:
        if limit <= 0:
            return
        entry = (item.absdiff, self._top_counter, item)
        self._top_counter += 1
        if len(self.top_abs_heap) < limit:
            heapq.heappush(self.top_abs_heap, entry)
        elif item.absdiff > self.top_abs_heap[0][0]:
            heapq.heapreplace(self.top_abs_heap, entry)

    @property
    def rms_abs(self) -> float:
        if self.finite_elements == 0:
            return float("nan")
        return math.sqrt(self.sum_sq_abs / self.finite_elements)

    @property
    def rms_rel(self) -> float:
        if self.finite_elements == 0:
            return float("nan")
        return math.sqrt(self.sum_sq_rel / self.finite_elements)

    @property
    def l1_rel(self) -> float:
        if self.sum_abs_ref == 0.0:
            return 0.0 if self.sum_abs_diff == 0.0 else float("inf")
        return self.sum_abs_diff / self.sum_abs_ref

    @property
    def passed(self) -> bool:
        return self.fail_elements == 0 and self.nonfinite_elements == 0

    def top_diffs(self) -> list[TopDiff]:
        return [entry[2] for entry in sorted(self.top_heap, reverse=True)]

    def top_abs_diffs(self) -> list[TopDiff]:
        return [entry[2] for entry in sorted(self.top_abs_heap, reverse=True)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CPU and CUDA tracer-evolving output directories."
    )
    parser.add_argument("cpu_dir", type=Path, help="CPU output directory")
    parser.add_argument("cuda_dir", type=Path, help="CUDA output directory")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated datasets to compare. Default: CRE,CRP,eSyn,eGamma",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-30)
    parser.add_argument(
        "--rel-floor",
        type=float,
        default=1.0e-300,
        help="Denominator floor for symmetric relative-error reporting.",
    )
    parser.add_argument(
        "--ignore-below",
        type=float,
        default=0.0,
        help=(
            "Ignore elements with max(abs(CPU),abs(CUDA)) below this absolute "
            "floor when counting failures and relative-difference top entries."
        ),
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
        help="Tracer rows processed at a time per rank in --match rank mode.",
    )
    parser.add_argument(
        "--match",
        choices=("id", "rank"),
        default="id",
        help="Compare by tracer ID across different rank layouts, or by rank/order.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of tracer IDs to sample in --match id mode. 0 compares all.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed for --sample-size.",
    )
    parser.add_argument(
        "--tracer-ids",
        default=None,
        help="Comma-separated tracer IDs, or a text file with IDs, to compare in --match id mode.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Number of largest relative differences to print per dataset.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    return parser.parse_args()


def rank_map_for_ids(run_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in run_dir.iterdir():
        match = TRACER_RE.match(path.name)
        if match:
            out[match.group("rank")] = path
    return out


def bin_map(run_dir: Path, dataset: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in run_dir.iterdir():
        match = BIN_RE.match(path.name)
        if match and match.group("name") == dataset:
            out[match.group("rank")] = path
    return out


def load_ids(path: Path) -> np.ndarray:
    ids = np.loadtxt(path, dtype=np.int64)
    return np.atleast_1d(ids)


def file_doubles(path: Path) -> int:
    size = path.stat().st_size
    if size % 8 != 0:
        raise ValueError(f"{path} size is not a multiple of sizeof(double): {size}")
    return size // 8


def memmap_rank(path: Path, ntracer: int) -> tuple[np.memmap, int]:
    if ntracer <= 0:
        raise ValueError(f"{path}: ntracer must be positive")
    ndouble = file_doubles(path)
    if ndouble % ntracer != 0:
        raise ValueError(
            f"{path}: {ndouble} doubles is not divisible by ntracer={ntracer}"
        )
    ncol = ndouble // ntracer
    arr = np.memmap(path, dtype=np.float64, mode="r", shape=(ntracer, ncol))
    return arr, ncol


@dataclass(frozen=True)
class TracerLoc:
    rank: str
    row: int


@dataclass
class RunIndex:
    ranks: list[str]
    ids_by_rank: dict[str, np.ndarray]
    loc_by_id: dict[int, TracerLoc]


def build_run_index(run_dir: Path, label: str) -> RunIndex:
    rank_files = rank_map_for_ids(run_dir)
    if not rank_files:
        raise ValueError(f"No tracerid_coreNN.txt files found in {run_dir}")

    ids_by_rank: dict[str, np.ndarray] = {}
    loc_by_id: dict[int, TracerLoc] = {}
    for rank in sorted(rank_files):
        ids = load_ids(rank_files[rank])
        ids_by_rank[rank] = ids
        for row, tracer_id_value in enumerate(ids):
            tracer_id = int(tracer_id_value)
            if tracer_id in loc_by_id:
                prev = loc_by_id[tracer_id]
                raise ValueError(
                    f"{label}: duplicate tracer ID {tracer_id} in "
                    f"rank {prev.rank}/row {prev.row} and rank {rank}/row {row}"
                )
            loc_by_id[tracer_id] = TracerLoc(rank=rank, row=row)
    return RunIndex(sorted(rank_files), ids_by_rank, loc_by_id)


def parse_requested_ids(spec: str | None) -> set[int] | None:
    if spec is None:
        return None
    path = Path(spec)
    if path.exists():
        ids = np.loadtxt(path, dtype=np.int64)
        return {int(x) for x in np.atleast_1d(ids)}
    return {int(x.strip()) for x in spec.split(",") if x.strip()}


def select_ids(
    common_ids: set[int],
    *,
    requested: set[int] | None,
    sample_size: int,
    sample_seed: int,
) -> list[int]:
    if requested is not None:
        missing = sorted(requested - common_ids)
        if missing:
            preview = missing[:10]
            raise ValueError(f"requested tracer IDs are missing from one run: {preview}")
        selected = sorted(requested)
    else:
        selected = sorted(common_ids)

    if sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    if sample_size == 0 or sample_size >= len(selected):
        return selected

    rng = np.random.default_rng(sample_seed)
    idx = rng.choice(len(selected), size=sample_size, replace=False)
    return [selected[i] for i in sorted(idx)]


def dataset_ncols_by_rank(
    dataset: str,
    run_dir: Path,
    run_index: RunIndex,
    label: str,
) -> tuple[dict[str, Path], dict[str, int]]:
    paths = bin_map(run_dir, dataset)
    expected = set(run_index.ranks)
    if not paths:
        return {}, {}
    if set(paths) != expected:
        missing = sorted(expected - set(paths))
        extra = sorted(set(paths) - expected)
        raise ValueError(f"{label} {dataset} ranks mismatch: missing={missing}, extra={extra}")

    ncols: dict[str, int] = {}
    for rank, path in paths.items():
        ntracer = len(run_index.ids_by_rank[rank])
        ndouble = file_doubles(path)
        if ntracer <= 0 or ndouble % ntracer != 0:
            raise ValueError(
                f"{label} {dataset} rank {rank}: {ndouble} doubles is not "
                f"divisible by ntracer={ntracer}"
            )
        ncols[rank] = ndouble // ntracer
    return paths, ncols


def update_metrics(
    metrics: DatasetMetrics,
    cpu: np.ndarray,
    cuda: np.ndarray,
    ids: np.ndarray,
    rank: str,
    row0: int,
    *,
    rtol: float,
    atol: float,
    rel_floor: float,
    ignore_below: float,
    top: int,
) -> None:
    cpu_f = np.asarray(cpu)
    cuda_f = np.asarray(cuda)
    finite = np.isfinite(cpu_f) & np.isfinite(cuda_f)
    both_equal = cpu_f == cuda_f
    nonfinite_bad = ~finite & ~both_equal

    metrics.elements += cpu_f.size
    metrics.finite_elements += int(finite.sum())
    metrics.nonfinite_elements += int(nonfinite_bad.sum())

    if finite.any():
        cpu_v = cpu_f[finite]
        cuda_v = cuda_f[finite]
        diff = cuda_v - cpu_v
        absdiff = np.abs(diff)
        ref_for_tol = np.abs(cpu_v)
        ref_scale = np.maximum(np.abs(cpu_v), np.abs(cuda_v))
        denom = np.maximum(ref_scale, rel_floor)
        rel = absdiff / denom
        significant = ref_scale >= ignore_below
        fail = significant & (absdiff > (atol + rtol * ref_for_tol))

        metrics.ignored_elements += int((~significant).sum())
        metrics.fail_elements += int(fail.sum())
        metrics.max_abs = max(metrics.max_abs, float(absdiff.max(initial=0.0)))
        metrics.max_rel = max(metrics.max_rel, float(rel.max(initial=0.0)))
        metrics.sum_sq_abs += float(np.dot(absdiff, absdiff))
        metrics.sum_sq_rel += float(np.dot(rel, rel))
        metrics.sum_abs_diff += float(absdiff.sum())
        metrics.sum_abs_ref += float(np.maximum(np.abs(cpu_v), np.abs(cuda_v)).sum())

        if top > 0 and rel.size > 0:
            changed_idx = np.flatnonzero((absdiff > 0.0) & significant)
            take = min(top, changed_idx.size)
            top_idx = changed_idx[np.argpartition(rel[changed_idx], -take)[-take:]] if take else []
            finite_coords = np.argwhere(finite)
            for idx in top_idx:
                r, c = finite_coords[int(idx)]
                tracer_index = row0 + int(r)
                metrics.add_top(
                    TopDiff(
                        rel=float(rel[idx]),
                        absdiff=float(absdiff[idx]),
                        rank=rank,
                        tracer_index=tracer_index,
                        tracer_id=int(ids[tracer_index]),
                        offset=int(c),
                        cpu=float(cpu_f[r, c]),
                        cuda=float(cuda_f[r, c]),
                    ),
                    top,
                )
            abs_changed_idx = np.flatnonzero(absdiff > 0.0)
            take_abs = min(top, abs_changed_idx.size)
            top_abs_idx = (
                abs_changed_idx[np.argpartition(absdiff[abs_changed_idx], -take_abs)[-take_abs:]]
                if take_abs
                else []
            )
            for idx in top_abs_idx:
                r, c = finite_coords[int(idx)]
                tracer_index = row0 + int(r)
                metrics.add_top_abs(
                    TopDiff(
                        rel=float(rel[idx]),
                        absdiff=float(absdiff[idx]),
                        rank=rank,
                        tracer_index=tracer_index,
                        tracer_id=int(ids[tracer_index]),
                        offset=int(c),
                        cpu=float(cpu_f[r, c]),
                        cuda=float(cuda_f[r, c]),
                    ),
                    top,
                )

    if nonfinite_bad.any():
        coords = np.argwhere(nonfinite_bad)
        metrics.fail_elements += int(nonfinite_bad.sum())
        for r, c in coords[:top]:
            tracer_index = row0 + int(r)
            metrics.add_top(
                TopDiff(
                    rel=float("inf"),
                    absdiff=float("inf"),
                    rank=rank,
                    tracer_index=tracer_index,
                    tracer_id=int(ids[tracer_index]),
                    offset=int(c),
                    cpu=float(cpu_f[r, c]),
                    cuda=float(cuda_f[r, c]),
                ),
                top,
            )


def compare_dataset(
    dataset: str,
    cpu_dir: Path,
    cuda_dir: Path,
    ranks: Iterable[str],
    ids_by_rank: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
    rel_floor: float,
    ignore_below: float,
    chunk_rows: int,
    top: int,
) -> DatasetMetrics | None:
    cpu_bins = bin_map(cpu_dir, dataset)
    cuda_bins = bin_map(cuda_dir, dataset)
    all_bin_ranks = set(cpu_bins) | set(cuda_bins)
    if not all_bin_ranks:
        return None

    expected_ranks = set(ranks)
    if set(cpu_bins) != expected_ranks:
        missing = sorted(expected_ranks - set(cpu_bins))
        extra = sorted(set(cpu_bins) - expected_ranks)
        raise ValueError(f"CPU {dataset} ranks mismatch: missing={missing}, extra={extra}")
    if set(cuda_bins) != expected_ranks:
        missing = sorted(expected_ranks - set(cuda_bins))
        extra = sorted(set(cuda_bins) - expected_ranks)
        raise ValueError(f"CUDA {dataset} ranks mismatch: missing={missing}, extra={extra}")

    metrics = DatasetMetrics(dataset)
    for rank in sorted(expected_ranks):
        ids = ids_by_rank[rank]
        ntracer = len(ids)
        cpu_mm, cpu_ncol = memmap_rank(cpu_bins[rank], ntracer)
        cuda_mm, cuda_ncol = memmap_rank(cuda_bins[rank], ntracer)
        if cpu_ncol != cuda_ncol:
            raise ValueError(
                f"{dataset} rank {rank}: column mismatch CPU={cpu_ncol}, CUDA={cuda_ncol}"
            )

        metrics.ranks += 1
        for row0 in range(0, ntracer, chunk_rows):
            row1 = min(row0 + chunk_rows, ntracer)
            update_metrics(
                metrics,
                cpu_mm[row0:row1],
                cuda_mm[row0:row1],
                ids,
                rank,
                row0,
                rtol=rtol,
                atol=atol,
                rel_floor=rel_floor,
                ignore_below=ignore_below,
                top=top,
            )
    return metrics


def compare_dataset_by_id(
    dataset: str,
    cpu_dir: Path,
    cuda_dir: Path,
    cpu_index: RunIndex,
    cuda_index: RunIndex,
    selected_ids: list[int],
    *,
    rtol: float,
    atol: float,
    rel_floor: float,
    ignore_below: float,
    top: int,
) -> DatasetMetrics | None:
    cpu_paths, cpu_ncols = dataset_ncols_by_rank(dataset, cpu_dir, cpu_index, "CPU")
    cuda_paths, cuda_ncols = dataset_ncols_by_rank(dataset, cuda_dir, cuda_index, "CUDA")
    if not cpu_paths and not cuda_paths:
        return None
    if bool(cpu_paths) != bool(cuda_paths):
        raise ValueError(f"{dataset}: present in only one output directory")

    cpu_cache: dict[str, np.memmap] = {}
    cuda_cache: dict[str, np.memmap] = {}

    def get_row(
        paths: dict[str, Path],
        ncols: dict[str, int],
        index: RunIndex,
        cache: dict[str, np.memmap],
        tracer_id: int,
    ) -> np.ndarray:
        loc = index.loc_by_id[tracer_id]
        if loc.rank not in cache:
            ntracer = len(index.ids_by_rank[loc.rank])
            cache[loc.rank] = np.memmap(
                paths[loc.rank],
                dtype=np.float64,
                mode="r",
                shape=(ntracer, ncols[loc.rank]),
            )
        return cache[loc.rank][loc.row : loc.row + 1]

    metrics = DatasetMetrics(dataset)
    metrics.ranks = len(set(cpu_paths) | set(cuda_paths))
    for tracer_id in selected_ids:
        cpu_loc = cpu_index.loc_by_id[tracer_id]
        cuda_loc = cuda_index.loc_by_id[tracer_id]
        cpu_ncol = cpu_ncols[cpu_loc.rank]
        cuda_ncol = cuda_ncols[cuda_loc.rank]
        if cpu_ncol != cuda_ncol:
            raise ValueError(
                f"{dataset} tracer {tracer_id}: column mismatch "
                f"CPU rank {cpu_loc.rank} has {cpu_ncol}, "
                f"CUDA rank {cuda_loc.rank} has {cuda_ncol}"
            )
        update_metrics(
            metrics,
            get_row(cpu_paths, cpu_ncols, cpu_index, cpu_cache, tracer_id),
            get_row(cuda_paths, cuda_ncols, cuda_index, cuda_cache, tracer_id),
            cpu_index.ids_by_rank[cpu_loc.rank],
            f"cpu{cpu_loc.rank}/cuda{cuda_loc.rank}",
            cpu_loc.row,
            rtol=rtol,
            atol=atol,
            rel_floor=rel_floor,
            ignore_below=ignore_below,
            top=top,
        )
    return metrics


def compare_tracer_ids(cpu_dir: Path, cuda_dir: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    cpu_ids = rank_map_for_ids(cpu_dir)
    cuda_ids = rank_map_for_ids(cuda_dir)
    if not cpu_ids:
        raise ValueError(f"No tracerid_coreNN.txt files found in {cpu_dir}")
    if set(cpu_ids) != set(cuda_ids):
        missing = sorted(set(cpu_ids) - set(cuda_ids))
        extra = sorted(set(cuda_ids) - set(cpu_ids))
        raise ValueError(f"Tracer-ID ranks mismatch: CUDA missing={missing}, CUDA extra={extra}")

    ids_by_rank: dict[str, np.ndarray] = {}
    for rank in sorted(cpu_ids):
        left = load_ids(cpu_ids[rank])
        right = load_ids(cuda_ids[rank])
        if left.shape != right.shape:
            raise ValueError(
                f"rank {rank}: tracer-ID count mismatch CPU={left.size}, CUDA={right.size}"
            )
        if not np.array_equal(left, right):
            first = int(np.flatnonzero(left != right)[0])
            raise ValueError(
                f"rank {rank}: tracer-ID order mismatch at row {first}: "
                f"CPU={left[first]}, CUDA={right[first]}"
            )
        ids_by_rank[rank] = left
    return sorted(cpu_ids), ids_by_rank


def print_report(metrics_list: list[DatasetMetrics], *, rtol: float, atol: float) -> None:
    print(f"Tolerance: rtol={rtol:.3e}, atol={atol:.3e}")
    print(
        "dataset ranks elements ignored failures max_abs rms_abs max_rel rms_rel l1_rel status"
    )
    for m in metrics_list:
        status = "PASS" if m.passed else "FAIL"
        print(
            f"{m.name:7s} {m.ranks:5d} {m.elements:12d} {m.ignored_elements:8d} "
            f"{m.fail_elements:8d} "
            f"{m.max_abs:.6e} {m.rms_abs:.6e} {m.max_rel:.6e} "
            f"{m.rms_rel:.6e} {m.l1_rel:.6e} {status}"
        )
        if m.top_abs_diffs() and (m.max_abs > 0.0 or m.nonfinite_elements > 0):
            print(f"  largest absolute differences for {m.name}:")
            for item in m.top_abs_diffs():
                print(
                    "    "
                    f"abs={item.absdiff:.6e} rel={item.rel:.6e} "
                    f"rank={item.rank} tracer_index={item.tracer_index} "
                    f"tracer_id={item.tracer_id} offset={item.offset} "
                    f"cpu={item.cpu:.17e} cuda={item.cuda:.17e}"
                )
        if m.top_diffs() and (m.max_abs > 0.0 or m.nonfinite_elements > 0):
            print(f"  largest relative differences for {m.name}:")
            for item in m.top_diffs():
                print(
                    "    "
                    f"rel={item.rel:.6e} abs={item.absdiff:.6e} "
                    f"rank={item.rank} tracer_index={item.tracer_index} "
                    f"tracer_id={item.tracer_id} offset={item.offset} "
                    f"cpu={item.cpu:.17e} cuda={item.cuda:.17e}"
                )


def write_json(path: Path, metrics_list: list[DatasetMetrics]) -> None:
    payload = []
    for m in metrics_list:
        payload.append(
            {
                "dataset": m.name,
                "ranks": m.ranks,
                "elements": m.elements,
                "finite_elements": m.finite_elements,
                "nonfinite_elements": m.nonfinite_elements,
                "ignored_elements": m.ignored_elements,
                "fail_elements": m.fail_elements,
                "max_abs": m.max_abs,
                "rms_abs": m.rms_abs,
                "max_rel": m.max_rel,
                "rms_rel": m.rms_rel,
                "l1_rel": m.l1_rel,
                "passed": m.passed,
                "top_diffs": [item.__dict__ for item in m.top_diffs()],
                "top_abs_diffs": [item.__dict__ for item in m.top_abs_diffs()],
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cpu_dir = args.cpu_dir
    cuda_dir = args.cuda_dir
    if not cpu_dir.is_dir():
        print(f"error: CPU output directory not found: {cpu_dir}", file=sys.stderr)
        return 2
    if not cuda_dir.is_dir():
        print(f"error: CUDA output directory not found: {cuda_dir}", file=sys.stderr)
        return 2
    if args.chunk_rows <= 0:
        print("error: --chunk-rows must be positive", file=sys.stderr)
        return 2

    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        print(f"error: unknown datasets: {unknown}", file=sys.stderr)
        return 2

    try:
        metrics_list = []
        if args.match == "rank":
            ranks, ids_by_rank = compare_tracer_ids(cpu_dir, cuda_dir)
            ntracer = sum(len(ids) for ids in ids_by_rank.values())
            print(f"Tracer IDs: PASS ({len(ranks)} ranks, {ntracer} tracers)")
            for dataset in datasets:
                metrics = compare_dataset(
                    dataset,
                    cpu_dir,
                    cuda_dir,
                    ranks,
                    ids_by_rank,
                    rtol=args.rtol,
                    atol=args.atol,
                    rel_floor=args.rel_floor,
                    ignore_below=args.ignore_below,
                    chunk_rows=args.chunk_rows,
                    top=args.top,
                )
                if metrics is not None:
                    metrics_list.append(metrics)
        else:
            cpu_index = build_run_index(cpu_dir, "CPU")
            cuda_index = build_run_index(cuda_dir, "CUDA")
            cpu_ids = set(cpu_index.loc_by_id)
            cuda_ids = set(cuda_index.loc_by_id)
            if cpu_ids != cuda_ids:
                missing = sorted(cpu_ids - cuda_ids)
                extra = sorted(cuda_ids - cpu_ids)
                raise ValueError(
                    f"Tracer-ID set mismatch: CUDA missing={missing[:10]}, "
                    f"CUDA extra={extra[:10]}"
                )
            requested = parse_requested_ids(args.tracer_ids)
            selected_ids = select_ids(
                cpu_ids,
                requested=requested,
                sample_size=args.sample_size,
                sample_seed=args.sample_seed,
            )
            print(
                "Tracer IDs: PASS "
                f"(CPU ranks={len(cpu_index.ranks)}, CUDA ranks={len(cuda_index.ranks)}, "
                f"total tracers={len(cpu_ids)}, compared tracers={len(selected_ids)})"
            )
            for dataset in datasets:
                metrics = compare_dataset_by_id(
                    dataset,
                    cpu_dir,
                    cuda_dir,
                    cpu_index,
                    cuda_index,
                    selected_ids,
                    rtol=args.rtol,
                    atol=args.atol,
                    rel_floor=args.rel_floor,
                    ignore_below=args.ignore_below,
                    top=args.top,
                )
                if metrics is not None:
                    metrics_list.append(metrics)
        if not metrics_list:
            raise ValueError("No requested binary datasets were found in either directory")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_report(metrics_list, rtol=args.rtol, atol=args.atol)
    if args.json is not None:
        write_json(args.json, metrics_list)
        print(f"JSON report: {args.json}")

    return 0 if all(m.passed for m in metrics_list) else 1


if __name__ == "__main__":
    raise SystemExit(main())

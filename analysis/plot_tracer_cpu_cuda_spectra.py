#!/usr/bin/env python3
"""Plot CPU/CUDA CRe and synch spectra for selected tracer IDs.

For each tracer, the figure has two columns (CRe, synch).  Each column shows
CPU/CUDA spectra on top and relative error on the bottom.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

plt = None


def ensure_matplotlib():
    global plt
    if plt is not None:
        return plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "error: matplotlib is required for plotting. Install it in the Python "
            "environment used to run this script."
        ) from exc
    plt = pyplot
    return plt

from compare_tracer_outputs import (
    RunIndex,
    bin_map,
    build_run_index,
    dataset_ncols_by_rank,
    parse_requested_ids,
    select_ids,
)


PARAM_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^#\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CPU/CUDA CRe and synch spectra for tracer IDs."
    )
    parser.add_argument("cpu_dir", type=Path, help="CPU output directory")
    parser.add_argument("cuda_dir", type=Path, help="CUDA output directory")
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Parameter file used for the runs. Used for p/nu axes.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("compare_plots"))
    parser.add_argument(
        "--tracer-ids",
        default=None,
        help="Comma-separated tracer IDs, or a text file with IDs. If omitted, sample IDs are used.",
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--npe", type=int, default=128)
    parser.add_argument("--nfreq", type=int, default=None)
    parser.add_argument("--pmin", type=float, default=None, help="log10 minimum p/mc")
    parser.add_argument("--pmax", type=float, default=None, help="log10 maximum p/mc")
    parser.add_argument("--nu-min-s", type=float, default=None, help="log10 minimum synch frequency")
    parser.add_argument("--nu-max-s", type=float, default=None, help="log10 maximum synch frequency")
    parser.add_argument(
        "--cre-snapshot",
        type=int,
        default=-1,
        help="CRe snapshot index. -1 means final; CRe has nsnap+1 slots.",
    )
    parser.add_argument(
        "--synch-snapshot",
        type=int,
        default=-1,
        help="Synch snapshot index. -1 means final; synch has nsnap slots.",
    )
    parser.add_argument(
        "--rel-floor",
        type=float,
        default=1.0e-300,
        help="Denominator floor for symmetric relative error.",
    )
    parser.add_argument(
        "--rel-peak-floor-frac",
        type=float,
        default=0.0,
        help=(
            "Use at least this fraction of each plotted spectrum peak as the "
            "relative-error denominator floor."
        ),
    )
    parser.add_argument(
        "--ylim-rel",
        type=float,
        default=None,
        help="Optional y-limit for relative-error panels.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output image format for per-tracer files.",
    )
    return parser.parse_args()


def read_params(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    vals: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PARAM_RE.match(line)
        if not match:
            continue
        key, raw = match.groups()
        try:
            vals[key] = float(raw)
        except ValueError:
            continue
    return vals


def param_value(args: argparse.Namespace, params: dict[str, float], attr: str, keys: tuple[str, ...], default: float | None) -> float | None:
    value = getattr(args, attr)
    if value is not None:
        return value
    for key in keys:
        if key in params:
            return params[key]
    return default


def fp_log_grid(log_min: float, log_max: float, n: int) -> np.ndarray:
    dlog = (log_max - log_min) / float(n)
    return np.power(10.0, log_min + dlog * np.arange(n, dtype=np.float64))


def resolve_snapshot(index: int, nslot: int, label: str) -> int:
    if nslot <= 0:
        raise ValueError(f"{label}: no snapshots available")
    resolved = nslot + index if index < 0 else index
    if resolved < 0 or resolved >= nslot:
        raise ValueError(f"{label}: snapshot index {index} is outside 0..{nslot - 1}")
    return resolved


class DatasetReader:
    def __init__(self, dataset: str, run_dir: Path, index: RunIndex, label: str) -> None:
        paths, ncols = dataset_ncols_by_rank(dataset, run_dir, index, label)
        if not paths:
            raise ValueError(f"{label}: {dataset} output files were not found")
        self.paths = paths
        self.ncols = ncols
        self.index = index
        self.cache: dict[str, np.memmap] = {}

    def row(self, tracer_id: int) -> tuple[np.ndarray, str, int]:
        loc = self.index.loc_by_id[tracer_id]
        if loc.rank not in self.cache:
            ntracer = len(self.index.ids_by_rank[loc.rank])
            self.cache[loc.rank] = np.memmap(
                self.paths[loc.rank],
                dtype=np.float64,
                mode="r",
                shape=(ntracer, self.ncols[loc.rank]),
            )
        return np.asarray(self.cache[loc.rank][loc.row]), loc.rank, loc.row


def infer_slots(ncol: int, bins: int, label: str) -> int:
    if bins <= 0 or ncol % bins != 0:
        raise ValueError(f"{label}: ncol={ncol} is not divisible by bins={bins}")
    return ncol // bins


def rel_error(cpu: np.ndarray, cuda: np.ndarray, floor: float) -> np.ndarray:
    denom = np.maximum(np.maximum(np.abs(cpu), np.abs(cuda)), floor)
    return np.abs(cuda - cpu) / denom


def positive_ylim(*arrays: np.ndarray) -> tuple[float, float]:
    tiny = 1.0e-300
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return tiny, 1.0
    vmin = max(float(vals.min()), tiny)
    vmax = max(float(vals.max()), vmin * 10.0)
    ymin = max(10.0 ** math.floor(math.log10(vmin) - 0.5), tiny)
    ymax = 10.0 ** math.ceil(math.log10(vmax) + 0.5)
    if not math.isfinite(ymax) or ymax <= ymin:
        ymax = ymin * 10.0
    return ymin, ymax


def plot_one(
    tracer_id: int,
    *,
    out_path: Path,
    p_axis: np.ndarray,
    nu_axis: np.ndarray,
    cre_cpu: np.ndarray,
    cre_cuda: np.ndarray,
    syn_cpu: np.ndarray,
    syn_cuda: np.ndarray,
    cre_snapshot: int,
    syn_snapshot: int,
    cpu_rank_row: tuple[str, int],
    cuda_rank_row: tuple[str, int],
    rel_floor: float,
    rel_peak_floor_frac: float,
    ylim_rel: float | None,
) -> None:
    pyplot = ensure_matplotlib()
    cre_peak = max(float(np.nanmax(np.abs(cre_cpu))), float(np.nanmax(np.abs(cre_cuda))), 0.0)
    syn_peak = max(float(np.nanmax(np.abs(syn_cpu))), float(np.nanmax(np.abs(syn_cuda))), 0.0)
    cre_rel = rel_error(cre_cpu, cre_cuda, max(rel_floor, cre_peak * rel_peak_floor_frac))
    syn_rel = rel_error(syn_cpu, syn_cuda, max(rel_floor, syn_peak * rel_peak_floor_frac))

    fig, axes = pyplot.subplots(
        2,
        2,
        figsize=(11.5, 7.0),
        gridspec_kw={"height_ratios": [3.0, 1.2]},
        constrained_layout=True,
    )
    ax_cre, ax_syn = axes[0]
    ax_cre_rel, ax_syn_rel = axes[1]

    ax_cre.loglog(p_axis, cre_cpu, label="CPU", color="#1f77b4", lw=1.8)
    ax_cre.loglog(p_axis, cre_cuda, label="CUDA", color="#d62728", lw=1.3, ls="--")
    ax_cre.set_title(f"CRe spectrum, snapshot {cre_snapshot}")
    ax_cre.set_xlabel("p / mc")
    ax_cre.set_ylabel("N_e")
    ax_cre.set_ylim(*positive_ylim(cre_cpu, cre_cuda))
    ax_cre.grid(True, which="both", alpha=0.25)
    ax_cre.legend(loc="best")

    ax_syn.loglog(nu_axis, syn_cpu, label="CPU", color="#1f77b4", lw=1.8)
    ax_syn.loglog(nu_axis, syn_cuda, label="CUDA", color="#d62728", lw=1.3, ls="--")
    ax_syn.set_title(f"Synch spectrum, snapshot {syn_snapshot}")
    ax_syn.set_xlabel("nu [Hz]")
    ax_syn.set_ylabel("epsilon_syn")
    ax_syn.set_ylim(*positive_ylim(syn_cpu, syn_cuda))
    ax_syn.grid(True, which="both", alpha=0.25)
    ax_syn.legend(loc="best")

    ax_cre_rel.semilogx(p_axis, cre_rel, color="#111111", lw=1.2)
    ax_cre_rel.set_xlabel("p / mc")
    ax_cre_rel.set_ylabel("|CUDA-CPU|/max")
    ax_cre_rel.grid(True, which="both", alpha=0.25)

    ax_syn_rel.semilogx(nu_axis, syn_rel, color="#111111", lw=1.2)
    ax_syn_rel.set_xlabel("nu [Hz]")
    ax_syn_rel.set_ylabel("|CUDA-CPU|/max")
    ax_syn_rel.grid(True, which="both", alpha=0.25)

    if ylim_rel is not None:
        ax_cre_rel.set_ylim(0.0, ylim_rel)
        ax_syn_rel.set_ylim(0.0, ylim_rel)
    else:
        max_rel = max(float(np.nanmax(cre_rel)), float(np.nanmax(syn_rel)), 1.0e-16)
        for ax in (ax_cre_rel, ax_syn_rel):
            ax.set_ylim(0.0, max_rel * 1.1)

    cpu_rank, cpu_row = cpu_rank_row
    cuda_rank, cuda_row = cuda_rank_row
    fig.suptitle(
        f"tracer_id={tracer_id}  CPU rank/row={cpu_rank}/{cpu_row}  "
        f"CUDA rank/row={cuda_rank}/{cuda_row}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    pyplot.close(fig)


def main() -> int:
    args = parse_args()
    if not args.cpu_dir.is_dir():
        print(f"error: CPU output directory not found: {args.cpu_dir}", file=sys.stderr)
        return 2
    if not args.cuda_dir.is_dir():
        print(f"error: CUDA output directory not found: {args.cuda_dir}", file=sys.stderr)
        return 2
    if args.sample_size < 0:
        print("error: --sample-size must be non-negative", file=sys.stderr)
        return 2

    try:
        params = read_params(args.params)
        nfreq_value = param_value(args, params, "nfreq", ("nfreq", "N_freq"), None)
        pmin = param_value(args, params, "pmin", ("pmin",), -1.0)
        pmax = param_value(args, params, "pmax", ("pmax",), 8.0)
        nu_min_s = param_value(args, params, "nu_min_s", ("nu_min_s",), 5.5)
        nu_max_s = param_value(args, params, "nu_max_s", ("nu_max_s",), 10.5)
        if nfreq_value is None:
            nfreq = 128
        else:
            nfreq = int(nfreq_value)
        npe = int(args.npe)

        cpu_index = build_run_index(args.cpu_dir, "CPU")
        cuda_index = build_run_index(args.cuda_dir, "CUDA")
        cpu_ids = set(cpu_index.loc_by_id)
        cuda_ids = set(cuda_index.loc_by_id)
        if cpu_ids != cuda_ids:
            missing = sorted(cpu_ids - cuda_ids)
            extra = sorted(cuda_ids - cpu_ids)
            raise ValueError(
                f"Tracer-ID set mismatch: CUDA missing={missing[:10]}, CUDA extra={extra[:10]}"
            )
        requested = parse_requested_ids(args.tracer_ids)
        selected_ids = select_ids(
            cpu_ids,
            requested=requested,
            sample_size=args.sample_size if requested is None else 0,
            sample_seed=args.sample_seed,
        )
        if not selected_ids:
            raise ValueError("no tracer IDs selected for plotting")

        cre_cpu = DatasetReader("CRE", args.cpu_dir, cpu_index, "CPU")
        cre_cuda = DatasetReader("CRE", args.cuda_dir, cuda_index, "CUDA")
        syn_cpu = DatasetReader("eSyn", args.cpu_dir, cpu_index, "CPU")
        syn_cuda = DatasetReader("eSyn", args.cuda_dir, cuda_index, "CUDA")

        first_id = selected_ids[0]
        cre_nslot = infer_slots(cre_cpu.ncols[cre_cpu.index.loc_by_id[first_id].rank], npe, "CRE")
        syn_nslot = infer_slots(syn_cpu.ncols[syn_cpu.index.loc_by_id[first_id].rank], nfreq, "eSyn")
        cre_snapshot = resolve_snapshot(args.cre_snapshot, cre_nslot, "CRE")
        syn_snapshot = resolve_snapshot(args.synch_snapshot, syn_nslot, "eSyn")

        p_axis = fp_log_grid(float(pmin), float(pmax), npe)
        nu_axis = fp_log_grid(float(nu_min_s), float(nu_max_s), nfreq)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for tracer_id in selected_ids:
            cre_cpu_row, cpu_cre_rank, cpu_cre_row = cre_cpu.row(tracer_id)
            cre_cuda_row, cuda_cre_rank, cuda_cre_row = cre_cuda.row(tracer_id)
            syn_cpu_row, _, _ = syn_cpu.row(tracer_id)
            syn_cuda_row, _, _ = syn_cuda.row(tracer_id)

            cre_cpu_nslot = infer_slots(cre_cpu_row.size, npe, "CRE")
            cre_cuda_nslot = infer_slots(cre_cuda_row.size, npe, "CRE")
            syn_cpu_nslot = infer_slots(syn_cpu_row.size, nfreq, "eSyn")
            syn_cuda_nslot = infer_slots(syn_cuda_row.size, nfreq, "eSyn")
            if cre_cpu_nslot != cre_cuda_nslot or syn_cpu_nslot != syn_cuda_nslot:
                raise ValueError(f"tracer {tracer_id}: CPU/CUDA snapshot count mismatch")

            cre_isnap = resolve_snapshot(args.cre_snapshot, cre_cpu_nslot, "CRE")
            syn_isnap = resolve_snapshot(args.synch_snapshot, syn_cpu_nslot, "eSyn")
            cre_cpu_spec = cre_cpu_row.reshape(cre_cpu_nslot, npe)[cre_isnap]
            cre_cuda_spec = cre_cuda_row.reshape(cre_cuda_nslot, npe)[cre_isnap]
            syn_cpu_spec = syn_cpu_row.reshape(syn_cpu_nslot, nfreq)[syn_isnap]
            syn_cuda_spec = syn_cuda_row.reshape(syn_cuda_nslot, nfreq)[syn_isnap]

            out_path = args.out_dir / f"tracer_{tracer_id}_cre_syn.{args.format}"
            plot_one(
                tracer_id,
                out_path=out_path,
                p_axis=p_axis,
                nu_axis=nu_axis,
                cre_cpu=cre_cpu_spec,
                cre_cuda=cre_cuda_spec,
                syn_cpu=syn_cpu_spec,
                syn_cuda=syn_cuda_spec,
                cre_snapshot=cre_isnap,
                syn_snapshot=syn_isnap,
                cpu_rank_row=(cpu_cre_rank, cpu_cre_row),
                cuda_rank_row=(cuda_cre_rank, cuda_cre_row),
                rel_floor=args.rel_floor,
                rel_peak_floor_frac=args.rel_peak_floor_frac,
                ylim_rel=args.ylim_rel,
            )
            print(out_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

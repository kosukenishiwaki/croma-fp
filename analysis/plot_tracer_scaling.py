#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ModuleNotFoundError:
    plt = None
    HAVE_MPL = False


@dataclass
class CaseSummary:
    label: str
    run_tag: str
    run_dir: Path
    nranks: int
    timing_file_count: int
    global_ntracer: int
    local_ntracer_min: int
    local_ntracer_mean: float
    local_ntracer_max: int
    wall_ms_min: float
    wall_ms_mean: float
    wall_ms_max: float
    wall_ms_p10: float
    wall_ms_p50: float
    wall_ms_p90: float
    local_tp_min: float
    local_tp_mean: float
    local_tp_max: float
    local_tp_p10: float
    local_tp_p50: float
    local_tp_p90: float
    global_throughput: float
    coeff_ms_mean: float
    solve_ms_mean: float
    synch_ms_mean: float
    total_staged_ms_mean: float
    heavy_count: int | None
    heavy_sum_nsub_min: int | None
    heavy_sum_nsub_mean: float | None
    heavy_sum_nsub_max: int | None
    warnings: list[str]


def output_plot_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{stem}.png" if HAVE_MPL else output_dir / f"{stem}.svg"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_simple_svg_line_chart(
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict],
    width: int = 900,
    height: int = 520,
    x_log: bool = False,
) -> None:
    left = 90
    right = 30
    top = 55
    bottom = 75
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = [x for s in series for x in s["x"]]
    all_y = []
    for s in series:
        all_y.extend(s["y"])
        if "y_lo" in s:
            all_y.extend(s["y_lo"])
        if "y_hi" in s:
            all_y.extend(s["y_hi"])
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    ypad = 0.08 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    if x_log:
        if xmin <= 0.0:
            raise ValueError("x_log=True requires positive x values")
        log_pad = 0.03 * (math.log10(xmax) - math.log10(xmin))
        log_xmin = math.log10(xmin) - log_pad
        log_xmax = math.log10(xmax) + log_pad
        xmin = 10 ** log_xmin
        xmax = 10 ** log_xmax
        log_xmin = math.log10(xmin)
        log_xmax = math.log10(xmax)

        def sx(x: float) -> float:
            return left + (math.log10(x) - log_xmin) / (log_xmax - log_xmin) * plot_w
    else:
        xpad = 0.03 * (xmax - xmin)
        xmin -= xpad
        xmax += xpad

        def sx(x: float) -> float:
            return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcf8"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="Helvetica">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222" stroke-width="1.5"/>',
    ]

    for i in range(5):
        frac = i / 4.0
        yv = ymin + frac * (ymax - ymin)
        yp = sy(yv)
        parts.append(f'<line x1="{left}" y1="{yp:.2f}" x2="{left+plot_w}" y2="{yp:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-8}" y="{yp+4:.2f}" text-anchor="end" font-size="12" font-family="Helvetica">{yv:.3g}</text>')
    for i in range(5):
        frac = i / 4.0
        if x_log:
            xv = 10 ** (log_xmin + frac * (log_xmax - log_xmin))
        else:
            xv = xmin + frac * (xmax - xmin)
        xp = sx(xv)
        parts.append(f'<line x1="{xp:.2f}" y1="{top}" x2="{xp:.2f}" y2="{top+plot_h}" stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{xp:.2f}" y="{top+plot_h+20}" text-anchor="middle" font-size="12" font-family="Helvetica">{xv:.3g}</text>')

    for s in series:
        pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(s["x"], s["y"]))
        parts.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2.5" points="{pts}"/>')
        if "y_lo" in s and "y_hi" in s:
            for x, ylo, yhi in zip(s["x"], s["y_lo"], s["y_hi"]):
                xp = sx(x)
                parts.append(f'<line x1="{xp:.2f}" y1="{sy(ylo):.2f}" x2="{xp:.2f}" y2="{sy(yhi):.2f}" stroke="{s["color"]}" stroke-width="1.2"/>')
        for x, y in zip(s["x"], s["y"]):
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.8" fill="{s["color"]}"/>')

    legend_x = left + 10
    legend_y = top + 16
    for idx, s in enumerate(series):
        y = legend_y + idx * 18
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+18}" y2="{y}" stroke="{s["color"]}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x+24}" y="{y+4}" font-size="12" font-family="Helvetica">{svg_escape(s["label"])}</text>')

    parts.append(f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="15" font-family="Helvetica">{svg_escape(x_label)}</text>')
    parts.append(
        f'<text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="15" font-family="Helvetica" transform="rotate(-90 22,{height/2:.1f})">{svg_escape(y_label)}</text>'
    )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot tracer throughput / scalability from timing_core*.tsv outputs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Directories containing timing_core*.tsv, e.g. test_out/tracer_10k/run_100/out_10k",
    )
    parser.add_argument(
        "--output-dir",
        default="test_out/plots",
        help="Directory for summary CSV and PNG outputs.",
    )
    parser.add_argument(
        "--title",
        default="Tracer FP Throughput and Scalability",
        help="Figure title prefix.",
    )
    return parser.parse_args()


def percentile(sorted_values: list[float], frac: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * frac
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    alpha = idx - lo
    return sorted_values[lo] * (1.0 - alpha) + sorted_values[hi] * alpha


def infer_label(run_dir: Path) -> str:
    text = str(run_dir)
    match = re.search(r"tracer_(\d+k)", text)
    if match:
        return match.group(1)
    return run_dir.name


def infer_run_tag(run_dir: Path) -> str:
    text = str(run_dir)
    match = re.search(r"(run_\d+)", text)
    if match:
        return match.group(1)
    return "run"


def infer_expected_ranks_from_run_tag(run_tag: str) -> int | None:
    match = re.fullmatch(r"run_(\d+)", run_tag)
    if not match:
        return None
    return int(match.group(1))


def display_run_label(run_tag: str) -> str:
    expected = infer_expected_ranks_from_run_tag(run_tag)
    if expected is not None:
        return f"{expected} ranks"
    return run_tag


def discover_default_run_dirs() -> list[Path]:
    base = Path("test_out")
    problem_sizes = ("10k", "100k", "1000k")
    discovered: list[Path] = []

    for size in problem_sizes:
        tracer_root = base / f"tracer_{size}"
        if not tracer_root.exists():
            continue
        for run_dir in sorted(tracer_root.glob("run_*")):
            candidate = run_dir / f"out_{size}"
            if candidate.exists() and any(candidate.glob("timing_core*.tsv")):
                discovered.append(candidate)
    return discovered


def cases_for_problem_size_plots(cases: list[CaseSummary]) -> list[CaseSummary]:
    allowed = {"run_100", "run_200", "run_300"}
    return [case for case in cases if case.run_tag in allowed]


def read_heavy_summary(path: Path) -> tuple[int | None, int | None, float | None, int | None]:
    if not path.exists():
        return None, None, None, None

    heavy_count = None
    sums: list[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# heavy_count"):
            heavy_count = int(line.split()[-1])
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        sums.append(int(parts[2]))

    if not sums:
        return heavy_count, None, None, None
    return heavy_count, min(sums), sum(sums) / len(sums), max(sums)


def read_run_summary(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            return row
    return None


def heavy_imbalance_ratio(case: CaseSummary) -> float:
    if (
        case.heavy_sum_nsub_min is None
        or case.heavy_sum_nsub_max is None
        or case.heavy_sum_nsub_min <= 0
    ):
        return math.nan
    return case.heavy_sum_nsub_max / case.heavy_sum_nsub_min


def amdahl_relative_speedup(serial_fraction: float, nranks: float, ref_ranks: float) -> float:
    def amdahl_speedup(p: float) -> float:
        return 1.0 / (serial_fraction + (1.0 - serial_fraction) / p)

    return amdahl_speedup(nranks) / amdahl_speedup(ref_ranks)


def fit_amdahl_serial_fraction(nranks: list[int], speedup: list[float]) -> float:
    best_f = 0.0
    best_err = float("inf")
    for i in range(5001):
        f = i / 5000.0
        err = 0.0
        for p, s in zip(nranks, speedup):
            model = amdahl_relative_speedup(f, float(p), float(nranks[0]))
            err += (model - s) ** 2
        if err < best_err:
            best_err = err
            best_f = f
    return best_f


def strong_scaling_cases(cases: list[CaseSummary], label: str = "10k") -> list[CaseSummary]:
    group = [case for case in cases if case.label == label]
    return sorted(group, key=lambda c: c.nranks)


def weak_scaling_groups(cases: list[CaseSummary], ratio_tol: float = 1.25) -> list[tuple[str, list[CaseSummary]]]:
    ordered = sorted(cases, key=lambda c: c.local_ntracer_mean)
    groups: list[list[CaseSummary]] = []
    current: list[CaseSummary] = []
    current_min = 0.0
    current_max = 0.0

    for case in ordered:
        load = case.local_ntracer_mean
        if not current:
            current = [case]
            current_min = load
            current_max = load
            continue
        new_min = min(current_min, load)
        new_max = max(current_max, load)
        if new_min > 0.0 and new_max / new_min <= ratio_tol:
            current.append(case)
            current_min = new_min
            current_max = new_max
        else:
            if len(current) >= 2:
                groups.append(sorted(current, key=lambda c: c.nranks))
            current = [case]
            current_min = load
            current_max = load
    if len(current) >= 2:
        groups.append(sorted(current, key=lambda c: c.nranks))

    labeled: list[tuple[str, list[CaseSummary]]] = []
    for group in groups:
        mean_load = sum(case.local_ntracer_mean for case in group) / len(group)
        labeled.append((f"~{mean_load:.0f} tracers/rank", group))
    return labeled


def load_case(run_dir: Path) -> CaseSummary:
    timing_files = sorted(run_dir.glob("timing_core*.tsv"))
    if not timing_files:
        raise FileNotFoundError(f"no timing_core*.tsv found in {run_dir}")

    rows: list[dict[str, str]] = []
    for path in timing_files:
        with path.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if not row or row.get("wall_ms") in (None, ""):
                    continue
                rows.append(row)

    if not rows:
        raise ValueError(f"no timing rows found in {run_dir}")

    run_summary = read_run_summary(run_dir / "run_summary.tsv")

    wall = sorted(float(r["wall_ms"]) for r in rows)
    local_tp = sorted(float(r["local_throughput_tracers_per_s"]) for r in rows)
    local_ntracer = sorted(int(r["local_ntracer"]) for r in rows)

    heavy_count, heavy_sum_min, heavy_sum_mean, heavy_sum_max = read_heavy_summary(
        run_dir / "heavy_summary.txt"
    )
    warnings: list[str] = []
    expected_ranks = infer_expected_ranks_from_run_tag(infer_run_tag(run_dir))
    summary_nranks = int(run_summary["mpi_size"]) if run_summary is not None else None
    timing_file_count = len(timing_files)
    resolved_nranks = summary_nranks if summary_nranks is not None else timing_file_count

    if summary_nranks is not None and summary_nranks != timing_file_count:
        warnings.append(
            f"run_summary mpi_size={summary_nranks} but found {timing_file_count} timing_core files"
        )
    if expected_ranks is not None and resolved_nranks != expected_ranks:
        warnings.append(
            f"run tag suggests {expected_ranks} ranks but resolved nranks={resolved_nranks}"
        )

    return CaseSummary(
        label=infer_label(run_dir),
        run_tag=infer_run_tag(run_dir),
        run_dir=run_dir,
        nranks=resolved_nranks,
        timing_file_count=timing_file_count,
        global_ntracer=int(rows[0]["global_ntracer"]),
        local_ntracer_min=min(local_ntracer),
        local_ntracer_mean=sum(local_ntracer) / len(local_ntracer),
        local_ntracer_max=max(local_ntracer),
        wall_ms_min=min(wall),
        wall_ms_mean=sum(wall) / len(wall),
        wall_ms_max=max(wall),
        wall_ms_p10=percentile(wall, 0.10),
        wall_ms_p50=percentile(wall, 0.50),
        wall_ms_p90=percentile(wall, 0.90),
        local_tp_min=min(local_tp),
        local_tp_mean=sum(local_tp) / len(local_tp),
        local_tp_max=max(local_tp),
        local_tp_p10=percentile(local_tp, 0.10),
        local_tp_p50=percentile(local_tp, 0.50),
        local_tp_p90=percentile(local_tp, 0.90),
        global_throughput=float(rows[0]["global_throughput_tracers_per_s"]),
        coeff_ms_mean=sum(float(r["coeff_ms"]) for r in rows) / len(rows),
        solve_ms_mean=sum(float(r["solve_ms"]) for r in rows) / len(rows),
        synch_ms_mean=sum(float(r["synch_ms"]) for r in rows) / len(rows),
        total_staged_ms_mean=sum(float(r["total_staged_ms"]) for r in rows) / len(rows),
        heavy_count=heavy_count,
        heavy_sum_nsub_min=heavy_sum_min,
        heavy_sum_nsub_mean=heavy_sum_mean,
        heavy_sum_nsub_max=heavy_sum_max,
        warnings=warnings,
    )


def write_summary_csv(cases: Iterable[CaseSummary], out_path: Path) -> None:
    fields = [
        "label",
        "run_tag",
        "run_dir",
        "nranks",
        "timing_file_count",
        "global_ntracer",
        "local_ntracer_min",
        "local_ntracer_mean",
        "local_ntracer_max",
        "wall_ms_min",
        "wall_ms_mean",
        "wall_ms_max",
        "wall_ms_p10",
        "wall_ms_p50",
        "wall_ms_p90",
        "per_rank_tp_min",
        "per_rank_tp_mean",
        "per_rank_tp_max",
        "per_rank_tp_p10",
        "per_rank_tp_p50",
        "per_rank_tp_p90",
        "global_throughput",
        "coeff_ms_mean",
        "solve_ms_mean",
        "synch_ms_mean",
        "total_staged_ms_mean",
        "heavy_count",
        "heavy_sum_nsub_min",
        "heavy_sum_nsub_mean",
        "heavy_sum_nsub_max",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "label": case.label,
                    "run_tag": case.run_tag,
                    "run_dir": str(case.run_dir),
                    "nranks": case.nranks,
                    "timing_file_count": case.timing_file_count,
                    "global_ntracer": case.global_ntracer,
                    "local_ntracer_min": case.local_ntracer_min,
                    "local_ntracer_mean": f"{case.local_ntracer_mean:.6f}",
                    "local_ntracer_max": case.local_ntracer_max,
                    "wall_ms_min": f"{case.wall_ms_min:.6f}",
                    "wall_ms_mean": f"{case.wall_ms_mean:.6f}",
                    "wall_ms_max": f"{case.wall_ms_max:.6f}",
                    "wall_ms_p10": f"{case.wall_ms_p10:.6f}",
                    "wall_ms_p50": f"{case.wall_ms_p50:.6f}",
                    "wall_ms_p90": f"{case.wall_ms_p90:.6f}",
                    "per_rank_tp_min": f"{case.local_tp_min:.6f}",
                    "per_rank_tp_mean": f"{case.local_tp_mean:.6f}",
                    "per_rank_tp_max": f"{case.local_tp_max:.6f}",
                    "per_rank_tp_p10": f"{case.local_tp_p10:.6f}",
                    "per_rank_tp_p50": f"{case.local_tp_p50:.6f}",
                    "per_rank_tp_p90": f"{case.local_tp_p90:.6f}",
                    "global_throughput": f"{case.global_throughput:.6f}",
                    "coeff_ms_mean": f"{case.coeff_ms_mean:.6f}",
                    "solve_ms_mean": f"{case.solve_ms_mean:.6f}",
                    "synch_ms_mean": f"{case.synch_ms_mean:.6f}",
                    "total_staged_ms_mean": f"{case.total_staged_ms_mean:.6f}",
                    "heavy_count": "" if case.heavy_count is None else case.heavy_count,
                    "heavy_sum_nsub_min": "" if case.heavy_sum_nsub_min is None else case.heavy_sum_nsub_min,
                    "heavy_sum_nsub_mean": "" if case.heavy_sum_nsub_mean is None else f"{case.heavy_sum_nsub_mean:.6f}",
                    "heavy_sum_nsub_max": "" if case.heavy_sum_nsub_max is None else case.heavy_sum_nsub_max,
                }
            )


def write_text_summary(cases: list[CaseSummary], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("Tracer scaling summary")
    lines.append("")

    for case in cases:
        lines.append(f"[{case.label} / {case.run_tag}]")
        lines.append(f"run_dir = {case.run_dir}")
        lines.append(f"nranks = {case.nranks}")
        lines.append(f"timing_file_count = {case.timing_file_count}")
        lines.append(f"global_ntracer = {case.global_ntracer}")
        lines.append(
            "wall_time_s min/mean/max = "
            f"{case.wall_ms_min/1.0e3:.3f} / {case.wall_ms_mean/1.0e3:.3f} / {case.wall_ms_max/1.0e3:.3f}"
        )
        lines.append(
            "per_rank_throughput min/mean/max = "
            f"{case.local_tp_min:.6f} / {case.local_tp_mean:.6f} / {case.local_tp_max:.6f}"
        )
        lines.append(f"global_throughput = {case.global_throughput:.6f}")
        lines.append(
            "solve_ms_mean / synch_ms_mean / total_staged_ms_mean = "
            f"{case.solve_ms_mean:.6f} / {case.synch_ms_mean:.6f} / {case.total_staged_ms_mean:.6f}"
        )
        lines.append(
            "local_ntracer min/mean/max = "
            f"{case.local_ntracer_min} / {case.local_ntracer_mean:.3f} / {case.local_ntracer_max}"
        )
        if case.heavy_count is not None:
            lines.append(f"heavy_count = {case.heavy_count}")
        if case.heavy_sum_nsub_min is not None and case.heavy_sum_nsub_max is not None:
            ratio = (
                case.heavy_sum_nsub_max / case.heavy_sum_nsub_min
                if case.heavy_sum_nsub_min > 0
                else math.nan
            )
            lines.append(
                "heavy_sum_nsub min/mean/max = "
                f"{case.heavy_sum_nsub_min} / "
                f"{case.heavy_sum_nsub_mean:.3f} / "
                f"{case.heavy_sum_nsub_max}"
            )
            lines.append(f"heavy_sum_nsub max/min = {ratio:.6f}")
        for warning in case.warnings:
            lines.append(f"warning = {warning}")
        lines.append("")

    groups: dict[int, list[CaseSummary]] = {}
    for case in cases:
        groups.setdefault(case.global_ntracer, []).append(case)

    scalable_groups = [sorted(group, key=lambda c: c.nranks) for group in groups.values() if len(group) >= 2]
    if scalable_groups:
        lines.append("Rank scaling")
        lines.append("")
        for group in scalable_groups:
            base = group[0]
            lines.append(f"[{base.label}]")
            lines.append(f"reference = {base.run_tag} ({base.nranks} ranks)")
            for case in group:
                ideal = base.global_throughput * (case.nranks / base.nranks)
                efficiency = case.global_throughput / ideal if ideal > 0.0 else math.nan
                lines.append(
                    f"{case.run_tag}: nranks={case.nranks} "
                    f"global_throughput={case.global_throughput:.6f} "
                    f"wall_time_max_s={case.wall_ms_max/1.0e3:.3f} "
                    f"efficiency={efficiency:.6f}"
                )
            lines.append("")

    out_path.write_text("\n".join(lines))


def plot_problem_size(cases: list[CaseSummary], output_dir: Path, title: str) -> None:
    cases = cases_for_problem_size_plots(cases)
    if not cases:
        return
    groups: dict[str, list[CaseSummary]] = {}
    for case in cases:
        groups.setdefault(case.run_tag, []).append(case)
    ordered_groups = sorted(groups.items())

    if HAVE_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        colors = ["#0b6e4f", "#355070", "#c84c09", "#6d597a", "#264653"]

        for idx, (run_tag, group) in enumerate(ordered_groups):
            color = colors[idx % len(colors)]
            legend_label = display_run_label(run_tag)
            group = sorted(group, key=lambda c: c.global_ntracer)
            x = [c.global_ntracer for c in group]
            axes[0, 0].plot(x, [c.global_throughput for c in group], marker="o", linewidth=2,
                            color=color, label=legend_label)
            axes[0, 1].plot(x, [c.wall_ms_max / 1.0e3 for c in group], marker="o", linewidth=2,
                            color=color, label=legend_label)
            axes[1, 0].errorbar(
                x,
                [c.local_tp_p50 for c in group],
                yerr=[
                    [c.local_tp_p50 - c.local_tp_p10 for c in group],
                    [c.local_tp_p90 - c.local_tp_p50 for c in group],
                ],
                fmt="o-",
                linewidth=2,
                capsize=4,
                color=color,
                label=legend_label,
            )
            axes[1, 1].errorbar(
                x,
                [c.wall_ms_p50 / 1.0e3 for c in group],
                yerr=[
                    [(c.wall_ms_p50 - c.wall_ms_p10) / 1.0e3 for c in group],
                    [(c.wall_ms_p90 - c.wall_ms_p50) / 1.0e3 for c in group],
                ],
                fmt="o-",
                linewidth=2,
                capsize=4,
                color=color,
                label=legend_label,
            )
        axes[0, 0].set_title("Global Throughput")
        axes[0, 0].set_xlabel("Global Tracer Count")
        axes[0, 0].set_ylabel("tracer / s")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()
        axes[0, 1].set_title("Critical-Path Wall Time")
        axes[0, 1].set_xlabel("Global Tracer Count")
        axes[0, 1].set_ylabel("max wall time [s]")
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].legend()
        axes[1, 0].set_title("Per-Rank Throughput Spread")
        axes[1, 0].set_xlabel("Global Tracer Count")
        axes[1, 0].set_ylabel("per-rank tracer / s")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()
        axes[1, 1].set_title("Wall-Time Spread")
        axes[1, 1].set_xlabel("Global Tracer Count")
        axes[1, 1].set_ylabel("per-rank wall time [s]")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()

        fig.suptitle(title)
        fig.savefig(output_plot_path(output_dir, "throughput_vs_problem_size"), dpi=180)
        plt.close(fig)
        return

    colors = ["#0b6e4f", "#355070", "#c84c09", "#6d597a", "#264653"]
    throughput_series = []
    wall_series = []
    local_tp_series = []
    wall_spread_series = []
    for idx, (run_tag, group) in enumerate(ordered_groups):
        color = colors[idx % len(colors)]
        legend_label = display_run_label(run_tag)
        group = sorted(group, key=lambda c: c.global_ntracer)
        x = [c.global_ntracer for c in group]
        throughput_series.append(
            {"label": legend_label, "x": x, "y": [c.global_throughput for c in group], "color": color}
        )
        wall_series.append(
            {"label": legend_label, "x": x, "y": [c.wall_ms_max / 1.0e3 for c in group], "color": color}
        )
        local_tp_series.append(
            {
                "label": legend_label,
                "x": x,
                "y": [c.local_tp_p50 for c in group],
                "y_lo": [c.local_tp_p10 for c in group],
                "y_hi": [c.local_tp_p90 for c in group],
                "color": color,
            }
        )
        wall_spread_series.append(
            {
                "label": legend_label,
                "x": x,
                "y": [c.wall_ms_p50 / 1.0e3 for c in group],
                "y_lo": [c.wall_ms_p10 / 1.0e3 for c in group],
                "y_hi": [c.wall_ms_p90 / 1.0e3 for c in group],
                "color": color,
            }
        )

    write_simple_svg_line_chart(
        output_plot_path(output_dir, "global_throughput_vs_problem_size"),
        f"{title}: Global Throughput",
        "Global Tracer Count",
        "tracer / s",
        throughput_series,
    )
    write_simple_svg_line_chart(
        output_plot_path(output_dir, "walltime_vs_problem_size"),
        f"{title}: Critical-Path Wall Time",
        "Global Tracer Count",
        "max wall time [s]",
        wall_series,
    )
    write_simple_svg_line_chart(
        output_plot_path(output_dir, "local_throughput_spread"),
        f"{title}: Per-Rank Throughput Spread",
        "Global Tracer Count",
        "per-rank tracer / s",
        local_tp_series,
    )
    write_simple_svg_line_chart(
        output_plot_path(output_dir, "walltime_spread"),
        f"{title}: Wall-Time Spread",
        "Global Tracer Count",
        "per-rank wall time [s]",
        wall_spread_series,
    )


def plot_rank_scaling(cases: list[CaseSummary], output_dir: Path, title: str) -> None:
    groups: dict[int, list[CaseSummary]] = {}
    for case in cases:
        groups.setdefault(case.global_ntracer, []).append(case)

    scalable_groups = [sorted(group, key=lambda c: c.nranks) for group in groups.values() if len(group) >= 2]
    if not scalable_groups:
        return

    if HAVE_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

        for group in scalable_groups:
            label = group[0].label
            x = [c.nranks for c in group]
            y = [c.global_throughput for c in group]
            base = y[0]
            ideal = [base * (nr / x[0]) for nr in x]
            eff = [actual / ideal_val if ideal_val > 0.0 else math.nan for actual, ideal_val in zip(y, ideal)]

            axes[0].plot(x, y, marker="o", linewidth=2, label=label)
            axes[0].plot(x, ideal, linestyle="--", linewidth=1.2, alpha=0.5)
            axes[1].plot(x, eff, marker="o", linewidth=2, label=label)

        axes[0].set_title("Global Throughput vs MPI Ranks")
        axes[0].set_xlabel("MPI ranks")
        axes[0].set_ylabel("tracer / s")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].set_title("Parallel Efficiency")
        axes[1].set_xlabel("MPI ranks")
        axes[1].set_ylabel("efficiency")
        axes[1].set_ylim(0.0, 1.1)
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle(f"{title}: Rank Scaling")
        fig.savefig(output_plot_path(output_dir, "throughput_vs_ranks"), dpi=180)
        plt.close(fig)
        return

    throughput_series = []
    efficiency_series = []
    colors = ["#0b6e4f", "#355070", "#c84c09", "#6d597a", "#264653"]
    for idx, group in enumerate(scalable_groups):
        label = group[0].label
        x = [c.nranks for c in group]
        y = [c.global_throughput for c in group]
        base = y[0]
        ideal = [base * (nr / x[0]) for nr in x]
        eff = [actual / ideal_val if ideal_val > 0.0 else math.nan for actual, ideal_val in zip(y, ideal)]
        color = colors[idx % len(colors)]
        throughput_series.append({"label": label, "x": x, "y": y, "color": color})
        efficiency_series.append({"label": label, "x": x, "y": eff, "color": color})

    write_simple_svg_line_chart(
        output_plot_path(output_dir, "throughput_vs_ranks"),
        f"{title}: Rank Scaling",
        "MPI ranks",
        "tracer / s",
        throughput_series,
    )
    write_simple_svg_line_chart(
        output_plot_path(output_dir, "parallel_efficiency_vs_ranks"),
        f"{title}: Parallel Efficiency",
        "MPI ranks",
        "efficiency",
        efficiency_series,
    )


def plot_load_balance(cases: list[CaseSummary], output_dir: Path, title: str) -> None:
    lb_cases = [c for c in cases if c.heavy_count is not None and c.heavy_sum_nsub_min is not None]
    if not lb_cases:
        return

    groups: dict[str, list[CaseSummary]] = {}
    for case in lb_cases:
        groups.setdefault(case.run_tag, []).append(case)
    ordered_groups = sorted(groups.items())

    if HAVE_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        colors = ["#264653", "#355070", "#c84c09", "#6d597a", "#0b6e4f"]
        for idx, (run_tag, group) in enumerate(ordered_groups):
            color = colors[idx % len(colors)]
            legend_label = display_run_label(run_tag)
            group = sorted(group, key=lambda c: c.global_ntracer)
            x = [c.global_ntracer for c in group]
            y = [c.local_ntracer_mean for c in group]
            ymin = [c.local_ntracer_min for c in group]
            ymax = [c.local_ntracer_max for c in group]
            heavy_imbalance = [heavy_imbalance_ratio(c) for c in group]
            axes[0].errorbar(
                x,
                y,
                yerr=[
                    [mean - lo for mean, lo in zip(y, ymin)],
                    [hi - mean for mean, hi in zip(y, ymax)],
                ],
                fmt="o-",
                linewidth=2,
                capsize=4,
                color=color,
                label=legend_label,
            )
            axes[1].plot(x, heavy_imbalance, marker="o", linewidth=2, color=color, label=legend_label)
        axes[0].set_title("Final Local Tracer Count Spread")
        axes[0].set_xlabel("Global Tracer Count")
        axes[0].set_ylabel("local_ntracer")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        axes[1].set_title("Heavy-Load Imbalance")
        axes[1].set_xlabel("Global Tracer Count")
        axes[1].set_ylabel("max/min heavy_sum_nsub")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle(f"{title}: Load Balancing")
        fig.savefig(output_plot_path(output_dir, "load_balance_summary"), dpi=180)
        plt.close(fig)
        return

    colors = ["#264653", "#355070", "#c84c09", "#6d597a", "#0b6e4f"]
    local_ntracer_series = []
    heavy_imbalance_series = []
    for idx, (run_tag, group) in enumerate(ordered_groups):
        color = colors[idx % len(colors)]
        legend_label = display_run_label(run_tag)
        group = sorted(group, key=lambda c: c.global_ntracer)
        x = [c.global_ntracer for c in group]
        local_ntracer_series.append(
            {
                "label": legend_label,
                "x": x,
                "y": [c.local_ntracer_mean for c in group],
                "y_lo": [c.local_ntracer_min for c in group],
                "y_hi": [c.local_ntracer_max for c in group],
                "color": color,
            }
        )
        heavy_imbalance_series.append(
            {
                "label": legend_label,
                "x": x,
                "y": [heavy_imbalance_ratio(c) for c in group],
                "color": color,
            }
        )
    write_simple_svg_line_chart(
        output_plot_path(output_dir, "local_ntracer_spread"),
        f"{title}: Final Local Tracer Count Spread",
        "Global Tracer Count",
        "local_ntracer",
        local_ntracer_series,
    )
    write_simple_svg_line_chart(
        output_plot_path(output_dir, "heavy_imbalance"),
        f"{title}: Heavy-Load Imbalance",
        "Global Tracer Count",
        "max/min heavy_sum_nsub",
        heavy_imbalance_series,
    )


def plot_scaling_verification(cases: list[CaseSummary], output_dir: Path, title: str) -> None:
    strong_cases = strong_scaling_cases(cases, "10k")
    weak_groups = weak_scaling_groups(cases)

    if HAVE_MPL:
        if strong_cases:
            ref = strong_cases[0]
            x = [case.nranks for case in strong_cases]
            speedup = [ref.wall_ms_max / case.wall_ms_max for case in strong_cases]
            ideal = [case.nranks / ref.nranks for case in strong_cases]
            eff = [s / i if i > 0.0 else math.nan for s, i in zip(speedup, ideal)]
            amdahl_f = fit_amdahl_serial_fraction(x, speedup)
            x_fit = sorted(set(x + [v for v in range(min(x), max(x) + 1) if v > 0]))
            amdahl_fit = [amdahl_relative_speedup(amdahl_f, float(v), float(ref.nranks)) for v in x_fit]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
            axes[0].plot(x, speedup, marker="o", linewidth=2, color="#0b6e4f", label="measured")
            axes[0].plot(x, ideal, linestyle="--", linewidth=1.5, color="#c84c09", label="ideal")
            axes[0].plot(x_fit, amdahl_fit, linewidth=1.8, color="#355070",
                         label=f"Amdahl fit (f={amdahl_f:.3f})")
            axes[0].set_title("Strong Scaling Speedup (10k)")
            axes[0].set_xlabel("MPI ranks")
            axes[0].set_ylabel("speedup")
            axes[0].set_xscale("log")
            axes[0].grid(alpha=0.3)
            axes[0].legend()

            axes[1].plot(x, eff, marker="o", linewidth=2, color="#355070")
            axes[1].set_title("Strong Scaling Efficiency (10k)")
            axes[1].set_xlabel("MPI ranks")
            axes[1].set_ylabel("efficiency")
            axes[1].set_xscale("log")
            axes[1].set_ylim(0.0, 1.1)
            axes[1].grid(alpha=0.3)

            fig.suptitle(f"{title}: Strong Scaling Verification")
            fig.savefig(output_plot_path(output_dir, "strong_scaling_10k"), dpi=180)
            plt.close(fig)

        if weak_groups:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
            colors = ["#0b6e4f", "#355070", "#c84c09", "#6d597a", "#264653"]
            for idx, (group_label, group) in enumerate(weak_groups):
                color = colors[idx % len(colors)]
                ref = group[0]
                x = [case.nranks for case in group]
                wall = [case.wall_ms_max / 1.0e3 for case in group]
                normalized = [case.wall_ms_max / ref.wall_ms_max for case in group]
                axes[0].plot(x, wall, marker="o", linewidth=2, color=color, label=group_label)
                axes[1].plot(x, normalized, marker="o", linewidth=2, color=color, label=group_label)

            axes[0].set_title("Weak Scaling Wall Time")
            axes[0].set_xlabel("MPI ranks")
            axes[0].set_ylabel("max wall time [s]")
            axes[0].grid(alpha=0.3)
            axes[0].legend()

            axes[1].axhline(1.0, linestyle="--", linewidth=1.2, color="#444444", alpha=0.7)
            axes[1].set_title("Weak Scaling Normalized Wall Time")
            axes[1].set_xlabel("MPI ranks")
            axes[1].set_ylabel("T / T_ref")
            axes[1].grid(alpha=0.3)
            axes[1].legend()

            fig.suptitle(f"{title}: Weak Scaling Verification")
            fig.savefig(output_plot_path(output_dir, "weak_scaling"), dpi=180)
            plt.close(fig)
        return

    if strong_cases:
        ref = strong_cases[0]
        x = [case.nranks for case in strong_cases]
        speedup = [ref.wall_ms_max / case.wall_ms_max for case in strong_cases]
        ideal = [case.nranks / ref.nranks for case in strong_cases]
        eff = [s / i if i > 0.0 else math.nan for s, i in zip(speedup, ideal)]
        amdahl_f = fit_amdahl_serial_fraction(x, speedup)
        x_fit = sorted(set(x + [v for v in range(min(x), max(x) + 1) if v > 0]))
        amdahl_fit = [amdahl_relative_speedup(amdahl_f, float(v), float(ref.nranks)) for v in x_fit]
        write_simple_svg_line_chart(
            output_plot_path(output_dir, "strong_scaling_10k_speedup"),
            f"{title}: Strong Scaling Speedup (10k)",
            "MPI ranks",
            "speedup",
            [
                {"label": "measured", "x": x, "y": speedup, "color": "#0b6e4f"},
                {"label": "ideal", "x": x, "y": ideal, "color": "#c84c09"},
                {"label": f"Amdahl fit (f={amdahl_f:.3f})", "x": x_fit, "y": amdahl_fit, "color": "#355070"},
            ],
            x_log=True,
        )
        write_simple_svg_line_chart(
            output_plot_path(output_dir, "strong_scaling_10k_efficiency"),
            f"{title}: Strong Scaling Efficiency (10k)",
            "MPI ranks",
            "efficiency",
            [{"label": "efficiency", "x": x, "y": eff, "color": "#355070"}],
            x_log=True,
        )

    if weak_groups:
        colors = ["#0b6e4f", "#355070", "#c84c09", "#6d597a", "#264653"]
        wall_series = []
        norm_series = []
        for idx, (group_label, group) in enumerate(weak_groups):
            color = colors[idx % len(colors)]
            ref = group[0]
            x = [case.nranks for case in group]
            wall_series.append(
                {"label": group_label, "x": x, "y": [case.wall_ms_max / 1.0e3 for case in group], "color": color}
            )
            norm_series.append(
                {"label": group_label, "x": x, "y": [case.wall_ms_max / ref.wall_ms_max for case in group], "color": color}
            )
        write_simple_svg_line_chart(
            output_plot_path(output_dir, "weak_scaling_walltime"),
            f"{title}: Weak Scaling Wall Time",
            "MPI ranks",
            "max wall time [s]",
            wall_series,
        )
        write_simple_svg_line_chart(
            output_plot_path(output_dir, "weak_scaling_normalized_walltime"),
            f"{title}: Weak Scaling Normalized Wall Time",
            "MPI ranks",
            "T / T_ref",
            norm_series,
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(run_dir) for run_dir in args.run_dirs] if args.run_dirs else discover_default_run_dirs()
    if not run_dirs:
        raise FileNotFoundError("no timing data found under the hard-coded test_out sweep paths")

    cases = [load_case(run_dir) for run_dir in run_dirs]
    cases = sorted(cases, key=lambda c: (c.run_tag, c.global_ntracer, c.nranks))
    write_summary_csv(cases, output_dir / "tracer_scaling_summary.csv")
    write_text_summary(cases, output_dir / "tracer_scaling_summary.txt")
    plot_problem_size(cases, output_dir, args.title)
    plot_rank_scaling(cases, output_dir, args.title)
    plot_load_balance(cases, output_dir, args.title)
    plot_scaling_verification(cases, output_dir, args.title)

    print(f"Wrote summary CSV to {output_dir / 'tracer_scaling_summary.csv'}")
    print(f"Wrote summary TXT to {output_dir / 'tracer_scaling_summary.txt'}")
    print(f"Wrote plots under {output_dir}")
    print("Used run directories:")
    for run_dir in run_dirs:
        print(f"  {run_dir}")


if __name__ == "__main__":
    main()

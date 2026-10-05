#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ModuleNotFoundError:
    plt = None
    HAVE_MPL = False


DEFAULT_INPUT = Path("test_out/bucketstats_example/load_balance_rank_totals.txt")
DEFAULT_OUTPUT_DIR = Path("test_out/bucketstats_example/plots")
DEFAULT_TITLE = "Example 100k: virtual 10-rank total nsub"


@dataclass
class RankLoad:
    rank: int
    base_count: int
    final_count: int
    base_sum_nsub: int
    final_sum_nsub: int


def load_rank_totals(path: Path) -> list[RankLoad]:
    rows: list[RankLoad] = []
    with path.open() as fh:
        next(fh)
        for line in fh:
            parts = line.split()
            if len(parts) < 9:
                continue
            rows.append(
                RankLoad(
                    rank=int(parts[0]),
                    base_count=int(parts[1]),
                    final_count=int(parts[4]),
                    base_sum_nsub=int(parts[5]),
                    final_sum_nsub=int(parts[8]),
                )
            )
    if not rows:
        raise ValueError(f"no rank rows found in {path}")
    return rows


def imbalance_ratio(values: list[int]) -> float:
    vmin = min(values)
    vmax = max(values)
    if vmin <= 0:
        return math.nan
    return vmax / vmin


def write_summary_csv(rows: list[RankLoad], out_path: Path) -> None:
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["rank", "base_count", "final_count", "base_sum_nsub", "final_sum_nsub"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row.rank,
                    "base_count": row.base_count,
                    "final_count": row.final_count,
                    "base_sum_nsub": row.base_sum_nsub,
                    "final_sum_nsub": row.final_sum_nsub,
                }
            )


def write_simple_svg(rows: list[RankLoad], out_path: Path, title: str) -> None:
    width = 960
    height = 720
    left = 80
    right = 30
    top = 60
    bottom = 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    ranks = [row.rank for row in rows]
    base = [row.base_sum_nsub for row in rows]
    final = [row.final_sum_nsub for row in rows]
    ymax = max(max(base), max(final)) * 1.08
    bar_slot = plot_w / max(len(rows), 1)
    bar_w = min(26.0, 0.38 * bar_slot)

    def sy(val: float) -> float:
        return top + plot_h - (val / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcf8"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="22" font-family="Helvetica">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222" stroke-width="1.5"/>',
    ]

    for i in range(5):
        frac = i / 4.0
        val = frac * ymax
        yp = sy(val)
        parts.append(f'<line x1="{left}" y1="{yp:.2f}" x2="{left+plot_w}" y2="{yp:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-8}" y="{yp+4:.2f}" text-anchor="end" font-size="12" font-family="Helvetica">{val/1.0e6:.1f}M</text>')

    for idx, row in enumerate(rows):
        cx = left + (idx + 0.5) * bar_slot
        x0 = cx - bar_w - 2
        x1 = cx + 2
        yb = sy(row.base_sum_nsub)
        yf = sy(row.final_sum_nsub)
        parts.append(f'<rect x="{x0:.2f}" y="{yb:.2f}" width="{bar_w:.2f}" height="{top+plot_h-yb:.2f}" fill="#d97b29"/>')
        parts.append(f'<rect x="{x1:.2f}" y="{yf:.2f}" width="{bar_w:.2f}" height="{top+plot_h-yf:.2f}" fill="#1f6f8b"/>')
        parts.append(f'<text x="{cx:.2f}" y="{top+plot_h+20}" text-anchor="middle" font-size="12" font-family="Helvetica">{row.rank}</text>')

    parts.append(f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="15" font-family="Helvetica">rank</text>')
    parts.append(f'<text x="24" y="{height/2:.1f}" text-anchor="middle" font-size="15" font-family="Helvetica" transform="rotate(-90 24,{height/2:.1f})">total nsub</text>')
    parts.append(f'<rect x="{left+10}" y="{top+8}" width="16" height="10" fill="#d97b29"/>')
    parts.append(f'<text x="{left+32}" y="{top+17}" font-size="12" font-family="Helvetica">without LB (base_sum_nsub)</text>')
    parts.append(f'<rect x="{left+250}" y="{top+8}" width="16" height="10" fill="#1f6f8b"/>')
    parts.append(f'<text x="{left+272}" y="{top+17}" font-size="12" font-family="Helvetica">with LB (final_sum_nsub)</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def write_simple_svg_single(
    rows: list[RankLoad],
    out_path: Path,
    title: str,
    values: list[int],
    color: str,
    legend: str,
    ymax: float,
) -> None:
    width = 960
    height = 720
    left = 80
    right = 30
    top = 60
    bottom = 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_slot = plot_w / max(len(rows), 1)
    bar_w = min(48.0, 0.62 * bar_slot)

    def sy(val: float) -> float:
        return top + plot_h - (val / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcf8"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="22" font-family="Helvetica">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222" stroke-width="1.5"/>',
    ]

    for i in range(5):
        frac = i / 4.0
        val = frac * ymax
        yp = sy(val)
        parts.append(f'<line x1="{left}" y1="{yp:.2f}" x2="{left+plot_w}" y2="{yp:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-8}" y="{yp+4:.2f}" text-anchor="end" font-size="12" font-family="Helvetica">{val/1.0e6:.1f}M</text>')

    for idx, row in enumerate(rows):
        cx = left + (idx + 0.5) * bar_slot
        x0 = cx - bar_w / 2
        y0 = sy(values[idx])
        parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" height="{top+plot_h-y0:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{cx:.2f}" y="{top+plot_h+20}" text-anchor="middle" font-size="12" font-family="Helvetica">{row.rank}</text>')

    parts.append(f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="15" font-family="Helvetica">rank</text>')
    parts.append(f'<text x="24" y="{height/2:.1f}" text-anchor="middle" font-size="15" font-family="Helvetica" transform="rotate(-90 24,{height/2:.1f})">total nsub</text>')
    parts.append(f'<rect x="{left+10}" y="{top+8}" width="16" height="10" fill="{color}"/>')
    parts.append(f'<text x="{left+32}" y="{top+17}" font-size="12" font-family="Helvetica">{legend}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def plot(rows: list[RankLoad], out_dir: Path, title: str) -> None:
    ranks = [row.rank for row in rows]
    base = [row.base_sum_nsub for row in rows]
    final = [row.final_sum_nsub for row in rows]
    base_ratio = imbalance_ratio(base)
    final_ratio = imbalance_ratio(final)
    ymax = max(max(base), max(final)) * 1.08

    if HAVE_MPL:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), constrained_layout=True, height_ratios=[3.0, 1.5])

        x = list(range(len(rows)))
        width = 0.42
        axes[0].bar([v - width / 2 for v in x], base, width=width, color="#d97b29", label="without LB")
        axes[0].bar([v + width / 2 for v in x], final, width=width, color="#1f6f8b", label="with LB")
        axes[0].set_xticks(x, [str(r) for r in ranks])
        axes[0].set_xlabel("rank")
        axes[0].set_ylabel("total nsub")
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].text(
            0.01,
            0.98,
            f"max/min without LB = {base_ratio:.3f}\nmax/min with LB = {final_ratio:.3f}",
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

        axes[1].plot(ranks, base, marker="o", linewidth=2, color="#d97b29", label="without LB")
        axes[1].plot(ranks, final, marker="o", linewidth=2, color="#1f6f8b", label="with LB")
        axes[1].set_xlabel("rank")
        axes[1].set_ylabel("total nsub")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.savefig(out_dir / "virtual_10rank_total_nsub.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), constrained_layout=True)
        ax.bar(ranks, base, color="#d97b29")
        ax.set_xlabel("rank")
        ax.set_ylabel("total nsub")
        ax.set_title(f"{title}: without LB")
        ax.set_ylim(0.0, ymax)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.01, 0.98, f"max/min = {base_ratio:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
        fig.savefig(out_dir / "virtual_10rank_total_nsub_before_lb.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), constrained_layout=True)
        ax.bar(ranks, final, color="#1f6f8b")
        ax.set_xlabel("rank")
        ax.set_ylabel("total nsub")
        ax.set_title(f"{title}: with LB")
        ax.set_ylim(0.0, ymax)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.01, 0.98, f"max/min = {final_ratio:.6f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"})
        fig.savefig(out_dir / "virtual_10rank_total_nsub_after_lb.png", dpi=180)
        plt.close(fig)
    else:
        write_simple_svg(rows, out_dir / "virtual_10rank_total_nsub.svg", title)
        write_simple_svg_single(
            rows,
            out_dir / "virtual_10rank_total_nsub_before_lb.svg",
            f"{title}: without LB",
            base,
            "#d97b29",
            "without LB (base_sum_nsub)",
            ymax,
        )
        write_simple_svg_single(
            rows,
            out_dir / "virtual_10rank_total_nsub_after_lb.svg",
            f"{title}: with LB",
            final,
            "#1f6f8b",
            "with LB (final_sum_nsub)",
            ymax,
        )


def main() -> int:
    input_path = DEFAULT_INPUT
    output_dir = DEFAULT_OUTPUT_DIR
    title = DEFAULT_TITLE

    if not input_path.exists():
        raise SystemExit(
            f"missing input: {input_path}\n"
            "place a 10-rank load_balance_rank_totals.txt there, then rerun this script"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rank_totals(input_path)
    rows.sort(key=lambda row: row.rank)

    write_summary_csv(rows, output_dir / "virtual_10rank_total_nsub.csv")
    plot(rows, output_dir, title)

    base = [row.base_sum_nsub for row in rows]
    final = [row.final_sum_nsub for row in rows]
    print(f"input                = {input_path}")
    print(f"nranks               = {len(rows)}")
    print(f"base max/min         = {imbalance_ratio(base):.6f}")
    print(f"balanced max/min     = {imbalance_ratio(final):.6f}")
    print(f"summary csv          = {output_dir / 'virtual_10rank_total_nsub.csv'}")
    if HAVE_MPL:
        print(f"plot                 = {output_dir / 'virtual_10rank_total_nsub.png'}")
        print(f"plot before LB       = {output_dir / 'virtual_10rank_total_nsub_before_lb.png'}")
        print(f"plot after LB        = {output_dir / 'virtual_10rank_total_nsub_after_lb.png'}")
    else:
        print(f"plot                 = {output_dir / 'virtual_10rank_total_nsub.svg'}")
        print(f"plot before LB       = {output_dir / 'virtual_10rank_total_nsub_before_lb.svg'}")
        print(f"plot after LB        = {output_dir / 'virtual_10rank_total_nsub_after_lb.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

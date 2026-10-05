#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate bucketstats top-tracer TSV into heavy tracer ID lists."
    )
    parser.add_argument("input_tsv", type=Path, help="bucketstats_top_tracers.tsv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for output files (default: input file directory)",
    )
    parser.add_argument(
        "--min-heavy-fraction",
        type=float,
        default=0.30,
        help="minimum fraction of snapshots in which a tracer must appear in the top set",
    )
    parser.add_argument(
        "--min-heavy-count",
        type=int,
        default=None,
        help="minimum number of heavy snapshots; overrides --min-heavy-fraction if larger",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("count", "sum_nsub"),
        default="count",
        help="heavy-tracer selection rule",
    )
    parser.add_argument(
        "--heavy-nsub-share",
        type=float,
        default=0.30,
        help="for --selection-mode=sum_nsub, select top tracers until this fraction of total sum_nsub is covered",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_tsv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    by_tracer: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "max_nsub": 0.0,
            "sum_nsub": 0.0,
            "sum_target_nsub": 0.0,
            "sum_inflate_sum": 0.0,
            "avg_nsub_sum": 0.0,
            "avg_target_sum": 0.0,
            "avg_inflate_sum": 0.0,
            "avg_ngas_sum": 0.0,
            "max_ngas": 0.0,
        }
    )
    snapshot_ids: set[int] = set()

    with args.input_tsv.open("r", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            snapshot = int(row["snapshot_index"])
            tracer_id = int(row["tracer_id"])
            nsub = float(row["nsub"])
            target = float(row["target_nsub"])
            inflate = float(row["inflate"])
            n_gas = float(row["n_gas"])

            snapshot_ids.add(snapshot)
            rec = by_tracer[tracer_id]
            rec["count"] += 1
            rec["max_nsub"] = max(rec["max_nsub"], nsub)
            rec["sum_nsub"] += nsub
            rec["sum_target_nsub"] += target
            rec["sum_inflate_sum"] += max(nsub - target, 0.0)
            rec["avg_nsub_sum"] += nsub
            rec["avg_target_sum"] += target
            rec["avg_inflate_sum"] += inflate
            rec["avg_ngas_sum"] += n_gas
            rec["max_ngas"] = max(rec["max_ngas"], n_gas)

    total_snapshots = len(snapshot_ids)
    if total_snapshots == 0:
        raise SystemExit("no snapshot rows found")
    if not (0.0 < args.heavy_nsub_share <= 1.0):
        raise SystemExit("--heavy-nsub-share must be in (0, 1]")

    min_count = int(total_snapshots * args.min_heavy_fraction + 0.999999)
    if args.min_heavy_count is not None:
        min_count = max(min_count, args.min_heavy_count)
    if min_count < 1:
        min_count = 1

    rows = []
    heavy_ids = []
    for tracer_id, rec in by_tracer.items():
        count = int(rec["count"])
        frac = count / total_snapshots
        row = {
            "tracer_id": tracer_id,
            "heavy_snapshot_count": count,
            "heavy_snapshot_fraction": frac,
            "max_nsub": rec["max_nsub"],
            "sum_nsub": rec["sum_nsub"],
            "sum_target_nsub": rec["sum_target_nsub"],
            "sum_nsub_overhead": rec["sum_inflate_sum"],
            "avg_nsub": rec["avg_nsub_sum"] / count,
            "avg_target_nsub": rec["avg_target_sum"] / count,
            "avg_inflate": rec["avg_inflate_sum"] / count,
            "avg_n_gas": rec["avg_ngas_sum"] / count,
            "max_n_gas": rec["max_ngas"],
        }
        rows.append(row)

    if args.selection_mode == "count":
        for row in rows:
            if row["heavy_snapshot_count"] >= min_count:
                heavy_ids.append(row["tracer_id"])
        rows.sort(key=lambda r: (-r["heavy_snapshot_count"], -r["max_nsub"], -r["sum_nsub"], r["tracer_id"]))
        heavy_ids.sort()
    else:
        total_sum_nsub = sum(row["sum_nsub"] for row in rows)
        target_sum_nsub = args.heavy_nsub_share * total_sum_nsub
        running_sum_nsub = 0.0
        rows.sort(key=lambda r: (-r["sum_nsub"], -r["max_nsub"], -r["heavy_snapshot_count"], r["tracer_id"]))
        for row in rows:
            if running_sum_nsub >= target_sum_nsub:
                break
            heavy_ids.append(row["tracer_id"])
            running_sum_nsub += row["sum_nsub"]
        heavy_ids.sort()

    summary_path = output_dir / "heavy_tracer_summary.tsv"
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "tracer_id",
                "heavy_snapshot_count",
                "heavy_snapshot_fraction",
                "max_nsub",
                "sum_nsub",
                "sum_target_nsub",
                "sum_nsub_overhead",
                "avg_nsub",
                "avg_target_nsub",
                "avg_inflate",
                "avg_n_gas",
                "max_n_gas",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    ids_path = output_dir / "heavy_tracer_ids.txt"
    with ids_path.open("w") as fp:
        for tracer_id in heavy_ids:
            fp.write(f"{tracer_id}\n")

    print(f"total snapshots       = {total_snapshots}")
    print(f"candidate tracers     = {len(rows)}")
    if args.selection_mode == "count":
        print(f"selection mode        = count")
        print(f"min heavy count       = {min_count}")
    else:
        heavy_id_set = set(heavy_ids)
        selected_sum_nsub = sum(row["sum_nsub"] for row in rows if row["tracer_id"] in heavy_id_set)
        total_sum_nsub = sum(row["sum_nsub"] for row in rows)
        covered = selected_sum_nsub / total_sum_nsub if total_sum_nsub > 0.0 else 0.0
        print(f"selection mode        = sum_nsub")
        print(f"target nsub share     = {args.heavy_nsub_share:.6f}")
        print(f"covered nsub share    = {covered:.6f}")
    print(f"selected heavy tracers= {len(heavy_ids)}")
    print(f"summary               = {summary_path}")
    print(f"heavy ids             = {ids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

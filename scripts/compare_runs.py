#!/usr/bin/env python3
"""Compare proxy_eval tables of several runs, per object and overall (reads CSVs only; laptop-safe).

    python3 scripts/compare_runs.py results/object_v3_w5d03/proxy_eval.csv results/joint_v1/proxy_eval.csv [...]

A run's table is <run>/proxy_eval.csv or any CSV path. Episodes missing from a table are left out of every
column of the comparison (only episodes present in all tables count).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["pen", "gap", "acc_o", "iou_o", "dep_o", "acc_h", "iou_h", "dep_h"]


def load(path: str) -> tuple[str, dict[int, dict]]:
    p = Path(path)
    if p.is_dir():
        p = p / "proxy_eval.csv"
    rows = {int(r["episode"]): {k: float(v) if v not in ("", "nan") else float("nan") for k, v in r.items()
                                if k != "episode"} for r in csv.DictReader(p.open())}
    name = p.parent.name if p.name == "proxy_eval.csv" else f"{p.parent.name}/{p.stem}"
    return name, rows


def main():
    tables = [load(a) for a in sys.argv[1:]]
    common = sorted(set.intersection(*(set(t) for _, t in tables)))
    meta = ROOT / "data" / "video_to_data_challenge" / "track_1" / "meta" / "episodes_metadata.jsonl"
    objects = {}
    if meta.exists():
        for line in meta.read_text().splitlines():
            if line:
                r = json.loads(line)
                objects[r["episode_index"]] = r["object"]
    groups = {}
    for e in common:
        groups.setdefault(objects.get(e, "?"), []).append(e)
    width = max(len(n) for n, _ in tables)
    for col in COLUMNS:
        if not all(col in t[common[0]] for _, t in tables):
            continue
        print(f"\n== {col}")
        print(" " * 18 + "".join(f"{n[:width]:>{width + 2}}" for n, _ in tables))
        for obj, eps in list(groups.items()) + [("all", common)]:
            vals = [sum(t[e][col] for e in eps) / len(eps) for _, t in tables]
            print(f"{obj[:17]:<18}" + "".join(f"{v:>{width + 2}.3f}" for v in vals))


if __name__ == "__main__":
    main()

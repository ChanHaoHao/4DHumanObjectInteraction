#!/usr/bin/env python3
"""Is a cached object track the handled object or static scenery? No reconstruction, masks only.

    uv run --with numpy python scripts/track_check.py cache results/e2e_v1/cache --episodes 0-29

stillness: median over frames of IoU(object mask, the track's median mask), both with the person's pixels of that
frame removed (occlusion by the actor is not motion). Furniture stays ~1; an object that is carried, rolled or
lifted drops. Coverage: fraction of frames with a non-empty object mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

STEP = 4  # masks are 576x768; 1/4 of that is plenty for a stillness score


def stillness(obj: np.ndarray, human: np.ndarray) -> tuple[float, float]:
    """obj, human: [T,H,W] bool. Returns (stillness, coverage)."""
    seen = obj.reshape(len(obj), -1).any(1)
    if seen.sum() < 2:
        return float("nan"), float(seen.mean())
    median = obj[seen].mean(0) >= 0.5
    free = ~human[seen]
    a, b = obj[seen] & free, median[None] & free
    union = (a | b).sum((1, 2))
    iou = np.where(union > 0, (a & b).sum((1, 2)) / np.maximum(union, 1), 1.0)
    return float(np.median(iou)), float(seen.mean())


def check(folder: Path) -> tuple[float, float, str] | None:
    path = folder / "masks.npz"
    if not path.exists():
        return None
    import json

    m = np.load(path)
    prompt = json.loads((folder / "meta.json").read_text()).get("prompts", {}).get("object", "?")
    return (*stillness(m["object"][:, ::STEP, ::STEP], m["human"][:, ::STEP, ::STEP]), prompt)


def _selftest():
    t = np.zeros((10, 20, 20), bool)
    static, moving = t.copy(), t.copy()
    static[:, 5:10, 5:10] = True
    for i in range(10):
        moving[i, 5:10, i:i + 5] = True
    assert stillness(static, t)[0] == 1.0
    assert stillness(moving, t)[0] < 0.5
    occluded = static.copy()
    human = t.copy()
    human[::2, 5:10, 5:8] = True  # the actor covers part of the static object every other frame
    occluded &= ~human
    assert stillness(occluded, human)[0] == 1.0


def main():
    _selftest()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("caches", nargs="+", type=Path)
    p.add_argument("--episodes", default="0-29")
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from step4_colab import parse_episodes

    print("ep  " + "".join(f"{c.parts[-2] if c.name == 'cache' and len(c.parts) > 1 else c.name:>34}" for c in args.caches))
    for e in parse_episodes(args.episodes):
        row = []
        for c in args.caches:
            r = check(c / f"episode_{e:06d}")
            row.append("-" if r is None else f"{r[2][:16]:>16} still {r[0]:.2f} cov {r[1]:.2f}")
        print(f"{e:<3} " + "".join(f"{x:>34}" for x in row), flush=True)


if __name__ == "__main__":
    main()

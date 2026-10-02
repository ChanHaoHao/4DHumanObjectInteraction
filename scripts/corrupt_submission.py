#!/usr/bin/env python3
"""Write a deliberately corrupted copy of a Track 1 episodes folder, to check that a metric reacts.

    uv run --with numpy --with scipy scripts/corrupt_submission.py SRC DST --kind jitter [--amount A]

Kinds (amount defaults in brackets):
  jitter  i.i.d. noise per frame: human root and object translation sigma A metres, body pose and
          object rotation sigma 2A radians                                             [0.01]
  offset  object moved A metres along +x, relative to the human                       [0.10]
  scale   object_scale multiplied by A                                                [1.5]
  drift   human root translation drifts linearly to A metres along +x by the last frame [0.20]
          (the object stays: unforgiven by the single first-frame alignment)

Mesh and camera files are copied unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULTS = {"jitter": 0.01, "offset": 0.10, "scale": 1.5, "drift": 0.20}
UNITS_PER_M = 10.0  # MHR root translation: 1 unit = 10 cm (measured), native frame, x not flipped


def corrupt(data: dict, kind: str, amount: float, rng: np.random.Generator) -> dict:
    data = {k: np.array(v) for k, v in data.items()}
    T = len(data["pose"])
    if kind == "jitter":
        data["pose"][:, 0:3] += rng.normal(0, amount * UNITS_PER_M, (T, 3))
        data["pose"][:, 6:] += rng.normal(0, 2 * amount, (T, data["pose"].shape[1] - 6))
        data["object_translation"] += rng.normal(0, amount, (T, 3))
        noise = Rotation.from_rotvec(rng.normal(0, 2 * amount, (T, 3))).as_matrix()
        data["object_rotation"] = noise @ data["object_rotation"]
    elif kind == "offset":
        data["object_translation"][:, 0] += amount
    elif kind == "scale":
        data["object_scale"] = np.asarray(data["object_scale"]) * amount
    elif kind == "drift":
        data["pose"][:, 0] += np.linspace(0, amount * UNITS_PER_M, T)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return data


def corrupt_folder(src: Path, dst: Path, kind: str, amount: float | None = None, seed: int = 0) -> Path:
    amount = DEFAULTS[kind] if amount is None else amount
    dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for path in sorted(src.iterdir()):
        if path.suffix == ".npz" and not path.stem.endswith("_camera"):
            np.savez(dst / path.name, **corrupt(dict(np.load(path)), kind, amount, rng))
        elif path.is_file():
            shutil.copy2(path, dst / path.name)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--kind", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--amount", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    corrupt_folder(args.src, args.dst, args.kind, args.amount, args.seed)


if __name__ == "__main__":
    main()

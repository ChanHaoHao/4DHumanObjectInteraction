#!/usr/bin/env python3
"""Validate a Track 1 episodes directory against the official sample submission.

Checks every episode and reports all problems at once (the kit's packer stops at the first one).
Uses the kit's own rotation and mesh-budget code, so anything that passes here packs.

    uv run --with numpy --with pandas --with pyarrow --with trimesh --with fast_simplification \
        scripts/validate_submission.py results/<run>/episodes [--pack out.parquet]

Expected per episode (see notes/dataset.md):
    episode_XXXXXX.npz         pose [T,136], scales [68], shape [45],
                               object_rotation [T,3,3], object_translation [T,3], object_scale ()
    episode_XXXXXX_object.*    glb / obj / ply / stl / off
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KIT = ROOT / "data" / "v2d_submission_kit"
MESH_SUFFIXES = (".glb", ".obj", ".ply", ".stl", ".off")
DUMMY_COMMIT = "https://github.com/x/y/commit/" + "0" * 40


def scored_frames(sample: Path, episode_frame: int) -> dict[int, np.ndarray]:
    parts = pd.read_parquet(sample, columns=["row_id"]).row_id.str.split("_", expand=True)
    episodes, frames = parts[1].astype(int), parts[2].astype(int)
    keep = frames != episode_frame
    out = {}
    for episode, group in frames[keep].groupby(episodes[keep]):
        out[int(episode)] = np.unique(group.to_numpy())
    return out


def stretches(frames: np.ndarray) -> int:
    return int((np.diff(frames) != 1).sum()) + 1


def check_episode(directory: Path, episode: int, frames: np.ndarray, ms, budget_mesh,
                  faces_budget: int, vertices_budget: int) -> list[str]:
    errors = []
    path = directory / f"episode_{episode:06d}.npz"
    if not path.exists():
        return [f"missing {path.name}"]
    data = dict(np.load(path))
    expected = {"pose": (None, 136), "scales": (68,), "shape": (45,),
                "object_rotation": (None, 3, 3), "object_translation": (None, 3),
                "object_scale": ()}
    for key, shape in expected.items():
        if key not in data:
            errors.append(f"{path.name}: lacks '{key}'")
            continue
        actual = np.shape(data[key])
        if key == "object_scale" and actual in ((), (1,)):
            continue
        if len(actual) != len(shape) or any(s is not None and s != a for s, a in zip(shape, actual)):
            errors.append(f"{path.name}: '{key}' has shape {actual}, expected "
                          f"{tuple('T' if s is None else s for s in shape)}")
    if errors:
        return errors

    last = int(frames[-1])
    for key in ("pose", "object_rotation", "object_translation"):
        if len(data[key]) <= last:
            errors.append(f"{path.name}: '{key}' has {len(data[key])} frames, "
                          f"but frame {last} is scored")
    if errors:
        return errors

    for key in ("pose", "object_rotation", "object_translation"):
        bad = ~np.isfinite(np.asarray(data[key], float)[frames]).reshape(len(frames), -1).all(1)
        if bad.any():
            errors.append(f"{path.name}: '{key}' is non-finite on {bad.sum()} scored frame(s), "
                          f"first {frames[bad][0]}")
    for key in ("scales", "shape"):
        if not np.isfinite(data[key]).all():
            errors.append(f"{path.name}: '{key}' is non-finite")
    scale = float(np.asarray(data["object_scale"]).reshape(-1)[0])
    if not (np.isfinite(scale) and scale > 0):
        errors.append(f"{path.name}: object_scale must be positive, got {scale}")
    rotation = np.asarray(data["object_rotation"], float)[frames]
    if np.isfinite(rotation).all():
        try:
            ms._rotations(rotation)
        except ValueError as error:
            errors.append(f"{path.name}: {error}")

    meshes = [directory / f"episode_{episode:06d}_object{s}" for s in MESH_SUFFIXES]
    mesh = next((m for m in meshes if m.exists()), None)
    if mesh is None:
        errors.append(f"missing episode_{episode:06d}_object.{{{','.join(s[1:] for s in MESH_SUFFIXES)}}}")
    else:
        try:
            v, f = budget_mesh(mesh, faces_budget, vertices_budget)
            used = f[(f != 0).any(1)]
            area = 0.5 * np.linalg.norm(np.cross(v[used[:, 1]] - v[used[:, 0]],
                                                 v[used[:, 2]] - v[used[:, 0]]), axis=1).sum()
            if not np.isfinite(v).all():
                errors.append(f"{mesh.name}: non-finite vertices")
            elif not area > 0:
                errors.append(f"{mesh.name}: zero surface area")
        except Exception as error:  # trimesh raises many types on unreadable files
            errors.append(f"{mesh.name}: {error}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--sample", type=Path, default=None,
                        help="defaults to <kit>/data/track_1_sample_submission.parquet")
    parser.add_argument("--pack", type=Path, default=None,
                        help="also run the official packer and write this parquet")
    parser.add_argument("--commit", default=DUMMY_COMMIT,
                        help="commit URL for --pack (a dummy is fine for a local check)")
    args = parser.parse_args()
    sample = args.sample or args.kit / "data" / "track_1_sample_submission.parquet"

    sys.path.insert(0, str(args.kit))
    from v2dlb import mhr_submission as ms
    from v2dlb.mesh_budget import budget_mesh

    parts = pd.read_parquet(sample, columns=["row_id"]).row_id.str.split("_", expand=True)
    one = parts[(parts[1] == parts[1].iloc[0]) & (parts[2] == str(ms.EPISODE_FRAME))]
    order = ms.array_order()
    faces_budget = int((one[3].astype(int) == order.index("object_mesh_faces")).sum())
    vertices_budget = int((one[3].astype(int) == order.index("object_mesh_vertices")).sum())

    print(f"{'episode':>7}  {'scored':>6}  {'span':>11}  {'stretches':>9}  status")
    failures = 0
    for episode, frames in scored_frames(sample, ms.EPISODE_FRAME).items():
        errors = check_episode(args.episodes, episode, frames, ms, budget_mesh,
                               faces_budget, vertices_budget)
        status = "ok" if not errors else f"FAIL ({len(errors)})"
        print(f"{episode:>7}  {len(frames):>6}  {frames[0]:>5}-{frames[-1]:<5}  "
              f"{stretches(frames):>9}  {status}")
        for error in errors:
            print(f"           {error}")
        failures += bool(errors)
    print(f"\n{failures} episode(s) failed" if failures else "\nall episodes valid")
    if failures or args.pack is None:
        return int(bool(failures))

    sys.stdout.flush()
    result = subprocess.run(
        [sys.executable, "eval_reconstruction.py", "--episodes", str(args.episodes.resolve()),
         "--sample", str(sample.resolve()), "--commit", args.commit,
         "--out", str(args.pack.resolve())],
        cwd=args.kit)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

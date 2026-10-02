#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pyarrow", "scipy", "torch", "trimesh", "fast_simplification", "numba"]
# ///
"""Sanity check for proxy_eval: corrupted outputs must score worse than the clean one.

    uv run scripts/proxy_sanity.py [--out results/proxy_sanity] [--episodes 0,6,12]

Builds a synthetic clean result on the real scored frames (MHR moving smoothly, a box held at one
hand, a known camera), renders its own masks and depth as the cache (denser samples, different
seed, so clean IoU is high but not exactly 1), writes jitter / offset / scale / drift copies with
corrupt_submission.py, scores everything, and checks that each corruption worsens the metrics
it should. Also checks reference mode: clean vs itself is ~0, and each corruption is caught.
Exit code 0 when every check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy_eval as pe  # noqa: E402
from corrupt_submission import corrupt_folder  # noqa: E402

K = np.array([[1400.0, 0, 768], [0, 1400, 576], [0, 0, 1]])
MASK_SIZE = (384, 288)
BOX = (0.30, 0.20, 0.10)  # metres; the thin axis faces away from the palm
FPS = 30.0

# metric -> must get worse under this corruption (lower is better except IoU)
EXPECT = {
    "jitter": ["acc_h", "acc_o", "iou_h", "accE_h", "accE_o"],
    # a slide along the box face keeps the hand in contact: only reprojection sees it
    "offset": ["iou_o", "dep_o", "cd_o"],
    "scale": ["pen+gap", "iou_o", "cd_o"],
    "drift": ["iou_h", "dep_h", "cd_h"],
}


def smooth_signal(rng, T, count, amplitude):
    t = np.arange(T)[:, None] / FPS
    out = np.zeros((T, count))
    for _ in range(3):
        freq, phase = rng.uniform(0.1, 0.5, count), rng.uniform(0, 2 * np.pi, count)
        out += amplitude * np.sin(2 * np.pi * freq * t + phase)
    return out


def frame_from_axis(d):
    """Rotations whose third column is d [T,3] (unit), built against a fixed up vector."""
    up = np.array([0.0, -1.0, 0.0])
    x = np.cross(up, d)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.stack([x, np.cross(d, x), d], axis=2)


def synthesize(out: Path, cache: Path, episode: int, frames: np.ndarray, body: pe.MHRBody,
               stride: int) -> None:
    import trimesh

    rng = np.random.default_rng(episode)
    T = int(frames[-1]) + 1
    t = np.arange(T)
    pose = np.zeros((T, 136))
    pose[:, 6:] = smooth_signal(rng, T, 130, 0.08)
    pose[:, 4] = 0.2 * np.sin(2 * np.pi * t / 200)  # root yaw
    rest = body(np.concatenate([pose[:1], np.zeros((1, 68))], 1), np.zeros(45))[0][0]
    target = np.array([0.0, 0.0, 3.5]) - rest.mean(0)  # body centre 3.5 m in front of the camera
    pose[:, 0:3] = target * pe.MHRBody.FLIP * 10.0  # Track 1 metres -> MHR root units (10 cm)
    pose[:, 0] += 2.0 * np.sin(2 * np.pi * t / 120)  # +-20 cm
    pose[:, 2] += 1.0 * np.sin(2 * np.pi * t / 170)  # +-10 cm

    verts, _ = body(np.concatenate([pose, np.zeros((T, 68))], 1), np.zeros(45))
    hand = verts[:, body.roles["right_hand"]].mean(1)
    away = hand - verts.mean(1)
    away /= np.linalg.norm(away, axis=1, keepdims=True)
    rotation = frame_from_axis(away)
    translation = hand + away * (BOX[2] / 2 + 0.01)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"episode_{episode:06d}.npz", pose=pose.astype(np.float32),
             scales=np.zeros(68, np.float32), shape=np.zeros(45, np.float32),
             object_rotation=rotation, object_translation=translation, object_scale=np.float64(1.0))
    trimesh.creation.box(extents=BOX).export(out / f"episode_{episode:06d}_object.obj")
    np.savez(out / f"episode_{episode:06d}_camera.npz", K=K, world_to_camera=np.eye(4),
             image_size=np.array(pe.IMAGE_SIZE))

    # Cache: masks and depth rendered from the clean scene with independent, denser samples.
    box = trimesh.creation.box(extents=BOX)
    box_v, box_f = np.asarray(box.vertices), np.asarray(box.faces)
    human_samples = pe.surface_samples(verts[frames[0]], body.faces, 120000, 7)
    box_local = pe.at_samples(box_v, box_f, pe.surface_samples(box_v, box_f, 60000, 8))
    Ks = K.copy()
    Ks[0] *= MASK_SIZE[0] / pe.IMAGE_SIZE[0]
    Ks[1] *= MASK_SIZE[1] / pe.IMAGE_SIZE[1]
    kept = frames[::stride]
    human_masks, object_masks, depths = [], [], []
    for f in kept:
        label, depth = pe.render_labels([pe.at_samples(verts[f], body.faces, human_samples),
                                         box_local @ rotation[f].T + translation[f]],
                                        Ks, np.eye(4), MASK_SIZE)
        human_masks.append(label == 1)
        object_masks.append(label == 2)
        depths.append(np.where(np.isfinite(depth), depth, 0).astype(np.float32))
    folder = cache / f"episode_{episode:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "masks.npz", frames=kept, human=np.array(human_masks),
                        object=np.array(object_masks))
    np.savez_compressed(folder / "depth.npz", frames=kept, depth=np.array(depths))


def worse(metric: str, clean: pd.Series, bad: pd.Series) -> bool:
    if metric == "pen+gap":
        return bad["pen"] + bad["gap"] > clean["pen"] + clean["gap"]
    if metric in pe.HIGHER_IS_BETTER:
        return bad[metric] < clean[metric]
    return bad[metric] > clean[metric]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=pe.ROOT / "results" / "proxy_sanity")
    parser.add_argument("--episodes", default="0,6,12")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--device", default=None, help="default: cuda if available")
    args = parser.parse_args()
    episodes = [int(e) for e in args.episodes.split(",")]
    body = pe.MHRBody(device=args.device)
    clean, cache = args.out / "clean" / "episodes", args.out / "cache"
    scored = pe.scored_frames(pe.DEFAULT_SAMPLE)
    for episode in episodes:
        synthesize(clean, cache, episode, scored[episode], body, args.stride)

    runs = {"clean": clean}
    for kind in EXPECT:
        runs[kind] = corrupt_folder(clean, args.out / kind / "episodes", kind)
    means = {}
    for name, folder in runs.items():
        table = pe.evaluate(folder, cache=cache, reference=clean, episodes=episodes,
                            stride=args.stride, body=body)
        print(f"\n== {name} ==\n{pe.format_table(table)}")
        means[name] = table.mean(numeric_only=True)

    summary = pd.DataFrame(means).T.drop(columns=["episode", "scored"])
    print("\n== means per run ==")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    failures = []
    zero = summary.loc["clean", ["cd_h", "cd_o", "accE_h", "accE_o"]].abs().max()
    if not zero < 1e-4:  # cm; float32 MHR vertices through the Sim(3) fit leave ~1e-6
        failures.append(f"reference mode: clean vs itself is {zero:.2e}, not ~0")
    print("\n== checks ==")
    for kind, metrics in EXPECT.items():
        for metric in metrics:
            ok = worse(metric, summary.loc["clean"], summary.loc[kind])
            print(f"{'ok  ' if ok else 'FAIL'} {kind:>6}: {metric} worse than clean")
            if not ok:
                failures.append(f"{kind}: {metric}")
    print(f"\n{len(failures)} check(s) failed: {failures}" if failures else "\nall checks passed")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())

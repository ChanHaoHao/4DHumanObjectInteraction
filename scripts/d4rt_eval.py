#!/usr/bin/env python3
"""Compare depth caches (Step 3v1 MoGe-2 vs D4RT) against the Step 4 body (runs on the Colab VM).

    python scripts/d4rt_eval.py --episodes 0 [--run results/human_v1] [--caches cache,cache/d4rt]
                                [--out results/d4rt_eval]

The human_v1 body was solved without depth (SAM 3D Body joints + smoothing), so it is an independent,
temporally smooth reference. Per cache and episode, over the scored frames:
  k            median over frames of r_t = median(cache depth / body depth) on actor pixels (Step 4's k_moge)
  k_p10_p90    spread of r_t: a depth that moves with the body has a narrow band
  logk_std     std of log r_t
  logk_jit     median |log r_t - log r_t-1| over consecutive frames (frame-to-frame depth jitter vs the body)
Then proxy_eval.py's reprojection columns (iou_h, dep_h, ...) with depth_scale = 1 / k of that cache, from
<out>/<cache>/episodes (links to the run's submission files plus a camera npz carrying that depth_scale).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from human_post import moge_ratio  # noqa: E402
from proxy_eval import DEFAULT_SAMPLE, MHRBody, evaluate, format_table, scored_frames  # noqa: E402


def ratio_series(run: Path, cache: Path, episode: int, frames: np.ndarray, body: MHRBody) -> np.ndarray:
    name = f"episode_{episode:06d}"
    data = dict(np.load(run / "episodes" / f"{name}.npz"))
    K = np.load(run / "episodes" / f"{name}_camera.npz")["K"]
    pose = np.asarray(data["pose"], np.float64)[frames]
    params = np.concatenate([pose, np.broadcast_to(data["scales"], (len(pose), 68))], 1)
    verts, _ = body(params, data["shape"])  # world = camera frame
    depth = np.load(cache / name / "depth.npz")["depth"]
    human = np.load(cache / name / "masks.npz")["human"]
    return np.array([moge_ratio(verts[i], K, depth[f].astype(np.float32), human[f]) for i, f in enumerate(frames)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--run", type=Path, default=REPO / "results/human_v1")
    parser.add_argument("--caches", default="cache,cache/d4rt")
    parser.add_argument("--out", type=Path, default=REPO / "results/d4rt_eval")
    args = parser.parse_args()
    episodes = [int(e) for e in args.episodes.split(",")]
    caches = [REPO / c for c in args.caches.split(",")]
    body = MHRBody()
    scored = scored_frames(DEFAULT_SAMPLE)
    summary = {}
    for cache in caches:
        tag = cache.name
        ep_dir = args.out / tag / "episodes"
        ep_dir.mkdir(parents=True, exist_ok=True)
        rows = {}
        for e in episodes:
            name = f"episode_{e:06d}"
            frames = scored[e]
            r = ratio_series(args.run, cache, e, frames, body)
            ok = np.isfinite(r)
            lr = np.log(r[ok])
            consecutive = np.diff(frames[ok]) == 1
            k = float(np.median(r[ok]))
            rows[e] = {"k": round(k, 4), "k_p10_p90": np.round(np.percentile(r[ok], [10, 90]), 4).tolist(),
                       "logk_std": round(float(lr.std()), 4),
                       "logk_jit": round(float(np.median(np.abs(np.diff(lr))[consecutive])), 5),
                       "frames": int(ok.sum())}
            np.save(args.out / tag / f"{name}_ratio.npy", np.stack([frames, r]))
            for suffix in (".npz", "_object.obj", "_object.glb", "_object.ply"):
                srcf = (args.run / "episodes" / f"{name}{suffix}").resolve()
                link = ep_dir / f"{name}{suffix}"
                if srcf.exists() and not link.exists():
                    link.symlink_to(srcf)
            cam = dict(np.load(args.run / "episodes" / f"{name}_camera.npz"))
            cam["depth_scale"] = np.float64(1 / k)
            np.savez(ep_dir / f"{name}_camera.npz", **cam)
            print(f"{tag} ep {e}: {rows[e]}", flush=True)
        table = evaluate(ep_dir, cache=cache, episodes=episodes, body=body)
        print(f"== {tag}\n{format_table(table)}", flush=True)
        table.to_csv(args.out / tag / "proxy_eval.csv", index=False)
        for _, row in table.iterrows():
            rows[int(row.episode)].update({c: round(float(row[c]), 4) for c in ("iou_h", "dep_h", "iou_o", "dep_o")
                                           if c in row and np.isfinite(row[c])})
        summary[tag] = rows
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

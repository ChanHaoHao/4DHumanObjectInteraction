#!/usr/bin/env python3
"""Why does the object float off the hand? Per scored frame, split the proxy gap into its causes (runs on the VM).

    python float_diag.py <repo>/results/joint_v5/episodes --cache <repo>/cache --out <repo>/results/joint_v5/float_diag \
        [--episodes 0-29] [--stride 2] [--float-cm 3] [--near-px 4]

Per scored frame (every --stride-th), with the 512 proxy hand points (proxy_eval's gap):
  gap      closest hand point to the object surface, cm, and the offset from that hand point to its nearest
           surface point, split into along the hand's camera ray (signed: + = object behind the hand) and sideways.
  mask_px  2D distance (half-resolution px) from the projected hand points to the cache's object mask.
  rend_px  the same to our rendered object silhouette (proxy_eval's point splat).
  iou      rendered object vs object mask.
A floating frame (gap > --float-cm) is
  depth    the mask and our render both reach the hand in the image: only the distance along the ray is wrong
           (what 7b's ray placement / a contact term can fix);
  pose     the mask reaches the hand, our render does not: wrong rotation, flip or missing part (shape);
  mask     the mask does not reach the hand: no contact in the image (true release, or the mask misses the part
           that is held, or the hand is wrong).
Writes <out>/episode_XXXXXX.csv (per frame) and <out>/summary.csv (per episode: fraction of frames and share of
the gap per kind, median along/sideways offset and IoU on floating frames); prints the per-episode table.
Needs only masks.npz in the cache (no depth).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_vm import parse_episodes  # noqa: E402
from proxy_eval import (DEFAULT_CACHE, DEFAULT_SAMPLE, MHRBody, ReprojectionInputs, at_samples, iou,  # noqa: E402
                        load_camera, load_episode, mesh_budget, render_labels, scored_frames, surface_samples)

KINDS = ["contact", "depth", "pose", "mask"]


def mask_distance(mask: np.ndarray, uv: np.ndarray) -> float:
    """Min over projected points of the distance to the mask, px (inf: empty mask or no point in the image)."""
    if not mask.any():
        return np.inf
    h, w = mask.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    if not ok.any():
        return np.inf
    dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return float(dt[np.rint(uv[ok, 1]).astype(int), np.rint(uv[ok, 0]).astype(int)].min())


def diagnose(episode: int, frames: np.ndarray, args, body: MHRBody, budget) -> pd.DataFrame:
    ep = load_episode(args.episodes_dir, episode, frames, budget, body)
    K, w2c, size, _ = load_camera(args.episodes_dir, episode)
    inputs = ReprojectionInputs(args.cache, episode)
    local = at_samples(ep["mesh_vertices"], ep["mesh_faces"],
                       surface_samples(ep["mesh_vertices"], ep["mesh_faces"], 20000, 0))
    tree = cKDTree(local)
    rows = []
    for i in range(0, len(frames), args.stride):
        R, t, s = ep["rotation"][i], ep["translation"][i], ep["scale"]
        hands = ep["verts"][i, body.hands]
        d, j = tree.query((hands - t) @ R / s, workers=-1)
        k = int(np.argmin(d))
        h = hands[k] @ w2c[:3, :3].T + w2c[:3, 3]
        o = (s * local[j[k]] @ R.T + t) @ w2c[:3, :3].T + w2c[:3, 3]
        ray = h / np.linalg.norm(h)
        v = o - h
        along = float(v @ ray)
        side = float(np.linalg.norm(v - along * ray))
        row = {"frame": int(frames[i]), "gap_cm": float(d[k] * s * 100), "along_cm": along * 100, "side_cm": side * 100,
               "mask_px": np.nan, "rend_px": np.nan, "iou": np.nan}
        masks = inputs.frame("masks", frames[i])
        if masks is not None:
            obj_mask = masks[1].astype(bool)
            mh, mw = obj_mask.shape
            Ks = K.copy()
            Ks[0] *= mw / size[0]
            Ks[1] *= mh / size[1]
            obj = s * local @ R.T + t
            label, _ = render_labels([obj], Ks, w2c, (mw, mh))
            rend = label == 1
            hc = hands @ w2c[:3, :3].T + w2c[:3, 3]
            hc = hc[hc[:, 2] > 1e-3]
            uv = hc[:, :2] / hc[:, 2:] @ Ks[:2, :2].T + Ks[:2, 2]
            row.update(mask_px=mask_distance(obj_mask, uv), rend_px=mask_distance(rend, uv), iou=iou(rend, obj_mask))
        rows.append(row)
    df = pd.DataFrame(rows)
    floating = df.gap_cm > args.float_cm
    near_mask = df.mask_px <= args.near_px
    near_rend = df.rend_px <= args.near_px
    df["kind"] = np.select([~floating, near_mask & near_rend, near_mask & ~near_rend], KINDS[:3], KINDS[3])
    return df


def summarize(episode: int, df: pd.DataFrame) -> dict:
    n = len(df)
    row = {"episode": episode, "frames": n, "gap": df.gap_cm.mean()}
    for kind in KINDS:
        part = df[df.kind == kind]
        row[f"{kind}_frac"] = len(part) / n
        row[f"{kind}_gap_share"] = part.gap_cm.sum() / max(df.gap_cm.sum(), 1e-9)  # share of the episode's gap
    fl = df[df.kind != "contact"]
    row["float_along_med"] = fl.along_cm.median() if len(fl) else np.nan
    row["float_side_med"] = fl.side_cm.median() if len(fl) else np.nan
    row["iou_contact"] = df[df.kind == "contact"].iou.mean()
    row["iou_float"] = fl.iou.mean() if len(fl) else np.nan
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes_dir", type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--float-cm", type=float, default=3.0, help="gap above this is floating")
    parser.add_argument("--near-px", type=float, default=4.0, help="half-res px: hand 'reaches' a mask (joint_fit's contact_px)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    body = MHRBody()
    budget = mesh_budget(args.sample)
    scored = scored_frames(args.sample)
    rows = []
    for e in parse_episodes(args.episodes):
        if not (args.episodes_dir / f"episode_{e:06d}.npz").exists():
            continue
        df = diagnose(e, scored[e], args, body, budget)
        df.to_csv(args.out / f"episode_{e:06d}.csv", index=False)
        rows.append(summarize(e, df))
        print(pd.DataFrame(rows[-1:]).round(2).to_string(index=False, header=len(rows) == 1), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

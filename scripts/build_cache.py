#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pyarrow", "scipy", "opencv-python-headless"]
# ///
"""Step 3 local stage: raw SAM 3 tracks + MoGe-2 depth (from preprocess_vm.py) -> the per-clip cache.

    python build_cache.py --raw /content/raw --out /content/cache --data /content/v2d/track_1 \
        --sample track_1_sample_submission.parquet [--episodes 0-29] [--overlay /content/cache_check]

Runs on the Colab VM next to preprocess_vm.py (it holds every mask track and the depth stack in RAM,
several GB per episode; too much for the laptop). Only the finished cache is downloaded.

Per episode, writes <out>/episode_XXXXXX/:
  masks.npz   frames [T], human [T,576,768] bool, object [T,576,768] bool   (1/2 resolution, every frame)
  depth.npz   frames [T], depth [T,288,384] float16 metres (0 = invalid; 1/4 resolution, every frame,
              temporally stabilized), scale [T] (per-frame correction applied), background [288,384]
  camera.npz  K [3,3] at video resolution (1536x1152), image_size (W, H), fov_x (deg)
  meta.json   chosen tracks, prompts, coverage, depth-scale stats, floor plane, camera and actor height,
              `warning` when no object track touches the actor. Written last: it marks the episode complete

Track selection. SAM 3 returns every instance of "person" and of the object prompt, including
bystanders and look-alikes (other desks, tables, chairs). Among the foreground people (by mask area),
the actor is the one touching an object track (in the image and within DEPTH_GAP in depth) on most
scored frames, which are the hand-object contact span; that track is the object. Other tracks of the
same class fill frames where the chosen track is missing (ID switches after occlusion) when they
overlap its nearest box in time.

Depth stabilization. The camera is static, so the background has one depth. A reference background is
the median over frames with every candidate person/object masked out; each frame is rescaled by the
median ratio to it on visible background, which removes MoGe's per-frame metric-scale jitter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
EPISODE_FRAME = 999999  # row_id frame of the per-episode constants (v2dlb.mhr_submission)
CONTACT_PX = 6  # touching = within 6 px at 1/8 resolution (~48 px in the video, ~18 cm at 3 m)
ACTOR_AREA = 0.4  # actor candidates: at least this fraction of the largest person's area
MIN_CONTACT = 0.3  # fraction of scored frames; below it the episode is flagged
DEPTH_GAP = 0.75  # m; touching tracks must also be this close in median depth
FILL_IOU = 0.3  # a spare track fills gaps when its box overlaps the chosen track's nearest box


def parse_episodes(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def scored_frames(sample: Path) -> dict[int, np.ndarray]:
    """Same as proxy_eval.scored_frames, without the kit import."""
    parts = pd.read_parquet(sample, columns=["row_id"]).row_id.str.split("_", expand=True)
    episodes, frames = parts[1].astype(int), parts[2].astype(int)
    keep = frames != EPISODE_FRAME
    return {int(e): np.unique(g.to_numpy()) for e, g in frames[keep].groupby(episodes[keep])}


def load_tracks(*paths: Path) -> dict:
    """Tracks from one or more SAM outputs, pooled; ids become "<file index>:<SAM id>" when pooled."""
    parts = []
    for n, path in enumerate(paths):
        data = np.load(path)
        masks = np.unpackbits(data["masks"], axis=-1)[..., : int(data["image_size"][0]) // 2].astype(bool)
        ids = [f"{n}:{i}" if len(paths) > 1 else str(i) for i in data["obj_ids"]]
        parts.append((ids, data["probs"].astype(np.float32), masks, str(data["prompt"])))
    return {"ids": np.array(sum((p[0] for p in parts), [])),
            "probs": np.concatenate([p[1] for p in parts]), "masks": np.concatenate([p[2] for p in parts]),
            "present": np.concatenate([p[1] for p in parts]) > 0, "prompt": " | ".join(p[3] for p in parts)}


# ------------------------------------------------------------------------------------ track selection
def boxes(masks: np.ndarray) -> np.ndarray:
    """[T,h,w] -> [T,4] (x0, y0, x1, y1), NaN where empty."""
    out = np.full((len(masks), 4), np.nan)
    rows, cols = masks.any(2), masks.any(1)
    for t in np.flatnonzero(rows.any(1)):
        r, c = np.flatnonzero(rows[t]), np.flatnonzero(cols[t])
        out[t] = c[0], r[0], c[-1] + 1, r[-1] + 1
    return out


def box_iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def track_depth(tracks, frames, depth) -> np.ndarray:
    """[K,len(frames)] depth (m) of each track's front surface (20th percentile under the mask: thin
    objects like a hoop pick up background depth at 1/4 resolution), NaN where absent or invalid."""
    out = np.full((len(tracks["masks"]), len(frames)), np.nan)
    for k, track in enumerate(tracks["masks"]):
        for n, f in enumerate(frames):
            d = depth[f][track[f, ::2, ::2]]
            d = d[d > 0]
            if d.size:
                out[k, n] = np.percentile(d.astype(np.float32), 20)
    return out


def contact_matrix(people, objects, frames, depth=None) -> np.ndarray:
    """[Kp,Ko] fraction of `frames` where person k touches object j: masks within CONTACT_PX in the
    image and, when depth is given, front-surface depths within DEPTH_GAP (a desk behind the actor is not touched)."""
    small = lambda m: m[:, frames, ::4, ::4]  # noqa: E731  1/8 resolution
    p, o = small(people["masks"]), small(objects["masks"])
    grow = ndimage.generate_binary_structure(2, 1)
    p = np.stack([[ndimage.binary_dilation(m, grow, CONTACT_PX) for m in track] for track in p])
    near = np.ones((len(p), len(o), len(frames)), bool)
    if depth is not None:
        dp, do = track_depth(people, frames, depth), track_depth(objects, frames, depth)
        gap = np.abs(dp[:, None] - do[None])
        near = np.where(np.isnan(gap), True, gap < DEPTH_GAP)  # thin objects may have no valid depth
    out = np.zeros((len(p), len(o)))
    for i in range(len(p)):
        for j in range(len(o)):
            out[i, j] = ((p[i] & o[j]).any((1, 2)) & near[i, j]).mean()
    return out


def stitch(tracks: dict, main: int) -> tuple[np.ndarray, list[int]]:
    """The chosen track's masks, with gaps filled by other tracks that sit where it last was."""
    masks = tracks["masks"][main].copy()
    have = tracks["present"][main].copy()
    used = [str(tracks["ids"][main])]
    order = np.argsort(-tracks["present"].sum(1))
    for k in order:
        if k == main:
            continue
        gaps = np.flatnonzero(tracks["present"][k] & ~have)
        if not len(gaps):
            continue
        ref = boxes(masks)
        known = np.flatnonzero(~np.isnan(ref[:, 0]))
        if not len(known):
            break
        theirs = boxes(tracks["masks"][k])
        nearest = known[np.abs(known[None, :] - gaps[:, None]).argmin(1)]
        ok = np.array([box_iou(theirs[g], ref[n]) >= FILL_IOU for g, n in zip(gaps, nearest)])
        if ok.mean() >= 0.5:
            fill = gaps[ok]
            masks[fill], have[fill] = tracks["masks"][k][fill], True
            used.append(str(tracks["ids"][k]))
    return masks, used


def select(people, objects, scored, depth=None) -> dict:
    """Actor/object pair. Candidates for the actor are the foreground people (median area over scored
    frames >= ACTOR_AREA of the largest; seated bystanders are smaller); among them, the pair touching on
    most scored frames wins (a camera operator can be as large as the actor). Without any touching pair,
    the largest person is the actor and the object mask stays empty."""
    sample = scored[::2]
    info = {"person_tracks": people["present"].sum(1).tolist(), "object_tracks": objects["present"].sum(1).tolist()}
    if not len(people["ids"]):
        raise RuntimeError("no person track")
    area = np.median(people["masks"][:, sample].sum((2, 3)), axis=1)
    info["person_area_scored"] = area.astype(int).tolist()
    if not len(objects["ids"]):
        human, human_ids = stitch(people, int(area.argmax()))
        return {"human": human, "object": np.zeros_like(human), **info, "human_ids": human_ids, "object_ids": [],
                "contact": 0.0, "warning": "no object track"}
    contact = contact_matrix(people, objects, sample, depth)
    candidates = np.flatnonzero(area >= ACTOR_AREA * area.max())
    i = int(candidates[contact[candidates].max(1).argmax()])
    if contact[i].max() < MIN_CONTACT:
        i = int(area.argmax())
    j = int(contact[i].argmax())
    human, human_ids = stitch(people, i)
    obj, object_ids = stitch(objects, j)
    out = {"human": human, "object": obj, **info, "human_ids": human_ids, "object_ids": object_ids,
           "contact": float(contact[i, j]), "contact_matrix": np.round(contact, 3).tolist()}
    if contact[i, j] < MIN_CONTACT:  # an empty mask is safer downstream than a look-alike's
        out.update(object=np.zeros_like(human), object_ids=[], best_untouched=object_ids,
                   warning="no object track touches the actor")
    return out


def draw_tracks(episode: int, people, objects, scored, data: Path, out: Path):
    """Every candidate track (p<id> red shades, o<id> green shades) on three scored frames."""
    cap = cv2.VideoCapture(str(data / "videos/chunk-000/observation.images.exo_camera" / f"episode_{episode:06d}.mp4"))
    tiles = []
    for f in scored[[0, len(scored) // 2, -1]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        h, w = people["masks"].shape[2:]
        img = cv2.resize(img, (w, h))
        for prefix, tracks, base in (("p", people, (60, 60, 255)), ("o", objects, (60, 255, 60))):
            for k, obj_id in enumerate(tracks["ids"]):
                m = tracks["masks"][k, f]
                if not m.any():
                    continue
                color = np.array(base) * (1 - 0.25 * (k % 3))
                img[m] = (0.45 * img[m] + 0.55 * color).astype(np.uint8)
                ys, xs = np.nonzero(m)
                cv2.putText(img, f"{prefix}{obj_id}", (int(xs.mean()), int(ys.min()) - 4), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"ep{episode} f{f}", (10, 30), 0, 0.9, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (512, 384)))
    cap.release()
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"episode_{episode:06d}_tracks.jpg"), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------------------------------------- depth
def stabilize(depth: np.ndarray, moving: np.ndarray, stride: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """depth [T,h,w] (0 = invalid), moving [T,h,w] bool -> (stabilized depth, per-frame scale, background)."""
    valid = (depth > 0) & ~moving
    stack = np.where(valid[::stride], depth[::stride].astype(np.float32), np.nan)
    background = np.nanmedian(stack, axis=0)
    scale = np.ones(len(depth), np.float32)
    for t in range(len(depth)):
        v = valid[t] & np.isfinite(background)
        if v.sum() > 1000:
            scale[t] = np.median(background[v] / depth[t][v].astype(np.float32))
    out = (depth.astype(np.float32) * scale[:, None, None]).astype(np.float16)
    return out, scale, background


def pixel_rays(K: np.ndarray, shape: tuple[int, int], step: int) -> np.ndarray:
    """Camera rays (z = 1) for depth pixels sampled at full-resolution centres step*i + step//2."""
    h, w = shape
    u = (np.arange(w) * step + step // 2)[None, :].repeat(h, 0)
    v = (np.arange(h) * step + step // 2)[:, None].repeat(w, 1)
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)


def fit_floor(points: np.ndarray, rng=np.random.default_rng(0), iters: int = 400, tol: float = 0.03):
    """RANSAC plane among roughly horizontal candidates; returns (n, d) with n.p + d = 0, n up (-y)."""
    best, best_count = None, 0
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < 0.8:  # camera y points down; the floor normal is close to it
            continue
        d = -n @ a
        count = (np.abs(points @ n + d) < tol).sum()
        if count > best_count:
            best, best_count = (n, d), count
    n, d = best
    inl = points[np.abs(points @ n + d) < tol]
    centroid = inl.mean(0)
    n = np.linalg.svd(inl - centroid)[2][-1]
    if n[1] > 0:
        n = -n
    return n, float(-n @ centroid), float(len(inl) / len(points))


def geometry_stats(background, K, human, depth, scored, step=4) -> dict:
    rays = pixel_rays(K, background.shape, step)
    h = background.shape[0]
    lower = np.zeros_like(background, bool)
    lower[int(h * 0.6):] = True
    pts = (rays * background[..., None])[lower & np.isfinite(background) & (background > 0)]
    n, d, inliers = fit_floor(pts)
    tops = []
    for t in scored[::5]:
        m = ndimage.binary_erosion(human[t, ::2, ::2], iterations=1) & (depth[t] > 0)
        if m.sum() < 50:
            continue
        z = depth[t][m].astype(np.float32)
        p = rays[m] * z[:, None]
        p = p[np.abs(z - np.median(z)) < 0.5]
        tops.append(np.percentile(p @ n + d, 98))
    return {"floor_normal": np.round(n, 4).tolist(), "floor_inliers": round(inliers, 3),
            "camera_height_m": round(abs(d), 3),
            "actor_top_m_p90": round(float(np.percentile(tops, 90)), 3) if tops else None,
            "actor_top_m_median": round(float(np.median(tops)), 3) if tops else None}


# -------------------------------------------------------------------------------------------- overlay
def overlay(episode: int, folder: Path, out: Path, scored: np.ndarray, data: Path, count: int = 8):
    masks = np.load(folder / "masks.npz")
    human, obj = masks["human"], masks["object"]
    has_depth = (folder / "depth.npz").exists()
    dep = np.load(folder / "depth.npz")["depth"] if has_depth else np.zeros((len(human), 288, 384), np.float16)
    picks = np.unique(np.r_[max(scored[0] - 30, 0), scored[np.linspace(0, len(scored) - 1, count - 1).astype(int)]])
    cap = cv2.VideoCapture(str(data / "videos/chunk-000/observation.images.exo_camera" / f"episode_{episode:06d}.mp4"))
    finite = dep[picks][dep[picks] > 0].astype(np.float32)
    lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
    top, bottom = [], []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        img = cv2.resize(img, (human.shape[2], human.shape[1]))
        for m, color in ((human[f], (60, 60, 255)), (obj[f], (60, 255, 60))):
            img[m] = (0.45 * img[m] + 0.55 * np.array(color)).astype(np.uint8)
            edges = m ^ ndimage.binary_erosion(m, iterations=2)
            img[edges] = color
        tag = "scored" if f in set(scored.tolist()) else "unscored"
        cv2.putText(img, f"ep{episode} f{f} {tag}", (10, 30), 0, 0.9, (255, 255, 255), 2)
        top.append(cv2.resize(img, (384, 288)))
        d = np.clip((dep[f].astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
        col = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[dep[f] <= 0] = 0
        cv2.putText(col, f"{lo:.1f}-{hi:.1f} m", (10, 30), 0, 0.8, (255, 255, 255), 2)
        bottom.append(col)
    cap.release()
    rows = []
    for i in range(0, len(top), 4):
        pad = lambda tiles: tiles + [np.zeros_like(tiles[0])] * (4 - len(tiles))  # noqa: E731
        rows.append(np.concatenate(pad(top[i:i + 4]), 1))
        if has_depth:
            rows.append(np.concatenate(pad(bottom[i:i + 4]), 1))
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"episode_{episode:06d}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------------------------------------- main
def build(episode: int, raw: Path, out: Path, scored: np.ndarray, data: Path, check: Path | None,
          objects: list[str] | None = None) -> dict:
    src, dst = raw / f"episode_{episode:06d}", out / f"episode_{episode:06d}"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta.json").unlink(missing_ok=True)
    people = load_tracks(src / "sam_person.npz")
    objects = load_tracks(*([src / f for f in objects] if objects else
                            [src / "sam_object.npz", *sorted(src.glob("sam_object_*.npz"))]))
    T = people["masks"].shape[1]
    raw_depth = dict(np.load(src / "depth_raw.npz")) if (src / "depth_raw.npz").exists() else None
    chosen = select(people, objects, scored, None if raw_depth is None else raw_depth["depth"])
    if check is not None:
        draw_tracks(episode, people, objects, scored[scored < T], data, check)
    frames = np.arange(T)
    np.savez_compressed(dst / "masks.npz", frames=frames, human=chosen["human"], object=chosen["object"])
    meta = {"episode": episode, "frames": T, "prompts": {"person": people["prompt"], "object": objects["prompt"]},
            **{k: v for k, v in chosen.items() if k not in ("human", "object")}}
    sc = scored[scored < T]
    meta["human_coverage_scored"] = round(float(chosen["human"][sc].any((1, 2)).mean()), 3)
    meta["object_coverage_scored"] = round(float(chosen["object"][sc].any((1, 2)).mean()), 3)
    meta["object_area_px_scored_median"] = int(np.median(chosen["object"][sc].sum((1, 2))))

    if raw_depth is not None:
        depth = raw_depth["depth"]
        everything = np.concatenate([people["masks"], objects["masks"]]).any(0)[:, ::2, ::2]
        moving = np.stack([ndimage.binary_dilation(m, iterations=3) for m in everything])
        depth, scale, background = stabilize(depth, moving)
        np.savez_compressed(dst / "depth.npz", frames=frames, depth=depth, scale=scale, background=background)
        K = raw_depth["K"]
        np.savez(dst / "camera.npz", K=K, image_size=raw_depth["image_size"], fov_x=raw_depth["fov_x"])
        p5, p95 = np.percentile(scale, [5, 95])
        meta.update({"fov_x": round(float(raw_depth["fov_x"]), 2),
                     "fov_x_spread": round(float(np.ptp(raw_depth["fov_x_samples"])), 2),
                     "fx": round(float(K[0, 0]), 1),
                     "depth_scale_p5_p95": [round(float(p5), 4), round(float(p95), 4)],
                     "depth_scale_std": round(float(scale.std()), 4),
                     **geometry_stats(background, K, chosen["human"], depth, sc)})
    (dst / "meta.json.partial").write_text(json.dumps(meta, indent=1))
    (dst / "meta.json.partial").replace(dst / "meta.json")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "cache" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "cache")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "video_to_data_challenge" / "track_1")
    parser.add_argument("--sample", type=Path,
                        default=ROOT / "data" / "v2d_submission_kit" / "data" / "track_1_sample_submission.parquet")
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--overlay", type=Path, help="also write a check image per episode here")
    parser.add_argument("--overlay-only", action="store_true", help="skip building; just draw overlays")
    parser.add_argument("--objects", nargs="+", help="object track files in the raw folder to use instead of "
                        "sam_object.npz + sam_object_*.npz (e.g. the motion candidates)")
    args = parser.parse_args()
    scored = scored_frames(args.sample)
    rows = []
    for episode in parse_episodes(args.episodes):
        if not args.overlay_only:
            if not (args.raw / f"episode_{episode:06d}" / (args.objects or ["sam_object.npz"])[0]).exists():
                print(f"ep {episode}: no raw SAM output, skipped")
                continue
            meta = build(episode, args.raw, args.out, scored[episode], args.data, args.overlay, args.objects)
            keys = ["warning", "contact", "human_ids", "object_ids", "human_coverage_scored", "object_coverage_scored",
                    "object_area_px_scored_median", "fov_x", "depth_scale_std", "camera_height_m", "actor_top_m_p90"]
            rows.append({"ep": episode, **{k: meta.get(k) for k in keys}})
            print(rows[-1], flush=True)
        if args.overlay:
            overlay(episode, args.out / f"episode_{episode:06d}", args.overlay, scored[episode], args.data)


if __name__ == "__main__":
    main()

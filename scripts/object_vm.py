#!/usr/bin/env python3
"""Step 5 geometry stages (run on the Colab VM): object crops from the video, then image-to-3D meshes.

    python object_vm.py --stage crops [--episodes 0-29] [--per-episode 3]   # CPU: best unoccluded views
    python object_vm.py --stage crops --episodes 6-8 --diverse 3            # + 3 views with other mask shapes
    python object_vm.py --stage mesh  [--episodes 0-29] [--steps 50]        # GPU: Hunyuan3D-2.1 shape
    python object_vm.py --stage clean [--episodes 0-29]                     # CPU: re-clean existing meshes

crops: every frame with an object mask (Step 3 cache, whole clip) is scored by
  min(area / p95 area, 1) * (1 - fraction of the mask outline touching the actor)^2,
rejecting masks cut by the image border or smaller than half the p95 area. The best `--per-episode`
frames at least 30 frames apart are cut from the full-resolution video as square RGBA crops (alpha =
the object mask, upsampled from 1/2 resolution), 512 px, object filling ~85% of the side.
Writes <out>/crops/episode_XXXXXX_k.png, episode_XXXXXX.json (frame, score, box) and a contact sheet
<out>/crops/sheet.jpg.

mesh: Hunyuan3D-2.1 shape model (`tencent/Hunyuan3D-2.1`, hunyuan3d-dit-v2-1; geometry only, no texture)
on each crop. The mesh is cleaned (floaters, degenerate faces; only the main piece and parts near it,
see main_part), normalized (bounding-box centre at the
origin, longest side 1) and written twice: <out>/meshes/episode_XXXXXX_k.obj decimated to <= 4000 faces
(the packer's budget is 4,096; this is the mesh that gets scored) and episode_XXXXXX_k_hi.ply (<= 40k
faces, for viewing). Skips crops whose mesh exists, so reruns resume.
Setup on the VM: clone Tencent-Hunyuan/Hunyuan3D-2.1 to /content/Hunyuan3D-2.1 (only hy3dshape is used).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from human_vm import parse_episodes, video_path  # noqa: E402

MASK_STEP = 2  # cache masks are at 1/2 resolution
CROP = 512
FILL = 0.85  # object's longer box side / crop side
MIN_GAP = 30  # frames between picked crops
MAX_FACES = 4000
HI_FACES = 40000
HUNYUAN = Path("/content/Hunyuan3D-2.1")


# ---------------------------------------------------------------------------------------------- crops
def view_scores(obj: np.ndarray, human: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score per frame (0 = unusable) and the object's area, from the half-resolution masks."""
    area = obj.reshape(len(obj), -1).sum(1)
    ref = np.percentile(area[area > 0], 95) if (area > 0).any() else 1
    h, w = obj.shape[1:]
    kernel = np.ones((7, 7), np.uint8)
    score = np.zeros(len(obj))
    for f in np.flatnonzero(area >= 0.5 * ref):
        m = obj[f].astype(np.uint8)
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        if rows[0] <= 1 or cols[0] <= 1 or rows[-1] >= h - 2 or cols[-1] >= w - 2:
            continue
        outline = m - cv2.erode(m, np.ones((3, 3), np.uint8))
        near_actor = cv2.dilate(human[f].astype(np.uint8), kernel)
        touch = (outline & near_actor).sum() / max(outline.sum(), 1)
        score[f] = min(area[f] / ref, 1.0) * (1 - touch) ** 2
    return score, area


def pick_frames(score: np.ndarray, count: int) -> list[int]:
    picks = []
    for f in np.argsort(-score):
        if score[f] <= 0 or len(picks) == count:
            break
        if all(abs(int(f) - p) >= MIN_GAP for p in picks):
            picks.append(int(f))
    return picks


def shape_descriptor(mask: np.ndarray) -> np.ndarray:
    """Elongation and principal direction of a mask: [r, (1 - r) cos 2a, (1 - r) sin 2a], r = minor/major."""
    ys, xs = np.nonzero(mask)
    cov = np.cov(np.stack([xs, ys]).astype(float))
    w, v = np.linalg.eigh(cov)
    r = float(np.sqrt(max(w[0], 1e-9) / max(w[1], 1e-9)))
    a = np.arctan2(v[1, 1], v[0, 1])
    return np.array([r, (1 - r) * np.cos(2 * a), (1 - r) * np.sin(2 * a)])


def pick_diverse(score: np.ndarray, obj: np.ndarray, count: int, skip: list[int], floor: float = 0.6) -> list[int]:
    """Frames with score >= floor x max whose mask shapes differ most (farthest-point sampling on
    shape_descriptor), at least MIN_GAP frames from each other and from `skip`."""
    good = np.flatnonzero(score >= floor * score.max())
    good = good[::max(1, len(good) // 300)]
    desc = {int(f): shape_descriptor(obj[f]) for f in good}
    picks = []
    seeds = [desc[f] for f in skip if f in desc] or [desc[int(good[np.argmax(score[good])])]]
    while len(picks) < count:
        far = [(min(np.linalg.norm(desc[f] - d) for d in seeds), f) for f in desc
               if all(abs(f - p) >= MIN_GAP for p in picks + skip)]
        if not far:
            break
        _, f = max(far)
        picks.append(f)
        seeds.append(desc[f])
    return picks


def read_frame(video: Path, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame} of {video}")
    return bgr


def rgba_crop(bgr: np.ndarray, mask_half: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Square BGRA crop around the object (alpha = upsampled mask), CROP x CROP, and its full-res box."""
    H, W = bgr.shape[:2]
    alpha = cv2.resize(mask_half.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha = (np.clip((alpha - 0.5) * 4 + 0.5, 0, 1) * 255).astype(np.uint8)
    ys, xs = np.nonzero(alpha > 127)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0) / FILL
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    src = np.float32([[cx - side / 2, cy - side / 2], [cx + side / 2, cy - side / 2], [cx - side / 2, cy + side / 2]])
    dst = np.float32([[0, 0], [CROP, 0], [0, CROP]])
    A = cv2.getAffineTransform(src, dst)
    rgb = cv2.warpAffine(bgr, A, (CROP, CROP), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(alpha, A, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=0)
    rgb = np.where(a[..., None] > 0, rgb, 255).astype(np.uint8)  # white background outside the object
    return np.dstack([rgb, a]), [int(x0), int(y0), int(x1), int(y1)]


def stage_crops(args, episodes):
    out = args.out / "crops"
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for e in episodes:
        masks = np.load(args.cache / f"episode_{e:06d}" / "masks.npz")
        obj, human = masks["object"], masks["human"]
        score, area = view_scores(obj, human)
        index = out / f"episode_{e:06d}.json"
        info = json.loads(index.read_text()) if args.diverse and index.exists() else []
        if args.diverse:  # more crops, added after the existing ones, with different mask shapes
            picks = pick_diverse(score, obj, args.diverse, [i["frame"] for i in info])
        else:
            picks, info = pick_frames(score, args.per_episode), []
        if not picks:
            print(f"ep {e}: no usable view", flush=True)
            continue
        for k, f in enumerate(picks, start=len(info)):
            rgba, box = rgba_crop(read_frame(video_path(args.data, e), f), obj[f])
            cv2.imwrite(str(out / f"episode_{e:06d}_{k}.png"), rgba)  # BGRA, as cv2 writes PNG alpha
            info.append({"k": k, "frame": f, "score": round(float(score[f]), 3), "area_half_px": int(area[f]),
                         "box_full_px": box})
            tile = cv2.resize(rgba[..., :3], (192, 192))
            cv2.putText(tile, f"ep{e} #{k} f{f}", (4, 16), 0, 0.45, (0, 0, 255), 1)
            tiles.append(tile)
        index.write_text(json.dumps(info, indent=1))
        print(f"ep {e}: " + ", ".join(f"f{i['frame']} s{i['score']} a{i['area_half_px']}" for i in info), flush=True)
    if not tiles:  # no episode had a usable object mask: nothing to draw
        return
    per_row = 3 * 3
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + per_row], 1) for i in range(0, len(tiles), per_row)]
    sheet = "sheet_diverse.jpg" if args.diverse else "sheet.jpg"
    cv2.imwrite(str(out / sheet), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------------------------------------- mesh
def main_part(mesh, margin: float = 0.2):
    """The largest connected piece (by area) plus pieces inside its box grown by `margin` of its size.
    Drops duplicates and floaters away from the object (Hunyuan sometimes emits a second copy),
    keeps attached details that are separate components (wheels, handles)."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    main = max(parts, key=lambda m: m.area)
    lo, hi = main.bounds
    pad = margin * (hi - lo).max()
    keep = [m for m in parts if m is main or (np.all(m.bounds[0] >= lo - pad) and np.all(m.bounds[1] <= hi + pad))]
    if len(keep) == len(parts):
        return mesh
    import trimesh

    return trimesh.util.concatenate(keep)


def normalize(mesh):
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    mesh.apply_scale(1.0 / float((hi - lo).max()))
    return mesh


def decimate(mesh, faces: int):
    import fast_simplification
    import trimesh

    if len(mesh.faces) <= faces:
        return mesh
    v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                        target_reduction=1 - faces / len(mesh.faces))
    out = trimesh.Trimesh(v, f, process=True)
    while len(out.faces) > faces:  # the reduction is approximate
        v, f = fast_simplification.simplify(out.vertices.astype(np.float32), out.faces.astype(np.int32),
                                            target_reduction=1 - 0.97 * faces / len(out.faces))
        out = trimesh.Trimesh(v, f, process=True)
    return out


def stage_mesh(args, episodes):
    import torch
    from PIL import Image

    sys.path.insert(0, str(HUNYUAN / "hy3dshape"))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    out = args.out / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        for crop in sorted((args.out / "crops").glob(f"episode_{e:06d}_*.png")):
            dst = out / f"{crop.stem}.obj"
            if dst.exists():
                continue
            try:
                t0 = time.time()
                image = Image.open(crop).convert("RGBA")
                mesh = pipe(image=image, num_inference_steps=args.steps, octree_resolution=args.octree,
                            generator=torch.manual_seed(args.seed), output_type="trimesh")[0]
                mesh = normalize(main_part(DegenerateFaceRemover()(FloaterRemover()(mesh))))
                decimate(mesh, HI_FACES).export(out / f"{crop.stem}_hi.ply")
                low = decimate(mesh, MAX_FACES)
                low.export(dst)
                print(f"ep {e} {crop.stem}: {len(mesh.faces)} -> {len(low.faces)} faces, "
                      f"watertight {low.is_watertight}, extent {np.round(low.extents, 3).tolist()}, "
                      f"{time.time() - t0:.0f} s", flush=True)
            except Exception:
                print(f"ep {e} {crop.stem} FAILED", flush=True)
                traceback.print_exc()
            torch.cuda.empty_cache()


def stage_clean(args, episodes):
    """Re-apply main_part to existing meshes (from the _hi copy) and rewrite both files."""
    import trimesh

    out = args.out / "meshes"
    for e in episodes:
        for hi in sorted(out.glob(f"episode_{e:06d}_*_hi.ply")):
            mesh = trimesh.load(hi, force="mesh")
            clean = normalize(main_part(mesh))
            if len(clean.faces) == len(mesh.faces):
                continue
            clean.export(hi)
            low = decimate(clean, MAX_FACES)
            low.export(out / f"{hi.stem[:-3]}.obj")
            print(f"ep {e} {hi.stem[:-3]}: {len(mesh.faces)} -> {len(clean.faces)} faces kept, "
                  f"extent {np.round(low.extents, 3).tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["crops", "mesh", "clean"], required=True)
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--per-episode", type=int, default=3)
    parser.add_argument("--diverse", type=int, default=0,
                        help="crops: add this many crops (indices after the existing ones) with different mask shapes")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--cache", type=Path, default=REPO / "cache")
    parser.add_argument("--out", type=Path, default=Path("/content/object_raw"))
    args = parser.parse_args()
    token = Path("/content/.hf_token")
    if token.exists():
        os.environ.setdefault("HF_TOKEN", token.read_text().strip())
    {"crops": stage_crops, "mesh": stage_mesh, "clean": stage_clean}[args.stage](args, parse_episodes(args.episodes))


if __name__ == "__main__":
    main()

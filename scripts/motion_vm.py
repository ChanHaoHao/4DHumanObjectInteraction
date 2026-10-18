#!/usr/bin/env python3
"""Name-free object finder (runs on the Colab VM, GPU): D4RT says what moves, SAM 3 turns that into whole objects.

    python motion_vm.py --stage motion --episodes 0,5 [--out DIR] [--grid 144x192] [--stride 2]
    python motion_vm.py --stage seed   --episodes 0,5 [--out DIR]
    python motion_vm.py --stage text   --episodes 0,5 [--out DIR]
    python motion_vm.py --stage box    --episodes 0,5 [--out DIR] [--locate DIR] [--compare CACHE_DIR]

motion  D4RT predicts, for pixel (u, v) of frame t, where that surface point is at time t' (camera of frame t).
        Every --stride-th frame, a 144x192 grid is queried at t, t +- 6 and t + 20 (t - 20 near the window end);
        motion_k = max 3D displacement over the gap-k targets / depth at t (scale-free). Each source frame is
        decoded in the 48-frame window where it is most central. Writes <out>/<ep>_motion.npz (frames, m6, m20,
        xyz0, xyz20, conf) and <out>/<ep>_motion_small.npz (frames, m6, m20).
seed    Uses the motion and the SAM 3 "person" tracks (cache/raw/<ep>/sam_person.npz):
        1. excess = motion - its per-cell median over the clip (static thin edges jitter every frame); moving =
           excess > --tau at either gap.
        2. actor = the person track with the most moving cells per frame.
        3. per frame, moving components (>= --min-cells) outside every person (dilated 1 cell) that touch the actor
           (dilated 3 cells); score = cells, halved if D4RT's 3D motion inside is not rigid (one rotation +
           translation from t to t + 20 leaves median residual > 0.5 x the median displacement).
        4. up to --seeds seed frames (best score, >= --seed-gap frames apart, over the whole clip).
        5. per seed, SAM 3's tracker on that frame with 3 prompt variants: positive points inside the component;
           + negative points on the actor next to it; + a box around the component. The variant kept covers the
           component best with little person overlap and a sane size; then it is tracked both ways over the clip.
        Writes <out>/<ep>_cand.npz (sam_object*.npz format, one track per seed), <out>/<ep>_seed.json and a check
        image <out>/<ep>_seed.jpg.
box     LocateAnything keyframe boxes -> SAM 3 tracker tracks voted on by the boxes (see stage_box).
text    SAM 3 text tracks for the dataset's own metadata fields (no hand-written names): `object` with
        underscores as spaces, and `object_prompt`. Writes <out>/<ep>_text.npz (both pooled; prompt per track).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cache import load_tracks, parse_episodes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
W, H = 1536, 1152
MASK_STEP = 2
GROW = ndimage.generate_binary_structure(2, 2)


def video_path(data: Path, episode: int) -> Path:
    return data / "videos/chunk-000/observation.images.exo_camera" / f"episode_{episode:06d}.mp4"


def save(path: Path, **arrays):
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ motion
def stage_motion(args, episodes):
    import torch
    from d4rt_vm import decode, load_model, read_frames, windows

    model, clip = load_model(args.repo, args.ckpt_dir, "cuda")
    gh, gw = args.grid
    for e in episodes:
        name = f"episode_{e:06d}"
        path = args.out / f"{name}_motion.npz"
        if path.exists():
            continue
        t0 = time.time()
        frames = read_frames(video_path(args.data, e))
        n = len(frames)
        xs = (np.arange(gw) + 0.5) * W / gw - 0.5
        ys = (np.arange(gh) + 0.5) * H / gh - 0.5
        gu, gv = np.meshgrid(xs / (W - 1), ys / (H - 1))
        gu = torch.tensor(gu.ravel(), device="cuda", dtype=torch.float32)
        gv = torch.tensor(gv.ravel(), device="cuda", dtype=torch.float32)
        q = gh * gw
        aspect = torch.tensor([[W / H]], device="cuda", dtype=torch.float32)
        wins = windows(n, clip, args.overlap)
        centres = np.array([(a + b - 1) / 2 for a, b in wins])
        src = np.arange(0, n, args.stride)
        owner = np.abs(src[:, None] - centres[None]).argmin(1)
        m6 = np.zeros((len(src), gh, gw), np.float16)
        m20 = np.zeros_like(m6)
        xyz0 = np.zeros((len(src), gh, gw, 3), np.float16)
        xyz20 = np.zeros_like(xyz0)
        conf = np.zeros_like(m6)
        for wi, (a, b) in enumerate(wins):
            mine = np.flatnonzero(owner == wi)
            if not len(mine):
                continue
            k = b - a
            video = torch.from_numpy(frames[a:b]).to("cuda").permute(0, 3, 1, 2)[None].float() / 255.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                memory = model.encode_video(video=video, aspect_ratio=aspect)
            loc = src[mine] - a
            far = np.where(loc + 20 < k, loc + 20, np.maximum(loc - 20, 0))
            tgt = np.stack([loc, np.maximum(loc - 6, 0), np.minimum(loc + 6, k - 1), far])  # [4,S]
            t_src = torch.tensor(np.repeat(np.tile(loc, len(tgt)), q), device="cuda")
            t_tgt = torch.tensor(np.repeat(tgt.ravel(), q), device="cuda")
            xyz, c = decode(model, video, memory, gu.repeat(t_src.numel() // q), gv.repeat(t_src.numel() // q),
                            t_src, t_tgt, t_src, args.chunk, True)
            xyz = xyz.view(len(tgt), len(mine), gh, gw, 3).cpu().numpy()
            base = xyz[0]
            depth = np.maximum(base[..., 2], 1e-6)
            d6 = np.maximum(np.linalg.norm(xyz[1] - base, axis=-1), np.linalg.norm(xyz[2] - base, axis=-1))
            m6[mine] = d6 / depth
            m20[mine] = np.linalg.norm(xyz[3] - base, axis=-1) / depth
            xyz0[mine], xyz20[mine] = base, xyz[3]
            conf[mine] = c.view(len(tgt), len(mine), gh, gw)[0].cpu().numpy()
            del video, memory
        save(path, frames=src, m6=m6, m20=m20, xyz0=xyz0, xyz20=xyz20, conf=conf)
        save(args.out / f"{name}_motion_small.npz", frames=src, m6=m6, m20=m20)
        print(f"ep {e} motion: {n} frames, {len(wins)} windows, {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()


def stage_change(args, episodes):
    """Model-free motion for a static camera: RGB change at the same pixel, t vs t +- 6 and t + 20 (mean abs
    difference over channels, 0-1, at the grid). Motion inside people is kept (it picks the actor); their 2-cell
    fringe and ghosts (cells a person covers at the other time only) are blanked. MoGe depth change was tried first:
    its frame-to-frame noise on the floor is > 1% and swamps a hand-held object (notes/preprocess.md).
    Same file as stage_motion without xyz (no rigidity check). Numpy + OpenCV only."""
    gh, gw = args.grid
    for e in episodes:
        name = f"episode_{e:06d}"
        path = args.out / f"{name}_motion.npz"
        if path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path(args.data, e)))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(bgr, (gw, gh), interpolation=cv2.INTER_AREA))
        cap.release()
        rgb = np.stack(frames).astype(np.float32) / 255.0
        n = len(rgb)
        inside = pool(load_tracks(args.cache / "raw" / name / "sam_person.npz")["masks"][:, :n].any(0), gh, gw)
        person = np.stack([dilate(m, 2) for m in inside])
        src = np.arange(0, n, args.stride)

        def change(tgt):
            c = np.abs(rgb[tgt] - rgb[src]).mean(-1)
            return np.where((person[src] | person[tgt]) & ~inside[src], 0.0, c)

        m6 = np.maximum(change(np.maximum(src - 6, 0)), change(np.minimum(src + 6, n - 1)))
        m20 = change(np.where(src + 20 < n, src + 20, np.maximum(src - 20, 0)))
        save(path, frames=src, m6=m6.astype(np.float16), m20=m20.astype(np.float16))
        print(f"ep {e} change: {n} frames", flush=True)


# -------------------------------------------------------------------------------------------------- seed
def pool(masks: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """[..., 576, 768] bool -> [..., gh, gw]: a cell is set if any of its pixels is."""
    *lead, h, w = masks.shape
    return masks.reshape(*lead, gh, h // gh, gw, w // gw).any((-3, -1))


def dilate(m: np.ndarray, it: int) -> np.ndarray:
    return ndimage.binary_dilation(m, GROW, it) if m.any() else m


def rigid_error(p: np.ndarray, q: np.ndarray) -> float:
    """Median residual of the best rotation + translation p -> q, over the median displacement (inf if static)."""
    if len(p) < 6:
        return float("inf")
    move = np.median(np.linalg.norm(q - p, axis=1))
    if move <= 0:
        return float("inf")
    mp, mq = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - mp).T @ (q - mq))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    R = vt.T @ d @ u.T
    res = np.linalg.norm((p - mp) @ R.T + mq - q, axis=1)
    return float(np.median(res) / move)


def spread_points(region: np.ndarray, k: int, avoid: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Up to k (row, col) cells deep inside `region`, spread apart (distance-transform maxima)."""
    dist = ndimage.distance_transform_edt(region)
    if avoid is not None:
        dist[avoid] = 0
    pts = []
    rr, cc = np.ogrid[:region.shape[0], :region.shape[1]]
    for _ in range(k):
        r, c = np.unravel_index(dist.argmax(), dist.shape)
        if dist[r, c] <= 0:
            break
        pts.append((int(r), int(c)))
        dist[(rr - r) ** 2 + (cc - c) ** 2 <= max(9.0, region.sum() / 10)] = 0
    return pts


def find_seeds(e: int, args) -> dict:
    """Actor and seed components from the motion and the person tracks (grid resolution)."""
    gh, gw = args.grid
    name = f"episode_{e:06d}"
    mo = np.load(args.out / f"{name}_motion.npz")
    frames = mo["frames"]
    ex = []
    for key in ("m6", "m20"):
        m = mo[key].astype(np.float32)
        ex.append(np.maximum(m - np.median(m, 0), 0))
    moving = (ex[0] > args.tau) | (ex[1] > args.tau)
    people = load_tracks(args.cache / "raw" / name / "sam_person.npz")
    persons = pool(people["masks"][:, frames], gh, gw)  # [K,S,gh,gw]
    present = persons.sum((2, 3)) >= 3
    counts = [(moving & persons[k]).sum((1, 2))[present[k]].mean() if present[k].any() else 0.0 for k in range(len(persons))]
    actor = int(np.argmax(counts))
    anyone = persons.any(0)
    xyz0, xyz20 = (mo["xyz0"], mo["xyz20"]) if "xyz0" in mo else (None, None)  # stage_change has no 3D points
    found = []
    for s in range(len(frames)):
        body = persons[actor, s]
        if body.sum() < 10:
            continue
        cand = moving[s] & ~dilate(anyone[s], 1)
        near = dilate(body, 3)
        lab, n = ndimage.label(cand, GROW)
        for i in range(1, n + 1):
            comp = lab == i
            cells = int(comp.sum())
            if cells < args.min_cells or not (comp & near).any():
                continue
            rig = 0.0 if xyz0 is None else rigid_error(xyz0[s][comp].astype(np.float32),
                                                       xyz20[s][comp].astype(np.float32))
            found.append({"s": s, "frame": int(frames[s]), "cells": cells, "rigid_err": round(rig, 3),
                          "score": cells * (1.0 if rig <= 0.5 else 0.5), "comp": comp})
    found.sort(key=lambda d: -d["score"])
    seeds = []
    for d in found:
        if len(seeds) == args.seeds:
            break
        if all(abs(d["frame"] - x["frame"]) >= args.seed_gap for x in seeds):
            seeds.append(d)
    return {"frames": frames, "actor": actor, "actor_id": str(people["ids"][actor]),
            "actor_counts": np.round(counts, 1).tolist(), "persons": persons, "moving": moving, "seeds": seeds,
            "components_found": len(found)}


def cell_px(rc) -> list[float]:
    """Grid cell (row, col) -> its centre in video pixels (x, y)."""
    r, c = rc
    return [(c + 0.5) * 8.0, (r + 0.5) * 8.0]


def add_prompt(tracker, state, frame, points, labels, box):
    """SAM 3 tracker prompt (relative coordinates; a box goes in as SAM 2's corner points) on one frame;
    returns that frame's mask [1152,1536] bool."""
    import torch

    rel = torch.tensor(np.array(points, float) / np.array([W, H]), dtype=torch.float32)
    b = None if box is None else torch.tensor(np.array(box, float).reshape(2, 2) / np.array([W, H]),
                                             dtype=torch.float32).reshape(-1)
    _, _, _, masks = tracker.add_new_points_or_box(inference_state=state, frame_idx=frame, obj_id=1, points=rel,
                                                   labels=torch.tensor(labels, dtype=torch.int32),
                                                   clear_old_points=True, box=b)
    return (masks.reshape(-1, *masks.shape[-2:])[0] > 0).cpu().numpy()


def stage_seed(args, episodes):
    import torch
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    gh, gw = args.grid
    for e in episodes:
        name = f"episode_{e:06d}"
        if (args.out / f"{name}_cand.npz").exists():
            continue
        t0 = time.time()
        info = find_seeds(e, args)
        video = video_path(args.data, e)
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        state = tracker.init_state(video_path=str(video))
        report = {"episode": e, "actor_track": info["actor_id"], "actor_counts": info["actor_counts"],
                  "components_found": info["components_found"], "seeds": []}
        tracks_probs, tracks_masks, prompts = [], [], []
        for seed in info["seeds"]:
            s, f, comp = seed["s"], seed["frame"], seed["comp"]
            body = info["persons"][info["actor"], s]
            anyone = info["persons"][:, s].any(0)
            pos = [cell_px(p) for p in spread_points(comp, 3)]
            # negatives: actor cells closest to the component (the arm holding it), spread apart
            ring = body & dilate(comp, 6)
            neg = [cell_px(p) for p in spread_points(ring if ring.any() else body, 3)]
            rows, cols = np.nonzero(comp)
            pad_r, pad_c = max(2, int(0.3 * (np.ptp(rows) + 1))), max(2, int(0.3 * (np.ptp(cols) + 1)))
            box = [max(0, cols.min() - pad_c) * 8.0, max(0, rows.min() - pad_r) * 8.0,
                   min(gw, cols.max() + 1 + pad_c) * 8.0, min(gh, rows.max() + 1 + pad_r) * 8.0]
            variants = {"pos": (pos, [1] * len(pos), None),
                        "pos_neg": (pos + neg, [1] * len(pos) + [0] * len(neg), None),
                        "box_pos_neg": (pos + neg, [1] * len(pos) + [0] * len(neg), box)}
            scored = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for v, (pts, labels, b) in variants.items():
                    tracker.clear_all_points_in_video(state)
                    try:
                        mask = add_prompt(tracker, state, f, pts, labels, b)
                    except Exception as err:  # noqa: BLE001
                        scored[v] = {"error": repr(err)[:200]}
                        continue
                    cells = pool(mask[::MASK_STEP, ::MASK_STEP], gh, gw)
                    area = cells.sum()
                    cover = float((cells & comp).sum() / comp.sum())
                    person = float((cells & anyone).sum() / max(area, 1))
                    big = area > 0.25 * gh * gw
                    scored[v] = {"cover": round(cover, 3), "person": round(person, 3), "cells": int(area),
                                 "score": round(cover - person - (1.0 if big else 0.0), 3)}
            ok = {v: d for v, d in scored.items() if "score" in d}
            best = max(ok, key=lambda v: ok[v]["score"]) if ok else None
            entry = {k: seed[k] for k in ("frame", "cells", "rigid_err", "score")}
            entry.update(pos=pos, neg=neg, box=box, variants=scored, chosen=best)
            if best is None or ok[best]["cover"] < 0.3 or ok[best]["person"] > 0.5:
                entry["dropped"] = "no variant covers the motion without the person"
                report["seeds"].append(entry)
                continue
            pts, labels, b = variants[best]
            probs = np.zeros(n, np.float16)
            packed = np.zeros((n, H // MASK_STEP, W // MASK_STEP // 8), np.uint8)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tracker.clear_all_points_in_video(state)
                add_prompt(tracker, state, f, pts, labels, b)
                for reverse in (False, True):
                    for fi, _, _, video_masks, scores in tracker.propagate_in_video(
                            state, start_frame_idx=f, max_frame_num_to_track=n, reverse=reverse, propagate_preflight=True):
                        mask = (video_masks[0] > 0).reshape(video_masks.shape[-2:]).cpu().numpy()
                        if mask.any():
                            probs[fi] = max(float(torch.sigmoid(torch.as_tensor(scores).float().flatten()[0])), 1e-3)
                            packed[fi] = np.packbits(mask[::MASK_STEP, ::MASK_STEP], axis=-1)
            entry["frames_tracked"] = int((probs > 0).sum())
            report["seeds"].append(entry)
            tracks_probs.append(probs)
            tracks_masks.append(packed)
            prompts.append(f"motion: frame {f} {best}")
        if tracks_probs:
            save(args.out / f"{name}_cand.npz", obj_ids=np.arange(1, len(tracks_probs) + 1), probs=np.stack(tracks_probs),
                 masks=np.stack(tracks_masks), prompt=" | ".join(prompts), image_size=np.array([W, H]))
        else:
            save(args.out / f"{name}_cand.npz", obj_ids=np.zeros(0, np.int64), probs=np.zeros((0, n), np.float16),
                 masks=np.zeros((0, n, H // MASK_STEP, W // MASK_STEP // 8), np.uint8), prompt="motion: none",
                 image_size=np.array([W, H]))
        report["seconds"] = round(time.time() - t0, 1)
        (args.out / f"{name}_seed.json").write_text(json.dumps(report, indent=1))
        check_image(e, info, report, tracks_masks, args)
        print(f"ep {e} seed: actor {info['actor_id']} {info['actor_counts']}, {info['components_found']} components, "
              f"seeds " + "; ".join(f"f{d['frame']} {d['cells']}c rig {d['rigid_err']} -> {d.get('chosen')} "
                                    f"{d.get('frames_tracked', d.get('dropped'))}" for d in report["seeds"])
              + f" ({report['seconds']}s)", flush=True)
        del state
        torch.cuda.empty_cache()


def check_image(e: int, info: dict, report: dict, tracks_masks: list, args):
    """Per seed: RGB with the moving component (red), prompts (green +, blue -, box) | the tracked mask there."""
    from d4rt_vm import read_full

    seeds = [d for d in report["seeds"]]
    if not seeds:
        return
    full = read_full(video_path(args.data, e), [d["frame"] for d in seeds])
    rows, k = [], 0
    for d, seed in zip(seeds, info["seeds"]):
        f = d["frame"]
        rgb = cv2.resize(full[f], (768, 576), interpolation=cv2.INTER_AREA)
        left = rgb.copy()
        comp = cv2.resize(seed["comp"].astype(np.uint8), (768, 576), interpolation=cv2.INTER_NEAREST) > 0
        left[comp] = (0.5 * left[comp] + [127, 0, 0]).astype(np.uint8)
        for x, y in d["pos"]:
            cv2.circle(left, (int(x / 2), int(y / 2)), 6, (0, 255, 0), -1)
        for x, y in d["neg"]:
            cv2.circle(left, (int(x / 2), int(y / 2)), 6, (0, 80, 255), -1)
        x0, y0, x1, y1 = (int(v / 2) for v in d["box"])
        cv2.rectangle(left, (x0, y0), (x1, y1), (255, 255, 0), 2)
        right = (rgb * 0.5).astype(np.uint8)
        if "frames_tracked" in d:
            m = np.unpackbits(tracks_masks[k][f], axis=-1)[:, :768].astype(bool)
            right[m] = (0.4 * rgb[m] + [0, 150, 0]).astype(np.uint8)
            k += 1
        row = np.concatenate([left, right], 1)
        cv2.putText(row, f"f{f} {d.get('chosen')} rig {d['rigid_err']}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        rows.append(row)
    cv2.imwrite(str(args.out / f"episode_{e:06d}_seed.jpg"), np.concatenate(rows, 0)[..., ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 85])


# -------------------------------------------------------------------------------------------------- text
def mask_box(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return None if not len(rows) else np.array([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], float)


def iou_xyxy(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def mean_iou(packed: np.ndarray, ref: Path) -> float | None:
    """Mean IoU of a packed track [n,576,96] vs a cache masks.npz object mask, over frames where either is non-empty."""
    if not ref.exists():
        return None
    r = np.load(ref)
    ours = np.unpackbits(packed[r["frames"]], axis=-1).astype(bool)
    theirs = r["object"]
    union = (ours | theirs).sum((1, 2))
    on = union > 0
    return round(float(((ours & theirs).sum((1, 2))[on] / union[on]).mean()), 3) if on.any() else None


STILL = 0.85  # same cut as e2e_vm's track check


def track_stillness(packed: np.ndarray, ref: Path) -> float:
    """track_check.stillness of a packed track [n,576,96] on the frames of a cache masks.npz, its actor removed."""
    from track_check import stillness

    r = np.load(ref)
    obj = np.unpackbits(packed[r["frames"], ::4], axis=-1)[:, :, ::4].astype(bool)
    return stillness(obj, r["human"][:, ::4, ::4])[0]


def pick(votes, still) -> int:
    """Most votes, unless that track is still and a moving one has >= 2 votes (the seed's own + one more box):
    then the moving one with most votes. Tables and monitors win box votes when the object rests on or near them."""
    votes, still = np.asarray(votes), np.asarray(still, float)
    best = int(votes.argmax())
    moving = (still < STILL) & (votes >= 2)
    if still[best] >= STILL and moving.any():
        return int(np.where(moving, votes, -1).argmax())
    return best


def stage_box(args, episodes):
    """LocateAnything keyframe boxes (<locate>/<ep>_locate.npz) -> SAM 3 tracker tracks. A track gets one vote per
    keyframe box its mask's box overlaps (IoU >= --vote-iou); a box no track explains seeds the next track (up to
    --seeds, boxes closest to the median box area first). Winner: see pick(). Writes <ep>_box.npz (sam_object*.npz
    format, all tracks, + votes, winner) and <ep>_box.json, with mean IoU vs <cache>/<ep>/masks.npz and --compare."""
    import torch
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    for e in episodes:
        name = f"episode_{e:06d}"
        if (args.out / f"{name}_box.npz").exists():
            continue
        t0 = time.time()
        loc = np.load(args.locate / f"{name}_locate.npz")
        kf, boxes = loc["frames"], loc["boxes"][:, :4]
        valid = loc["boxes"][:, 4] > 0
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = [int(i) for i in np.argsort(np.abs(np.log(area / np.median(area[valid])))) if valid[i]]
        video = video_path(args.data, e)
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        state = tracker.init_state(video_path=str(video))
        explained = ~valid
        tracks, report = [], []
        for i in order:
            if len(tracks) >= args.seeds:
                break
            if explained[i]:
                continue
            f = int(kf[i])
            probs = np.zeros(n, np.float16)
            packed = np.zeros((n, H // MASK_STEP, W // MASK_STEP // 8), np.uint8)
            at_kf = {}
            # shortcut: box prompt only, no negative points on the actor; add them if a box around a hoop grabs the person
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tracker.clear_all_points_in_video(state)
                add_prompt(tracker, state, f, np.zeros((0, 2)), [], boxes[i])
                for reverse in (False, True):
                    for fi, _, _, video_masks, scores in tracker.propagate_in_video(
                            state, start_frame_idx=f, max_frame_num_to_track=n, reverse=reverse, propagate_preflight=True):
                        mask = (video_masks[0] > 0).reshape(video_masks.shape[-2:]).cpu().numpy()
                        if mask.any():
                            probs[fi] = max(float(torch.sigmoid(torch.as_tensor(scores).float().flatten()[0])), 1e-3)
                            packed[fi] = np.packbits(mask[::MASK_STEP, ::MASK_STEP], axis=-1)
                        if fi in kf:
                            at_kf[int(fi)] = mask_box(mask)
            ious = [0.0 if at_kf.get(int(t)) is None else iou_xyxy(at_kf[int(t)], b) for t, b in zip(kf, boxes)]
            votes = valid & (np.array(ious) >= args.vote_iou)
            votes[i] = True
            explained |= votes
            tracks.append((probs, packed))
            report.append({"seed_frame": f, "seed_box": boxes[i].round(1).tolist(), "votes": int(votes.sum()),
                           "kf_iou": np.round(ious, 2).tolist(), "frames_tracked": int((probs > 0).sum())})
            print(f"ep {e} track {len(tracks)}: seed f{f}, votes {int(votes.sum())}/{int(valid.sum())}", flush=True)
        votes = np.array([r["votes"] for r in report])
        for r, (_, packed) in zip(report, tracks):
            r["still"] = round(track_stillness(packed, args.cache / name / "masks.npz"), 3)
        winner = pick(votes, [r["still"] for r in report])
        for r, (_, packed) in zip(report, tracks):
            r["iou_ref"] = mean_iou(packed, args.cache / name / "masks.npz")
            r["iou_compare"] = mean_iou(packed, args.compare / name / "masks.npz") if args.compare else None
        out = {"episode": e, "winner": winner, "tracks": report, "seconds": round(time.time() - t0, 1),
               "compare_vs_ref": None}
        if args.compare and (args.compare / name / "masks.npz").exists():
            c = np.load(args.compare / name / "masks.npz")
            full = np.zeros((n, H // MASK_STEP, W // MASK_STEP // 8), np.uint8)
            full[c["frames"]] = np.packbits(c["object"], axis=-1)
            out["compare_vs_ref"] = mean_iou(full, args.cache / name / "masks.npz")
        save(args.out / f"{name}_box.npz", obj_ids=np.arange(1, len(tracks) + 1), probs=np.stack([t[0] for t in tracks]),
             masks=np.stack([t[1] for t in tracks]), prompt=" | ".join(f"box: frame {r['seed_frame']}" for r in report),
             image_size=np.array([W, H]), votes=votes, winner=winner)
        (args.out / f"{name}_box.json").write_text(json.dumps(out, indent=1))
        print(f"ep {e} box: winner {winner + 1}, votes {votes.tolist()}, iou_ref "
              f"{[r['iou_ref'] for r in report]}, compare {out['compare_vs_ref']} ({out['seconds']}s)", flush=True)
        del state
        torch.cuda.empty_cache()


def stage_text(args, episodes):
    import torch
    from preprocess_vm import run_sam
    from sam3.model_builder import build_sam3_video_predictor

    meta = {}
    with open(args.data / "meta/episodes_metadata.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            meta[d["episode_index"]] = d
    predictor = build_sam3_video_predictor()
    for e in episodes:
        path = args.out / f"episode_{e:06d}_text.npz"
        if path.exists():
            continue
        video = video_path(args.data, e)
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        parts = []
        for prompt in (meta[e]["object"].replace("_", " "), meta[e]["object_prompt"].strip().rstrip(".")):
            t0 = time.time()
            r = run_sam(predictor, video, prompt, n)
            print(f"ep {e} text '{prompt}': tracks {dict(zip(r['obj_ids'].tolist(), (r['probs'] > 0).sum(1).tolist()))} "
                  f"{time.time() - t0:.0f}s", flush=True)
            parts.append((prompt, r))
            torch.cuda.empty_cache()
        save(path, obj_ids=np.concatenate([np.array([f"{i}:{x}" for x in r["obj_ids"]], dtype="U16")
                                           for i, (_, r) in enumerate(parts)]),
             probs=np.concatenate([r["probs"] for _, r in parts]), masks=np.concatenate([r["masks"] for _, r in parts]),
             prompt=" | ".join(p for p, _ in parts),
             track_prompt=np.array([p for p, r in parts for _ in r["obj_ids"]]), image_size=np.array([W, H]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["motion", "change", "seed", "text", "box"], required=True)
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--repo", type=Path, default=Path("/content/Open-d4rt"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/content/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--cache", type=Path, default=REPO / "cache")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "motion2")
    parser.add_argument("--grid", default="144x192")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--tau", type=float, default=0.01, help="moving: excess displacement / depth (D4RT); use 0.1 (RGB change, 0-1) with --stage change")
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-gap", type=int, default=30)
    parser.add_argument("--locate", type=Path, default=REPO / "results" / "locate", help="box stage: LocateAnything npz dir")
    parser.add_argument("--compare", type=Path, help="box stage: another cache dir whose object masks are scored too")
    parser.add_argument("--vote-iou", type=float, default=0.5)
    args = parser.parse_args()
    args.grid = tuple(int(x) for x in args.grid.split("x"))
    args.out.mkdir(parents=True, exist_ok=True)
    token = Path("/content/.hf_token")
    if token.exists():
        os.environ.setdefault("HF_TOKEN", token.read_text().strip())
    episodes = parse_episodes(args.episodes)
    if args.stage == "motion":
        stage_motion(args, episodes)
    elif args.stage == "change":
        stage_change(args, episodes)
    else:
        import torch

        with torch.inference_mode():
            {"seed": stage_seed, "text": stage_text, "box": stage_box}[args.stage](args, episodes)


if __name__ == "__main__":
    sys.exit(main())

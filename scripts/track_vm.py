#!/usr/bin/env python3
"""D4RT point tracking (runs on the Colab VM, GPU): long-range 2D + 3D tracks of query points through a whole episode.

    python track_vm.py --episodes 0 [--out DIR] [--query-frames 0 | --query-every 100] [--grid 30x40]
        [--mask-points 0] [--trail 10] [--no-video]

Query points, per query frame s (default: frame 0 only): a uniform --grid over the image, plus optionally
--mask-points each sampled inside the actor and the object masks of the D4RT cache (cache/d4rt/<ep>/masks.npz, eroded so points sit off the silhouette edge).

Tracking. OpenD4RT answers "(u, v, t_src) at t_tgt in t_cam" with the point's 3D position (xyz_3d), its 2D position
in frame t_tgt (uv_2d, normalized like the query), a visibility logit and a confidence. The checkpoint sees 48 frames,
so targets are cut into consecutive 47-frame segments and each segment is decoded in a clip made of the source frame
plus those frames (48 consecutive frames when the source falls inside) -- the anchor-clip scheme of OpenD4RT's
infer_track_3d.py, with one encode per segment instead of one per target. Every target is reached straight from the
source frame, so nothing drifts by chaining. t_cam = t_tgt (xyz in the camera of the target frame; the camera is
static here, see notes/preprocess.md).

Metric scale. D4RT's xyz has no scale and each clip gets its own; each (source, segment) clip is scaled by the robust
median of cache depth / z over its visible predictions (cache depth = the D4RT cache, metric via MoGe-2), sampled at
the predicted uv.

Writes <out>/<ep>_tracks.npz:
    query_frames [S]; src [P] (query frame of each point); kind [P] (0 grid, 1 actor, 2 object);
    query_uv [P,2] px at 1536x1152; uv [P,T,2] px; visible [P,T] bool; vis_prob, conf [P,T] float16;
    xyz [P,T,3] float16 metres (camera of frame t); clip_scale [S,G] (one per source and segment)
<out>/<ep>_summary.json (visibility, static-point drift, mask hit rates, scale spread, timing),
<out>/<ep>_tracks.mp4 (768x576 overlay: per frame the tracks of the nearest query frame, with trails) and
<out>/<ep>_check.jpg (6 frames of the same overlay).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d4rt_vm import IMAGE_SIZE, load_model, read_frames, robust_scale, save, video_path  # noqa: E402

W, H = IMAGE_SIZE
KIND_COLOR = {1: (255, 80, 220), 2: (255, 160, 0)}  # RGB: actor magenta, object orange; grid: rainbow by position


@torch.no_grad()
def decode_all(model, video, memory, u, v, t_src, t_tgt, chunk: int):
    """xyz [N,3], uv [N,2] (normalized), visibility probability [N], confidence [N] for flat query tensors."""
    outs = {"xyz_3d": [], "uv_2d": [], "visibility": [], "confidence": []}
    for a in range(0, len(u), chunk):
        q = {"u": u[a:a + chunk][None], "v": v[a:a + chunk][None], "t_src": t_src[a:a + chunk][None],
             "t_tgt": t_tgt[a:a + chunk][None], "t_cam": t_tgt[a:a + chunk][None]}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.decode_queries(video=video, query=q, memory=memory)
        for k in outs:
            outs[k].append(pred[k][0].float())
    xyz, uv = torch.cat(outs["xyz_3d"]), torch.cat(outs["uv_2d"])
    return xyz, uv, torch.sigmoid(torch.cat(outs["visibility"])), torch.sigmoid(torch.cat(outs["confidence"]))


def segments(n: int, clip: int, src: int) -> list[tuple[list[int], int, int]]:
    """(clip frame indices, first target, end target) covering [0, n) in clip - 1 frame steps, each clip holding src."""
    if n <= clip:
        return [(list(range(n)), 0, n)]
    out = []
    step = clip - 1
    for a in range(0, n, step):
        b = min(a + step, n)
        lo = min(a, n - step)
        if lo <= src < lo + step:  # source inside: 48 consecutive frames around the targets
            lo = min(lo, n - clip)
            ctx = list(range(lo, lo + clip))
        else:
            ctx = sorted([src] + list(range(lo, lo + step)))
        out.append((ctx, a, b))
    return out


def sample_mask(mask: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Up to k px positions (full resolution) inside an eroded half-resolution mask, spread by a stratified draw."""
    if k <= 0:
        return np.zeros((0, 2), np.float32)
    eroded = ndimage.binary_erosion(mask, iterations=2)
    if eroded.sum() < k:
        eroded = mask
    ys, xs = np.nonzero(eroded)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    order = np.lexsort((xs, ys))  # raster order, then one draw per stratum
    edges = np.linspace(0, len(order), min(k, len(order)) + 1).astype(int)
    pick = order[[rng.integers(lo, max(lo + 1, hi)) for lo, hi in zip(edges[:-1], edges[1:])]]
    sx, sy = W / mask.shape[1], H / mask.shape[0]
    return np.stack([(xs[pick] + 0.5) * sx - 0.5, (ys[pick] + 0.5) * sy - 0.5], 1).astype(np.float32)


def grid_points(rows: int, cols: int) -> np.ndarray:
    xs = np.linspace(0.03 * (W - 1), 0.97 * (W - 1), cols)
    ys = np.linspace(0.03 * (H - 1), 0.97 * (H - 1), rows)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2).astype(np.float32)


def rainbow(uv: np.ndarray) -> np.ndarray:
    """RGB colour per query position: hue along x, value along y."""
    hsv = np.zeros((len(uv), 1, 3), np.uint8)
    hsv[:, 0, 0] = (uv[:, 0] / W * 170).astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = (255 - uv[:, 1] / H * 100).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[:, 0]


def draw(rgb: np.ndarray, t: int, idx: np.ndarray, uv: np.ndarray, vis: np.ndarray, kind: np.ndarray,
         colors: np.ndarray, trail: int, scale: float) -> np.ndarray:
    img = rgb.copy()
    t0 = max(0, t - trail)
    for p in idx:
        if not vis[p, t]:
            continue
        c = tuple(int(x) for x in colors[p])
        seg = []
        for s in range(t0, t + 1):
            if vis[p, s]:
                seg.append(uv[p, s] * scale)
            elif seg:
                seg = []
        if len(seg) > 1:
            cv2.polylines(img, [np.round(np.array(seg)).astype(np.int32)], False, c, 1, cv2.LINE_AA)
        x, y = (uv[p, t] * scale).round().astype(int)
        cv2.circle(img, (x, y), 2 if kind[p] == 0 else 3, c, -1, cv2.LINE_AA)
    return img


def run_episode(model, clip: int, e: int, args) -> None:
    name = f"episode_{e:06d}"
    if (args.out / f"{name}_tracks.npz").exists():
        print(f"ep {e}: done already, skipped", flush=True)
        return
    t0 = time.time()
    device = "cuda"
    frames = read_frames(video_path(args.data, e), [(0, 0, W, H)])[0]  # whole frame, no tiles
    n = len(frames)
    cache = args.cache / name
    mz = np.load(cache / "masks.npz")
    human, obj = mz["human"], mz["object"]  # [T,576,768] bool
    depth = np.load(cache / "depth.npz")["depth"]  # [T,288,384] float16, metric
    if len(human) != n or len(depth) != n:
        raise SystemExit(f"ep {e}: video {n} frames, masks {len(human)}, depth {len(depth)}")
    dh, dw = depth.shape[1:]

    qframes = list(range(0, n, args.query_every)) if args.query_every else args.query_frames
    rng = np.random.default_rng(e)
    pts, src, kind = [], [], []
    for s in qframes:
        for k, p in ((0, grid_points(*args.grid)), (1, sample_mask(human[s], args.mask_points, rng)),
                     (2, sample_mask(obj[s], args.mask_points, rng))):
            pts.append(p), src.append(np.full(len(p), s)), kind.append(np.full(len(p), k))
    pts, src, kind = np.concatenate(pts), np.concatenate(src), np.concatenate(kind).astype(np.int8)
    P = len(pts)
    print(f"ep {e}: {n} frames, {P} query points on frames {qframes} "
          f"(grid {(kind == 0).sum()}, actor {(kind == 1).sum()}, object {(kind == 2).sum()})", flush=True)

    uv = np.full((P, n, 2), np.nan, np.float32)
    xyz = np.full((P, n, 3), np.nan, np.float32)
    vis_p = np.zeros((P, n), np.float32)
    conf = np.zeros((P, n), np.float32)
    nseg = len(segments(n, clip, 0))
    clip_scale = np.full((len(qframes), nseg), np.nan, np.float32)
    video_all = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2)[None].float() / 255.0
    aspect = torch.tensor([[W / H]], device=device, dtype=torch.float32)
    norm = np.array([W - 1, H - 1], np.float32)
    for si, s in enumerate(qframes):
        ts = time.time()
        sel = np.flatnonzero(src == s)
        qu = torch.tensor(pts[sel, 0] / norm[0], device=device)
        qv = torch.tensor(pts[sel, 1] / norm[1], device=device)
        for gi, (ctx, a, b) in enumerate(segments(n, clip, s)):
            video = video_all[:, ctx]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                memory = model.encode_video(video=video, aspect_ratio=aspect)
            local = {f: i for i, f in enumerate(ctx)}
            tg = torch.tensor([local[f] for f in range(a, b)], device=device)
            k = len(tg)
            out = decode_all(model, video, memory, qu.repeat_interleave(k), qv.repeat_interleave(k),
                             torch.full((len(sel) * k,), local[s], device=device), tg.repeat(len(sel)), args.chunk)
            x3, u2, vp, cf = (o.view(len(sel), k, -1).squeeze(-1).cpu().numpy() for o in out)
            u2 = u2 * norm
            # metric scale of this clip: cache depth at the predicted pixel over z, visible predictions only
            ci = np.clip(np.round((u2[..., 1] + 0.5) * dh / H - 0.5), 0, dh - 1).astype(int)
            cj = np.clip(np.round((u2[..., 0] + 0.5) * dw / W - 0.5), 0, dw - 1).astype(int)
            dref = depth[np.arange(a, b)[None, :], ci, cj].astype(np.float32)
            ok = (vp > 0.5) & (dref > 0) & (x3[..., 2] > 0)
            sc = robust_scale(dref[ok] / x3[..., 2][ok])
            clip_scale[si, gi] = sc
            uv[sel, a:b], vis_p[sel, a:b], conf[sel, a:b] = u2, vp, cf
            xyz[sel, a:b] = x3 * (sc if np.isfinite(sc) else np.nan)
            del video, memory
        print(f"ep {e} query frame {s}: {len(sel)} points, {time.time() - ts:.1f}s, "
              f"visible {np.mean(vis_p[sel] > 0.5):.3f}, clip scale median {np.nanmedian(clip_scale[si]):.3f} "
              f"CV {np.nanstd(clip_scale[si]) / np.nanmedian(clip_scale[si]):.3f}", flush=True)
    del video_all
    visible = vis_p > 0.5

    # diagnostics
    def in_mask(masks: np.ndarray, p: np.ndarray) -> tuple[int, int]:
        """(hits, tries): visible frames of points p whose predicted pixel lies in the (dilated) mask of that frame."""
        hit = tries = 0
        mh, mw = masks.shape[1:]
        for t in range(n):
            m = masks[t]
            if not m.any():
                continue
            pv = p[visible[p, t]]
            if len(pv) == 0:
                continue
            md = ndimage.binary_dilation(m, iterations=3)
            ii = np.clip(((uv[pv, t, 1] + 0.5) * mh / H).astype(int), 0, mh - 1)
            jj = np.clip(((uv[pv, t, 0] + 0.5) * mw / W).astype(int), 0, mw - 1)
            hit += int(md[ii, jj].sum())
            tries += len(pv)
        return hit, tries

    drift, occl_in_person, vis_off_person = [], [], []
    for s in qframes:
        dyn = ndimage.binary_dilation(human[s] | obj[s], iterations=6)
        g = np.flatnonzero((src == s) & (kind == 0))
        mh, mw = dyn.shape
        ii = np.clip(((pts[g, 1] + 0.5) * mh / H).astype(int), 0, mh - 1)
        jj = np.clip(((pts[g, 0] + 0.5) * mw / W).astype(int), 0, mw - 1)
        static = g[~dyn[ii, jj]]
        d = np.linalg.norm(uv[static] - pts[static, None], axis=-1)  # px from the query position (static camera)
        drift += [np.median(d[i][visible[p]]) for i, p in enumerate(static) if visible[p].any()]
        # static background points covered by the actor in a frame should be predicted occluded
        for t in range(0, n, 5):
            cov = human[t][ii[~dyn[ii, jj]], jj[~dyn[ii, jj]]]
            occl_in_person += list(~visible[static[cov], t])
            vis_off_person += list(visible[static[~cov], t])
    hh, ht = in_mask(human, np.flatnonzero(kind == 1))
    oh, ot = in_mask(obj, np.flatnonzero(kind == 2))
    drift = np.array(drift)
    summary = {
        "episode": e, "frames": n, "query_frames": qframes, "points": int(P),
        "points_by_kind": {k: int((kind == i).sum()) for i, k in enumerate(["grid", "actor", "object"])},
        "visible_fraction": {k: round(float(visible[kind == i].mean()), 4) for i, k in enumerate(["grid", "actor", "object"])},
        "static_grid_drift_px_median_p90": [round(float(np.median(drift)), 2), round(float(np.percentile(drift, 90)), 2)]
        if len(drift) else None,
        "static_grid_points": int(len(drift)),
        "static_occluded_when_actor_covers": round(float(np.mean(occl_in_person)), 4) if occl_in_person else None,
        "static_visible_when_uncovered": round(float(np.mean(vis_off_person)), 4) if vis_off_person else None,
        "actor_points_in_actor_mask": round(hh / ht, 4) if ht else None,
        "object_points_in_object_mask": round(oh / ot, 4) if ot else None,
        "clip_scale_median": round(float(np.nanmedian(clip_scale)), 4),
        "clip_scale_cv": round(float(np.nanstd(clip_scale) / np.nanmedian(clip_scale)), 4),
        "conf_median_visible": round(float(np.median(conf[visible])), 4),
        "seconds_tracking": round(time.time() - t0, 1),
    }
    print(f"ep {e}: " + json.dumps(summary), flush=True)

    save(args.out / f"{name}_tracks.npz", query_frames=np.array(qframes), src=src, kind=kind, query_uv=pts,
         uv=uv, visible=visible, vis_prob=vis_p.astype(np.float16), conf=conf.astype(np.float16),
         xyz=xyz.astype(np.float16), clip_scale=clip_scale)
    (args.out / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))

    # overlay: each frame shows the tracks of the nearest query frame
    colors = rainbow(pts)
    for i in (1, 2):
        colors[kind == i] = KIND_COLOR[i]
    near = np.array(qframes)[np.argmin(np.abs(np.arange(n)[:, None] - np.array(qframes)[None]), 1)]
    by_src = {s: np.flatnonzero(src == s) for s in qframes}
    ow, oh_ = W // 2, H // 2
    cap = cv2.VideoCapture(str(video_path(args.data, e)))
    tmp = args.out / f"{name}_tracks_raw.mp4"
    writer = None if args.no_video else cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (ow, oh_))
    picks = set(np.linspace(0, n - 1, 6).astype(int).tolist())
    tiles = []
    for t in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        if writer is None and t not in picks:
            continue
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (ow, oh_), interpolation=cv2.INTER_AREA)
        img = draw(rgb, t, by_src[near[t]], uv, visible, kind, colors, args.trail, ow / W)
        cv2.putText(img, f"f{t} query f{near[t]}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer is not None:
            writer.write(img[..., ::-1])
        if t in picks:
            tiles.append(img)
    cap.release()
    if writer is not None:
        writer.release()
        final = args.out / f"{name}_tracks.mp4"
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "23",
                            "-pix_fmt", "yuv420p", str(final)], check=True)
            tmp.unlink()
        else:
            tmp.rename(final)
    rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    cv2.imwrite(str(args.out / f"{name}_check.jpg"), np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"ep {e}: wrote {args.out}/{name}_* in {time.time() - t0:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--repo", type=Path, default=Path("/content/Open-d4rt"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/content/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--cache", type=Path, default=Path("/content/repo/cache/d4rt"))
    parser.add_argument("--out", type=Path, default=Path("/content/repo/results/track"))
    parser.add_argument("--query-every", type=int, default=0, help="query frames every N frames (0: --query-frames)")
    parser.add_argument("--query-frames", default="0", help="comma list of query frames")
    parser.add_argument("--grid", default="30x40", help="rows x cols of uniform query points per query frame")
    parser.add_argument("--mask-points", type=int, default=0, help="per query frame, each of actor and object")
    parser.add_argument("--chunk", type=int, default=16384, help="queries per decoder call")
    parser.add_argument("--trail", type=int, default=10, help="overlay trail length in frames")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.grid = tuple(int(x) for x in args.grid.lower().split("x"))
    args.query_frames = [int(x) for x in args.query_frames.split(",") if x]
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = []
    for part in args.episodes.split(","):
        lo, _, hi = part.partition("-")
        episodes += list(range(int(lo), int(hi or lo) + 1))
    model, clip = load_model(args.repo, args.ckpt_dir, "cuda")
    for e in episodes:
        run_episode(model, clip, e, args)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

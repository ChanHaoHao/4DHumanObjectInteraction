#!/usr/bin/env python3
"""Step 3 D4RT revision (runs on the Colab VM, GPU): temporally consistent depth from OpenD4RT, put on
MoGe-2's metric scale, written as a drop-in replacement for the Step 3v1 cache.

    python d4rt_vm.py --episodes 0 [--repo /content/Open-d4rt] [--ckpt-dir DIR] [--data /content/v2d/track_1]
        [--cache /content/repo/cache] [--out /content/repo/cache/d4rt] [--grid 288x384] [--overlap 16]
        [--chunk 16384] [--no-stabilize] [--fp32] [--max-windows N] [--tiles 2 --tile-overlap 0.125]
        [--video [CACHE ...]]

Per episode:
1. Frames are resized to 256x256 (the model input, with the 4:3 aspect token) and cut into sliding windows
   of 48 frames (the checkpoint's clip length) sharing --overlap frames.
2. Depth: every cell centre of the cache grid (1/4 resolution) is queried with t_src = t_tgt = t_cam = t;
   depth = z of the predicted point, plus its confidence (sigmoid).
3. Camera check: a 32x32 grid per frame queried in its own frame and in the window's first frame; a Sim(3)
   on background points gives camera t -> first camera (static camera: identity, scale 1).
   Intrinsics from the local points (fx = z |u - cx| / |x|, the paper's estimator).
4. Metric scale. D4RT's output has no scale (trained on mean-depth-normalized targets) and each window
   has its own. Window w gets s_w = median(B / z_w) over static pixels of all its frames, refit on
   |log ratio| < 0.1, where B is the Step 3v1 background (median over the clip of stabilized MoGe-2 depth,
   people/objects masked). B is pixel-aligned with the D4RT queries, so no feature matching is needed.
   Overlapping frames are blended with linear ramps. Then, as in Step 3v1, each frame is rescaled by the
   median ratio to B on its static pixels (--no-stabilize skips this; its std is reported either way).
Tiles (--tiles N): the frame is cut into NxN tiles, each 1+--tile-overlap times W/N by H/N (still 4:3, so the same
aspect token), and each tile is a separate D4RT run at 256x256 (2x the pixels per object for N=2). Only z is kept:
every tile cell is a cell of the full-frame grid, so x, y follow from the Step 3v1 K and the tiles share one camera.
Per window, each tile gets its own scale to the background (step 4); then tiles are placed most-static-pixels first
and a tile whose median ratio to the already placed tiles on the overlap cells (all cells, people included) is off
by more than --seam-tol is rescaled to them. Blend: confidence x a linear ramp to 0 at interior tile edges across the
overlap band. The camera check and the intrinsics estimate (3.) run only without tiles.
--video [CACHE ...]: <out>/check/episode_XXXXXX_depth.mp4, 2x2 panels RGB | Step 3v1 MoGe | each CACHE | this run,
one colour range per episode (MoGe's 2-98 percentiles).
5. Writes <out>/episode_XXXXXX/{masks.npz (copy), depth.npz, camera.npz (Step 3v1 K, which Steps 4-5
   were fit with), meta.json} and <out>/check/episode_XXXXXX.jpg (RGB | MoGe | D4RT | |rel diff| 0-20%).
   depth.npz also has `confidence` [T,h,w]. Inputs per episode: the Step 3v1 masks.npz, camera.npz, meta.json and
   depth.npz, or instead of depth.npz a slim background.npz (frames, background), which skips the *_moge diagnostics.
   meta.json["d4rt"] has the diagnostics:
     depth_scale_std        std of the per-frame correction (Step 3v1 reported 0.004-0.017 for MoGe)
     flicker_*              median |d_t - d_t+1| / d on static pixels, D4RT vs MoGe cache
     actor_acc_*            mean |2nd difference| of the actor's median depth, cm/frame^2, both depths
     absrel_vs_moge         median |d - d_moge| / d_moge over valid pixels (disagreement, not error)
     cam_rot_deg / cam_trans_rel / cam_scale    camera t -> window start (median, max)
     fx / fy / fov_x        D4RT intrinsics at 1536x1152 (Step 3v1 MoGe fov_x is in meta)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SIZE = (1536, 1152)  # W, H
MODEL_HW = (256, 256)
CAM_GRID = 32
STATIC_DILATE = 9  # px at 1/4 resolution around the actor/object masks


def video_path(data: Path, episode: int) -> Path:
    return data / "videos/chunk-000/observation.images.exo_camera" / f"episode_{episode:06d}.mp4"


def read_frames(path: Path, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    """Every frame cut to each (x0, y0, w, h) box and resized to the model input: per box [T,256,256,3] uint8 RGB."""
    cap = cv2.VideoCapture(str(path))
    frames = [[] for _ in boxes]
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for out, (x0, y0, w, h) in zip(frames, boxes):
            out.append(cv2.resize(rgb[y0:y0 + h, x0:x0 + w], MODEL_HW[::-1], interpolation=cv2.INTER_AREA))
    cap.release()
    return [np.stack(f) for f in frames]


def tile_boxes(n: int, overlap: float, step: int) -> list[tuple[int, int, int, int]]:
    """NxN boxes (x0, y0, w, h), each (1 + overlap) / N of the image per side, on the --grid cell raster (step px)."""
    W, H = IMAGE_SIZE
    if n == 1:
        return [(0, 0, W, H)]
    tw, th = (round(L / n * (1 + overlap) / step) * step for L in (W, H))
    xs = np.linspace(0, W - tw, n).round().astype(int) // step * step
    ys = np.linspace(0, H - th, n).round().astype(int) // step * step
    return [(int(x), int(y), tw, th) for y in ys for x in xs]


def edge_ramp(n: int, lo_inner: bool, hi_inner: bool, band: float) -> np.ndarray:
    """Weight of n cells along one tile axis: 1, falling linearly to 0 over `band` cells at edges inside the image."""
    i = np.arange(n) + 0.5
    w = np.ones(n)
    if lo_inner:
        w = np.minimum(w, i / band)
    if hi_inner:
        w = np.minimum(w, (n - i) / band)
    return np.clip(w, 1e-3, 1.0)


def read_full(path: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = {}
    for f in sorted(wanted):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, bgr = cap.read()
        if ok:
            out[f] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def save(path: Path, **arrays):
    """savez_compressed via a temporary name, so a concurrent download never sees a partial file."""
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------- model
def fast_local_patches(self, video, u, v, t_src):
    """QueryEmbedder._extract_local_patches without one frame copy per query (same output).

    The original gathers video[b, t_src] for every query ([M,3,256,256], ~3 GB per 4k queries);
    here each distinct source frame is sampled once for all of its queries.
    """
    bsz, _, channels, height, width = video.shape
    p = self.local_patch_size
    offsets = torch.linspace(-(p - 1) / 2.0, (p - 1) / 2.0, p, device=video.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(offsets * (2.0 / max(1, height - 1)), offsets * (2.0 / max(1, width - 1)), indexing="ij")
    out = video.new_empty((bsz, u.shape[1], channels * p * p))
    for b in range(bsz):
        for t in torch.unique(t_src[b]).tolist():
            sel = torch.nonzero(t_src[b] == t, as_tuple=True)[0]
            cx = (u[b, sel] * 2.0 - 1.0).float()
            cy = (v[b, sel] * 2.0 - 1.0).float()
            grid = torch.stack([gx[None] + cx[:, None, None], gy[None] + cy[:, None, None]], dim=-1)  # [M,p,p,2]
            patch = F.grid_sample(video[b, t][None].float(), grid.reshape(1, -1, p, 2), mode="bilinear",
                                  padding_mode="border", align_corners=True)  # [1,C,M*p,p]
            out[b, sel] = patch.view(channels, -1, p, p).permute(1, 0, 2, 3).reshape(-1, channels * p * p).to(out.dtype)
    return out


def load_model(repo: Path, ckpt_dir: Path, device: str):
    sys.path.insert(0, str(repo))
    from src.core import load_yaml_config
    from src.model import build_model

    cfg = load_yaml_config(ckpt_dir / "model.yaml")
    model = build_model(cfg["model"]).eval()
    small = ckpt_dir / "model_only.pt"  # weights only, written on first load (the release also has optimizer state)
    if small.exists():
        state = torch.load(small, map_location="cpu", mmap=True)
    else:
        try:
            payload = torch.load(ckpt_dir / "opend4rt.ckpt", map_location="cpu", mmap=True, weights_only=False)
        except RuntimeError:  # legacy (non-zip) checkpoints cannot be memory-mapped
            payload = torch.load(ckpt_dir / "opend4rt.ckpt", map_location="cpu", weights_only=False)
        state = next((payload[k] for k in ("state_dict", "model", "module", "network", "net")
                      if isinstance(payload, dict) and isinstance(payload.get(k), dict)), payload)
        state = {k: v for k, v in state.items() if torch.is_tensor(v)}
        torch.save(state, small)
        del payload
    res = model.load_state_dict(state, strict=False)
    print(f"checkpoint: missing {len(res.missing_keys)} {res.missing_keys[:5]}, "
          f"unexpected {len(res.unexpected_keys)} {res.unexpected_keys[:5]}", flush=True)
    if res.missing_keys:
        raise RuntimeError("checkpoint does not cover the model")
    del state
    model.to(device).eval()

    # swap in the cheaper patch sampler after checking it matches on random queries
    embedder = model.query_embedder
    video = torch.rand(1, 4, 3, *MODEL_HW, device=device)
    u, v = torch.rand(1, 300, device=device), torch.rand(1, 300, device=device)
    t = torch.randint(0, 4, (1, 300), device=device)
    ref = embedder._extract_local_patches(video, u, v, t)
    type(embedder)._extract_local_patches = fast_local_patches
    new = embedder._extract_local_patches(video, u, v, t)
    err = float((ref - new).abs().max())
    print(f"patch sampler check: max abs diff {err:.2e}", flush=True)
    if err > 1e-5:
        raise RuntimeError("fast patch sampler differs from the original")
    return model, int(cfg.get_path("model.input.clip_frames", 48))


@torch.no_grad()
def decode(model, video, memory, u, v, t_src, t_tgt, t_cam, chunk: int, amp: bool):
    """xyz [N,3] and confidence [N] (sigmoid) for flat query arrays (torch, on device)."""
    xyz, conf = [], []
    for a in range(0, len(u), chunk):
        q = {"u": u[a:a + chunk][None], "v": v[a:a + chunk][None], "t_src": t_src[a:a + chunk][None],
             "t_tgt": t_tgt[a:a + chunk][None], "t_cam": t_cam[a:a + chunk][None]}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model.decode_queries(video=video, query=q, memory=memory)
        xyz.append(pred["xyz_3d"][0].float())
        conf.append(torch.sigmoid(pred["confidence"][0].float()))
    return torch.cat(xyz), torch.cat(conf)


# ---------------------------------------------------------------------------------------- geometry
def windows(n: int, clip: int, overlap: int) -> list[tuple[int, int]]:
    if n <= clip:
        return [(0, n)]
    stride = max(1, clip - overlap)
    starts = list(range(0, n - clip + 1, stride))
    if starts[-1] != n - clip:
        starts.append(n - clip)
    return [(s, s + clip) for s in starts]


def ramp(a: int, b: int, n: int, overlap: int) -> np.ndarray:
    """Blend weight of each frame of window [a, b): linear ramps where it overlaps a neighbour."""
    i = np.arange(b - a, dtype=np.float64)
    left = np.ones_like(i) if a == 0 else (i + 1) / (overlap + 1)
    right = np.ones_like(i) if b == n else (b - a - i) / (overlap + 1)
    return np.clip(np.minimum(left, right), 1e-3, 1.0)


def sim3(src: np.ndarray, dst: np.ndarray):
    """Umeyama: dst ~ s R src + t. Returns (s, R, t) or None."""
    if len(src) < 10:
        return None
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(a.T @ b / len(src))
    d = np.array([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    R = vt.T @ np.diag(d) @ u.T
    var = (a ** 2).sum() / len(src)
    if var < 1e-12:
        return None
    s = float((sv * d).sum() / var)
    return s, R, md - s * R @ ms


def robust_scale(ratio: np.ndarray) -> float:
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if len(ratio) < 50:
        return math.nan
    s = np.median(ratio)
    keep = np.abs(np.log(ratio / s)) < 0.1
    return float(np.median(ratio[keep])) if keep.sum() >= 50 else float(s)


def colorize(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((255 * (1 - x)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    img[~(depth > 0)] = 0
    return img


def run_lengths_acc(series: np.ndarray, ok: np.ndarray) -> float:
    """Mean |second difference| (cm/frame^2) of a metre series within runs of consecutive ok frames."""
    accs = []
    idx = np.flatnonzero(ok)
    for run in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
        if len(run) >= 3:
            s = series[run]
            accs.append(np.abs(s[2:] - 2 * s[1:-1] + s[:-2]))
    return float(np.concatenate(accs).mean() * 100) if accs else math.nan


# ----------------------------------------------------------------------------------------- episode
def run_episode(model, clip: int, episode: int, args) -> None:
    name = f"episode_{episode:06d}"
    src, dst = args.cache / name, args.out / name
    if (dst / "meta.json").exists() and not args.max_windows:
        print(f"ep {episode}: done already, skipped", flush=True)
        return
    t0 = time.time()
    device = "cuda"
    gh, gw = args.grid
    step_x, step_y = IMAGE_SIZE[0] / gw, IMAGE_SIZE[1] / gh

    assert IMAGE_SIZE[0] / gw == IMAGE_SIZE[1] / gh, "tiles assume square grid cells"
    boxes = tile_boxes(args.tiles, args.tile_overlap, int(step_x))
    tiled = len(boxes) > 1
    tile_frames = read_frames(video_path(args.data, episode), boxes)
    n = len(tile_frames[0])
    meta = json.loads((src / "meta.json").read_text())
    mz = np.load(src / "masks.npz")
    if (src / "depth.npz").exists():
        dz = np.load(src / "depth.npz")
        moge = dz["depth"]  # [T,288,384] float16, stabilized MoGe-2 (Step 3v1)
    else:  # slim input (frames + background only): the MoGe comparisons are skipped
        dz, moge = np.load(src / "background.npz"), None
    cache_frames = dz["frames"]
    background = dz["background"].astype(np.float32)
    if background.shape != (gh, gw):
        raise SystemExit(f"--grid {gh}x{gw} must match the Step 3v1 depth grid {background.shape}")
    if len(cache_frames) != n:
        raise SystemExit(f"ep {episode}: video has {n} frames, cache {len(cache_frames)}")
    human, obj = mz["human"], mz["object"]  # [T,576,768] bool
    fy_m, fx_m = human.shape[1] // gh, human.shape[2] // gw
    kernel = np.ones((STATIC_DILATE, STATIC_DILATE), np.uint8)

    def dynamic(f: int) -> np.ndarray:
        m = (human[f] | obj[f])[::fy_m, ::fx_m].astype(np.uint8)
        return cv2.dilate(m, kernel) > 0

    # query grids (u, v normalized to [0, 1] with align_corners: x_px / (W - 1), pixel centres at integers)
    xs = (np.arange(gw) + 0.5) * step_x - 0.5
    ys = (np.arange(gh) + 0.5) * step_y - 0.5
    gu, gv = np.meshgrid(xs / (IMAGE_SIZE[0] - 1), ys / (IMAGE_SIZE[1] - 1))
    gu, gv = torch.tensor(gu.ravel(), device=device, dtype=torch.float32), torch.tensor(gv.ravel(), device=device, dtype=torch.float32)
    tiles = []  # per box: cell slices, queries (u, v in the tile, align_corners), blend ramp
    for x0, y0, w, h in boxes:
        c0, c1 = int(x0 // step_x), int((x0 + w) // step_x)
        r0, r1 = int(y0 // step_y), int((y0 + h) // step_y)
        tu, tv = np.meshgrid((xs[c0:c1] - x0) / (w - 1), (ys[r0:r1] - y0) / (h - 1))
        n1 = max(args.tiles - 1, 1)  # overlap band = tile side - stride between tiles
        band_x, band_y = (w - (IMAGE_SIZE[0] - w) / n1) / step_x, (h - (IMAGE_SIZE[1] - h) / n1) / step_y
        ramp2 = (edge_ramp(r1 - r0, r0 > 0, r1 < gh, band_y)[:, None] * edge_ramp(c1 - c0, c0 > 0, c1 < gw, band_x)[None])
        tiles.append({"rs": slice(r0, r1), "cs": slice(c0, c1), "shape": (r1 - r0, c1 - c0), "ramp": ramp2.astype(np.float32),
                      "u": torch.tensor(tu.ravel(), device=device, dtype=torch.float32),
                      "v": torch.tensor(tv.ravel(), device=device, dtype=torch.float32)})
    if not tiled:
        tiles[0]["u"], tiles[0]["v"] = gu, gv
    cxs = np.linspace(0.05, 0.95, CAM_GRID)
    cu, cv = np.meshgrid(cxs, cxs)
    cu, cv = torch.tensor(cu.ravel(), device=device, dtype=torch.float32), torch.tensor(cv.ravel(), device=device, dtype=torch.float32)
    cam_cells = (np.clip((cv.cpu().numpy() * (IMAGE_SIZE[1] - 1) + 0.5) / step_y, 0, gh - 1).astype(int),
                 np.clip((cu.cpu().numpy() * (IMAGE_SIZE[0] - 1) + 0.5) / step_x, 0, gw - 1).astype(int))
    aspect = torch.tensor([[IMAGE_SIZE[0] / IMAGE_SIZE[1]]], device=device, dtype=torch.float32)
    amp = not args.fp32

    wins = windows(n, clip, args.overlap)
    if args.max_windows:
        wins = wins[:args.max_windows]
    acc = np.zeros((n, gh, gw), np.float32)
    conf_acc = np.zeros((n, gh, gw), np.float32)
    wsum = np.zeros(n, np.float64)
    win_stats, fx_all, fy_all = [], [], []
    for wi, (a, b) in enumerate(wins):
        tw = time.time()
        k = b - a
        rots, trans, scales, placed = [], [], [], []
        for ti, tile in enumerate(tiles):
            video = torch.from_numpy(tile_frames[ti][a:b]).to(device).permute(0, 3, 1, 2)[None].float() / 255.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                memory = model.encode_video(video=video, aspect_ratio=aspect)
            q = len(tile["u"])
            t_local = torch.arange(k, device=device).repeat_interleave(q)
            xyz, conf = decode(model, video, memory, tile["u"].repeat(k), tile["v"].repeat(k), t_local, t_local, t_local,
                               args.chunk, amp)
            xyz = xyz.view(k, *tile["shape"], 3).cpu().numpy()
            z = xyz[..., 2]
            conf = conf.view(k, *tile["shape"]).cpu().numpy()

            if not tiled:
                # camera t -> window start, on static grid points
                tc = torch.arange(k, device=device).repeat_interleave(len(cu))
                loc, _ = decode(model, video, memory, cu.repeat(k), cv.repeat(k), tc, tc, tc, args.chunk, amp)
                ref, _ = decode(model, video, memory, cu.repeat(k), cv.repeat(k), tc, tc, torch.zeros_like(tc), args.chunk, amp)
                loc, ref = loc.view(k, -1, 3).cpu().numpy(), ref.view(k, -1, 3).cpu().numpy()
                for i in range(1, k):
                    static = ~dynamic(a + i)[cam_cells]
                    fit = sim3(loc[i][static], ref[i][static])
                    if fit is None:
                        continue
                    s, R, tr = fit
                    rots.append(math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
                    trans.append(float(np.linalg.norm(tr) / np.median(loc[i][static, 2])))
                    scales.append(s)

                # intrinsics from local points (pixels at full resolution, principal point at the centre)
                sub = xyz[::4, ::4, ::4].reshape(-1, 3)
                uu = np.broadcast_to(xs[None, None, ::4], z[::4, ::4, ::4].shape).ravel()
                vv = np.broadcast_to(ys[None, ::4, None], z[::4, ::4, ::4].shape).ravel()
                cx, cy = (IMAGE_SIZE[0] - 1) / 2, (IMAGE_SIZE[1] - 1) / 2
                okx = (np.abs(uu - cx) > 150) & (sub[:, 2] > 0)
                oky = (np.abs(vv - cy) > 150) & (sub[:, 2] > 0)
                fx_all.append(np.median(sub[okx, 2] * np.abs(uu[okx] - cx) / np.maximum(np.abs(sub[okx, 0]), 1e-6)))
                fy_all.append(np.median(sub[oky, 2] * np.abs(vv[oky] - cy) / np.maximum(np.abs(sub[oky, 1]), 1e-6)))

            # tile/window scale against the MoGe background on static pixels
            bg = background[tile["rs"], tile["cs"]]
            ratios = []
            for i in range(0, k, 2):
                ok = ~dynamic(a + i)[tile["rs"], tile["cs"]] & (bg > 0) & (z[i] > 0)
                ratios.append((bg[ok] / z[i][ok])[::3])
            ratios = np.concatenate(ratios)
            placed.append((ti, robust_scale(ratios), z, conf, len(ratios)))
            del video, memory

        # place the tiles, best anchored first; rescale a tile that disagrees with the placed ones on the overlap
        num = np.zeros((k, gh, gw), np.float32)
        den = np.zeros((k, gh, gw), np.float32)
        cnum = np.zeros((k, gh, gw), np.float32)
        tile_stats = []
        for ti, s_t, z, conf, n_static in sorted(placed, key=lambda p: -p[4]):
            tile = tiles[ti]
            rs, cs = tile["rs"], tile["cs"]
            base = s_t if np.isfinite(s_t) else 1.0
            ov = (den[:, rs, cs] > 0) & (z > 0)
            seam = None
            if ov.sum() >= 50:
                blend = num[:, rs, cs][ov] / den[:, rs, cs][ov]
                r = robust_scale((blend / (base * z[ov]))[::5])
                if np.isfinite(r):
                    seam = round(abs(math.log(r)), 4)
                    if not np.isfinite(s_t) or seam > args.seam_tol:
                        base *= r
            elif not np.isfinite(s_t):
                raise RuntimeError(f"ep {episode} window {a}-{b} tile {ti}: no static pixels and no overlap")
            wgt = tile["ramp"][None] * np.maximum(conf, 1e-3)
            num[:, rs, cs] += wgt * base * z
            den[:, rs, cs] += wgt
            cnum[:, rs, cs] += wgt * conf
            tile_stats.append({"tile": ti, "scale_bg": round(s_t, 5) if np.isfinite(s_t) else None,
                               "scale": round(base, 5), "static_px": n_static, "seam_logratio": seam})
        z_win = np.where(den > 0, num / np.maximum(den, 1e-12), 0)
        conf_win = np.where(den > 0, cnum / np.maximum(den, 1e-12), 0)
        s_w = float(np.median([t["scale"] for t in tile_stats]))
        w = ramp(a, b, n if not args.max_windows else wins[-1][1], args.overlap)
        acc[a:b] += (w[:, None, None] * z_win).astype(np.float32)
        conf_acc[a:b] += (w[:, None, None] * conf_win).astype(np.float32)
        wsum[a:b] += w
        win_stats.append({"start": a, "end": b, "scale": round(s_w, 5),
                          "cam_rot_deg_max": round(max(rots), 3) if rots else None,
                          "cam_trans_rel_max": round(max(trans), 4) if trans else None,
                          "cam_scale_dev_max": round(max(abs(s - 1) for s in scales), 4) if scales else None,
                          "conf_median": round(float(np.median(conf_win)), 4),
                          "tiles": tile_stats if tiled else None,
                          "rots": rots, "trans": trans, "scales": scales})
        dt = time.time() - tw
        seams = [t["seam_logratio"] for t in tile_stats if t["seam_logratio"] is not None]
        print(f"ep {episode} window {wi + 1}/{len(wins)} [{a},{b}) {dt:.1f}s scale {s_w:.4f} "
              f"rot max {win_stats[-1]['cam_rot_deg_max']} trans max {win_stats[-1]['cam_trans_rel_max']} "
              + (f"seams {seams} " if tiled else "") + f"ETA {dt * (len(wins) - wi - 1) / 60:.1f} min", flush=True)

    end = wins[-1][1]
    covered = wsum[:end] > 0
    depth = np.zeros((end, gh, gw), np.float32)
    depth[covered] = acc[:end][covered] / wsum[:end][covered, None, None]
    confidence = np.zeros_like(depth)
    confidence[covered] = conf_acc[:end][covered] / wsum[:end][covered, None, None]
    del acc, conf_acc

    # per-frame correction to the background (Step 3v1's stabilization), measured always
    corr = np.ones(end)
    for f in range(end):
        ok = ~dynamic(f) & (background > 0) & (depth[f] > 0)
        corr[f] = robust_scale((background[ok] / depth[f][ok])[::2])
    corr = np.where(np.isfinite(corr), corr, 1.0)
    if not args.no_stabilize:
        depth *= corr[:, None, None].astype(np.float32)
    depth[~np.isfinite(depth) | (depth < 0.05)] = 0

    # diagnostics: flicker on static pixels, actor depth smoothness, disagreement with MoGe
    flick_d, flick_m, absrel = [], [], []
    actor_d, actor_m, actor_ok = np.full(end, np.nan), np.full(end, np.nan), np.zeros(end, bool)
    area = np.array([human[f][::fy_m, ::fx_m].sum() for f in range(end)])
    ones = np.ones((gh, gw), np.float32)  # stands in for MoGe when it was not uploaded (always "valid")
    for f in range(end):
        hm = human[f][::fy_m, ::fx_m]
        mf = moge[f].astype(np.float32) if moge is not None else ones
        if area[f] > 0.5 * np.median(area[area > 0]):
            ok = hm & (depth[f] > 0) & (mf > 0)
            if ok.sum() > 30:
                actor_d[f], actor_m[f], actor_ok[f] = np.median(depth[f][ok]), np.median(mf[ok]), True
        both = (depth[f] > 0) & (mf > 0)
        if moge is not None:
            absrel.append(np.median(np.abs(depth[f][both] - mf[both]) / mf[both]))
        if f + 1 < end:
            m1 = moge[f + 1].astype(np.float32) if moge is not None else ones
            st = ~dynamic(f) & ~dynamic(f + 1) & (depth[f] > 0) & (depth[f + 1] > 0) & (mf > 0) & (m1 > 0)
            flick_d.append(np.median(np.abs(depth[f + 1][st] - depth[f][st]) / depth[f][st]))
            if moge is not None:
                flick_m.append(np.median(np.abs(m1[st] - mf[st]) / mf[st]))
    rots = [r for s in win_stats for r in s.pop("rots")]
    trans = [t for s in win_stats for t in s.pop("trans")]
    scl = [abs(c - 1) for s in win_stats for c in s.pop("scales")]
    fx, fy = (float(np.median(v)) if v else math.nan for v in (fx_all, fy_all))
    diag = {
        "checkpoint": args.ckpt_dir.name, "grid": [gh, gw], "clip": clip, "overlap": args.overlap,
        "tiles": args.tiles, "tile_overlap": args.tile_overlap if tiled else None, "seam_tol": args.seam_tol if tiled else None,
        "stabilized": not args.no_stabilize, "bf16": amp, "frames": end, "windows": win_stats,
        "depth_scale_std": round(float(np.std(corr)), 4),
        "depth_scale_p5_p95": np.round(np.percentile(corr, [5, 95]), 4).tolist(),
        "window_scale_cv": round(float(np.std([s["scale"] for s in win_stats]) / np.mean([s["scale"] for s in win_stats])), 4),
        "flicker_d4rt": round(float(np.median(flick_d)), 5) if flick_d else None,
        "flicker_moge": round(float(np.median(flick_m)), 5) if flick_m else None,
        "actor_acc_d4rt_cm": round(run_lengths_acc(actor_d, actor_ok), 3),
        "actor_acc_moge_cm": round(run_lengths_acc(actor_m, actor_ok), 3) if moge is not None else None,
        "actor_depth_ratio_median": round(float(np.nanmedian(actor_d / actor_m)), 4) if moge is not None else None,
        "absrel_vs_moge": round(float(np.median(absrel)), 4) if absrel else None,
        "conf_median": round(float(np.median(confidence[covered[:end]])), 4),
        "cam_rot_deg_median_max": [round(float(np.median(rots)), 3), round(float(np.max(rots)), 3)] if rots else None,
        "cam_trans_rel_median_max": [round(float(np.median(trans)), 4), round(float(np.max(trans)), 4)] if trans else None,
        "cam_scale_dev_median_max": [round(float(np.median(scl)), 4), round(float(np.max(scl)), 4)] if scl else None,
        "fx": round(fx, 1), "fy": round(fy, 1),
        "fov_x": round(math.degrees(2 * math.atan(IMAGE_SIZE[0] / 2 / fx)), 2),
        "fov_x_moge": meta.get("fov_x"),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"ep {episode}: " + json.dumps({k: v for k, v in diag.items() if k != "windows"}), flush=True)

    # overlay: 4 frames x (RGB | MoGe | D4RT | |rel diff| 0-20%)
    pick = np.linspace(0, end - 1, 4).astype(int).tolist()
    full = read_full(video_path(args.data, episode), pick)
    ref = moge[:end] if moge is not None else depth
    valid = ref[ref > 0].astype(np.float32)
    lo, hi = np.percentile(valid[::97], [2, 98])
    rows = []
    for f in pick:
        rgb = cv2.resize(full[f], (gw, gh), interpolation=cv2.INTER_AREA) if f in full else np.zeros((gh, gw, 3), np.uint8)
        mf = moge[f].astype(np.float32) if moge is not None else np.zeros((gh, gw), np.float32)
        rel = np.where((depth[f] > 0) & (mf > 0), np.abs(depth[f] - mf) / np.maximum(mf, 1e-6), 0)
        diff = cv2.applyColorMap((255 * np.clip(rel / 0.2, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
        row = np.concatenate([rgb, colorize(mf, lo, hi), colorize(depth[f], lo, hi), diff], 1)
        cv2.putText(row, f"f{f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(row)
    check = args.out / "check"
    check.mkdir(parents=True, exist_ok=True)
    tag = "_probe" if args.max_windows else ""
    cv2.imwrite(str(check / f"{name}{tag}.jpg"), np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])

    if args.max_windows:  # timing / sanity probe: no cache written
        (check / f"{name}_probe.json").write_text(json.dumps(diag, indent=1))
        return
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta.json").unlink(missing_ok=True)
    shutil.copy(src / "masks.npz", dst / "masks.npz")
    shutil.copy(src / "camera.npz", dst / "camera.npz")
    d4rt_bg = np.zeros((gh, gw), np.float32)
    stack = np.stack([np.where(dynamic(f) | (depth[f] <= 0), np.nan, depth[f]) for f in range(0, end, 4)])
    with np.errstate(all="ignore"):
        d4rt_bg = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0).astype(np.float32)
    del stack
    save(dst / "depth.npz", frames=cache_frames, depth=depth.astype(np.float16), scale=corr.astype(np.float32),
         background=d4rt_bg, confidence=confidence.astype(np.float16))
    meta["depth_source"] = "d4rt"
    meta["depth_scale_std"] = diag["depth_scale_std"]
    meta["depth_scale_p5_p95"] = diag["depth_scale_p5_p95"]
    meta["d4rt"] = diag
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))  # last: marks the episode complete
    print(f"ep {episode}: wrote {dst} in {time.time() - t0:.0f}s", flush=True)


def depth_video(args, episode: int, compare: list[Path]):
    """RGB | MoGe | compare caches | this run, 2x2 (or 2xN) panels at the cache grid, H.264 via ffmpeg."""
    import subprocess

    name = f"episode_{episode:06d}"
    panels = [("MoGe (3v1)", args.cache / name / "depth.npz")] + [(c.name, c / name / "depth.npz") for c in compare]
    panels.append((args.out.name + (f" {args.tiles}x{args.tiles} tiles" if args.tiles > 1 else ""), args.out / name / "depth.npz"))
    depths = [(label, np.load(p)["depth"]) for label, p in panels if p.exists()]
    gh, gw = depths[-1][1].shape[1:]
    ref = depths[0][1]
    lo, hi = np.percentile(ref[ref > 0][::97].astype(np.float32), [2, 98])
    T = min(len(d) for _, d in depths)
    cols = 2
    rows = (len(depths) + 1 + cols - 1) // cols
    out = args.out / "check" / f"{name}_depth.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{cols * gw}x{rows * gh}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "20",
                           "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(video_path(args.data, episode)))
    for f in range(T):
        ok, bgr = cap.read()
        if not ok:
            break
        tiles = [("RGB", cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (gw, gh), interpolation=cv2.INTER_AREA))]
        tiles += [(label, np.ascontiguousarray(colorize(d[f].astype(np.float32), lo, hi))) for label, d in depths]
        tiles += [("", np.zeros((gh, gw, 3), np.uint8))] * (rows * cols - len(tiles))
        for label, img in tiles:
            cv2.putText(img, f"{label} f{f}" if label else "", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        grid = np.concatenate([np.concatenate([t[1] for t in tiles[r * cols:(r + 1) * cols]], 1) for r in range(rows)], 0)
        ff.stdin.write(np.ascontiguousarray(grid).tobytes())
    cap.release()
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"ep {episode}: wrote {out} ({lo:.2f}-{hi:.2f} m)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--repo", type=Path, default=Path("/content/Open-d4rt"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/content/Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"))
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--cache", type=Path, default=Path("/content/repo/cache"))
    parser.add_argument("--out", type=Path, default=Path("/content/repo/cache/d4rt"))
    parser.add_argument("--grid", default="288x384", help="HxW query grid; must match the Step 3v1 depth grid")
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=16384, help="queries per decoder call")
    parser.add_argument("--no-stabilize", action="store_true", help="skip the per-frame background rescale")
    parser.add_argument("--fp32", action="store_true", help="no bf16 autocast")
    parser.add_argument("--max-windows", type=int, default=0, help="probe: only the first N windows, no cache")
    parser.add_argument("--tiles", type=int, default=1, help="NxN tiles, one D4RT run each (see Tiles above)")
    parser.add_argument("--tile-overlap", type=float, default=0.125)
    parser.add_argument("--seam-tol", type=float, default=0.02, help="|log ratio| on a tile overlap that triggers a rescale")
    parser.add_argument("--video", type=Path, nargs="*", help="render the depth video; optional caches to compare")
    args = parser.parse_args()
    args.grid = tuple(int(x) for x in args.grid.lower().split("x"))
    episodes = []
    for part in args.episodes.split(","):
        lo, _, hi = part.partition("-")
        episodes += list(range(int(lo), int(hi or lo) + 1))
    model, clip = load_model(args.repo, args.ckpt_dir, "cuda")
    for e in episodes:
        run_episode(model, clip, e, args)
        torch.cuda.empty_cache()
        if args.video is not None:
            depth_video(args, e, args.video)


if __name__ == "__main__":
    main()

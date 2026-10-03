#!/usr/bin/env python3
"""Step 5 pose fitting (runs on the Colab VM, GPU): one mesh and one scale per object, a 6D pose track per episode.

    python object_fit.py --human <repo>/results/human_v1/episodes --run <repo>/results/object_v1 \
        [--episodes 0-29] [--raw /content/object_raw] [--mesh best|E_K|path.obj]

Inputs: the candidate meshes from object_vm.py (<raw>/meshes/episode_XXXXXX_k.obj, normalized: centred,
longest side 1), the Step 3 cache (object/actor masks at 1/2 resolution, stabilized MoGe depth at 1/4)
and the Step 4 run (human params, K and depth_scale = 1 / k_moge, which brings the Step 3 depth to the
body's metric scale; world = camera frame).

Model: point_camera = s R p + t, p on the mesh. No rasterizer; per frame, with the mesh sampled as points:
  dep    data -> model: interior object depth pixels (mask eroded at 1/4 resolution, background-bleeding
         pixels dropped, x depth_scale) to the nearest model point, Geman-McClure with sigma --sigma (m)
  front  model -> data: front-facing model points that project onto that interior mask compared with
         the depth map there (Geman-McClure)
         Both are off on frames with < 5 interior pixels (thin or tiny objects: silhouette only).
  in     projected model points outside (object mask U dilated actor mask): distance-transform pixels
         / object size in pixels (sqrt of mask area), Geman-McClure with --sil-sigma (points off the
         image and behind the actor are free: occlusion)
  cov    object mask pixels to the nearest projected model point, / object size, Geman-McClure
All four terms lie in [0, 1).
Frames without an object mask carry no data term (the track interpolates through them).

Fitting, per object (the requested episodes showing it, usually 3, are fitted together):
1. Keyframes: --keyframes well-visible frames per episode in the scored span (large, unoccluded masks).
   Candidate meshes: every crop's mesh of the object over its episodes, minus near-flat ones
   (--min-thickness). Each is screened with --screen-hyps rotations (uniform on SO(3)) per keyframe,
   optimizing rotation, translation and log scale; the --finalists best get the full --hyps init. The mesh
   with the lowest mean keyframe loss over all the object's episodes wins (different views of one object).
   Scale: the weighted median of all keyframes' scales (frames with depth count 5x), shared by the
   episodes (--no-share-scale: per episode).
2. Per episode, the best keyframe is re-fitted at that scale, then tracked forward and backward: each
   frame starts from the previous frame's pose
   and 6 rotations of it (+-25 deg about the camera axes), --track-steps Adam steps; kept: the lowest
   loss + --w-jump x (rotation change / 0.5 rad)^2.
3. Joint refinement over the episode's frames: data terms + second-difference penalties on t (--acc-t,
   m/frame^2) and R (--acc-r, rad/frame^2), scale fixed.

Writes <run>/episodes/episode_XXXXXX.npz (the human from --human, object from here), _object.obj (the chosen
mesh, <= 4000 faces), _camera.npz (copied), <run>/object_meta/episode_XXXXXX.json (+ _series.npz) and
<run>/overlays/episode_XXXXXX.jpg (mesh points green, object mask cyan, actor mask red).
Objects whose episodes all have an object_meta json are skipped (reruns resume); several workers claim
objects with <run>/object_meta/object_<name>.lock (delete stale locks after a crash).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from human_vm import episode_range, parse_episodes, scored_frames, video_path  # noqa: E402

DEV = "cuda"
IMAGE_SIZE = (1536, 1152)
MASK_STEP, DEPTH_STEP = 2, 4
N_MODEL = 1000  # model points during init and tracking
N_REFINE = 600
N_MASK = 400  # object mask pixels per frame
N_DEPTH = 300  # depth points per frame
BLEED = 0.03  # a pixel within 3% of the static background depth shows background, not the object
OCCLUDER_DILATE = 5  # px at 1/2 resolution around the actor mask that may hide the object


# ------------------------------------------------------------------------------------------ rotations
def rot6d_to_mat(x: torch.Tensor) -> torch.Tensor:
    b1 = F.normalize(x[..., :3], dim=-1)
    a2 = x[..., 3:]
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)


def mat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def gm(d: torch.Tensor, sigma: float) -> torch.Tensor:
    return d ** 2 / (d ** 2 + sigma ** 2)


# ------------------------------------------------------------------------------------- observations
class Observations:
    """Per fitted frame, GPU tensors of everything the losses read (see module docstring)."""

    def __init__(self, cache: Path, episode: int, frames: np.ndarray, K: np.ndarray, depth_scale: float,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        folder = cache / f"episode_{episode:06d}"
        masks = np.load(folder / "masks.npz")
        dz = np.load(folder / "depth.npz")
        obj_all, hum_all = masks["object"], masks["human"]
        depth_all, background = dz["depth"], dz["background"].astype(np.float32)
        n = len(frames)
        h2, w2 = obj_all.shape[1:]
        h4, w4 = depth_all.shape[1:]
        self.frames, self.K, self.shape2, self.shape4 = frames, K, (h2, w2), (h4, w4)
        mask_pts = np.zeros((n, N_MASK, 2), np.float32)
        mask_w = np.zeros((n, N_MASK), np.float32)
        depth_pts = np.zeros((n, N_DEPTH, 3), np.float32)
        depth_w = np.zeros((n, N_DEPTH), np.float32)
        dt = np.zeros((n, h2, w2), np.float16)
        dmap = np.zeros((n, h4, w4), np.float16)
        omap = np.zeros((n, h4, w4), np.float16)
        area = np.zeros(n)
        z_raw = np.full(n, np.nan)  # median depth under the whole 1/4-res mask (outline included)
        kernel = np.ones((2 * OCCLUDER_DILATE + 1,) * 2, np.uint8)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        for i, f in enumerate(frames):
            obj, hum = obj_all[f], hum_all[f]
            area[i] = obj.sum()
            depth = depth_all[f].astype(np.float32)
            dmap[i] = depth * depth_scale
            cover = cv2.resize(obj.astype(np.float32), (w4, h4), interpolation=cv2.INTER_AREA)
            obj4 = cover > 0
            # interior only: fully covered 1/4-resolution pixels, eroded once more. A pixel on the outline
            # mixes object and background depth, so thin objects (hoop, pan handle) may have no usable depth
            inner = cv2.erode((cover > 0.999).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            omap[i] = inner
            if area[i] < 10:
                continue
            if (obj4 & (depth > 0)).any():
                z_raw[i] = np.median(depth[obj4 & (depth > 0)]) * depth_scale
            ys, xs = np.nonzero(obj)
            pick = rng.choice(len(xs), min(N_MASK, len(xs)), replace=False)
            mask_pts[i, :len(pick)] = np.stack([xs[pick], ys[pick]], 1)
            mask_w[i, :len(pick)] = 1
            allowed = (obj | (cv2.dilate(hum.astype(np.uint8), kernel) > 0)).astype(np.uint8)
            dt[i] = cv2.distanceTransform(1 - allowed, cv2.DIST_L2, 5)
            # depth points: interior pixels, valid, not background bleed
            sel = inner & (depth > 0)
            if background.shape == depth.shape:
                sel &= ~((background > 0) & (np.abs(depth - background) < BLEED * background))
            ys, xs = np.nonzero(sel)
            if len(xs) >= 5:
                pick = rng.choice(len(xs), min(N_DEPTH, len(xs)), replace=False)
                z = depth[ys[pick], xs[pick]] * depth_scale
                u = xs[pick] * DEPTH_STEP + (DEPTH_STEP - 1) / 2
                v = ys[pick] * DEPTH_STEP + (DEPTH_STEP - 1) / 2
                depth_pts[i, :len(pick)] = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
                depth_w[i, :len(pick)] = 1
        self.area, self.z_raw = area, z_raw
        ref = np.percentile(area[area > 0], 90) if (area > 0).any() else 1.0
        self.vis = np.clip(area / ref, 0, 1)
        t = lambda a: torch.as_tensor(a, device=DEV)  # noqa: E731
        self.mask_pts, self.mask_w = t(mask_pts), t(mask_w)
        self.depth_pts, self.depth_w = t(depth_pts), t(depth_w)
        self.dt, self.dmap, self.omap = t(dt), t(dmap), t(omap)
        self.size = t(np.sqrt(np.maximum(area, 1)).astype(np.float32))
        self.has = t((area >= 10).astype(np.float32))
        self.has_depth = t((depth_w.sum(1) >= 5).astype(np.float32))

    def init_translation(self, i: int) -> tuple[np.ndarray, float]:
        """Rough object centre (camera frame) and size (m) from the frame's mask and depth."""
        w = self.depth_w[i] > 0
        K = self.K
        m = self.mask_pts[i][self.mask_w[i] > 0].cpu().numpy()
        if w.sum() >= 5:
            z = float(self.depth_pts[i][w, 2].median())
        else:  # no interior depth: the whole mask's depth (biased toward what lies behind a thin object)
            z = float(self.z_raw[i]) if np.isfinite(self.z_raw[i]) else float(np.nanmedian(self.z_raw))
        u = m[:, 0].mean() * MASK_STEP + 0.5
        v = m[:, 1].mean() * MASK_STEP + 0.5
        lo, hi = m.min(0), m.max(0)
        diag_px = np.linalg.norm(hi - lo) * MASK_STEP
        size = diag_px / K[0, 0] * z
        centre = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        centre += centre / np.linalg.norm(centre) * 0.3 * size  # the centre lies behind the visible surface
        return centre, size


# ---------------------------------------------------------------------------------------------- loss
class Model:
    def __init__(self, mesh: trimesh.Trimesh, count: int, seed: int = 0):
        pts, face = trimesh.sample.sample_surface_even(mesh, count, seed=seed)
        if len(pts) < count:  # sample_surface_even may return fewer
            extra, efaces = trimesh.sample.sample_surface(mesh, count - len(pts), seed=seed)
            pts, face = np.r_[pts, extra], np.r_[face, efaces]
        self.points = torch.as_tensor(pts, dtype=torch.float32, device=DEV)
        self.normals = torch.as_tensor(mesh.face_normals[face], dtype=torch.float32, device=DEV)


def sample_map(maps: torch.Tensor, idx: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """maps [n,h,w]; idx [B]; u, v [B,N] pixel coordinates on that grid -> [B,N] bilinear samples
    (clamped to the border). Gathers from the flat map, so B samples of one frame never copy it."""
    h, w = maps.shape[1:]
    u, v = u.clamp(0, w - 1), v.clamp(0, h - 1)
    x0, y0 = u.detach().floor().clamp(max=w - 2), v.detach().floor().clamp(max=h - 2)
    ax, ay = u - x0, v - y0
    i00 = idx[:, None] * (h * w) + y0.long() * w + x0.long()
    flat = maps.reshape(-1)
    at = lambda i: flat[i].float()  # noqa: E731
    return (at(i00) * (1 - ax) * (1 - ay) + at(i00 + 1) * ax * (1 - ay)
            + at(i00 + w) * (1 - ax) * ay + at(i00 + w + 1) * ax * ay)


def data_loss(obs: Observations, model: Model, idx: torch.Tensor, R, t, s, args, parts=False):
    """Per-sample loss [B] for frames idx [B] at poses R [B,3,3], t [B,3], s [B]."""
    K = obs.K
    X = s[:, None, None] * model.points[None] @ R.transpose(1, 2) + t[:, None]
    normals = model.normals[None] @ R.transpose(1, 2)
    z = X[..., 2].clamp_min(0.05)
    uf = K[0, 0] * X[..., 0] / z + K[0, 2]
    vf = K[1, 1] * X[..., 1] / z + K[1, 2]
    # silhouette terms on the 1/2-resolution grid
    h2, w2 = obs.shape2
    u2, v2 = (uf - 0.5) / MASK_STEP, (vf - 0.5) / MASK_STEP
    inside = (u2 >= 0) & (u2 <= w2 - 1) & (v2 >= 0) & (v2 <= h2 - 1) & (X[..., 2] > 0.05)
    size = obs.size[idx][:, None]
    l_in = (gm(sample_map(obs.dt, idx, u2, v2) / size, args.sil_sigma) * inside).mean(1)
    c = torch.cdist(obs.mask_pts[idx], torch.stack([u2, v2], -1)).min(-1).values
    mw = obs.mask_w[idx]
    l_cov = (gm(c / size, args.sil_sigma) * mw).sum(1) / mw.sum(1).clamp_min(1)
    # depth, data -> model
    d = torch.cdist(obs.depth_pts[idx], X).min(-1).values
    dw = obs.depth_w[idx]
    l_dep = (gm(d, args.sigma) * dw).sum(1) / dw.sum(1).clamp_min(1)
    # depth, front-facing model points on the object mask -> depth map
    h4, w4 = obs.shape4
    u4, v4 = (uf - 1.5) / DEPTH_STEP, (vf - 1.5) / DEPTH_STEP
    facing = (normals * X).sum(-1) < 0
    on_obj = sample_map(obs.omap, idx, u4, v4) > 0.5
    dval = sample_map(obs.dmap, idx, u4, v4)
    fw = (facing & on_obj & inside & (dval > 0)).float()
    l_front = (gm(X[..., 2] - dval, args.sigma) * fw).sum(1) / fw.sum(1).clamp_min(1)
    has, has_d = obs.has[idx], obs.has_depth[idx]
    total = has * (args.w_sil * (l_in + l_cov) + has_d * args.w_dep * (l_dep + l_front))
    if parts:
        return total, {"in": l_in * has, "cov": l_cov * has, "dep": l_dep * has * has_d,
                       "front": l_front * has * has_d}
    return total


def optimize(obs, model, idx, R0, t0, logs0, args, steps, fit_scale=True, lr=(0.03, 0.01, 0.02)):
    """Adam on (6D rotation, translation, log scale) for B independent samples; returns final values + loss."""
    r6 = mat_to_rot6d(R0).clone().requires_grad_(True)
    t = t0.clone().requires_grad_(True)
    logs = logs0.clone().requires_grad_(fit_scale)
    groups = [{"params": [r6], "lr": lr[0]}, {"params": [t], "lr": lr[1]}]
    if fit_scale:
        groups.append({"params": [logs], "lr": lr[2]})
    opt = torch.optim.Adam(groups)
    for _ in range(steps):
        opt.zero_grad()
        loss = data_loss(obs, model, idx, rot6d_to_mat(r6), t, logs.exp(), args)
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        R = rot6d_to_mat(r6)
        loss = data_loss(obs, model, idx, R, t, logs.exp(), args)
    return R.detach(), t.detach(), logs.detach(), loss


# --------------------------------------------------------------------------------------- the stages
def object_names(data: Path) -> dict[int, str]:
    """Episode -> object name (episodes_metadata.jsonl)."""
    rows = [json.loads(line) for line in (data / "meta/episodes_metadata.jsonl").read_text().splitlines() if line]
    return {r["episode_index"]: r["object"] for r in rows}


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    c = np.cumsum(w[order])
    return float(x[order][np.searchsorted(c, c[-1] / 2)])


def candidate_meshes(args, episodes: list[int]) -> tuple[list[Path], list[str]]:
    """Mesh files to try and the names of near-flat ones dropped (Hunyuan sometimes returns a background
    plane or a sheet: thinnest side < --min-thickness x longest)."""
    if args.mesh.endswith(".obj"):  # a given mesh file (e.g. a primitive), normalized like the others
        return [Path(args.mesh)], []
    if args.mesh != "best":  # one generated mesh: E_K
        e, k = args.mesh.split("_")
        return [p for p in [args.raw / "meshes" / f"episode_{int(e):06d}_{k}.obj"] if p.exists()], []
    found = sorted(p for e in episodes for p in (args.raw / "meshes").glob(f"episode_{e:06d}_*.obj"))
    keep, flat = [], []
    for p in found:
        ext = np.sort(trimesh.load(p, force="mesh", process=False).extents)
        (flat if ext[0] < args.min_thickness * ext[2] else keep).append(p)
    return (keep or found), [p.name for p in flat]


def keyframes(obs: Observations, scored: np.ndarray, count: int) -> list[int]:
    """Indices (into obs.frames) of well-visible frames in the scored span, spread over it."""
    in_scored = np.isin(obs.frames, scored)
    good = np.flatnonzero(in_scored & (obs.vis >= 0.8) & (obs.has_depth.cpu().numpy() > 0))
    if len(good) < count:
        good = np.flatnonzero(in_scored & (obs.area >= 10))
    if not len(good):
        good = np.flatnonzero(obs.area >= 10)
    if not len(good):
        raise RuntimeError("no object mask in the fitted range")
    return sorted({int(good[int(round(q))]) for q in np.linspace(0, len(good) - 1, count)})


def init_keyframes(obs, model, kfs, args, hyps: int, steps: int):
    """Best (R, t, log s, loss) per keyframe over `hyps` random rotations."""
    H = hyps
    rots = torch.as_tensor(Rotation.random(H, random_state=args.seed).as_matrix(), dtype=torch.float32, device=DEV)
    idx, R0, t0, s0 = [], [], [], []
    for k in kfs:
        centre, size = obs.init_translation(k)
        idx += [k] * H
        R0.append(rots)
        t0.append(torch.as_tensor(centre, dtype=torch.float32, device=DEV).expand(H, 3))
        s0 += [np.log(max(size, 0.02) / 1.2)] * H  # normalized mesh: projected diagonal ~ 1.2 x scale
    idx = torch.as_tensor(idx, device=DEV)
    R, t, logs, loss = optimize(obs, model, idx, torch.cat(R0), torch.cat(t0),
                                torch.as_tensor(s0, dtype=torch.float32, device=DEV), args, steps)
    best = []
    for j, k in enumerate(kfs):
        b = j * H + int(loss[j * H:(j + 1) * H].argmin())
        best.append((k, R[b], t[b], float(logs[b]), float(loss[b])))
    return best


PERTURB = torch.as_tensor(np.concatenate([np.eye(3)[None], Rotation.from_rotvec(
    np.radians(25) * np.concatenate([np.eye(3), -np.eye(3)])).as_matrix()]), dtype=torch.float32)


def track(obs, model, start: int, R_start, t_start, logs: float, args):
    """Forward and backward from `start`; both chains advance in the same batch."""
    n = len(obs.frames)
    R_all = torch.zeros((n, 3, 3), device=DEV)
    t_all = torch.zeros((n, 3), device=DEV)
    loss_all = torch.zeros(n, device=DEV)
    R_all[start], t_all[start] = R_start, t_start
    perturb = PERTURB.to(DEV)
    P = len(perturb)
    s = torch.full((1,), float(logs), dtype=torch.float32, device=DEV)
    for j in range(1, max(start + 1, n - start)):
        pairs = [(i, i - d) for i, d in ((start + j, 1), (start - j, -1)) if 0 <= i < n]
        idx, R0, t0 = [], [], []
        for i, prev in pairs:
            # rotate about the object's centre: R' = Q R, t stays (the mesh is centred at its bbox centre)
            idx += [i] * P
            R0.append(perturb @ R_all[prev])
            t0.append(t_all[prev].expand(P, 3))
        idx_t = torch.as_tensor(idx, device=DEV)
        R, t, _, loss = optimize(obs, model, idx_t, torch.cat(R0), torch.cat(t0), s.expand(len(idx)), args,
                                 args.track_steps, fit_scale=False, lr=(0.02, 0.005, 0.0))
        # pick by loss plus a penalty on the rotation jump from the previous frame, so an occluded frame
        # does not flip to a pose that only fits its partial mask slightly better
        prev_R = torch.stack([R_all[prev] for _, prev in pairs]).repeat_interleave(P, 0)
        cos = ((prev_R.transpose(1, 2) @ R).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
        jump = torch.arccos(cos.clamp(-1, 1))
        score = loss + args.w_jump * (jump / 0.5) ** 2
        for p, (i, prev) in enumerate(pairs):
            if obs.has[i] < 1:  # nothing to fit: hold the previous pose
                R_all[i], t_all[i] = R_all[prev], t_all[prev]
                continue
            b = p * P + int(score[p * P:(p + 1) * P].argmin())
            R_all[i], t_all[i], loss_all[i] = R[b], t[b], loss[b]
    return R_all, t_all, loss_all


def refine(obs, model, R_init, t_init, logs: float, args, fit_scale: bool = False):
    n = len(obs.frames)
    idx = torch.arange(n, device=DEV)
    r6 = mat_to_rot6d(R_init).clone().requires_grad_(True)
    t = t_init.clone().requires_grad_(True)
    ls = torch.tensor(float(logs), dtype=torch.float32, device=DEV, requires_grad=fit_scale)
    groups = [{"params": [r6], "lr": 0.01}, {"params": [t], "lr": 0.002}]
    opt = torch.optim.Adam(groups + ([{"params": [ls], "lr": 0.005}] if fit_scale else []))
    for _ in range(args.refine_steps):
        opt.zero_grad()
        R = rot6d_to_mat(r6)
        data = data_loss(obs, model, idx, R, t, ls.exp().expand(n), args).mean()
        acc_t = ((t[2:] - 2 * t[1:-1] + t[:-2]).norm(dim=-1) / args.acc_t) ** 2
        acc_r = ((R[2:] - 2 * R[1:-1] + R[:-2]).flatten(1).norm(dim=-1) / args.acc_r) ** 2
        loss = data + args.w_smooth * (acc_t.mean() + acc_r.mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        R = rot6d_to_mat(r6)
        total, parts = data_loss(obs, model, idx, R, t, ls.exp().expand(n), args, parts=True)
    return R, t.detach(), float(ls.detach()), total, parts


# ------------------------------------------------------------------------------------------ outputs
def overlay(args, episode, obs, mesh, R, t, s, picks, out: Path, tile=(320, 320)):
    pts = torch.as_tensor(trimesh.sample.sample_surface(mesh, 6000, seed=1)[0], dtype=torch.float32, device=DEV)
    masks = np.load(args.cache / f"episode_{episode:06d}" / "masks.npz")
    K = obs.K
    cap = cv2.VideoCapture(str(video_path(args.data, episode)))
    tiles = []
    for i in picks:
        f = int(obs.frames[i])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (768, 576))
        X = (s * pts @ R[i].T + t[i]).cpu().numpy()
        u = (X[:, 0] / X[:, 2] * K[0, 0] + K[0, 2] - 0.5) / 2
        v = (X[:, 1] / X[:, 2] * K[1, 1] + K[1, 2] - 0.5) / 2
        ok = (u >= 0) & (u < 768) & (v >= 0) & (v < 576) & (X[:, 2] > 0)
        layer = img.copy()
        layer[v[ok].astype(int), u[ok].astype(int)] = (60, 230, 60)
        img = cv2.addWeighted(img, 0.35, layer, 0.65, 0)
        for m, color in ((masks["human"][f], (60, 60, 255)), (masks["object"][f], (255, 220, 0))):
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 1)
        ys, xs = np.nonzero(masks["object"][f])
        px, py = np.r_[xs, u[ok]], np.r_[ys, v[ok]]
        cx, cy = (np.median(px), np.median(py)) if len(px) else (384, 288)
        half = max(np.ptp(px) if len(px) else 0, np.ptp(py) if len(py) else 0, 60) * 0.9
        x0, y0 = int(np.clip(cx - half, 0, 767)), int(np.clip(cy - half, 0, 575))
        x1, y1 = int(np.clip(cx + half, x0 + 8, 768)), int(np.clip(cy + half, y0 + 8, 576))
        crop = cv2.resize(img[y0:y1, x0:x1], tile)
        cv2.putText(crop, f"ep{episode} f{f}", (6, 20), 0, 0.6, (255, 255, 255), 2)
        tiles.append(crop)
    cap.release()
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


class Episode:
    """One episode's inputs for the fit."""

    def __init__(self, episode: int, args, scored: np.ndarray):
        self.episode, self.scored = episode, scored
        self.human = dict(np.load(args.human / f"episode_{episode:06d}.npz"))
        cam = dict(np.load(args.human / f"episode_{episode:06d}_camera.npz"))
        self.K, self.depth_scale = cam["K"].astype(np.float64), float(cam.get("depth_scale", 1.0))
        self.total = len(self.human["pose"])
        self.frames = episode_range(scored, self.total)
        self.obs = Observations(args.cache, episode, self.frames, self.K, self.depth_scale, seed=args.seed)
        self.kfs = keyframes(self.obs, scored, args.keyframes)


def init_object(eps: list[Episode], mesh, args, hyps: int, steps: int):
    """Keyframe init in every episode with one mesh: ({episode: best per keyframe}, mean loss)."""
    model = Model(mesh, N_MODEL, args.seed)
    best = {ep.episode: init_keyframes(ep.obs, model, ep.kfs, args, hyps, steps) for ep in eps}
    return best, float(np.mean([np.mean([b[4] for b in v]) for v in best.values()]))


def process_object(eps: list[Episode], args) -> list[dict]:
    """One mesh and one scale for the object (all its episodes' keyframes), then a track per episode."""
    candidates, flat = candidate_meshes(args, [ep.episode for ep in eps])
    if not candidates:
        raise RuntimeError("no candidate mesh")
    timing = {}
    t1 = time.time()
    meshes = {p: trimesh.load(p, force="mesh", process=False) for p in candidates}
    screen = {}  # cheap pass over every candidate, then the full init on the best few
    if len(meshes) > 1:
        for path, mesh in meshes.items():
            screen[path] = init_object(eps, mesh, args, args.screen_hyps, args.screen_steps)[1]
    else:
        screen = {path: 0.0 for path in meshes}
    timing["screen"] = round(time.time() - t1, 1)
    t1 = time.time()
    trials = {path: init_object(eps, meshes[path], args, args.hyps, args.init_steps)
              for path in sorted(screen, key=screen.get)[:args.finalists]}
    timing["init"] = round(time.time() - t1, 1)
    path = min(trials, key=lambda p: trials[p][1])
    mesh, (best, _) = meshes[path], trials[path]

    # one scale for the object: weighted median over every keyframe, frames with depth count 5x
    kf_scale = np.array([np.exp(b[3]) for ep in eps for b in best[ep.episode]])
    kf_weight = np.array([1 + 4 * float(ep.obs.has_depth[b[0]]) for ep in eps for b in best[ep.episode]])
    s = weighted_median(kf_scale, kf_weight) if args.share_scale else None
    model = Model(mesh, N_MODEL, args.seed)
    stats = []
    for ep in eps:
        t1 = time.time()
        obs, kb = ep.obs, best[ep.episode]
        start = min(kb, key=lambda b: b[4])
        s_ep = s if s is not None else float(np.median([np.exp(b[3]) for b in kb]))
        # re-fit the start keyframe at the shared scale: the translation scales along the ray (same image)
        k, R_k, t_k, logs_k = start[0], start[1], start[2], start[3]
        P = len(PERTURB)
        R, t, _, loss = optimize(obs, model, torch.full((P,), k, dtype=torch.long, device=DEV), PERTURB.to(DEV) @ R_k,
                                 (t_k * float(s_ep / np.exp(logs_k))).expand(P, 3),
                                 torch.full((P,), float(np.log(s_ep)), dtype=torch.float32, device=DEV),
                                 args, args.init_steps, fit_scale=False)
        b = int(loss.argmin())
        R_tr, t_tr, loss_tr = track(obs, model, k, R[b], t[b], float(np.log(s_ep)), args)
        t_track = time.time() - t1
        t1 = time.time()
        R, t, logs, loss, parts = refine(obs, Model(mesh, N_REFINE, args.seed + 1), R_tr, t_tr, float(np.log(s_ep)), args,
                                         fit_scale=s is None)
        ep_timing = dict(timing, track=round(t_track, 1), refine=round(time.time() - t1, 1))
        stats.append(write_episode(ep, args, path, mesh, float(np.exp(logs)), R, t, loss, loss_tr, parts, {
            "object_episodes": [e.episode for e in eps], "shared_scale": s is not None,
            "screen": {p.name: round(v, 4) for p, v in sorted(screen.items(), key=lambda kv: kv[1])},
            "dropped_flat": flat,
            "finalists": {p.name: round(v[1], 4) for p, v in trials.items()},
            "keyframes": [int(ep.frames[b[0]]) for b in kb],
            "keyframe_scales_m": [round(float(np.exp(b[3])), 4) for b in kb],
            "keyframe_loss": [round(b[4], 4) for b in kb],
            "timing": ep_timing}))
    return stats


def write_episode(ep: Episode, args, path, mesh, s, R, t, loss, loss_tr, parts, extra) -> dict:
    """Submission npz + mesh + camera, per-frame series, overlay; returns the stats."""
    frames, total, obs = ep.frames, ep.total, ep.obs
    near = np.clip(np.searchsorted(frames, np.arange(total)), 0, len(frames) - 1)
    Rn = R.cpu().numpy().astype(np.float64)
    Rn = np.stack([u @ vt for u, _, vt in (np.linalg.svd(r) for r in Rn)])  # exactly orthonormal
    out = args.run / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    name = f"episode_{ep.episode:06d}"
    human = ep.human
    np.savez(out / f"{name}.npz", pose=human["pose"], scales=human["scales"], shape=human["shape"],
             object_rotation=Rn[near].astype(np.float32), object_translation=t.cpu().numpy()[near].astype(np.float32),
             object_scale=np.float32(s))
    mesh.export(out / f"{name}_object.obj")
    shutil.copy(args.human / f"{name}_camera.npz", out / f"{name}_camera.npz")

    in_scored = np.isin(frames, ep.scored)
    has = obs.has.cpu().numpy() > 0
    tr = t.cpu().numpy()
    acc = np.linalg.norm(tr[2:] - 2 * tr[1:-1] + tr[:-2], axis=-1) * 100
    stats = {
        "episode": ep.episode, "mesh": path.name, "scale_m": round(s, 4),
        "extent_m": np.round(np.asarray(mesh.extents) * s, 3).tolist(),
        "frames_fitted": len(frames), "frames_with_mask": int(has.sum()),
        "frames_with_depth": int((obs.has_depth.cpu().numpy() > 0).sum()),
        "scored_with_mask": round(float(has[in_scored].mean()), 3),
        "loss_track_median": round(float(np.median(loss_tr.cpu().numpy()[has])), 4),
        "loss_median": round(float(np.median(loss.cpu().numpy()[has])), 4),
        "parts_median": {k: round(float(np.median(v.cpu().numpy()[has])), 4) for k, v in parts.items()},
        "acc_t_cm_mean": round(float(acc.mean()), 4), "depth_scale": round(ep.depth_scale, 4), **extra,
    }
    meta = args.run / "object_meta"
    meta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(meta / f"{name}_series.npz", frames=frames, R=Rn, t=tr, loss=loss.cpu().numpy(),
                        loss_track=loss_tr.cpu().numpy(), vis=obs.vis,
                        **{f"part_{k}": v.cpu().numpy() for k, v in parts.items()})
    picks = np.flatnonzero(in_scored)
    picks = picks[np.linspace(0, len(picks) - 1, 8).astype(int)]
    overlay(args, ep.episode, obs, mesh, R, t, s, picks, args.run / "overlays" / f"{name}.jpg")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", type=Path, required=True, help="Step 4 run's episodes folder")
    parser.add_argument("--run", type=Path, required=True, help="results folder to write")
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--mesh", default="best",
                        help="best: lowest keyframe loss over every mesh of the object (all its episodes); "
                             "or one mesh: E_K (episode E's crop K) or a path to an .obj")
    parser.add_argument("--raw", type=Path, default=Path("/content/object_raw"))
    parser.add_argument("--cache", type=Path, default=REPO / "cache")
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--sample", type=Path,
                        default=REPO / "data/v2d_submission_kit/data/track_1_sample_submission.parquet")
    parser.add_argument("--keyframes", type=int, default=3)
    parser.add_argument("--hyps", type=int, default=256)
    parser.add_argument("--init-steps", type=int, default=150)
    parser.add_argument("--screen-hyps", type=int, default=64, help="rotations per keyframe when screening meshes")
    parser.add_argument("--screen-steps", type=int, default=80)
    parser.add_argument("--finalists", type=int, default=2, help="meshes that get the full --hyps init")
    parser.add_argument("--min-thickness", type=float, default=0.06,
                        help="drop candidate meshes whose thinnest side < this x longest (sheets, planes)")
    parser.add_argument("--no-share-scale", dest="share_scale", action="store_false",
                        help="fit the scale per episode instead of one per object")
    parser.add_argument("--track-steps", type=int, default=30)
    parser.add_argument("--refine-steps", type=int, default=300)
    parser.add_argument("--sigma", type=float, default=0.03, help="m, depth robust scale")
    parser.add_argument("--sil-sigma", type=float, default=0.1, help="silhouette robust scale, x object size")
    parser.add_argument("--w-sil", type=float, default=1.0)
    parser.add_argument("--w-dep", type=float, default=1.0)
    parser.add_argument("--w-smooth", type=float, default=0.1)
    parser.add_argument("--w-jump", type=float, default=0.05,
                        help="tracking: candidate score += w (rotation change from the previous frame / 0.5 rad)^2")
    parser.add_argument("--acc-t", type=float, default=0.005, help="m/frame^2 scale of the translation penalty")
    parser.add_argument("--acc-r", type=float, default=0.02, help="rad/frame^2 scale of the rotation penalty")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    scored = scored_frames(args.sample)
    meta = args.run / "object_meta"
    meta.mkdir(parents=True, exist_ok=True)
    names = object_names(args.data)
    wanted = parse_episodes(args.episodes)
    groups = {}  # object -> its requested episodes, in request order
    for e in wanted:
        groups.setdefault(names[e], []).append(e)
    for obj, episodes in groups.items():
        todo = [e for e in episodes if not (meta / f"episode_{e:06d}.json").exists()]
        lock = meta / f"object_{obj}.lock"  # several workers can share the GPU; each object goes to one
        if not todo:
            continue
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        try:
            eps = [Episode(e, args, scored[e]) for e in episodes]  # all requested episodes inform mesh + scale
            for stats in process_object(eps, args):
                stats["object"] = obj
                print("ep " + json.dumps(stats), flush=True)
                (meta / f"episode_{stats['episode']:06d}.json").write_text(json.dumps(stats, indent=1))
        except Exception:
            print(f"object {obj} (episodes {episodes}) FAILED", flush=True)
            traceback.print_exc()
            lock.unlink(missing_ok=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

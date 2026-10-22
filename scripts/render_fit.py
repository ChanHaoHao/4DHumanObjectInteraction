#!/usr/bin/env python3
"""Render-and-compare pose refinement (runs on the Colab VM, GPU): each frame's object pose fitted to the SAM 3
masks by rendering the object and the body with nvdiffrast, plus the body (contact, penetration, floor) and
smoothness. No depth map, no colour.

    python render_fit.py --human <run>_human/episodes --init <run>_j/episodes --run <run>_r [--episodes 2]
        [--cache /content/repo/cache] [--data /content/v2d/track_1] [--crop 512]

Per episode (fitted frames = scored span +- 15; mesh and scale from --init, fixed):
1. Crops. Per frame a square window around the object (its mask's box and the init pose's projection, +30%)
   rendered at --crop px, so thin parts get pixels. The object and actor masks (1/2 res) and the frame's Canny
   edges (distance transform) are resampled into it once; the body mesh is rendered into it once (depth z_b and
   coverage: the body does not move here).
2. Image term (the hybrid occlusion rule). Object coverage S (antialiased: soft edges with gradients), object
   depth z_o, visibility v = sigmoid((z_b - z_o) / --tau) where the body is drawn, 1 elsewhere. Each pixel is
   judged by the observed masks:
     object mask    1 - S v   missing, or drawn but behind the body
     actor only     S v       drawn in front of the body where only the body shows; only where the body mesh is
                              drawn (an actor pixel the mesh misses is don't-care)
     background     S         any object: the body mesh may not hide anything outside the actor mask
   Drawn pixels within --band px of a mask boundary count x0.2 (object mask pixels always count in full); on frames
   whose drawn body overlaps the actor mask in the crop by IoU < --trust the actor-only rule counts x0.2. Cost =
   1 - soft IoU = 1 - sum(object mask x S v) / (sum(object mask) + the weighted actor-only and background costs).
   Plus --w-edge x the mean image-edge distance (clamped at 20 px, / 20) under the visible outline away from the body.
3. Body terms: contact (joint_fit's nearest candidate outside the mesh, GM, on every fitted frame; it moves the
   object along its camera ray only), hand penetration (penetration_cm, the
   scorer's inside rule), object points inside the body mesh away from the hands (winding numbers, refreshed every
   --inside-every steps; cost = cm to the nearest body vertex), floor (cm below it).
4. Per frame, coarse to fine (masks blurred by --blur px per level), for each candidate: the init track, its tilt
   mirror (mirror_tilt) and the 180-degree turns / flips that change the shape (joint_fit stage 1b's distinct
   states). Viterbi over frames picks one per frame (unary: the per-frame loss; pair: --w-switch x
   min((surface point motion / 5 cm)^2, 9)).
4b. Tracking: forward then backward over the picked track, each frame also refitted from its neighbour's pose and
   that pose's constant-velocity extrapolation (--track-blur, --track-steps); the better wins unless the frame's
   own loss is lower by more than --track-margin. The init alone can drift by ~90 degrees (ep 2 frames 303+), and a
   swung hoop turns ~20 degrees a frame, past what one local refit recovers.
5. All frames jointly from that pick: steps 2-3 + --w-acc x GM(surface point acceleration, --acc-ref) + --w-vel x
   GM(relu(point speed - --vmax), --vmax), robust so a lost frame's jump does not drag its neighbours. Frames with no object mask are flagged unreliable: their image term is
   dropped (it would only push the object out of view), so neighbours, the body and smoothness carry them.

Writes <run>/episodes/ (npz, mesh, camera: joint_fit.write_episode, which also leaves <run>/joint_meta/*_series.npz),
<run>/render_meta/episode_XXXXXX.json (+ _series.npz: per frame IoU, actor-only violation, body trust, unreliable,
candidate) and <run>/videos/episode_XXXXXX_fit.mp4 (half res: visible outline green, object mask cyan, red border =
unreliable).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_vm import parse_episodes, video_path  # noqa: E402
from joint_fit import (BodyParts, Episode, axis_angle, build_parser, contact_report, farthest_points,  # noqa: E402
                       floor_depth, mirror_tilt, nearest, object_axes, outside_distance, viterbi, write_episode)
from object_fit import N_REFINE, Model, gm, mat_to_rot6d, rot6d_to_mat  # noqa: E402
from proxy_eval import MHRBody, mesh_budget, scored_frames  # noqa: E402
from pushout import penetration_cm, winding_numbers  # noqa: E402

DEV = "cuda"
NEAR, FAR = 0.05, 50.0  # m


# ------------------------------------------------------------------------------------------- render
class Renderer:
    """Meshes in camera coordinates -> square crops (x0, y0, side in full-resolution px) at res x res."""

    def __init__(self, K: np.ndarray, res: int):
        import nvdiffrast.torch as dr

        self.dr, self.ctx, self.res = dr, dr.RasterizeCudaContext(), res
        # python floats: numpy scalars would turn the clip coordinates float64
        self.fx, self.fy, self.cx, self.cy = (float(K[i, j]) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))

    def clip(self, X: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Pixel centres at integers (full resolution); crop pixel j's centre maps to NDC 2 (j + 0.5) / res - 1,
        row 0 at the crop's top (NDC y = -1: image v and NDC y grow together)."""
        x, y, z = X.unbind(-1)
        x0, y0, side = box[:, 0:1], box[:, 1:2], box[:, 2:3]
        cx = 2 * (self.fx * x + (self.cx - x0 + 0.5) * z) / side - z
        cy = 2 * (self.fy * y + (self.cy - y0 + 0.5) * z) / side - z
        cz = z * (FAR + NEAR) / (FAR - NEAR) - 2 * FAR * NEAR / (FAR - NEAR)
        return torch.stack([cx, cy, cz, z], -1).contiguous()

    def __call__(self, X: torch.Tensor, tri: torch.Tensor, box: torch.Tensor, soft: bool = True):
        """X [B,V,3] -> coverage [B,res,res] (antialiased if soft) and depth z [B,res,res] (FAR where empty)."""
        pos = self.clip(X, box)
        rast, _ = self.dr.rasterize(self.ctx, pos, tri, (self.res, self.res))
        hit = (rast[..., 3:] > 0).float()
        cov = self.dr.antialias(hit, rast, pos, tri)[..., 0] if soft else hit[..., 0]
        z, _ = self.dr.interpolate(X[..., 2:].contiguous(), rast, tri)
        return cov, torch.where(rast[..., 3] > 0, z[..., 0], torch.full_like(z[..., 0], FAR))


def check_orientation(K: np.ndarray, res: int = 128):
    """A small triangle at a known pixel must land on the crop pixel the clip mapping promises (rows = image v)."""
    rd = Renderer(K, res)
    z = 2.0
    u, v = K[0, 2] + 37.0, K[1, 2] - 21.0
    c = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    X = torch.as_tensor(c + np.array([[-0.01, -0.01, 0], [0.01, -0.01, 0], [0.0, 0.012, 0]]), dtype=torch.float32,
                        device=DEV)[None]
    box = torch.tensor([[float(K[0, 2]) - 60.0, float(K[1, 2]) - 110.0, 300.0]], dtype=torch.float32, device=DEV)
    cov, _ = rd(X, torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEV), box, soft=False)
    ys, xs = torch.nonzero(cov[0] > 0, as_tuple=True)
    want = ((u - float(box[0, 0]) + 0.5) * res / 300 - 0.5, (v - float(box[0, 1]) + 0.5) * res / 300 - 0.5)
    got = (float(xs.float().mean()), float(ys.float().mean()))
    assert abs(got[0] - want[0]) < 2 and abs(got[1] - want[1]) < 2, f"render orientation: got {got}, want {want}"


def blur(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """[B,H,W] float -> Gaussian-blurred (separable), same size."""
    if sigma <= 0:
        return m
    r = int(2 * sigma + 0.5)
    k = torch.exp(-torch.arange(-r, r + 1, device=m.device, dtype=m.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(m[:, None], k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))[:, 0]


# ------------------------------------------------------------------------------------------- setup
def crop_affine(box, res: int, step: float) -> np.ndarray:
    """2x3 map from crop pixel (j, i) to a 1/step-resolution image's pixel (centres at integers)."""
    x0, y0, side = box
    a = side / res
    return np.array([[a / step, 0, (x0 + 0.5 * a) / step - 0.5], [0, a / step, (y0 + 0.5 * a) / step - 0.5]])


class Scene:
    """One episode's fixed inputs on the GPU: crops, masks, edge distances, the body render, body meshes."""

    def __init__(self, ep: Episode, body: MHRBody, model: Model, args):
        self.ep, self.args = ep, args
        n, res = len(ep.frames), args.crop
        K = ep.cam["K"].astype(np.float64)
        self.K = K
        self.rd = Renderer(K, res)
        f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=DEV)  # noqa: E731
        self.V = f32(ep.vertices)  # the packed mesh, as scored
        self.tri = torch.as_tensor(ep.faces.astype(np.int32), device=DEV)
        self.s = float(ep.s0)
        pts = model.points.cpu().numpy()
        self.sub = f32(farthest_points(pts, args.points))
        self.mpts = model.points

        # body meshes, camera frame
        pose = np.asarray(ep.human["pose"], np.float64)[ep.frames]
        params = np.concatenate([pose, np.broadcast_to(ep.human["scales"], (n, 68))], 1)
        verts, _ = body(params, ep.human["shape"])
        self.bverts = f32(verts)
        self.btri = torch.as_tensor(body.faces.astype(np.int32), device=DEV)
        not_hand = np.setdiff1d(np.arange(verts.shape[1]), body.hand_all)
        self.bsub = self.bverts[:, not_hand[:: args.body_stride]]  # inside cost: nearest body vertex, hands excluded
        self.hands = self.bverts[:, body.hand_all[:: max(1, args.body_stride // 2)]]

        # crops: the object mask's box and the init pose's projection, +30%
        mz = np.load(ep.cache / ep.name / "masks.npz")
        obj_all, hum_all = mz["object"], mz["human"]
        X0 = self.s * ep.vertices[None] @ np.transpose(ep.R0, (0, 2, 1)) + ep.t0[:, None]
        z0 = np.maximum(X0[..., 2], 0.05)
        uv0 = np.stack([K[0, 0] * X0[..., 0] / z0 + K[0, 2], K[1, 1] * X0[..., 1] / z0 + K[1, 2]], -1)
        boxes = []
        for i, f in enumerate(ep.frames):
            ys, xs = np.nonzero(obj_all[f])
            P = np.concatenate([uv0[i], np.c_[2 * xs + 0.5, 2 * ys + 0.5]]) if len(xs) else uv0[i]
            lo, hi = P.min(0), P.max(0)
            side = float(np.clip((hi - lo).max() * 1.3, 64, 1400))
            c = (lo + hi) / 2
            boxes.append([c[0] - side / 2, c[1] - side / 2, side])
        self.boxes = np.array(boxes)
        self.box = f32(self.boxes)

        mo, ma, act, band = (np.zeros((n, res, res), np.uint8) for _ in range(4))  # 1 byte: a 6 GB GPU fits
        dt = np.zeros((n, res, res), np.float16)
        warp = lambda img, M, interp: cv2.warpAffine(img, M, (res, res), flags=interp | cv2.WARP_INVERSE_MAP,  # noqa: E731
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        dt[:] = self.edge_crops(ep, args)
        for i, f in enumerate(ep.frames):
            M2 = crop_affine(self.boxes[i], res, 2.0)
            o = warp(obj_all[f].astype(np.uint8), M2, cv2.INTER_NEAREST)
            h = warp(hum_all[f].astype(np.uint8), M2, cv2.INTER_NEAREST)
            mo[i], ma[i], act[i] = o, h & (1 - o), h
            # boundary band of either mask
            bnd = np.zeros((res, res), np.uint8)
            for m in (o, h):
                bnd |= (m ^ cv2.erode(m, np.ones((3, 3), np.uint8))) | (m ^ cv2.dilate(m, np.ones((3, 3), np.uint8)))
            band[i] = cv2.distanceTransform(1 - bnd, cv2.DIST_L2, 3) < args.band
        del obj_all, hum_all
        t = lambda a: torch.as_tensor(a, device=DEV)  # noqa: E731
        self.mo, self.ma, self.band, self.dt = t(mo), t(ma), t(band).bool(), t(dt)
        area = mo.reshape(n, -1).astype(np.float32).sum(1)
        self.area = area
        self.norm = t(np.maximum(area, 0.25 * np.median(area[area > 0]) if (area > 0).any() else 1.0).astype(np.float32))

        # the body, rendered once
        zb, br = [], []
        with torch.no_grad():
            for a in range(0, n, args.batch):
                cov, z = self.rd(self.bverts[a:a + args.batch], self.btri, self.box[a:a + args.batch], soft=False)
                zb.append(z.half()), br.append(cov > 0)
        self.zb, self.br = torch.cat(zb), torch.cat(br)
        actt = t(act).float()
        inter = (actt * self.br).sum((1, 2))
        union = (actt + self.br.float()).clamp(max=1).sum((1, 2))
        small = actt.sum((1, 2)) < 0.02 * res * res
        self.trust = torch.where(small, torch.ones_like(inter), inter / union.clamp_min(1))
        self.trust_w = torch.where(self.trust < args.trust, 0.2, 1.0)
        self.inside = torch.zeros((n, len(self.sub)), dtype=torch.bool, device=DEV)
        self.unrel_w = torch.ones(n, device=DEV)

    def edge_crops(self, ep: Episode, args) -> np.ndarray:
        """Per fitted frame, in its crop: distance (crop px) to the nearest Canny edge of the RGB frame (40 full-res
        px at most, and outside the image)."""
        res = args.crop
        cap = cv2.VideoCapture(str(video_path(args.data, ep.episode)))
        want = {int(f): i for i, f in enumerate(ep.frames)}
        out = np.zeros((len(ep.frames), res, res), np.float16)
        f = -1
        while want:
            ok, bgr = cap.read()
            if not ok:
                break
            f += 1
            if f in want:
                i = want.pop(f)
                e = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 50, 150)
                d = np.minimum(cv2.distanceTransform(255 - e, cv2.DIST_L2, 3), 40)
                out[i] = cv2.warpAffine(d, crop_affine(self.boxes[i], res, 1.0), (res, res),
                                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=40) * (res / self.boxes[i][2])
        cap.release()
        if want:
            raise RuntimeError(f"ep {ep.episode}: video ended before frames {sorted(want)[:5]}")
        return out

    # --------------------------------------------------------------------------------------- losses
    def points(self, R, t, P):
        return self.s * P[None] @ R.transpose(1, 2) + t[:, None]

    def image(self, idx, R, t, sigma: float, parts: bool = False):
        """Per frame [B]: the hybrid silhouette term + the edge term (see the docstring)."""
        args = self.args
        cov, zo = self.rd(self.points(R, t, self.V), self.tri, self.box[idx])
        br = self.br[idx]
        v = torch.where(br, torch.sigmoid((self.zb[idx].float() - zo) / args.tau), torch.ones_like(zo))
        vis = cov * v
        mo, ma = blur(self.mo[idx].float(), sigma), blur(self.ma[idx].float(), sigma)
        bg = (1 - mo - ma).clamp_min(0)
        w = torch.where(self.band[idx], 0.2, 1.0)
        act_only = ma * br * self.trust_w[idx][:, None, None]
        # 1 - soft IoU: drawing nothing costs 1, as much as no overlap, so shrinking or hiding the object never beats
        # a near miss (the old sum / mask area scored an empty render 0.42 under an offset ring's 0.55, ep 2 f317);
        # object pixels count in full: on a thin ring nearly every mask pixel lies in the boundary band
        tp = (mo * vis).sum((1, 2))
        fp = (w * (act_only * vis + bg * cov)).sum((1, 2))
        sil = 1 - tp / (mo.sum((1, 2)) + fp).clamp_min(1e-6)
        mag = (cov[:, :-1, 1:] - cov[:, :-1, :-1]).abs() + (cov[:, 1:, :-1] - cov[:, :-1, :-1]).abs()
        away = F.max_pool2d(br.half()[:, None], 7, 1, 3)[:, 0] == 0  # 3 px off the drawn body
        mag = mag * away[:, :-1, :-1]
        edge = (mag * self.dt[idx][:, :-1, :-1].float().clamp(max=20)).sum((1, 2)) / mag.sum((1, 2)).clamp_min(1) / 20
        total = sil + args.w_edge * edge
        if parts:
            with torch.no_grad():
                hv, ho = vis > 0.5, self.mo[idx] > 0
                iou = (hv & ho).sum((1, 2)) / (hv | ho).sum((1, 2)).clamp_min(1)
                viol = (hv & (self.ma[idx] > 0) & br).sum((1, 2)) / self.norm[idx]
            return total, {"sil": sil.detach(), "edge": edge.detach(), "iou": iou, "actor_violation": viol}
        return total

    def body_terms(self, idx, R, t):
        """Per frame [B]: contact + hand penetration + object inside the body + floor, weighted."""
        ep, args = self.ep, self.args
        s = torch.full((len(idx),), self.s, device=DEV)
        # contact moves the object along its camera ray only (joint_fit --contact-dof ray): the silhouette fixes the
        # image, the hands the distance; with all 6 DOF it dragged the hoop off its mask (ep 2 f363 IoU 0.60 -> 0.30)
        ray = F.normalize(t.detach(), dim=-1)
        t_c = t.detach() + ((t - t.detach()) * ray).sum(-1, keepdim=True) * ray
        contact = gm(nearest(outside_distance(ep.pts[idx], R.detach(), t_c, s, ep.sdf), ep.cand[idx]), args.sigma_contact)
        contact = contact * ep.has_contact[idx]
        pen = penetration_cm(ep.hands_all[idx], R, t, self.s, ep.sdf, args.margin)
        floor = floor_depth(self.points(R, t, self.mpts), ep, args.floor_tol).mean(1) * 100
        inside = torch.zeros(len(idx), device=DEV)
        flag = self.inside[idx]
        rows = torch.nonzero(flag.any(1), as_tuple=True)[0]
        if len(rows):
            X = self.points(R[rows], t[rows], self.sub)
            d = torch.cdist(X, self.bsub[idx[rows]]).min(-1).values * 100
            inside[rows] = (d * flag[rows]).sum(1) / len(self.sub)
        return (args.w_contact * contact + args.w_pen * pen + args.w_inside * inside + args.w_floor * floor)

    def refresh_inside(self, R, t, frames):
        """Object sample points inside the body mesh (winding number > 0.5) and > 3 cm from every hand vertex.
        R, t: poses of the frame indices `frames`."""
        with torch.no_grad():
            X = self.points(R, t, self.sub)
            for x, i in zip(X, frames.tolist()):
                wn = winding_numbers(x, self.bverts[i][self.btri.long()])
                far = torch.cdist(x[None], self.hands[i][None])[0].min(-1).values > 0.03
                self.inside[i] = (wn > 0.5) & far

    def temporal(self, R, t):
        args = self.args
        X = self.points(R, t, self.sub)
        acc = (X[2:] - 2 * X[1:-1] + X[:-2]).norm(dim=-1).mean(-1)
        vel = (X[1:] - X[:-1]).norm(dim=-1).mean(-1)
        # GM (robust): a jump to a lost frame costs at most w, so it cannot drag its good neighbours along
        return (args.w_acc * gm(acc, args.acc_ref).mean()
                + args.w_vel * gm(torch.relu(vel - args.vmax), args.vmax).mean())


# ---------------------------------------------------------------------------------------- fitting
def fit(sc: Scene, R0: np.ndarray, t0: np.ndarray, levels, steps: int, joint: bool, frames=None):
    """Adam on every frame's (6D rotation, translation). joint: + temporal terms and the unreliable weights.
    frames: the frame indices R0, t0 belong to (default all; joint needs all)."""
    args, n = sc.args, len(R0)
    sel = torch.arange(n, device=DEV) if frames is None else torch.as_tensor(frames, device=DEV)
    r6 = mat_to_rot6d(torch.as_tensor(R0, dtype=torch.float32, device=DEV)).clone().requires_grad_(True)
    t = torch.as_tensor(t0, dtype=torch.float32, device=DEV).clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [r6], "lr": args.lr_r}, {"params": [t], "lr": args.lr_t}])
    chunks = [torch.arange(a, min(a + args.batch, n), device=DEV) for a in range(0, n, args.batch)]
    k = 0
    for sigma in levels:
        for _ in range(steps):
            if k % args.inside_every == 0:
                sc.refresh_inside(rot6d_to_mat(r6).detach(), t.detach(), sel)
            k += 1
            opt.zero_grad()
            for j in chunks:
                idx = sel[j]
                R = rot6d_to_mat(r6[j])
                img = sc.image(idx, R, t[j], sigma)
                if joint:
                    img = img * sc.unrel_w[idx]
                ((img + sc.body_terms(idx, R, t[j])).sum() / n).backward()
            if joint:
                sc.temporal(rot6d_to_mat(r6), t).backward()
            opt.step()
    with torch.no_grad():
        R = rot6d_to_mat(r6)
        loss, parts = torch.zeros(n, device=DEV), {}
        for j in chunks:
            idx = sel[j]
            img, p = sc.image(idx, R[j], t[j], 0.0, parts=True)
            loss[j] = img + sc.body_terms(idx, R[j], t[j])
            for key, v in p.items():
                parts.setdefault(key, torch.zeros(n, device=DEV))[j] = v
    Rn = R.cpu().numpy().astype(np.float64)
    Rn = np.stack([u @ vt for u, _, vt in (np.linalg.svd(r) for r in Rn)])
    return Rn, t.detach().cpu().numpy().astype(np.float64), loss.cpu().numpy(), {k: v.cpu().numpy() for k, v in parts.items()}


def candidates(ep: Episode, model: Model, args) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The init track, its tilt mirror, and 180-degree turns about the thin axis / flips about the long axis that
    change the shape (a turn mapping the mesh onto itself within --sym-tol is the same pose), each about the centroid."""
    from scipy.spatial import cKDTree

    pts = model.points.cpu().numpy().astype(np.float64)
    thin, long = object_axes(pts)
    com = pts.mean(0)
    size = float(np.ptp(pts, 0).max())
    tree = cKDTree(pts - com)
    Q, names = [], []
    for flip in (0, 180):
        for yaw in (0, 180):
            q = axis_angle(long, flip) @ axis_angle(thin, yaw)
            if all(tree.query((pts - com) @ q.T @ p, workers=-1)[0].mean() > args.sym_tol * size for p in Q):
                Q.append(q), names.append(f"yaw{yaw}flip{flip}")
    M = mirror_tilt(ep.R0, ep.t0, thin)
    out = []
    for m in (0, 1):
        for q, name in zip(Q, names):
            R = (M if m else np.eye(3)) @ ep.R0 @ q
            out.append((name + ("_mirror" if m else ""), R, ep.t0 + ep.s0 * (ep.R0 @ com - R @ com)))
    return out


def track(sc: Scene, R, t, loss, iou, args):
    """A forward then a backward sweep: frame i also refits (--track-blur) from frame i-1's (i+1's) pose and, when
    the last two frames moved (> 2 deg or 1 cm), from their constant-velocity extrapolation (rotation step and
    centre step repeated), since a fast swing outruns what one refit can turn. The best start's fit is taken unless
    the frame's own pose's loss is lower by more than --track-margin, so a good pose carries on where the init
    drifts, and near ties (the tilt mirror) keep the neighbour's state. Returns the updated arrays + counts."""
    R, t, loss, iou = R.copy(), t.copy(), loss.copy(), iou.copy()
    levels = [float(b) for b in args.track_blur.split(",")]
    com = sc.mpts.mean(0).cpu().numpy().astype(np.float64)
    taken, predicted = [], []
    for order in (range(1, len(R)), range(len(R) - 2, -1, -1)):
        n_taken = n_pred = 0
        for i in order:
            j, k = i - order.step, i - 2 * order.step
            starts = [(R[j], t[j])]
            if 0 <= k < len(R):
                dR = R[j] @ R[k].T
                cj, ck = sc.s * R[j] @ com + t[j], sc.s * R[k] @ com + t[k]
                if np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) > 2 or np.linalg.norm(cj - ck) > 0.01:
                    Rp = dR @ R[j]
                    starts.append((Rp, 2 * cj - ck - sc.s * Rp @ com))
            fits = [fit(sc, R0[None], t0[None], levels, args.track_steps, joint=False, frames=[i]) for R0, t0 in starts]
            b = int(np.argmin([f[2][0] for f in fits]))
            Ri, ti, li, pi = fits[b]
            if li[0] < loss[i] + args.track_margin:
                R[i], t[i], loss[i], iou[i] = Ri[0], ti[0], li[0], pi["iou"][0]
                n_taken += 1
                n_pred += b
        taken.append(n_taken)
        predicted.append(n_pred)
    return R, t, loss, iou, taken, predicted


def overlay_video(sc: Scene, R, t, unreliable, iou, out: Path, args):
    """Half-resolution frames: visible object outline green, object mask cyan, red border on unreliable frames."""
    ep, res = sc.ep, args.crop
    n = len(ep.frames)
    vis = []
    with torch.no_grad():
        Rt = torch.as_tensor(R, dtype=torch.float32, device=DEV)
        tt = torch.as_tensor(t, dtype=torch.float32, device=DEV)
        for a in range(0, n, args.batch):
            idx = torch.arange(a, min(a + args.batch, n), device=DEV)
            cov, zo = sc.rd(sc.points(Rt[idx], tt[idx], sc.V), sc.tri, sc.box[idx], soft=False)
            v = torch.where(sc.br[idx], zo < sc.zb[idx].float(), torch.ones_like(zo, dtype=torch.bool))
            vis.append(((cov > 0) & v).cpu().numpy().astype(np.uint8))
    vis = np.concatenate(vis)
    obj = np.load(ep.cache / ep.name / "masks.npz")["object"]
    cap = cv2.VideoCapture(str(video_path(args.data, ep.episode)))
    raw = out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 30, (768, 576))
    want = {int(f): i for i, f in enumerate(ep.frames)}
    scored = set(int(f) for f in ep.scored)
    f = -1
    while want:
        ok, bgr = cap.read()
        if not ok:
            break
        f += 1
        if f not in want:
            continue
        i = want.pop(f)
        im = cv2.resize(bgr, (768, 576), interpolation=cv2.INTER_AREA)
        M = crop_affine(sc.boxes[i], res, 2.0)
        full = cv2.warpAffine(vis[i], M, (768, 576), flags=cv2.INTER_NEAREST)
        for m, col in ((obj[f].astype(np.uint8), (255, 255, 0)), (full, (0, 255, 0))):
            cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, col, 1)
        if unreliable[i]:
            cv2.rectangle(im, (0, 0), (767, 575), (0, 0, 255), 6)
        cv2.putText(im, f"ep {ep.episode} f{f} iou {iou[i]:.2f}" + (" scored" if f in scored else ""), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(im)
    writer.release()
    cap.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "24", str(out)], check=True)
    raw.unlink()


def process(e: int, args, jargs, body, parts, budget, scored) -> dict:
    t1 = time.time()
    ep = Episode(e, jargs, body, parts, budget, scored[e])
    model = Model(ep.mesh, N_REFINE, jargs.seed)
    sc = Scene(ep, body, model, args)
    t_load = time.time() - t1
    levels = [float(b) for b in args.blur.split(",")]
    cands = candidates(ep, model, args)
    fits = [fit(sc, R, t, levels, args.steps, joint=False) for _, R, t in cands]
    unary = np.stack([f[2] for f in fits], 1)  # [n,K]
    with torch.no_grad():
        X = torch.stack([sc.points(torch.as_tensor(f[0], dtype=torch.float32, device=DEV),
                                   torch.as_tensor(f[1], dtype=torch.float32, device=DEV), sc.sub) for f in fits], 1)
        d = (X[1:, :, None] - X[:-1, None, :]).norm(dim=-1).mean(-1).transpose(1, 2)  # [n-1, from k, to l]
        pair = (args.w_switch * ((d / 0.05) ** 2).clamp(max=9)).cpu().numpy()
    path, _ = viterbi(unary, pair)
    rows = np.arange(len(path))
    R1 = np.stack([fits[k][0][i] for i, k in zip(rows, path)])
    t1_ = np.stack([fits[k][1][i] for i, k in zip(rows, path)])
    iou1 = np.array([fits[k][3]["iou"][i] for i, k in zip(rows, path)])
    t_track = time.time()
    R1, t1_, _, iou2, taken, predicted = track(sc, R1, t1_, unary[rows, path], iou1, args)
    t_track = time.time() - t_track
    torch.cuda.empty_cache()  # nvdiffrast allocates outside torch's cache, which the 1-frame refits leave fragmented
    unreliable = sc.area < 10
    sc.unrel_w = torch.as_tensor(np.where(unreliable, 0.0, 1.0), dtype=torch.float32, device=DEV)
    R, t, loss, prt = fit(sc, R1, t1_, [0.0], args.joint_steps, joint=True)

    before = contact_report(ep, ep.R_in, ep.t_in, ep.s_in, model, jargs)
    after = contact_report(ep, R, t, sc.s, model, jargs)
    sel = ep.in_scored
    stats = write_episode(ep, jargs, R, t, sc.s, 1.0, {"render": True}, loss.astype(np.float32), before, after, {})
    meta = args.run / "render_meta"
    meta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(meta / f"{ep.name}_series.npz", frames=ep.frames, iou=prt["iou"], iou_pick=iou1, iou_perframe=iou2,
                        actor_violation=prt["actor_violation"], sil=prt["sil"], edge=prt["edge"],
                        trust=sc.trust.cpu().numpy(), unreliable=unreliable, candidate=path, in_scored=sel)
    (args.run / "videos").mkdir(parents=True, exist_ok=True)
    overlay_video(sc, R, t, unreliable, prt["iou"], args.run / "videos" / f"{ep.name}_fit.mp4", args)
    r2 = lambda x: round(float(x), 4)  # noqa: E731
    out = {"episode": e, "candidates": [c[0] for c in cands],
           "picked": {cands[k][0]: int((path == k).sum()) for k in range(len(cands)) if (path == k).any()},
           "tracked_fwd_bwd": taken, "predicted_fwd_bwd": predicted, "unreliable_frames": int(unreliable.sum()), "frames": len(ep.frames),
           "iou_scored": r2(prt["iou"][sel].mean()), "iou_all": r2(prt["iou"].mean()),
           "actor_violation_scored": r2(prt["actor_violation"][sel].mean()),
           "trust_scored": r2(sc.trust.cpu().numpy()[sel].mean()), "loss_scored": r2(loss[sel].mean()),
           "before": before, "after": after, "turn_deg_mean": stats["turn_deg_mean"],
           "shift_cm_mean": stats["shift_from_scaled_init_cm_mean"],
           "timing_s": {"load": round(t_load, 1), "track": round(t_track, 1), "total": round(time.time() - t1, 1)},
           "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    (meta / f"{ep.name}.json").write_text(json.dumps(out, indent=1))
    print(f"ep {e}: {json.dumps({k: v for k, v in out.items() if k not in ('before', 'after')})}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human", type=Path, required=True)
    p.add_argument("--init", type=Path, required=True, help="episodes dir of the run to start from (mesh, scale, track)")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--cache", type=Path, default=Path("/content/repo/cache"))
    p.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    p.add_argument("--sample", type=Path, default=Path("/content/repo/data/v2d_submission_kit/data/track_1_sample_submission.parquet"))
    p.add_argument("--crop", type=int, default=512, help="render resolution of the square crop")
    p.add_argument("--tau", type=float, default=0.03, help="m, soft front/behind transition")
    p.add_argument("--band", type=float, default=3.0, help="crop px around mask boundaries where drawn pixels count x0.2")
    p.add_argument("--trust", type=float, default=0.6, help="drawn body vs actor mask IoU below which actor-only counts x0.2")
    p.add_argument("--w-edge", type=float, default=0.3)
    p.add_argument("--w-contact", type=float, default=1.0)
    p.add_argument("--sigma-contact", type=float, default=0.08, help="m")
    p.add_argument("--w-pen", type=float, default=5.0, help="per cm (pushout's weight with free R, t)")
    p.add_argument("--margin", type=float, default=0.003)
    p.add_argument("--w-inside", type=float, default=1.0, help="per cm of object inside the body")
    p.add_argument("--w-floor", type=float, default=1.0, help="per cm below the floor")
    p.add_argument("--floor-tol", type=float, default=0.02)
    p.add_argument("--w-acc", type=float, default=1.0)
    p.add_argument("--acc-ref", type=float, default=0.02, help="m/frame^2")
    p.add_argument("--w-vel", type=float, default=1.0)
    p.add_argument("--vmax", type=float, default=0.15, help="m/frame of mean surface point motion")
    p.add_argument("--w-switch", type=float, default=1.0)
    p.add_argument("--sym-tol", type=float, default=0.02)
    p.add_argument("--blur", default="8,3,0", help="mask blur per coarse-to-fine level, crop px")
    p.add_argument("--track-blur", default="8,3,0", help="levels of the per-frame refit from a neighbour's pose")
    p.add_argument("--track-steps", type=int, default=15, help="per level")
    p.add_argument("--track-margin", type=float, default=0.02,
                   help="the neighbour's pose is taken unless the frame's own loss is lower by more than this")
    p.add_argument("--steps", type=int, default=40, help="per level, per candidate")
    p.add_argument("--joint-steps", type=int, default=150)
    p.add_argument("--lr-r", type=float, default=0.01)
    p.add_argument("--lr-t", type=float, default=0.003)
    p.add_argument("--points", type=int, default=128, help="surface samples for temporal / inside terms")
    p.add_argument("--body-stride", type=int, default=4, help="every n-th body vertex for the inside distance")
    p.add_argument("--inside-every", type=int, default=20, help="steps between inside-the-body refreshes")
    p.add_argument("--batch", type=int, default=48, help="frames per render chunk")
    args = p.parse_args()
    jargs = build_parser().parse_args(["--human", str(args.human), "--init", str(args.init), "--run", str(args.run),
                                       "--episodes", args.episodes, "--cache", str(args.cache), "--data", str(args.data),
                                       "--sample", str(args.sample)])
    jargs.sources = lambda e: (args.init, args.cache)
    jargs.contact_frames = "all"  # every fitted frame is optimized, so every frame needs its contact (ep 2 f374: held, none)
    body = MHRBody(device=DEV)
    parts = BodyParts(body, jargs.seed)
    budget = mesh_budget(args.sample)
    scored = scored_frames(args.sample)
    for e in parse_episodes(args.episodes):
        if (args.run / "render_meta" / f"episode_{e:06d}.json").exists():
            continue
        try:
            first = np.load(args.human / f"episode_{e:06d}_camera.npz")["K"].astype(np.float64)
            check_orientation(first)
            process(e, args, jargs, body, parts, budget, scored)
        except Exception:
            print(f"ep {e} FAILED", flush=True)
            traceback.print_exc()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

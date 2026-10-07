#!/usr/bin/env python3
"""Steps 7b + 7c: place each object relative to the body (contact, floor, one depth frame), runs on the VM (GPU).

    python joint_fit.py --human <repo>/results/human_v1/episodes --init <repo>/results/object_v3_w5d03/episodes \
        --run <repo>/results/joint_v1 [--episodes 0-29] [--cache <repo>/cache] \
        [--alt 0-2:<repo>/results/object_d4_push/episodes:<repo>/cache/d4rt]

The human is taken as is from --human (7c variants differ only in that run, e.g. human_d4dep, whose depth
follows the D4RT actor depth); the object track, mesh and scale of --init are the starting point (--alt: other
init run and cache for some episodes, e.g. the hoop from the D4RT pipeline). Per object (its episodes together):

0. One depth frame (7c). Per fitted frame, k_t = median(cache depth / rendered body depth) over the actor mask,
   log k_t Whittaker-smoothed (--lam-k); the object's depth data uses 1 / k_t, so object depth relative to the
   actor follows the depth map frame by frame (Step 5 used one k per episode: the body's own depth wander
   then shows up as hand-object gaps).
1. Contact candidates. On scored frames (the contact span), body points whose projection lies within
   --contact-px (half-resolution px, or 10% of the object's size) of the object mask: hand and sole points
   first; other body points only when no hand or foot is near (sitting, hugging).
1b. Rotation hypotheses (rotation_hypotheses): poses the image cannot tell apart, e.g. a ring's two tilts (same
   ellipse) or a pan's handle direction (a few thin pixels), and that Step 5's tracking carried from its keyframe.
   Per sampled frame, discrete states (turns about the thin axis, a flip, the tilt mirror) scored by data +
   support (static, unheld frames: the centre of mass must stand over the lowest points, as an object at rest
   does) [+ hand contact, --w-hand-h, off by default], Viterbi over frames with a rotation-jump cost.
   Turns that map the mesh onto itself are one state (--sym-tol); an episode leaves the init only if the path
   saves --hyp-min-gain per sampled frame, its re-fit fits the image within --hyp-fit-tol of the init and its
   support is not worse (--hyp-rest-tol). Untouched episodes keep their track exactly (--no-rot-hyps: off).
2. Depth along the ray (7b, per episode). Scaling the object and its track by alpha (s -> a s, t -> a t) leaves
   the image unchanged, so only depth, contact and the floor see it. alpha is chosen on a grid by
     mean data loss (depth terms, --w-dep-a, robust scale --sigma-a) + --w-contact-a x mean over contact frames of GM(nearest
     candidate's distance outside the mesh, --sigma-contact-a) + --w-floor-a x mean GM(object below the floor)
   plus --lam-share x (log scale - the object's shared log scale)^2, the shared scale minimizing the sum over
   its episodes. Fixes thin objects without depth (hoop) and wrong shared scales (pan).
3. Joint refinement per episode (scale fixed; --refine ray: per frame a depth along the camera ray only, R and
   the image position kept; full: R, t per frame): Step 5 data terms (per-frame depth scale) +
   second differences + --w-contact x GM(nearest candidate's distance, --sigma-contact) + --w-pen x 7a hand
   penetration hinge + --w-floor x object points below the floor (cm). Contact and floor act on depth only
   (--contact-dof ray): the silhouette fixes the image, the body fixes the distance.
   Floor: the Step 3 plane normal through the body's lowest points (--floor-pct percentile over visible frames),
   i.e. in the body's frame and scale, not MoGe's.

Writes <run>/episodes/ (npz: human from --human, object from here; mesh; camera npz with depth_scale = median 1/k_t
and depth_scale_frames = 1/k_t per video frame, which pushout.py uses),
<run>/joint_meta/episode_XXXXXX.json (+ _series.npz) and object_<name>.json (alpha curves).
Objects whose episodes all have a json are skipped (resume); --episodes picks objects (all their episodes).
A process fitting an object holds joint_meta/object_<name>.lock (its pid); a lock whose process is gone (killed
job) is taken over.
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
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_post import moge_ratio, whittaker  # noqa: E402
from human_vm import episode_range, parse_episodes  # noqa: E402
from object_fit import (MASK_STEP, N_REFINE, Model, Observations, data_loss, gm, mat_to_rot6d,  # noqa: E402
                        object_names, optimize, rot6d_to_mat)
from proxy_eval import (DEFAULT_CACHE, DEFAULT_SAMPLE, MESH_SUFFIXES, MHRBody, budget_mesh, mesh_budget, ms,  # noqa: E402
                        scored_frames)
from pushout import SDFGrid, exact_penetration, penetration_cm  # noqa: E402

DEV = "cuda"
REPO = Path(__file__).resolve().parents[1]
FAR = 1e3  # m; distance given to non-candidate points


# --------------------------------------------------------------------------------------------- body
class BodyParts:
    """Contact candidates on MHR: the 512 hand points (proxy/scorer style), up to 256 sole points, the 1024
    other body points. kind 0 hand, 1 foot, 2 body."""

    def __init__(self, body: MHRBody, seed: int = 0):
        rest = body(np.zeros((1, 204)), np.zeros(45))[0][0]  # Track 1 frame: y down, so soles have the largest y
        rng = np.random.default_rng(seed)
        soles = np.flatnonzero(rest[:, 1] > rest[:, 1].max() - 0.06)
        soles = np.sort(rng.choice(soles, min(256, len(soles)), replace=False))
        other = np.setdiff1d(body.roles["body"], soles)
        self.index = np.concatenate([body.hands, soles, other])
        self.kind = np.concatenate([np.zeros(len(body.hands)), np.ones(len(soles)), np.full(len(other), 2)]).astype(np.int8)


def sdf_far(sdf: SDFGrid, p: torch.Tensor) -> torch.Tensor:
    """Signed distance (mesh units) that keeps growing outside the grid: the grid value at the nearest grid point
    plus the distance to it, so far-away contact points still pull."""
    lo, hi = sdf.lo, sdf.lo + sdf.extent
    q = torch.maximum(torch.minimum(p, hi), lo)
    return sdf(q) + (p - q).norm(dim=-1)


def outside_distance(points, R, t, s, sdf):
    """[n,P,3] world points -> [n,P] distance outside the object mesh (0 inside), metres."""
    local = torch.einsum("tpi,tij->tpj", points - t[:, None], R) / s[:, None, None]
    return torch.relu(s[:, None] * sdf_far(sdf, local))


def nearest(d: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """[n,P] distances, [n,P] candidate mask -> [n] distance of the nearest candidate (FAR when there is none).
    A hard minimum: the gradient goes to the nearest point only, as in ICP (a soft-min over many points at
    distance 0 goes negative and would push them out)."""
    return torch.where(cand, d, torch.full_like(d, FAR)).min(1).values


def depth_scales(cache: Path, episode: int, frames, verts, K, lam: float, fallback: float):
    """Per fitted frame 1 / k_t (k_t = cache depth / body depth over the actor), smoothed in log; raw ratios; vis."""
    folder = cache / f"episode_{episode:06d}"
    human = np.load(folder / "masks.npz")["human"]
    area = human[frames].sum((1, 2)).astype(float)
    vis = np.clip(area / max(np.percentile(area, 90), 1), 0, 1)
    ratio = np.full(len(frames), np.nan)
    depth = np.load(folder / "depth.npz")["depth"]
    for i, f in enumerate(frames):
        if vis[i] > 0.3:
            ratio[i] = moge_ratio(verts[i], K, depth[f].astype(np.float32), human[f])
    del depth, human
    ok = np.isfinite(ratio) & (ratio > 0)
    if ok.sum() < 10:
        return np.full(len(frames), fallback), ratio, vis
    y = np.where(ok, np.log(np.where(ok, ratio, 1)), 0.0)
    w = ok * np.maximum(vis, 0.05)
    for _ in range(2):  # one robust pass: frames far from the smooth curve (occlusion, bad masks) drop out
        logk = whittaker(y[:, None], np.maximum(w, 1e-6), lam)[:, 0]
        res = np.abs(y - logk)
        mad = np.median(res[ok]) * 1.4826 + 1e-4
        w = ok * np.maximum(vis, 0.05) * (res < 3 * mad)
    return np.exp(-logk), ratio, vis


# ----------------------------------------------------------------------------------------- episodes
class Episode:
    """Everything one episode's fit reads, on the GPU where the losses need it."""

    def __init__(self, episode: int, args, body: MHRBody, parts: BodyParts, budget, scored: np.ndarray):
        self.episode, self.scored = episode, scored
        self.name = name = f"episode_{episode:06d}"
        self.init_dir, self.cache = args.sources(episode)
        self.human = dict(np.load(args.human / f"{name}.npz"))
        init = dict(np.load(self.init_dir / f"{name}.npz"))
        cam = dict(np.load(args.human / f"{name}_camera.npz"))
        self.cam = cam
        K = cam["K"].astype(np.float64)
        self.total = total = len(self.human["pose"])
        self.frames = frames = episode_range(scored, total)
        n = len(frames)
        self.mesh_path = next(p for s in MESH_SUFFIXES if (p := self.init_dir / f"{name}_object{s}").exists())
        self.vertices, self.faces = budget_mesh(self.mesh_path, *budget)
        self.sdf = SDFGrid(self.vertices, self.faces, DEV, res=args.res)
        self.mesh = trimesh.load(self.mesh_path, force="mesh")
        self.R0 = ms._rotations(np.asarray(init["object_rotation"], np.float64))[frames]
        self.t0 = np.asarray(init["object_translation"], np.float64)[frames]
        self.R_in, self.t_in = self.R0, self.t0  # the init track; stage 1b may replace R0/t0
        self.s0 = float(np.asarray(init["object_scale"]).reshape(-1)[0])

        pose = np.asarray(self.human["pose"], np.float64)[frames]
        params = np.concatenate([pose, np.broadcast_to(self.human["scales"], (n, 68))], 1)
        verts, _ = body(params, self.human["shape"])  # camera frame (world = camera)

        # 0. per-frame depth scale against this body
        fallback = float(cam.get("depth_scale", 1.0))
        if args.depth_scale == "perframe":
            self.ds, self.ratio, vis = depth_scales(self.cache, episode, frames, verts, K, args.lam_k, fallback)
        else:
            self.ds, self.ratio, vis = np.full(n, fallback), np.full(n, np.nan), np.ones(n)
        self.obs = Observations(self.cache, episode, frames, K, self.ds, seed=args.seed)

        # 1. contact candidates on scored frames
        obj_masks = np.load(self.cache / name / "masks.npz")["object"]
        h2, w2 = obj_masks.shape[1:]
        pts = verts[:, parts.index]
        cand = np.zeros(pts.shape[:2], bool)
        hand_near = np.zeros(pts.shape[:2], bool)  # every fitted frame: hand points at the object mask in 2D
        scored_set = set(int(f) for f in scored)
        for i, f in enumerate(frames):
            obj = obj_masks[f]
            area = obj.sum()
            if area < 10:
                continue
            dt = cv2.distanceTransform((~obj).astype(np.uint8), cv2.DIST_L2, 5)
            X = pts[i]
            z = np.maximum(X[:, 2], 1e-3)
            u = (K[0, 0] * X[:, 0] / z + K[0, 2] - 0.5) / MASK_STEP
            v = (K[1, 1] * X[:, 1] / z + K[1, 2] - 0.5) / MASK_STEP
            ok = (X[:, 2] > 0.1) & (u >= 0) & (u <= w2 - 1) & (v >= 0) & (v <= h2 - 1)
            d = np.full(len(X), np.inf)
            d[ok] = dt[np.rint(v[ok]).astype(int), np.rint(u[ok]).astype(int)]
            near = d <= max(args.contact_px, 0.1 * np.sqrt(area))
            hand_near[i] = near & (parts.kind == 0)
            if int(f) not in scored_set:
                continue
            limbs = near & (parts.kind < 2)
            cand[i] = limbs if limbs.any() else near & (parts.kind == 2)
        del obj_masks
        self.cand_kind = np.array([np.bincount(parts.kind[c], minlength=3) > 0 for c in cand])  # [n,3]

        # floor in the body's frame: Step 3 normal, offset from the body's lowest points
        meta = json.loads((self.cache / name / "meta.json").read_text())
        normal = np.asarray(meta["floor_normal"], np.float64)
        normal /= np.linalg.norm(normal)
        if normal[1] > 0:  # up is -y in the camera frame
            normal = -normal
        low = (verts @ normal).min(1)
        good = vis > 0.7
        self.normal = normal
        self.floor = float(np.percentile(low[good] if good.sum() >= 10 else low, args.floor_pct))

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEV)  # noqa: E731
        self.pts, self.cand = t(pts), torch.as_tensor(cand, device=DEV)
        self.hand_near = torch.as_tensor(hand_near, device=DEV)
        self.has_contact = self.cand.any(1).float()
        self.hands_all = t(verts[:, body.hand_all])
        self.hands_512 = verts[:, body.hands]  # numpy, for the exact PEN check
        self.normal_t = t(normal)
        self.in_scored = np.isin(frames, scored)
        del verts


# ------------------------------------------------------------------------------------------- losses
def floor_depth(X: torch.Tensor, ep: Episode, tol: float) -> torch.Tensor:
    """[n,N,3] object points -> [n,N] depth below the floor beyond tol, metres."""
    return torch.relu(ep.floor - tol - X @ ep.normal_t)


# ------------------------------------------------------------------------------ 1b. rotation hypotheses
def axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    return Rotation.from_rotvec(np.radians(deg) * axis / np.linalg.norm(axis)).as_matrix()


def object_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thinnest and longest principal axes of the model points (object frame)."""
    c = points - points.mean(0)
    _, vecs = np.linalg.eigh(c.T @ c)
    return vecs[:, 0], vecs[:, 2]


def mirror_tilt(R: np.ndarray, t: np.ndarray, thin: np.ndarray) -> np.ndarray:
    """[n,3,3] camera-frame rotations taking the object's thin axis to its reflection across the plane normal to
    the viewing ray: the two tilts of a circle that project to the same ellipse (identity where they coincide)."""
    n = R @ thin
    ray = t / np.linalg.norm(t, axis=1, keepdims=True)
    m = n - 2 * (n * ray).sum(1, keepdims=True) * ray
    axis = np.cross(n, m)
    sin = np.linalg.norm(axis, axis=1)
    out = np.repeat(np.eye(3)[None], len(R), 0)
    ok = sin > 1e-4
    ang = np.arctan2(sin[ok], (n[ok] * m[ok]).sum(1))
    out[ok] = Rotation.from_rotvec(axis[ok] / sin[ok, None] * ang[:, None]).as_matrix()
    return out


def static_frames(R: np.ndarray, t: np.ndarray, args) -> np.ndarray:
    """Frames where the init track barely moves (median over +-5 frames of the per-frame motion)."""
    v = np.linalg.norm(np.gradient(t, axis=0), axis=1)
    dR = np.einsum("tij,tij->t", R[1:], R[:-1])
    w = np.arccos(np.clip((dR - 1) / 2, -1, 1))
    w = np.r_[w[:1], w]
    med = lambda x: np.array([np.median(x[max(0, i - 5):i + 6]) for i in range(len(x))])  # noqa: E731
    return (med(v) < args.static_v) & (med(w) < args.static_w)


def support_cost(X: np.ndarray, com: np.ndarray, up: np.ndarray, size: float, args) -> float:
    """An object at rest must stand on its lowest points: distance of the centre of mass (projected along `up`)
    outside the convex hull of the points within --support-tol x size of the lowest, plus --w-low x the centre's
    height above them; both / size. 0 for a ring lying flat, ~0.4 for a ring tilted 40 deg on one point of its rim;
    a stool on its legs pays only the height part."""
    h = X @ up
    sup = X[h < h.min() + args.support_tol * size]
    e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    P = np.stack([sup @ e1, sup @ e2], 1)
    c = np.array([com @ e1, com @ e2])
    try:
        eq = ConvexHull(P).equations
        out = max(float((eq[:, :2] @ c + eq[:, 2]).max()), 0.0)
    except Exception:  # fewer than 3 or collinear support points
        out = float(np.linalg.norm(P - c, axis=1).min())
    return (out + args.w_low * float(com @ up - h.min())) / size


def viterbi(unary: np.ndarray, pair: np.ndarray) -> tuple[np.ndarray, float]:
    """unary [S,K], pair [S-1,K,K] (from state at s-1 to state at s) -> min-cost path [S] and its cost."""
    cost = unary[0].copy()
    back = np.zeros(unary.shape, int)
    for s in range(1, len(unary)):
        c = cost[:, None] + pair[s - 1]
        back[s] = c.argmin(0)
        cost = c.min(0) + unary[s]
    path = np.zeros(len(unary), int)
    path[-1] = int(cost.argmin())
    for s in range(len(unary) - 1, 0, -1):
        path[s - 1] = back[s, path[s]]
    return path, float(cost.min())


def refit_track(ep: Episode, model: Model, R_init: np.ndarray, t_init: np.ndarray, args):
    """Data terms + second differences + a pull to the given track (R, t free, scale fixed): settles a
    hypothesis track on the image."""
    n = len(ep.frames)
    idx = torch.arange(n, device=DEV)
    s = torch.full((n,), ep.s0, device=DEV)
    Ri = torch.as_tensor(R_init, dtype=torch.float32, device=DEV)
    ti = torch.as_tensor(t_init, dtype=torch.float32, device=DEV)
    r6 = mat_to_rot6d(Ri).clone().requires_grad_(True)
    t = ti.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [r6], "lr": 0.01}, {"params": [t], "lr": 0.002}])
    for _ in range(args.hyp_refine_steps):
        opt.zero_grad()
        R = rot6d_to_mat(r6)
        acc_t = ((t[2:] - 2 * t[1:-1] + t[:-2]).norm(dim=-1) / args.acc_t) ** 2
        acc_r = ((R[2:] - 2 * R[1:-1] + R[:-2]).flatten(1).norm(dim=-1) / args.acc_r) ** 2
        prior = ((R - Ri) ** 2).sum((1, 2)).mean() + (((t - ti) / 0.05) ** 2).sum(1).mean()
        loss = (data_loss(ep.obs, model, idx, R, t, s, args).mean() + args.w_smooth_h * (acc_t.mean() + acc_r.mean())
                + args.w_prior * prior)
        loss.backward()
        opt.step()
    with torch.no_grad():
        R = rot6d_to_mat(r6).cpu().numpy().astype(np.float64)
    R = np.stack([u @ vt for u, _, vt in (np.linalg.svd(r) for r in R)])
    return R, t.detach().cpu().numpy().astype(np.float64)


def rotation_hypotheses(ep: Episode, model: Model, args) -> dict:
    """Stage 1b: rotations the image cannot tell apart, settled by contact and support (see the docstring).

    States: the init track x {0, 90, 180, 270 deg about the object's thinnest axis} x {as is, flipped about its
    longest axis} x {as is, tilt mirrored across the viewing ray}; state 0 is the init. Each is turned about the
    model's centroid and re-fitted on every --hyp-stride-th frame (--hyp-steps, R and t). Per sampled frame:
      data   the re-fitted data loss
      hand   frames with hand points at the object mask in 2D: the nearest of them outside the mesh,
             min(d / 10 cm, 3) x --w-hand-h (linear: far hands still count), at one alpha per episode
             (the best of a grid, as in stage 2: the image cannot fix depth)
      rest   static frames without a hand at the mask: support_cost x --w-rest
    plus --w-jump-h x (rotation change / 0.5 rad)^2 between samples; Viterbi picks the path. If it leaves
    state 0 anywhere, the new track (states spread to the frames between samples) is re-fitted with refit_track
    and replaces ep.R0, ep.t0."""
    pts_obj = model.points.cpu().numpy().astype(np.float64)
    thin, long = object_axes(pts_obj)
    com = pts_obj.mean(0)
    Q = [axis_angle(long, flip) @ axis_angle(thin, yaw) for flip in (0, 180) for yaw in (0, 90, 180, 270)]
    # turns that map the mesh onto itself (a ring about its axis, a roller end over end) are the same state
    size_obj = float(np.ptp(pts_obj, 0).max())
    tree = cKDTree(pts_obj - com)
    distinct = []
    for q in range(len(Q)):
        moved = (pts_obj - com) @ Q[q].T
        if all(tree.query(moved @ Q[p], workers=-1)[0].mean() > args.sym_tol * size_obj for p in distinct):
            distinct.append(q)
    states = [(q, m) for m in (0, 1) for q in distinct]
    K = len(states)
    n = len(ep.frames)
    sample = np.arange(0, n, args.hyp_stride)
    sample = sample[ep.obs.has.cpu().numpy()[sample] > 0]
    if len(sample) < 3:
        return {"changed": False, "reason": "fewer than 3 frames with a mask"}
    S = len(sample)
    M = mirror_tilt(ep.R0, ep.t0, thin)
    R_c = np.empty((S, K, 3, 3))
    t_c = np.empty((S, K, 3))
    for j, (q, m) in enumerate(states):
        Rn = (M[sample] if m else np.eye(3)) @ ep.R0[sample] @ Q[q]
        R_c[:, j] = Rn
        t_c[:, j] = ep.t0[sample] + ep.s0 * (ep.R0[sample] @ com - Rn @ com)  # centroid stays put

    # re-fit every (sample, state) a little
    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    flat_R, flat_t = f32(R_c.reshape(-1, 3, 3)), f32(t_c.reshape(-1, 3))
    flat_i = torch.as_tensor(np.repeat(sample, K), device=DEV)
    logs = torch.full((len(flat_i),), float(np.log(ep.s0)), device=DEV)
    Rs, ts, ds = [], [], []
    for a in range(0, len(flat_i), args.hyp_batch):
        sl = slice(a, a + args.hyp_batch)
        R, t, _, loss = optimize(ep.obs, model, flat_i[sl], flat_R[sl], flat_t[sl], logs[sl], args, args.hyp_steps,
                                 fit_scale=False, lr=(0.02, 0.005, 0.0))
        Rs.append(R), ts.append(t), ds.append(loss)
    R = torch.cat(Rs).reshape(S, K, 3, 3)
    t = torch.cat(ts).reshape(S, K, 3)
    data = torch.cat(ds).reshape(S, K).cpu().numpy()

    # hand contact, per alpha
    hand = ep.hand_near[torch.as_tensor(sample, device=DEV)]
    held = hand.any(1)
    pts = ep.pts[torch.as_tensor(sample, device=DEV)]
    alphas = np.exp(np.linspace(*np.log(args.hyp_alpha), args.hyp_alpha_steps)) if args.w_hand_h > 0 else np.ones(1)
    contact = np.zeros((len(alphas), S, K))
    with torch.no_grad():
        for ai, a in enumerate(alphas):
            for j in range(K):
                d = nearest(outside_distance(pts, R[:, j], a * t[:, j], torch.full((S,), a * ep.s0, device=DEV),
                                             ep.sdf), hand)
                contact[ai, :, j] = ((d / 0.1).clamp(max=3) * held).cpu().numpy()

    # support on static, unheld frames
    rest = static_frames(ep.R0, ep.t0, args)[sample] & ~held.cpu().numpy()
    Rn, tn = R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64)
    size = ep.s0 * float(np.ptp(pts_obj, 0).max())
    stab = np.zeros((S, K))
    for i in np.flatnonzero(rest):
        for j in range(K):
            X = ep.s0 * pts_obj @ Rn[i, j].T + tn[i, j]
            stab[i, j] = support_cost(X, ep.s0 * Rn[i, j] @ com + tn[i, j], ep.normal, size, args)

    # rotation change between consecutive samples, every pair of states: pair[s-1, k, l] from k to l
    cos = np.einsum("skij,slij->skl", Rn[:-1], Rn[1:])  # trace(R_a^T R_b)
    ang = np.arccos(np.clip((cos - 1) / 2, -1, 1))
    pair = args.w_jump_h * (ang / 0.5) ** 2

    best = None
    for ai in range(len(alphas)):
        unary = data + args.w_hand_h * contact[ai] + args.w_rest * stab
        path, cost = viterbi(unary, pair)
        if best is None or cost < best[1]:
            best = (ai, cost, path, unary)
    ai, cost, path, unary = best
    keep_cost = float(unary[:, 0].sum() + pair[np.arange(S - 1), 0, 0].sum())
    rows = np.arange(S)
    pick = lambda arr: float(arr[rows, path].sum())  # noqa: E731
    keep = lambda arr: float(arr[:, 0].sum())  # noqa: E731
    info = {"samples": S, "rest_samples": int(rest.sum()), "held_samples": int(held.sum()),
            "alpha": round(float(alphas[ai]), 3), "cost_keep": round(keep_cost, 3), "cost_best": round(cost, 3),
            "states": {f"q{q}{'m' if m else ''}": int((path == j).sum()) for j, (q, m) in enumerate(states)
                       if (path == j).any()},
            "data": [round(keep(data), 3), round(pick(data), 3)],
            "hand": [round(keep(contact[ai]) * args.w_hand_h, 3), round(pick(contact[ai]) * args.w_hand_h, 3)],
            "rest": [round(keep(stab) * args.w_rest, 3), round(pick(stab) * args.w_rest, 3)]}
    turn0 = np.degrees(np.arccos(np.clip((np.einsum("skij,sij->sk", Rn, Rn[:, 0]) - 1) / 2, -1, 1)))
    info["per_state"] = {f"q{q}{'m' if m else ''}": {  # whole-episode sums if the state were kept throughout
        "data": round(float(data[:, j].sum()), 2), "hand": round(float(contact[ai][:, j].sum() * args.w_hand_h), 2),
        "rest": round(float(stab[:, j].sum() * args.w_rest), 2), "deg_from_q0": round(float(turn0[:, j].mean()), 1)}
        for j, (q, m) in enumerate(states)}
    if not path.any() or keep_cost - cost < args.hyp_min_gain * S:
        return {"changed": False, **info}

    # spread the states to all fitted frames, re-fit the track
    near = np.clip(np.searchsorted(sample, np.arange(n)), 0, S - 1)
    left = np.clip(near - 1, 0, S - 1)
    near = np.where(np.abs(sample[left] - np.arange(n)) < np.abs(sample[near] - np.arange(n)), left, near)
    state = path[near]
    R_new = np.empty_like(ep.R0)
    t_new = np.empty_like(ep.t0)
    for j, (q, m) in enumerate(states):
        f = state == j
        if f.any():
            R_new[f] = (M[f] if m else np.eye(3)) @ ep.R0[f] @ Q[q]
            t_new[f] = ep.t0[f] + ep.s0 * (ep.R0[f] @ com - R_new[f] @ com)
    shift = tn[rows, path] - t_c[rows, path]  # what the short re-fit moved, interpolated in between
    t_new += np.stack([np.interp(np.arange(n), sample, shift[:, k]) for k in range(3)], 1)
    R_fit, t_fit = refit_track(ep, model, R_new, t_new, args)
    turn = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", R_fit, ep.R0) - 1) / 2, -1, 1)))
    # guard: the new track must fit the image about as well as the init (support breaks ties, it does not
    # buy a worse silhouette)
    with torch.no_grad():
        idx = torch.arange(n, device=DEV)
        s = torch.full((n,), ep.s0, device=DEV)
        fit_old = float(data_loss(ep.obs, model, idx, f32(ep.R0), f32(ep.t0), s, args).mean())
        fit_new = float(data_loss(ep.obs, model, idx, f32(R_fit), f32(t_fit), s, args).mean())
    info.update(frames_changed=int((state != 0).sum()), turn_deg_mean=round(float(turn.mean()), 1),
                turn_deg_p90=round(float(np.percentile(turn, 90)), 1), fit_old=round(fit_old, 4),
                fit_new=round(fit_new, 4))
    if fit_new > fit_old + args.hyp_fit_tol:
        return {"changed": False, "rejected": "image fit worse", **info}
    if pick(stab) * args.w_rest > keep(stab) * args.w_rest + args.hyp_rest_tol:  # data alone moved it, and the
        return {"changed": False, "rejected": "support worse", **info}  # scene got less plausible (hoop ep 2)
    ep.R0, ep.t0 = R_fit, t_fit
    return {"changed": True, **info}


def stage_a(ep: Episode, model: Model, alphas: np.ndarray, args, stride: int = 2) -> dict[str, np.ndarray]:
    """Per alpha (the track and scale times alpha), mean data / contact / floor costs on every `stride`-th frame."""
    idx_np = np.arange(0, len(ep.frames), stride)
    idx = torch.as_tensor(idx_np, device=DEV)
    R = torch.as_tensor(ep.R0[idx_np], dtype=torch.float32, device=DEV)
    t0 = torch.as_tensor(ep.t0[idx_np], dtype=torch.float32, device=DEV)
    pts, cand, has_c = ep.pts[idx], ep.cand[idx], ep.has_contact[idx]
    a_args = argparse.Namespace(**{**vars(args), "w_dep": args.w_dep_a, "sigma": args.sigma_a})
    out = {"data": [], "contact": [], "floor": []}
    with torch.no_grad():
        for a in alphas:
            s = torch.full((len(idx),), a * ep.s0, device=DEV)
            t = a * t0
            out["data"].append(float(data_loss(ep.obs, model, idx, R, t, s, a_args).mean()))
            d = nearest(outside_distance(pts, R, t, s, ep.sdf), cand)
            c = (gm(d, args.sigma_contact_a) * has_c).sum() / has_c.sum().clamp_min(1)
            out["contact"].append(float(c))
            X = s[:, None, None] * model.points[None] @ R.transpose(1, 2) + t[:, None]
            below = floor_depth(X, ep, args.floor_tol).max(1).values
            out["floor"].append(float(gm(below, 0.05).mean()))
    return {k: np.array(v) for k, v in out.items()}


def choose_alphas(eps: list[Episode], curves: dict[int, dict], grid: np.ndarray, args) -> tuple[dict, dict]:
    """Per episode alpha on a common log-scale grid: own cost + --lam-share x (u - shared u)^2."""
    log_s = np.log(np.median([ep.s0 for ep in eps])) + np.log(grid)  # absolute log scale u
    cost = {}
    for ep in eps:
        c = curves[ep.episode]
        e = c["data"] + args.w_contact_a * c["contact"] + args.w_floor_a * c["floor"]
        cost[ep.episode] = e
    shared = int(np.argmin(sum(cost.values())))
    alphas, info = {}, {"log_scale_grid": log_s.round(4).tolist(), "shared_scale_m": float(np.exp(log_s[shared]))}
    for ep in eps:
        total = cost[ep.episode] + args.lam_share * (log_s - log_s[shared]) ** 2
        j = int(np.argmin(total))
        alphas[ep.episode] = float(np.exp(log_s[j]) / ep.s0)
        info[str(ep.episode)] = {"alpha": alphas[ep.episode], "scale_m": float(np.exp(log_s[j])),
                                 "own_best_scale_m": float(np.exp(log_s[int(np.argmin(cost[ep.episode]))])),
                                 "at_grid_edge": j in (0, len(grid) - 1),
                                 **{k: np.round(v, 5).tolist() for k, v in curves[ep.episode].items()}}
    return alphas, info


def refine(ep: Episode, model: Model, alpha: float, args):
    """All fitted frames jointly at scale alpha s0: data + smoothness + contact + penetration + floor.

    --refine ray: R and the image position stay the init's; per frame only a log-depth offset d_t along the
    camera ray (t = alpha t0 exp(d_t)). --refine full: R and t free (contact and floor still act along the ray
    with --contact-dof ray)."""
    n = len(ep.frames)
    idx = torch.arange(n, device=DEV)
    s_val = alpha * ep.s0
    s = torch.full((n,), s_val, device=DEV)
    R0 = torch.as_tensor(ep.R0, dtype=torch.float32, device=DEV)
    t0 = alpha * torch.as_tensor(ep.t0, dtype=torch.float32, device=DEV)
    if args.refine == "ray":
        d = torch.zeros(n, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([{"params": [d], "lr": args.lr_d}])
        pose = lambda: (R0, t0 * d.exp()[:, None])  # noqa: E731
    else:
        r6 = mat_to_rot6d(R0).clone().requires_grad_(True)
        t_free = t0.clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [r6], "lr": args.lr_r}, {"params": [t_free], "lr": args.lr_t}])
        pose = lambda: (rot6d_to_mat(r6), t_free)  # noqa: E731
    has_c = ep.has_contact

    def terms():
        R, t = pose()
        data = data_loss(ep.obs, model, idx, R, t, s, args)
        acc_t = ((t[2:] - 2 * t[1:-1] + t[:-2]).norm(dim=-1) / args.acc_t) ** 2
        acc_r = ((R[2:] - 2 * R[1:-1] + R[:-2]).flatten(1).norm(dim=-1) / args.acc_r) ** 2
        if args.refine == "full" and args.contact_dof == "ray":  # contact and floor only move along the ray
            ray = torch.nn.functional.normalize(t.detach(), dim=-1)
            t_c = t.detach() + ((t - t.detach()) * ray).sum(-1, keepdim=True) * ray
            R_c = R.detach()
        else:
            t_c, R_c = t, R
        dist = nearest(outside_distance(ep.pts, R_c, t_c, s, ep.sdf), ep.cand)
        contact = (gm(dist, args.sigma_contact) * has_c).sum() / has_c.sum().clamp_min(1)
        pen = penetration_cm(ep.hands_all, R, t, s_val, ep.sdf, args.margin)
        X = s_val * model.points[None] @ R_c.transpose(1, 2) + t_c[:, None]
        floor = floor_depth(X, ep, args.floor_tol).mean(1) * 100
        return R, t, data, acc_t.mean() + acc_r.mean(), contact, pen, floor

    for _ in range(args.steps):
        opt.zero_grad()
        _, _, data, smooth, contact, pen, floor = terms()
        loss = (data.mean() + args.w_smooth * smooth + args.w_contact * contact + args.w_pen * pen.mean()
                + args.w_floor * floor.mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        R, t, data, smooth, contact, pen, floor = terms()
    Rn = R.cpu().numpy().astype(np.float64)
    Rn = np.stack([u @ vt for u, _, vt in (np.linalg.svd(r) for r in Rn)])  # exactly orthonormal
    parts = {"data": float(data.mean()), "smooth": float(smooth), "contact": float(contact),
             "pen_margin_cm": float(pen.mean()), "floor_cm": float(floor.mean())}
    return Rn, t.detach().cpu().numpy().astype(np.float64), s_val, parts, data.cpu().numpy()


# ------------------------------------------------------------------------------------------ reports
def contact_report(ep: Episode, R, t, s, model: Model, args) -> dict:
    """On scored frames: hand gap (closest of the 512 hand points to the surface, grid SDF), candidate gap,
    floor violation, exact PEN; all cm."""
    sel = np.flatnonzero(ep.in_scored)
    with torch.no_grad():
        Rt = torch.as_tensor(R[sel], dtype=torch.float32, device=DEV)
        tt = torch.as_tensor(t[sel], dtype=torch.float32, device=DEV)
        st = torch.full((len(sel),), s, device=DEV)
        hands = torch.as_tensor(ep.hands_512[sel], dtype=torch.float32, device=DEV)
        local = torch.einsum("tpi,tij->tpj", hands - tt[:, None], Rt) / s
        gap = (s * sdf_far(ep.sdf, local)).abs().min(1).values * 100
        dist = outside_distance(ep.pts[sel], Rt, tt, st, ep.sdf)
        cand = ep.cand[sel]
        cgap = nearest(dist, cand) * 100
        has = cand.any(1)
        X = s * model.points[None] @ Rt.transpose(1, 2) + tt[:, None]
        below = floor_depth(X, ep, 0.0).max(1).values * 100
    pen, depth = exact_penetration(ep.hands_512[sel], R[sel], t[sel], s, ep.vertices, ep.faces)
    return {"gap_hand_cm": float(gap.mean()), "gap_hand_median_cm": float(gap.median()),
            "gap_contact_cm": float(cgap[has].mean()) if has.any() else None,
            "below_floor_cm": float(below.mean()), "pen_cm": pen, "pen_max_depth_cm": depth}


def write_episode(ep: Episode, args, R, t, s, alpha, parts, data_frames, before, after, extra) -> dict:
    frames, total = ep.frames, ep.total
    near = np.clip(np.searchsorted(frames, np.arange(total)), 0, len(frames) - 1)
    out = args.run / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    h = ep.human
    np.savez(out / f"{ep.name}.npz", pose=h["pose"], scales=h["scales"], shape=h["shape"],
             object_rotation=R[near].astype(np.float32), object_translation=t[near].astype(np.float32),
             object_scale=np.float32(s))
    shutil.copy(ep.mesh_path, out / ep.mesh_path.name)
    cam = dict(ep.cam)
    cam["depth_scale"] = np.float64(np.median(ep.ds))  # the cache this episode was fit to, at its median k_t
    cam["depth_scale_frames"] = ep.ds[near].astype(np.float64)  # per video frame 1 / k_t, for pushout.py
    np.savez(out / f"{ep.name}_camera.npz", **cam)

    sc = ep.in_scored
    shift = np.linalg.norm(t[sc] - alpha * ep.t0[sc], axis=1) * 100
    turn = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", R[sc], ep.R0[sc]) - 1) / 2, -1, 1)))
    acc = np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=-1) * 100
    kinds = ep.cand_kind[sc]
    stats = {
        "episode": ep.episode, "init": str(ep.init_dir), "cache": str(ep.cache), "human": str(args.human),
        "mesh": ep.mesh_path.name, "scale_init_m": round(ep.s0, 4), "alpha": round(alpha, 4), "scale_m": round(s, 4),
        "frames_fitted": len(frames), "contact_frames": int(kinds.any(1).sum()), "scored": int(sc.sum()),
        "contact_kind_frames": {"hand": int(kinds[:, 0].sum()), "foot": int(kinds[:, 1].sum()),
                                "body_only": int((kinds[:, 2] & ~kinds[:, 0] & ~kinds[:, 1]).sum())},
        "k_t_median": round(float(np.median(1 / ep.ds)), 4), "k_t_p5_p95": np.round(np.percentile(1 / ep.ds, [5, 95]), 4).tolist(),
        "floor_offset_m": round(ep.floor, 4), "before": before, "after": after,
        "shift_from_scaled_init_cm_mean": round(float(shift.mean()), 2), "turn_deg_mean": round(float(turn.mean()), 2),
        "acc_t_cm_mean": round(float(acc.mean()), 4), "loss": parts, **extra,
    }
    meta = args.run / "joint_meta"
    meta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(meta / f"{ep.name}_series.npz", frames=frames, R=R, t=t, depth_scale=ep.ds, ratio=ep.ratio,
                        data=data_frames, contact_kind=ep.cand_kind, in_scored=sc)
    return stats


# --------------------------------------------------------------------------------------------- main
def take_lock(lock: Path) -> bool:
    """Create the lock with this pid; a lock left by a process that is gone (killed job) is replaced."""
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                os.kill(int(lock.read_text()), 0)
                return False  # holder alive
            except (ValueError, ProcessLookupError, FileNotFoundError):  # empty/old-style lock or dead holder
                print(f"stale lock {lock.name}, taken over", flush=True)
                lock.unlink(missing_ok=True)
                continue
            except PermissionError:  # alive, another user's
                return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    return False


def process_object(name: str, episodes: list[int], args, body, parts, budget, scored) -> list[dict]:
    t1 = time.time()
    eps = [Episode(e, args, body, parts, budget, scored[e]) for e in episodes]
    t_load = time.time() - t1
    grid = np.exp(np.linspace(np.log(args.alpha_range[0]), np.log(args.alpha_range[1]), args.alpha_steps))
    models = {ep.episode: Model(ep.mesh, N_REFINE, args.seed) for ep in eps}
    hyps = {}
    for ep in eps:
        t1 = time.time()
        hyps[ep.episode] = rotation_hypotheses(ep, models[ep.episode], args) if args.rot_hyps else {"changed": False}
        hyps[ep.episode]["seconds"] = round(time.time() - t1, 1)
        print(f"ep {ep.episode:2d} {name}: rotation hypotheses {hyps[ep.episode]}", flush=True)
    curves = {}
    for ep in eps:
        rel = np.exp(np.log(np.median([e.s0 for e in eps])) + np.log(grid)) / ep.s0  # common absolute grid
        curves[ep.episode] = stage_a(ep, models[ep.episode], rel, args) if args.stage_a else None
    if args.stage_a:
        alphas, info = choose_alphas(eps, curves, grid, args)
    else:
        alphas, info = {ep.episode: 1.0 for ep in eps}, {}
    info["object"] = name
    (args.run / "joint_meta" / f"object_{name}.json").write_text(json.dumps(info))
    stats = []
    for ep in eps:
        t1 = time.time()
        model = models[ep.episode]
        before = contact_report(ep, ep.R_in, ep.t_in, ep.s0, model, args)
        R, t, s, parts_, data_frames = refine(ep, model, alphas[ep.episode], args)
        after = contact_report(ep, R, t, s, model, args)
        st = write_episode(ep, args, R, t, s, alphas[ep.episode], parts_, data_frames, before, after,
                           {"object": name, "object_episodes": episodes, "rotation_hypotheses": hyps[ep.episode],
                            "timing_s": {"load_all": round(t_load, 1), "fit": round(time.time() - t1, 1)}})
        stats.append(st)
        print(f"ep {ep.episode:2d} {name}: alpha {st['alpha']:.3f} (scale {ep.s0:.3f} -> {s:.3f} m), k_t "
              f"{st['k_t_p5_p95']}, contact frames {st['contact_frames']}/{st['scored']}; hand gap "
              f"{before['gap_hand_cm']:.1f} -> {after['gap_hand_cm']:.1f} cm, below floor {before['below_floor_cm']:.1f} -> "
              f"{after['below_floor_cm']:.1f} cm, pen {before['pen_cm']:.4f} -> {after['pen_cm']:.4f}, data "
              f"{parts_['data']:.4f}", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human", type=Path, required=True, help="episodes dir of the human run (kept as is)")
    parser.add_argument("--init", type=Path, required=True, help="episodes dir of the object run to start from")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", default="0-29", help="objects with any of these episodes (all their episodes)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--alt", action="append", default=[],
                        help="EPISODES:INIT_DIR:CACHE_DIR, another init run and cache for these episodes")
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    # 0. depth frame
    parser.add_argument("--depth-scale", choices=["perframe", "global"], default="perframe")
    parser.add_argument("--lam-k", type=float, default=100.0, help="Whittaker lambda on log k_t")
    # 1. contact candidates
    parser.add_argument("--contact-px", type=float, default=4.0, help="half-resolution px from the object mask")
    # 1b. rotation hypotheses
    parser.add_argument("--no-rot-hyps", dest="rot_hyps", action="store_false")
    parser.add_argument("--hyp-stride", type=int, default=3)
    parser.add_argument("--hyp-steps", type=int, default=30, help="re-fit steps per (frame, state)")
    parser.add_argument("--hyp-batch", type=int, default=256)
    parser.add_argument("--hyp-alpha", type=float, nargs=2, default=[0.6, 1.8])
    parser.add_argument("--hyp-alpha-steps", type=int, default=13)
    parser.add_argument("--w-hand-h", type=float, default=0.0,
                        help="per frame, x min(hand distance / 10 cm, 3); off: at one depth per episode the distance is mostly depth error (pan: every state 10 cm off), and it flipped hoop ep 2 / foam roll ep 21 to worse silhouettes")
    parser.add_argument("--w-rest", type=float, default=1.0, help="per static unheld frame, x support_cost")
    parser.add_argument("--w-low", type=float, default=0.1, help="centre height in support_cost (a tie-break)")
    parser.add_argument("--support-tol", type=float, default=0.1,
                        help="x object size: points this close to the lowest support it (0.03 read a flat pan tilted 5 deg as unstable)")
    parser.add_argument("--static-v", type=float, default=0.005, help="m/frame")
    parser.add_argument("--static-w", type=float, default=0.02, help="rad/frame")
    parser.add_argument("--w-jump-h", type=float, default=1.0)
    parser.add_argument("--hyp-min-gain", type=float, default=0.03, help="per sampled frame, to leave the init")
    parser.add_argument("--hyp-fit-tol", type=float, default=0.02,
                        help="mean data loss the new track may lose against the init, else it is rejected "
                             "(hoop ep 0, fixed: +0.014; foam roll ep 21, worse silhouette: +0.095)")
    parser.add_argument("--hyp-rest-tol", type=float, default=0.1,
                        help="support cost (episode sum) the new path may add, else it is rejected")
    parser.add_argument("--sym-tol", type=float, default=0.02, help="x size: turns this close are one state")
    parser.add_argument("--hyp-refine-steps", type=int, default=200)
    parser.add_argument("--w-smooth-h", type=float, default=0.1, help="Step 5's smoothness in the hypothesis re-fit")
    parser.add_argument("--w-prior", type=float, default=1.0, help="pull of the re-fit to the hypothesis track")
    # 2. alpha search
    parser.add_argument("--no-stage-a", dest="stage_a", action="store_false")
    parser.add_argument("--alpha-range", type=float, nargs=2, default=[0.4, 2.5])
    parser.add_argument("--alpha-steps", type=int, default=129)
    parser.add_argument("--w-dep-a", type=float, default=1.0)
    parser.add_argument("--sigma-a", type=float, default=0.15,
                        help="m, depth robust scale in the alpha search (Step 5's 0.03 saturates 1%% off the init "
                             "at 3 m: the depth curve is then a spike at alpha 1)")
    parser.add_argument("--w-contact-a", type=float, default=3.0,
                        help="contact wins where it and the depth map disagree (black pan: MoGe puts it 40 cm in front of the hands)")
    parser.add_argument("--sigma-contact-a", type=float, default=0.10, help="m")
    parser.add_argument("--w-floor-a", type=float, default=3.0, help="as strong as contact: feet inside a hoop must not sink it")
    parser.add_argument("--lam-share", type=float, default=1.0, help="pull of each episode's log scale to the object's")
    # 3. refinement
    parser.add_argument("--refine", choices=["ray", "full"], default="ray",
                        help="ray: only a per-frame depth along the camera ray (full re-fits lost IoU on the hoop)")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr-d", type=float, default=0.002, help="log-depth step (--refine ray)")
    parser.add_argument("--lr-t", type=float, default=0.002)
    parser.add_argument("--lr-r", type=float, default=0.01)
    parser.add_argument("--w-contact", type=float, default=1.0)
    parser.add_argument("--sigma-contact", type=float, default=0.08, help="m")
    parser.add_argument("--contact-dof", choices=["ray", "full"], default="ray",
                        help="ray: contact and floor gradients move the object only along its camera ray (depth); "
                             "the image position and rotation stay the data terms' (a thin ring pulled sideways "
                             "toward feet standing inside it lost half its IoU)")
    parser.add_argument("--w-pen", type=float, default=20.0, help="7a used 5 with free R, t; ray-only needs more")
    parser.add_argument("--margin", type=float, default=0.003)
    parser.add_argument("--w-floor", type=float, default=5.0, help="per cm below the floor")
    parser.add_argument("--floor-tol", type=float, default=0.02, help="m below the body floor that is free")
    parser.add_argument("--floor-pct", type=float, default=30.0)
    parser.add_argument("--res", type=int, default=128)
    # object_fit.py data / smoothness terms (7a's submitted settings: w_dep 0.3)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--sil-sigma", type=float, default=0.1)
    parser.add_argument("--w-sil", type=float, default=1.0)
    parser.add_argument("--w-dep", type=float, default=0.3)
    parser.add_argument("--w-smooth", type=float, default=3.0, help="0.1 (Step 5) lets contact jitter the depth")
    parser.add_argument("--acc-t", type=float, default=0.005)
    parser.add_argument("--acc-r", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    alt = {}
    for spec in args.alt:
        eps, init_dir, cache_dir = spec.split(":")
        for e in parse_episodes(eps):
            alt[e] = (Path(init_dir), Path(cache_dir))
    args.sources = lambda e: alt.get(e, (args.init, args.cache))

    body = MHRBody(device=DEV)
    parts = BodyParts(body, args.seed)
    budget = mesh_budget(args.sample)
    scored = scored_frames(args.sample)
    meta = args.run / "joint_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (args.run / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items() if k != "sources"}, indent=1))
    names = object_names(args.data)
    groups = {}
    for e in range(30):
        groups.setdefault(names[e], []).append(e)
    wanted = set(parse_episodes(args.episodes))
    for name, episodes in groups.items():
        if not wanted & set(episodes):
            continue
        if all((meta / f"episode_{e:06d}.json").exists() for e in episodes):
            continue
        lock = meta / f"object_{name}.lock"
        if not take_lock(lock):
            print(f"object {name}: locked by a running process, skipped", flush=True)
            continue
        if all((meta / f"episode_{e:06d}.json").exists() for e in episodes):  # finished while we checked
            lock.unlink(missing_ok=True)
            continue
        try:
            for st in process_object(name, episodes, args, body, parts, budget, scored):
                (meta / f"episode_{st['episode']:06d}.json").write_text(json.dumps(st, indent=1))
        except Exception:
            print(f"object {name} FAILED", flush=True)
            traceback.print_exc()
        finally:
            lock.unlink(missing_ok=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

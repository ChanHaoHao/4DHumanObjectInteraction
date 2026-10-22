#!/usr/bin/env python3
"""Last stage before packing (Colab VM, GPU): smooth the object translation, keep the hands out of the object.

    python finalize.py --src results/<run>/episodes --run results/<run>_fin [--episodes 0-29] [--lam-o 1000]
        [--lam-h 0,0,0] [--margin 0.004]

ACC-O scores the second difference of object_translation only, against a GT whose own acceleration is ~0.074 cm/f^2
(the score of a static object on the leaderboard); our tracks carry ~0.10 of mostly noise. CD-O is ~30 cm, so moving
the object ~1 cm to smooth it costs nothing measurable there. Per episode, on the fitted frames (scored span +- 15):

1. Mesh origin -> area centroid of the packed surface (t += s R c, so the posed surface is unchanged): ACC-O
   then follows the object's centre, not a corner of it (the desk's origin sat 17-27 cm off its centre).
2. Human (optional, --lam-h t,r,p > 0): human_post's Whittaker smoothing once more, on top of the existing one.
3. Object translation: argmin_t |t - t0|^2 + lam_o |D2 t - g|^2 + w_pen sum_t mean_p relu(margin - SDF(hand p))^2.
   g = the holding wrists' acceleration, weighted per frame by how close each hand is to the object surface
   (--follow-d: full weight within d0, none beyond d1; weights smoothed in time), 0 away from the hands. Smoothing t
   alone (g = 0) only approaches the static-object score from above (ACC-O 0.119 -> 0.088 at lam 1000, floor
   0.074): beating it needs the true accelerations, and a held object moves with the hand, which SAM 3D Body tracks
   far better than our object fit (ACC-H 0.137).
   (L-BFGS, warm-started from the Whittaker solution and from each lighter weight), rotation and scale kept. Hand
   points = every vertex within --hand-radius of a vertex the finger parameters move (pushout's set plus palm and
   wrist: the host's 512 hand points are not ours). Weights 0 (Whittaker only) and --w-pen x 1, 10, 100; kept: the
   lowest exact penetration (kit check on proxy_eval's 512 hand points, which tracked Kaggle's PEN within 5%) among
   those whose track is no rougher than the input, stopping at <= 1e-5 cm. A grip that wraps a thin part (hoop tube, iron handle) cannot be cleared by a translation: a stiff
   weight there only made the track jitter (ep 1 acc 0.09 -> 0.71).
4. Fingers: on frames where a hand point is still inside, the finger parameters (MHR params 68:122) alone are
   optimized out of the object (squared SDF hinge, prior on the change). CD-H samples body vertices the finger
   parameters do not move and ACC-H scores 22 joints up to the wrists, so only PEN sees them.

Writes <run>/episodes/ (npz, recentred mesh, camera) and <run>/finalize.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_post import euler_convention, smooth_sequence, whittaker  # noqa: E402
from human_vm import episode_range, parse_episodes  # noqa: E402
from proxy_eval import DEFAULT_SAMPLE, MESH_SUFFIXES, MHRBody, budget_mesh, mesh_budget, ms, scored_frames  # noqa: E402
from pushout import SDFGrid, exact_penetration  # noqa: E402
from v2dlb.mhr_metrics import MHR_TABLE3_BODY_JOINT_INDICES  # noqa: E402  (kit path set by proxy_eval)

JOINTS = list(MHR_TABLE3_BODY_JOINT_INDICES)  # the 22 joints ACC-H scores
WRISTS = [78, 42]  # l_wrist, r_wrist (MHR joints; the last two of JOINTS)


def surface_centroid(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces[(faces != 0).any(1)]]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return (area[:, None] * tri.mean(1)).sum(0) / area.sum()


def acc_cm(x: np.ndarray, frames: np.ndarray) -> float:
    """Mean |second difference| over the scored stretches, cm/frame^2 (the kit's ACC without alignment)."""
    cut = np.flatnonzero(np.diff(frames) != 1) + 1
    parts = [x[f] for f in np.split(frames, cut) if len(f) >= 3]
    return float(np.concatenate([np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) for p in parts]).mean() * 100)


def follow_target(t0, g, lam) -> np.ndarray:
    """argmin_t |t - t0|^2 + lam |D2 t - g|^2 (g [n-2, 3]), closed form."""
    n = len(t0)
    d = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    return splu((sparse.eye(n) + lam * (d.T @ d)).tocsc()).solve(t0 + lam * (d.T @ g))


def hand_acceleration(wrists, hands_lr, R, t, s, sdf, d0, d1, lam_w=100.0) -> np.ndarray:
    """g [n-2, 3]: per frame sum_k w_k D2 wrist_k with w_k = closeness of hand k to the object, sum <= 1."""
    w = []
    with torch.no_grad():
        Rt = torch.as_tensor(R, dtype=torch.float32, device="cuda")
        for pts in hands_lr:
            local = torch.einsum("tpi,tij->tpj", torch.as_tensor(pts - t[:, None], dtype=torch.float32, device="cuda"),
                                 Rt) / s
            d = (s * sdf(local)).min(1).values.cpu().numpy()
            w.append(np.clip(whittaker(np.clip((d1 - d) / (d1 - d0), 0, 1)[:, None], np.ones(len(d)), lam_w)[:, 0], 0, 1))
    w = np.stack(w, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1)
    acc = wrists[2:] - 2 * wrists[1:-1] + wrists[:-2]  # [n-2, 2, 3]
    return (w[1:-1, :, None] * acc).sum(1), w


def solve_translation(t0, init, hands, R, s, sdf, lam, margin, w_pen, g) -> np.ndarray:
    """Smooth t that keeps every hand point at least margin outside the object (all arrays on the fitted frames)."""
    T0 = torch.as_tensor(t0, dtype=torch.float64, device="cuda")
    G = torch.as_tensor(g, dtype=torch.float64, device="cuda")
    H = torch.as_tensor(hands, dtype=torch.float32, device="cuda")
    Rt = torch.as_tensor(R, dtype=torch.float32, device="cuda")
    t = torch.tensor(np.ascontiguousarray(init), dtype=torch.float64, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=1, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        local = torch.einsum("tpi,tij->tpj", H - t[:, None].float(), Rt) / s
        inside = torch.relu(margin - s * sdf(local)).double()
        loss = (((t - T0) ** 2).sum() + lam * ((t[2:] - 2 * t[1:-1] + t[:-2] - G) ** 2).sum()
                + w_pen * (inside ** 2).mean(1).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return t.detach().cpu().numpy()


def open_fingers(body: MHRBody, params, shape, hand_pts, R, t, s, sdf, margin, w_prior, steps, chunk=32):
    """Finger parameters (params[:, 68:122]) that move the hand points out of the object, per frame."""
    out = params.copy()
    flip = torch.as_tensor(body.FLIP / 100.0, dtype=torch.float32, device="cuda")
    shape_t = torch.as_tensor(np.asarray(shape).reshape(1, 45), dtype=torch.float32, device="cuda")
    cols = list(body.HAND_COLS)
    for a in range(0, len(params), chunk):
        P = torch.as_tensor(params[a:a + chunk], dtype=torch.float32, device="cuda")
        Rt = torch.as_tensor(R[a:a + chunk], dtype=torch.float32, device="cuda")
        tt = torch.as_tensor(t[a:a + chunk], dtype=torch.float32, device="cuda")
        theta0 = P[:, cols].clone()
        theta = theta0.clone().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=0.02)
        n = len(P)
        for _ in range(steps):
            opt.zero_grad()
            full = torch.cat([P[:, :cols[0]], theta, P[:, cols[-1] + 1:]], 1)
            v, _ = body.model(shape_t.expand(n, -1).contiguous(), full, torch.zeros(n, 72, device="cuda"), True)
            local = torch.einsum("tpi,tij->tpj", v[:, hand_pts] * flip - tt[:, None], Rt) / s
            inside = torch.relu(margin - s * sdf(local))
            loss = 1e4 * (inside ** 2).mean(1).sum() + w_prior * ((theta - theta0) ** 2).sum()
            loss.backward()
            opt.step()
        out[a:a + chunk, cols] = theta.detach().cpu().numpy()
    return out


def process(episode: int, args, body: MHRBody, budget, scored: np.ndarray, hand_pts: np.ndarray) -> dict:
    src, dst = Path(args.src), Path(args.run) / "episodes"
    name = f"episode_{episode:06d}"
    data = dict(np.load(src / f"{name}.npz"))
    mesh_path = next(p for s in MESH_SUFFIXES if (p := src / f"{name}_object{s}").exists())
    total = len(data["pose"])
    frames = episode_range(scored, total)
    near = np.clip(np.searchsorted(frames, np.arange(total)), 0, len(frames) - 1)

    # 1. mesh origin at the packed surface's centroid
    vertices, faces = budget_mesh(mesh_path, *budget)
    c = surface_centroid(vertices, faces)
    R = ms._rotations(np.asarray(data["object_rotation"], np.float64))
    t = np.asarray(data["object_translation"], np.float64)
    s = float(np.asarray(data["object_scale"]).reshape(-1)[0])
    t = t + s * R @ c
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_translation(-c)
    dst.mkdir(parents=True, exist_ok=True)
    out_mesh = dst / f"{name}_object.obj"
    mesh.export(out_mesh)
    vertices, faces = budget_mesh(out_mesh, *budget)
    sdf = SDFGrid(vertices, faces, "cuda", res=args.res)

    # 2. human
    pose = np.asarray(data["pose"], np.float64)
    joints_in = body(np.concatenate([pose, np.broadcast_to(data["scales"], (total, 68))], 1), data["shape"])[1][:, JOINTS]
    if any(args.lam_h):
        lam = dict(zip("trp", args.lam_h))
        t_s, rot_s, pose_s = smooth_sequence(pose[frames, 0:3], pose[frames, 3:6], pose[frames, 6:136],
                                             np.ones(len(frames)), lam, args.conv)
        pose[frames] = np.concatenate([t_s, rot_s, pose_s], 1)
        pose = pose[frames][near]
        data["pose"] = pose.astype(np.float32)
    verts, joints = body(np.concatenate([pose, np.broadcast_to(data["scales"], (total, 68))], 1), data["shape"])
    wrists = joints[:, WRISTS]
    joints = joints[:, JOINTS]
    hands = verts[:, hand_pts]
    hand_sides = args.hand_sides

    # 3. object translation: smooth + hands out. <= 1e-5 cm per episode is 0.00000 on the board (mean of 30); ep 5
    # keeps 3e-6 at any weight (a spot the 128^3 grid does not resolve).
    t_in = t.copy()
    acc_in = acc_cm(t_in, scored)
    probe = verts[scored][:, body.hands]  # the exact (CPU) check on the 512 proxy points; the fit uses all hand points
    pen_before, _ = exact_penetration(probe, R[scored], t[scored], s, vertices, faces)
    if args.follow_d:
        g, w_hand = hand_acceleration(wrists[frames], [hands[frames][:, side] for side in hand_sides], R[frames],
                                      t_in[frames], s, sdf, *args.follow_d)
    else:
        g, w_hand = np.zeros((len(frames) - 2, 3)), np.zeros((len(frames), 2))
    tf = follow_target(t_in[frames], g, args.lam_o)
    best = None
    for w_pen in (0, args.w_pen, 10 * args.w_pen, 100 * args.w_pen) if args.w_pen else (0,):
        if w_pen:
            tf = solve_translation(t_in[frames], tf, hands[frames], R[frames], s, sdf, args.lam_o, args.margin, w_pen, g)
        cand = tf[near]
        pen, depth = exact_penetration(probe, R[scored], cand[scored], s, vertices, faces)
        acc = acc_cm(cand, scored)
        print(f"   w_pen {w_pen:g}: pen {pen:.6f}, max depth {depth:.2f} cm, acc_o {acc:.4f}", flush=True)
        if best is None or (acc <= acc_in and pen < best[1]):
            best = (cand, pen, depth, w_pen)
        if pen <= 1e-5:
            break
    t, pen, depth, w_pen = best

    # 4. fingers out where a hand point is still inside (grid check on every hand point, scored frames)
    pen_t = pen
    with torch.no_grad():
        local = torch.einsum("tpi,tij->tpj", torch.as_tensor(hands[scored] - t[scored, None], dtype=torch.float32,
                             device="cuda"), torch.as_tensor(R[scored], dtype=torch.float32, device="cuda")) / s
        touch = scored[(s * sdf(local) < args.margin).any(1).cpu().numpy()]
    if args.finger_steps and len(touch) and pen > 1e-5:
        params = np.concatenate([pose, np.broadcast_to(data["scales"], (total, 68))], 1)
        params[touch] = open_fingers(body, params[touch], data["shape"], hand_pts, R[touch], t[touch], s, sdf,
                                     args.margin, args.w_finger, args.finger_steps)
        cand = verts.copy()
        cand[touch] = body(params[touch], data["shape"])[0]
        pen_f, depth_f = exact_penetration(cand[scored][:, body.hands], R[scored], t[scored], s, vertices, faces)
        if pen_f < pen:
            pose[touch, 68:122] = params[touch, 68:122]
            data["pose"] = pose.astype(np.float32)
            pen, depth = pen_f, depth_f

    data.update(object_translation=t.astype(np.float32))
    np.savez(dst / f"{name}.npz", **data)
    shutil.copy(src / f"{name}_camera.npz", dst / f"{name}_camera.npz")
    move = np.linalg.norm(t[scored] - t_in[scored], axis=1) * 100
    acc_h = [np.mean([acc_cm(j[:, k], scored) for k in range(len(JOINTS))]) for j in (joints_in, joints)]
    out = {"episode": episode, "centroid_offset_cm": float(np.linalg.norm(s * c) * 100),
           "held_fraction": float(np.minimum(w_hand.sum(1), 1)[np.isin(frames, scored)].mean()),
           "acc_h_before": acc_h[0], "acc_h_after": acc_h[1],
           "joint_move_cm": float(np.linalg.norm(joints[scored] - joints_in[scored], axis=-1).mean() * 100),
           "acc_o_before": acc_cm(t_in, scored), "acc_o_after": acc_cm(t, scored),
           "move_cm_mean": float(move.mean()), "move_cm_max": float(move.max()), "w_pen": w_pen, "pen_before": pen_before,
           "pen_512_after_t": pen_t, "finger_frames": int(len(touch)), "pen_512": pen,
           "max_depth_cm": depth}
    print(f"ep {episode:2d}: acc_o {out['acc_o_before']:.4f} -> {out['acc_o_after']:.4f}, move mean "
          f"{out['move_cm_mean']:.2f} max {out['move_cm_max']:.2f} cm, held {out['held_fraction']:.2f}, pen {pen_before:.6f} -> {pen_t:.6f} -> fingers ({len(touch)} frames) "
          f"{pen:.6f}, w_pen {w_pen:g}; acc_h {acc_h[0]:.4f} -> {acc_h[1]:.4f}, joints move "
          f"{out['joint_move_cm']:.2f} cm", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--episodes", default="0-29")
    p.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    p.add_argument("--lam-o", type=float, default=1000.0, help="Whittaker lambda on the object translation")
    p.add_argument("--lam-h", type=lambda v: [float(x) for x in v.split(",")], default=[0.0, 0.0, 0.0],
                   help="extra human smoothing (translation, root rotation, body pose); 0 = untouched")
    p.add_argument("--margin", type=float, default=0.004, help="m kept between hand points and the object surface")
    p.add_argument("--follow-d", type=lambda v: [float(x) for x in v.split(",")] if v else [], default=[0.03, 0.08],
                   help="m: a hand within d0 of the object surface holds it fully, beyond d1 not at all; '' = off")
    p.add_argument("--hand-radius", type=float, default=0.03, help="m around the finger-driven vertices (rest pose)")
    p.add_argument("--w-pen", type=float, default=10.0, help="lightest penetration weight tried (x 1, 10, 100)")
    p.add_argument("--w-finger", type=float, default=0.1, help="prior on the finger parameter change (rad^2)")
    p.add_argument("--finger-steps", type=int, default=150)
    p.add_argument("--res", type=int, default=128)
    args = p.parse_args()

    body = MHRBody()
    args.conv = euler_convention(body)
    rest = body(np.zeros((1, 204)), np.zeros(45))[0][0]
    hand_pts = np.flatnonzero(cKDTree(rest[body.hand_all]).query(rest)[0] < args.hand_radius)
    args.hand_sides = [np.flatnonzero(rest[hand_pts, 0] > 0), np.flatnonzero(rest[hand_pts, 0] < 0)]  # left, right
    print(f"{len(hand_pts)} hand points ({len(body.hand_all)} finger-driven)", flush=True)
    budget = mesh_budget(args.sample)
    scored = scored_frames(args.sample)
    report = [process(e, args, body, budget, scored[e], hand_pts) for e in parse_episodes(args.episodes)]
    (args.run / "finalize.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                        "episodes": report}, indent=1))
    keys = ["acc_o_before", "acc_o_after", "move_cm_mean", "pen_before", "pen_512_after_t", "pen_512", "acc_h_before",
            "acc_h_after", "joint_move_cm"]
    print("mean " + "  ".join(f"{k} {np.mean([r[k] for r in report]):.5f}" for k in keys))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 7a: refit the object pose with a hand-penetration term, to remove contact penetration (PEN).

    python pushout.py --src results/object_v2/episodes --run results/object_v3 [--episodes 0-29] [--w-pen 1]

PEN is the mean depth of the host's 512 MHR hand points inside our *packed* object mesh (generalized winding
number > 0.5, depth = distance to the nearest triangle). The host's hand points are not ours, so every vertex the
finger parameters move is pushed out, with a margin. The human is not touched; the object scale is kept.

Per episode, on the fitted frames (scored span +- 15, as in Step 5), starting from the --src object track:
  1. The packed mesh (kit budget_mesh, as the packer will write it) -> signed distance grid in mesh coordinates,
     same inside rule as the scorer (GPU winding numbers, exact point-triangle distances).
  2. Step 5's joint refinement (object_fit.py: silhouette + depth data terms, second-difference smoothness on
     the absolute track) plus  --w-pen x mean over hand points of relu(margin - signed distance), in cm.
     The data terms keep the object where the video puts it; without them (an earlier version with only a
     pose prior) the foam block moved 7 cm and turned 53 deg to clear a hand sunk 10 cm into it.
  3. Exact check with the kit's penetration_depth on scored frames: our 512 proxy hand points (the official
     metric with alignment scale 1) and all hand vertices.

Writes <run>/episodes/ (npz with the new object track, mesh and camera copied) and <run>/pushout.json.
Needs the Step 3 cache (masks.npz, depth.npz) and the camera npz (K, depth_scale) next to each episode; a
depth_scale_frames array there (joint_fit.py: per frame 1 / k_t) is used instead of the single depth_scale.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from object_fit import N_REFINE, Model, Observations, data_loss, mat_to_rot6d, rot6d_to_mat  # noqa: E402
from human_vm import episode_range  # noqa: E402
from proxy_eval import (DEFAULT_CACHE, DEFAULT_SAMPLE, MESH_SUFFIXES, MHRBody, budget_mesh, mesh_budget, ms,  # noqa: E402
                        scored_frames)


def parse_episodes(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


# ------------------------------------------------------------------------------------------ SDF grid
def winding_numbers(points: torch.Tensor, tri: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Generalized winding number of each point w.r.t. triangles tri [F,3,3] (Van Oosterom-Strackee)."""
    out = []
    for a in range(0, len(points), chunk):
        p = points[a:a + chunk, None, None, :]  # [n,1,1,3]
        v = tri[None] - p  # [n,F,3,3]
        x, y, z = v[:, :, 0], v[:, :, 1], v[:, :, 2]
        lx, ly, lz = x.norm(dim=-1), y.norm(dim=-1), z.norm(dim=-1)
        det = (x * torch.cross(y, z, dim=-1)).sum(-1)
        den = lx * ly * lz + (x * y).sum(-1) * lz + (y * z).sum(-1) * lx + (z * x).sum(-1) * ly
        out.append((2 * torch.atan2(det, den)).sum(1) / (4 * math.pi))
    return torch.cat(out)


def segment_distance(p, a, b):
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-20)).clamp(0, 1)
    return (p - (a + t[..., None] * ab)).norm(dim=-1)


def triangle_distance(points: torch.Tensor, tri: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Unsigned distance from each point to the nearest triangle (face interior or edges)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = torch.cross(b - a, c - a, dim=-1)
    area2 = n.norm(dim=-1)
    good = area2 > 1e-12
    n = n / area2.clamp_min(1e-20)[:, None]
    out = []
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk, None, :]  # [n,1,3]
        h = ((p - a) * n).sum(-1)  # signed plane distance [n,F]
        q = p - h[..., None] * n

        def side(u, v, q0):  # (v - u) x (q - u) along n, >= 0 on the inner side of edge u->v
            return (torch.cross((v - u).expand_as(q), q - q0, dim=-1) * n).sum(-1) >= 0

        inside = good & side(a, b, a) & side(b, c, b) & side(c, a, c)
        edge = torch.minimum(torch.minimum(segment_distance(p, a, b), segment_distance(p, b, c)),
                             segment_distance(p, c, a))
        d = torch.where(inside, h.abs(), edge)
        out.append(d.min(1).values)
    return torch.cat(out)


class SDFGrid:
    """Signed distance (negative inside, |winding| > 0.5 like the scorer) on a regular grid, mesh units."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device, res: int = 128, pad: float = 0.06):
        faces = faces[(faces != 0).any(1)]  # the packer's zero-area padding
        tri = torch.as_tensor(vertices[faces], dtype=torch.float32, device=device)
        lo, hi = vertices.min(0) - pad, vertices.max(0) + pad
        self.h = float((hi - lo).max() / (res - 1))
        dims = np.ceil((hi - lo) / self.h).astype(int) + 1
        self.lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
        self.extent = torch.as_tensor((dims - 1) * self.h, dtype=torch.float32, device=device)
        axes = [torch.arange(d, device=device, dtype=torch.float32) * self.h + float(l) for d, l in zip(dims, lo)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        dist = triangle_distance(grid, tri)
        inside = winding_numbers(grid, tri).abs() > 0.5
        self.values = torch.where(inside, -dist, dist).reshape(*dims)[None, None].permute(0, 1, 4, 3, 2).contiguous()
        self.inside_fraction = float(inside.float().mean())

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        """p [..., 3] in mesh units -> signed distance [...] (border value outside the grid, always positive)."""
        u = (p - self.lo) / self.extent * 2 - 1
        shape = u.shape[:-1]
        out = torch.nn.functional.grid_sample(self.values, u.reshape(1, 1, 1, -1, 3), align_corners=True,
                                              padding_mode="border")
        return out.reshape(shape)




# ------------------------------------------------------------------------------------------ fitting
def penetration_cm(hands: torch.Tensor, R: torch.Tensor, t: torch.Tensor, s: float, sdf: SDFGrid,
                   margin: float) -> torch.Tensor:
    """Per frame, mean over hand points of relu(margin - signed distance), cm. hands [n,P,3]."""
    local = torch.einsum("tpi,tij->tpj", hands - t[:, None], R) / s
    return torch.relu(margin - s * sdf(local)).mean(1) * 100


def refit(obs: Observations, model: Model, hands, R0, t0, s: float, sdf: SDFGrid, args):
    """object_fit.refine with a penetration term; all fitted frames jointly, scale fixed."""
    n = len(obs.frames)
    idx = torch.arange(n, device="cuda")
    H = torch.as_tensor(hands, dtype=torch.float32, device="cuda")
    r6 = mat_to_rot6d(torch.as_tensor(R0, dtype=torch.float32, device="cuda")).clone().requires_grad_(True)
    t = torch.as_tensor(t0, dtype=torch.float32, device="cuda").clone().requires_grad_(True)
    scale = torch.full((n,), s, device="cuda")
    opt = torch.optim.Adam([{"params": [r6], "lr": args.lr_r}, {"params": [t], "lr": args.lr_t}])

    def terms():
        R = rot6d_to_mat(r6)
        data = data_loss(obs, model, idx, R, t, scale, args)
        acc_t = ((t[2:] - 2 * t[1:-1] + t[:-2]).norm(dim=-1) / args.acc_t) ** 2
        acc_r = ((R[2:] - 2 * R[1:-1] + R[:-2]).flatten(1).norm(dim=-1) / args.acc_r) ** 2
        pen = penetration_cm(H, R, t, s, sdf, args.margin)
        return R, data, acc_t.mean() + acc_r.mean(), pen

    for _ in range(args.steps):
        opt.zero_grad()
        _, data, smooth, pen = terms()
        loss = data.mean() + args.w_smooth * smooth + args.w_pen * pen.mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        R, data, smooth, pen = terms()
    Rn = R.cpu().numpy().astype(np.float64)
    Rn = np.stack([u @ vt for u, _, vt in (np.linalg.svd(r) for r in Rn)])  # exactly orthonormal
    return Rn, t.detach().cpu().numpy().astype(np.float64), {
        "data": float(data.mean()), "smooth": float(smooth), "pen_margin_cm": float(pen.mean())}


def exact_penetration(hands, R, t, s, vertices, faces) -> tuple[float, float]:
    """Kit PEN on these hand points (alignment scale 1), cm, and the max depth, cm."""
    faces = faces[(faces != 0).any(1)]
    local = np.einsum("tpi,tij->tpj", hands - t[:, None], R) / s
    depth = ms.penetration_depth(local.reshape(-1, 3), vertices, faces).reshape(hands.shape[:2]) * s
    pen = float(np.mean(ms.sdf_interpenetration_distance_per_frame(-depth)))
    return pen, float(depth.max() * 100)


def process(episode: int, args, body: MHRBody, budget, scored: np.ndarray) -> dict:
    src, dst = Path(args.src), Path(args.run) / "episodes"
    name = f"episode_{episode:06d}"
    data = dict(np.load(src / f"{name}.npz"))
    cam = dict(np.load(src / f"{name}_camera.npz"))
    mesh_path = next(p for s in MESH_SUFFIXES if (p := src / f"{name}_object{s}").exists())
    vertices, faces = budget_mesh(mesh_path, *budget)
    t0 = time.time()
    sdf = SDFGrid(vertices, faces, "cuda", res=args.res)
    t_sdf = time.time() - t0

    total = len(data["pose"])
    frames = episode_range(scored, total)
    if "depth_scale_frames" in cam:  # joint_fit.py: 1 / k_t per frame, the depth frame the object was placed in
        depth_scale = np.asarray(cam["depth_scale_frames"], np.float64)[frames]
    else:
        depth_scale = float(cam.get("depth_scale", 1.0))
    obs = Observations(args.cache, episode, frames, cam["K"].astype(np.float64), depth_scale, seed=args.seed)
    model = Model(trimesh.load(mesh_path, force="mesh"), N_REFINE, args.seed)

    pose = np.asarray(data["pose"], np.float64)
    params = np.concatenate([pose, np.broadcast_to(data["scales"], (total, 68))], 1)
    verts, _ = body(params, data["shape"])
    hands_all, hands_512 = verts[:, body.hand_all], verts[:, body.hands]
    R = ms._rotations(np.asarray(data["object_rotation"], np.float64))
    t = np.asarray(data["object_translation"], np.float64)
    s = float(np.asarray(data["object_scale"]).reshape(-1)[0])

    Rf, tf, parts = refit(obs, model, hands_all[frames], R[frames], t[frames], s, sdf, args)
    near = np.clip(np.searchsorted(frames, np.arange(total)), 0, len(frames) - 1)
    Rn, tn = Rf[near], tf[near]

    before = exact_penetration(hands_512[scored], R[scored], t[scored], s, vertices, faces)
    after = exact_penetration(hands_512[scored], Rn[scored], tn[scored], s, vertices, faces)
    after_all = exact_penetration(hands_all[scored], Rn[scored], tn[scored], s, vertices, faces)
    shift = np.linalg.norm(tn[scored] - t[scored], axis=1) * 100
    turn = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", Rn[scored], R[scored]) - 1) / 2, -1, 1)))

    data.update(object_rotation=Rn.astype(np.float32), object_translation=tn.astype(np.float32))
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / f"{name}.npz", **data)
    shutil.copy(mesh_path, dst / mesh_path.name)
    shutil.copy(src / f"{name}_camera.npz", dst / f"{name}_camera.npz")
    out = {
        "episode": episode, "frames_fitted": len(frames), "hand_points": int(hands_all.shape[1]),
        "pen_512_before": before[0], "pen_512_after": after[0], "max_depth_cm_before": before[1],
        "max_depth_cm_after_all": after_all[1], "pen_all_after": after_all[0],
        "shift_cm_mean": float(shift.mean()), "shift_cm_max": float(shift.max()),
        "turn_deg_mean": float(turn.mean()), "turn_deg_max": float(turn.max()), "sdf_seconds": round(t_sdf, 1),
        **{f"loss_{k}": v for k, v in parts.items()},
    }
    print(f"ep {episode:2d}: pen {before[0]:.4f} -> {after[0]:.4f} cm (all hand verts {after_all[0]:.4f}, max depth "
          f"{before[1]:.2f} -> {after_all[1]:.2f} cm), shift mean {out['shift_cm_mean']:.2f} max "
          f"{out['shift_cm_max']:.2f} cm, turn mean {out['turn_deg_mean']:.1f} max {out['turn_deg_max']:.1f} deg, "
          f"data {parts['data']:.4f}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="episodes dir of the run to correct")
    parser.add_argument("--run", required=True, help="output run dir (writes <run>/episodes)")
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--res", type=int, default=128, help="SDF grid cells along the longest side")
    parser.add_argument("--margin", type=float, default=0.003, help="m; hand points end at least this far out")
    parser.add_argument("--w-pen", type=float, default=1.0, help="weight of the penetration term (cm)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr-t", type=float, default=0.002)
    parser.add_argument("--lr-r", type=float, default=0.01)
    # object_fit.py's data/smoothness settings (same defaults)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--sil-sigma", type=float, default=0.1)
    parser.add_argument("--w-sil", type=float, default=1.0)
    parser.add_argument("--w-dep", type=float, default=1.0)
    parser.add_argument("--w-smooth", type=float, default=0.1)
    parser.add_argument("--acc-t", type=float, default=0.005)
    parser.add_argument("--acc-r", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    body = MHRBody(device="cuda")
    budget = mesh_budget(args.sample)
    scored = scored_frames(args.sample)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    report = run / "pushout.json"
    results = json.loads(report.read_text()) if report.exists() else {}
    results["args"] = {k: str(v) for k, v in vars(args).items()}
    for episode in parse_episodes(args.episodes):
        results[str(episode)] = process(episode, args, body, budget, scored[episode])
        report.write_text(json.dumps(results, indent=1))
        torch.cuda.empty_cache()
    rows = [v for k, v in results.items() if k != "args"]
    print(f"mean over {len(rows)}: pen_512 {np.mean([r['pen_512_before'] for r in rows]):.4f} -> "
          f"{np.mean([r['pen_512_after'] for r in rows]):.4f} cm")


if __name__ == "__main__":
    main()

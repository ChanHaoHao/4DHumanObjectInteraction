#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pyarrow", "scipy", "torch", "trimesh", "fast_simplification", "numba"]
# ///
"""Score a Track 1 results folder with GT-free proxy metrics and print a per-episode table.

    uv run scripts/proxy_eval.py results/<run>/episodes [--cache cache] [--reference DIR] [--csv out.csv] [--depth-scale-from DIR]

GT is held out, so every number here is a proxy (see notes/proxy_eval.md). Columns, all over scored frames:

  plausibility (always; needs only the submission)
    pen    contact penetration, cm: the official PEN with the alignment scale taken as 1 (metric human)
    gap    hand-object contact gap, cm: per frame, the closest hand point to the object surface.
           Scored frames are the hand-object contact span, so this should be ~0
    acc_h  mean |second difference| of the 22 scored joints, cm/frame^2, within stretches.
           The organizers' GT averages 0.12-0.13; well above that means jitter
    acc_o  same for object translation

  reprojection (when <cache>/episode_XXXXXX/masks.npz and <episodes>/episode_XXXXXX_camera.npz exist)
    iou_h, iou_o   silhouette IoU of the rendered human / object against the masks (occlusion-aware)
    dep_h, dep_o   median |rendered depth - depth map| on agreeing pixels, cm (needs depth.npz)

  vs reference (with --reference DIR: another results folder taken as pseudo-GT, official metric code)
    cd_h, cd_o, accE_h, accE_o, penR

Inputs beyond the submission (both optional; Step 3 writes the cache):
  <episodes>/episode_XXXXXX_camera.npz   K [3,3] at video resolution, world_to_camera [4,4] (metres,
                                         submission world frame), optional image_size (W, H),
                                         optional depth_scale (cache depth x this = world metres)
  <cache>/episode_XXXXXX/masks.npz       frames [N], human [N,h,w] bool, object [N,h,w] bool
  <cache>/episode_XXXXXX/depth.npz       frames [N], depth [N,h,w] metres (0 or non-finite = invalid)

The host's vertex roles (alignment, body, hand points) are not in the kit; this uses its own
(see MHRBody), so absolute values differ slightly from the official ones.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "data" / "v2d_submission_kit"
DEFAULT_MODEL = ROOT / "body_models" / "mhr" / "mhr_model.pt"
DEFAULT_SAMPLE = KIT / "data" / "track_1_sample_submission.parquet"
DEFAULT_CACHE = ROOT / "cache"
IMAGE_SIZE = (1536, 1152)
MESH_SUFFIXES = (".glb", ".obj", ".ply", ".stl", ".off")
GT_ACC_H = 0.12  # cm/frame^2, organizers' GT joint acceleration (tools/track1/MESH_TO_MHR_PARAMS.md)

sys.path.insert(0, str(KIT))
from v2dlb import mhr_submission as ms  # noqa: E402
from v2dlb.mesh_budget import budget_mesh  # noqa: E402
from v2dlb.mhr_metrics import MHR_TABLE3_BODY_JOINT_INDICES  # noqa: E402

PLAUSIBILITY = ["pen", "gap", "acc_h", "acc_o"]
REPROJECTION = ["iou_h", "iou_o", "dep_h", "dep_o"]
REFERENCE = ["cd_h", "cd_o", "accE_h", "accE_o", "penR"]
HIGHER_IS_BETTER = {"iou_h", "iou_o"}


class MHRBody:
    """Meta's MHR TorchScript model in the Track 1 frame (metres, y and z flipped), kit-style API.

    `body(params [T,204], shape [45]) -> (vertices [T,18439,3], joints [T,127,3])`, plus `roles`:
    alignment = all vertices; hands = 256 points per hand among the vertices the finger parameters
    move; body = 1024 of the remaining vertices. Fixed seeds, so roles are stable across runs.
    """

    FLIP = np.array([1.0, -1.0, -1.0])
    HAND_COLS = range(68, 122)  # finger parameters (tools/track1/mesh_to_mhr_params.py)

    def __init__(self, path: Path = DEFAULT_MODEL, device: str | None = None, chunk: int = 128):
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not path.exists():
            raise SystemExit(f"{path} missing: see notes/proxy_eval.md (MHR v1.0.1 assets.zip)")
        self.torch, self.device, self.chunk = torch, device, chunk
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.faces = dict(self.model.named_buffers())["character_torch.mesh.faces"].cpu().numpy()
        rest = self(np.zeros((1, 204)), np.zeros(45))[0][0]
        posed = np.zeros((len(self.HAND_COLS), 204))
        posed[np.arange(len(posed)), list(self.HAND_COLS)] = 0.3
        moved = np.linalg.norm(self(posed, np.zeros(45))[0] - rest, axis=-1).max(0) > 1e-5
        rng = np.random.default_rng(0)
        hand = np.flatnonzero(moved)
        left, right = hand[rest[hand, 0] > 0], hand[rest[hand, 0] < 0]
        self.roles = {
            "alignment": np.arange(len(rest)),
            "left_hand": np.sort(rng.choice(left, 256, replace=False)),
            "right_hand": np.sort(rng.choice(right, 256, replace=False)),
            "body": np.sort(rng.choice(np.flatnonzero(~moved), 1024, replace=False)),
        }
        self.hands = np.concatenate([self.roles["left_hand"], self.roles["right_hand"]])
        self.hand_all = hand  # every vertex the finger parameters move (both hands)

    def __call__(self, params, shape):
        torch = self.torch
        params = torch.as_tensor(np.atleast_2d(params), dtype=torch.float32, device=self.device)
        shape = torch.as_tensor(np.asarray(shape).reshape(1, 45), dtype=torch.float32, device=self.device)
        verts, joints = [], []
        with torch.no_grad():
            for a in range(0, len(params), self.chunk):
                p = params[a:a + self.chunk]
                v, skel = self.model(shape.expand(len(p), -1).contiguous(), p,
                                     torch.zeros(len(p), 72, device=self.device), True)
                verts.append(v.cpu().numpy())
                joints.append(skel[..., :3].cpu().numpy())
        flip = self.FLIP / 100.0
        return np.concatenate(verts) * flip, np.concatenate(joints) * flip


# ------------------------------------------------------------------------------------------- loading
def scored_frames(sample: Path) -> dict[int, np.ndarray]:
    parts = pd.read_parquet(sample, columns=["row_id"]).row_id.str.split("_", expand=True)
    episodes, frames = parts[1].astype(int), parts[2].astype(int)
    keep = frames != ms.EPISODE_FRAME
    return {int(e): np.unique(g.to_numpy()) for e, g in frames[keep].groupby(episodes[keep])}


def mesh_budget(sample: Path) -> tuple[int, int]:
    parts = pd.read_parquet(sample, columns=["row_id"]).row_id.str.split("_", expand=True)
    one = parts[(parts[1] == parts[1].iloc[0]) & (parts[2] == str(ms.EPISODE_FRAME))]
    order = ms.array_order()
    count = lambda name: int((one[3].astype(int) == order.index(name)).sum())  # noqa: E731
    return count("object_mesh_faces"), count("object_mesh_vertices")


def load_episode(directory: Path, episode: int, frames: np.ndarray, budget, body: MHRBody) -> dict:
    """The submission on its scored frames, posed: human vertices/joints, object pose and mesh."""
    data = dict(np.load(directory / f"episode_{episode:06d}.npz"))
    mesh = next(p for s in MESH_SUFFIXES if (p := directory / f"episode_{episode:06d}_object{s}").exists())
    vertices, faces = budget_mesh(mesh, *budget)
    faces = faces[(faces != 0).any(1)]  # drop the packer's zero-area padding
    pose = np.asarray(data["pose"], np.float64)[frames]
    params = np.concatenate([pose, np.broadcast_to(data["scales"], (len(pose), 68))], 1)
    verts, joints = body(params, data["shape"])
    return {
        "verts": verts,
        "joints": joints[:, list(MHR_TABLE3_BODY_JOINT_INDICES)],
        "rotation": ms._rotations(np.asarray(data["object_rotation"], np.float64)[frames]),
        "translation": np.asarray(data["object_translation"], np.float64)[frames],
        "scale": float(np.asarray(data["object_scale"]).reshape(-1)[0]),
        "mesh_vertices": vertices,
        "mesh_faces": faces,
    }


def surface_samples(vertices, faces, count, seed):
    """Area-weighted barycentric samples: (face index, barycentric weights [count, 3])."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=count, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    return chosen, np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)


def at_samples(vertices, faces, samples):
    """Points at fixed barycentric samples; vertices [..., V, 3] -> [..., count, 3]."""
    chosen, w = samples
    tri = vertices[..., faces[chosen], :]  # [..., count, 3, 3]
    return np.einsum("...kij,ki->...kj", tri, w)


def object_points(ep, local):
    return ep["scale"] * np.einsum("tij,pj->tpi", ep["rotation"], local) + ep["translation"][:, None]


# ------------------------------------------------------------------------------------- plausibility
def stretch_acceleration(points, frames) -> float:
    """Mean |x[t-1] - 2x[t] + x[t+1]| in cm, within contiguous stretches (the ACC-H/ACC-O operator)."""
    values = []
    for part in ms.stretches(frames):
        x = points[part]
        if len(x) >= 3:
            values.append(np.linalg.norm(x[:-2] - 2 * x[1:-1] + x[2:], axis=-1).reshape(len(x) - 2, -1).mean(1))
    return float(np.concatenate(values).mean() * 100) if values else math.nan


def penetration(ep, hands, align: float = 1.0) -> float:
    """The official PEN (kit code): hand points' depth inside the object's own mesh, cm."""
    local = np.einsum("tpi,tij->tpj", hands - ep["translation"][:, None], ep["rotation"]) / ep["scale"]
    depth = ms.penetration_depth(local.reshape(-1, 3), ep["mesh_vertices"], ep["mesh_faces"])
    depth = depth.reshape(hands.shape[:2])
    return float(np.mean(ms.sdf_interpenetration_distance_per_frame(-depth * ep["scale"] * align)))


def contact_gap(ep, hands) -> float:
    """Mean over frames of the closest hand point's distance to the object surface, cm."""
    from scipy.spatial import cKDTree

    local_surface = at_samples(ep["mesh_vertices"], ep["mesh_faces"],
                               surface_samples(ep["mesh_vertices"], ep["mesh_faces"], 20000, 0))
    local = np.einsum("tpi,tij->tpj", hands - ep["translation"][:, None], ep["rotation"]) / ep["scale"]
    distance, _ = cKDTree(local_surface).query(local.reshape(-1, 3), workers=-1)
    return float((distance.reshape(hands.shape[:2]).min(1) * ep["scale"]).mean() * 100)


# ------------------------------------------------------------------------------------- reprojection
def load_camera(directory: Path, episode: int):
    path = directory / f"episode_{episode:06d}_camera.npz"
    if not path.exists():
        return None
    data = np.load(path)
    size = tuple(int(x) for x in data["image_size"]) if "image_size" in data else IMAGE_SIZE
    scale = float(data["depth_scale"]) if "depth_scale" in data else 1.0
    return np.asarray(data["K"], np.float64), np.asarray(data["world_to_camera"], np.float64), size, scale


def render_labels(points_by_label, K, world_to_camera, size):
    """Point-splat z-buffer. points_by_label: list of [P,3] world points, label = index + 1.

    Returns (label [h,w] uint8, 0 = background; depth [h,w], inf = background). Points are dense
    surface samples, and a 3x3 closing fills the gaps between them.
    """
    from scipy import ndimage

    w, h = size
    pts = np.concatenate(points_by_label)
    labels = np.concatenate([np.full(len(p), i + 1, np.uint8) for i, p in enumerate(points_by_label)])
    cam = pts @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = cam[:, 2]
    front = z > 1e-3
    uv = cam[front, :2] / z[front, None] @ K[:2, :2].T + K[:2, 2]
    u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    flat, zz, ll = (v * w + u)[inside], z[front][inside], labels[front][inside]
    order = np.argsort(-zz)  # nearest written last; fancy assignment keeps the last write
    depth = np.full(h * w, np.inf)
    label = np.zeros(h * w, np.uint8)
    depth[flat[order]] = zz[order]
    label[flat[order]] = ll[order]
    depth, label = depth.reshape(h, w), label.reshape(h, w)
    for value in range(1, len(points_by_label) + 1):  # fill splat holes per label, keep the nearest
        hole = ndimage.binary_closing(label == value, np.ones((3, 3), bool)) & (label == 0)
        if hole.any():
            filled = ndimage.grey_erosion(np.where(label == value, depth, np.inf), size=(3, 3))
            label[hole], depth[hole] = value, filled[hole]
    return label, depth


def iou(a, b) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else math.nan


class ReprojectionInputs:
    def __init__(self, cache: Path, episode: int):
        folder = cache / f"episode_{episode:06d}"
        self.masks = dict(np.load(folder / "masks.npz")) if (folder / "masks.npz").exists() else None
        self.depth = dict(np.load(folder / "depth.npz")) if (folder / "depth.npz").exists() else None

    def frame(self, kind: str, frame: int):
        data = self.masks if kind == "masks" else self.depth
        if data is None:
            return None
        hit = np.flatnonzero(data["frames"] == frame)
        if not hit.size:
            return None
        return ((data["human"][hit[0]], data["object"][hit[0]]) if kind == "masks"
                else data["depth"][hit[0]])


def reprojection(ep, frames, camera, inputs: ReprojectionInputs, body: MHRBody, stride: int) -> dict:
    K, world_to_camera, image_size, depth_scale = camera
    human_samples = surface_samples(ep["verts"][0], body.faces, 60000, 0)
    object_local = at_samples(ep["mesh_vertices"], ep["mesh_faces"],
                              surface_samples(ep["mesh_vertices"], ep["mesh_faces"], 30000, 1))
    out = {key: [] for key in REPROJECTION}
    for i in range(0, len(frames), stride):
        masks = inputs.frame("masks", frames[i])
        if masks is None:
            continue
        h, w = masks[0].shape
        Ks = K.copy()
        Ks[0] *= w / image_size[0]
        Ks[1] *= h / image_size[1]
        human = at_samples(ep["verts"][i], body.faces, human_samples)
        obj = ep["scale"] * object_local @ ep["rotation"][i].T + ep["translation"][i]
        label, depth = render_labels([human, obj], Ks, world_to_camera, (w, h))
        depth_map = inputs.frame("depth", frames[i])
        if depth_map is not None:
            depth_map = depth_map.astype(np.float32) * depth_scale
        if depth_map is not None and depth_map.shape != (h, w):  # depth may be coarser than masks
            rows = np.arange(h) * depth_map.shape[0] // h
            cols = np.arange(w) * depth_map.shape[1] // w
            depth_map = depth_map[rows[:, None], cols[None, :]]
        for value, (name, mask) in enumerate(zip("ho", masks), 1):
            out[f"iou_{name}"].append(iou(label == value, mask.astype(bool)))
            if depth_map is not None:
                agree = (label == value) & mask.astype(bool) & np.isfinite(depth_map) & (depth_map > 0)
                if agree.any():
                    out[f"dep_{name}"].append(np.median(np.abs(depth[agree] - depth_map[agree])) * 100)
    return {key: float(np.nanmean(v)) if len(v) and not np.isnan(v).all() else math.nan
            for key, v in out.items()}


# ---------------------------------------------------------------------------------------- reference
def kit_arrays(ep, body: MHRBody, episode: int):
    """The arrays the official scorers read (v2dlb.mhr_submission._to_frames), from a loaded episode."""
    local = ms.sample_mesh_surface(ep["mesh_vertices"], ep["mesh_faces"], len(body.roles["body"]), episode)
    return {
        "mhr_vertices": ep["verts"][:, body.roles["alignment"]],
        "mhr_joints": ep["joints"],
        "human_surface_points": ep["verts"][:, body.roles["body"]],
        "object_surface_points": object_points(ep, local),
        "object_translation": ep["translation"],
    }


def reference_metrics(ep, ref, frames, body: MHRBody, episode: int) -> dict:
    """ms.episode_metrics, with the Chamfer series on torch (same numbers, ~8x faster on a GPU)."""
    torch = body.torch
    pred, target = kit_arrays(ep, body, episode), kit_arrays(ref, body, episode)
    transform = ms.fit_similarity_transform(pred["mhr_vertices"][0], target["mhr_vertices"][0])

    def tensors(arrays):  # only frame 0 of mhr_vertices is read; expand instead of copying T frames
        out = {k: torch.as_tensor(v, dtype=torch.float64, device=body.device) for k, v in arrays.items()
               if k != "mhr_vertices"}
        first = torch.as_tensor(arrays["mhr_vertices"][:1], dtype=torch.float64, device=body.device)
        return out | {"mhr_vertices": first.expand(len(frames), -1, -1)}

    series = ms.mhr_table3_metric_series(tensors(pred), tensors(target), body_joint_indices=None)
    pred_joints = ms.apply_similarity_transform(pred["mhr_joints"], transform)
    pred_translation = ms.apply_similarity_transform(pred["object_translation"], transform)
    acc_h, acc_o = [], []
    for part in ms.stretches(frames):
        acc_h.append(ms.acceleration_error_per_frame(pred_joints[part], target["mhr_joints"][part]))
        acc_o.append(ms.acceleration_error_per_frame(pred_translation[part], target["object_translation"][part]))
    return {"cd_h": float(series["cd_h_cm"].mean()), "cd_o": float(series["cd_o_cm"].mean()),
            "accE_h": float(np.concatenate(acc_h).mean() * 100), "accE_o": float(np.concatenate(acc_o).mean() * 100),
            "penR": penetration(ep, ep["verts"][:, body.hands], float(transform[0]))}


# --------------------------------------------------------------------------------------------- main
def evaluate(episodes_dir: Path, *, cache: Path = DEFAULT_CACHE, reference: Path | None = None,
             episodes: list[int] | None = None, sample: Path = DEFAULT_SAMPLE, stride: int = 5,
             body: MHRBody | None = None, depth_scale_from: Path | None = None, log=print) -> pd.DataFrame:
    if (episodes_dir / "episodes").is_dir():
        episodes_dir = episodes_dir / "episodes"
    if reference is not None and (reference / "episodes").is_dir():
        reference = reference / "episodes"
    body = body or MHRBody()
    budget = mesh_budget(sample)
    rows = []
    for episode, frames in scored_frames(sample).items():
        if episodes is not None and episode not in episodes:
            continue
        if not (episodes_dir / f"episode_{episode:06d}.npz").exists():
            if episodes is not None:
                log(f"episode {episode}: no npz, skipped")
            continue
        ep = load_episode(episodes_dir, episode, frames, budget, body)
        hands = ep["verts"][:, body.hands]
        row = {"episode": episode, "scored": len(frames),
               "pen": penetration(ep, hands), "gap": contact_gap(ep, hands),
               "acc_h": stretch_acceleration(ep["joints"], frames),
               "acc_o": stretch_acceleration(ep["translation"], frames)}
        camera = load_camera(episodes_dir, episode)
        if camera is not None and depth_scale_from is not None:  # another run's scale for this cache's depth
            camera = camera[:3] + (load_camera(depth_scale_from, episode)[3],)
        inputs = ReprojectionInputs(cache, episode)
        if camera is not None and inputs.masks is not None:
            row.update(reprojection(ep, frames, camera, inputs, body, stride))
        if reference is not None and (reference / f"episode_{episode:06d}.npz").exists():
            ref = load_episode(reference, episode, frames, budget, body)
            row.update(reference_metrics(ep, ref, frames, body, episode))
        rows.append(row)
    table = pd.DataFrame(rows)
    for key in PLAUSIBILITY + REPROJECTION + (REFERENCE if reference is not None else []):
        if key not in table:
            table[key] = math.nan
    return table


def format_table(table: pd.DataFrame) -> str:
    columns = ["episode", "scored"] + [c for c in PLAUSIBILITY + REPROJECTION + REFERENCE
                                       if c in table and table[c].notna().any()]
    mean = table[columns].mean(numeric_only=True).astype(object)
    mean["episode"], mean["scored"] = "mean", int(table["scored"].sum())
    shown = pd.concat([table[columns], mean.to_frame().T], ignore_index=True)
    lines = [shown.to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep="-")]
    groups = [("plausibility", PLAUSIBILITY), ("reprojection", REPROJECTION), ("vs reference", REFERENCE)]
    summary = [f"{name}: " + ", ".join(f"{k} {table[k].mean():.3f}" for k in keys if k in columns)
               for name, keys in groups if any(k in columns for k in keys)]
    lines += ["", *summary, f"(lower is better except {', '.join(sorted(HIGHER_IS_BETTER))}; "
              f"GT acc_h is ~{GT_ACC_H})"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes_dir", type=Path, help="results/<run>/episodes (or results/<run>)")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--reference", type=Path, default=None, help="results folder used as pseudo-GT")
    parser.add_argument("--episodes", default=None, help="comma-separated subset, e.g. 0,3,12")
    parser.add_argument("--stride", type=int, default=5, help="score every Nth frame for reprojection")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument("--csv", type=Path, default=None, help="also write the per-episode table here")
    parser.add_argument("--depth-scale-from", type=Path, default=None,
                        help="episodes dir whose camera npz depth_scale to use (scoring against another cache)")
    args = parser.parse_args()
    subset = [int(e) for e in args.episodes.split(",")] if args.episodes else None
    table = evaluate(args.episodes_dir, cache=args.cache, reference=args.reference, episodes=subset,
                     sample=args.sample, stride=args.stride, body=MHRBody(args.model, args.device),
                     depth_scale_from=args.depth_scale_from)
    if table.empty:
        print(f"no episodes found in {args.episodes_dir}")
        return 1
    print(format_table(table))
    if args.csv:
        table.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Step 4 post-processing (runs on the Colab VM): per-frame SAM 3D Body -> one MHR sequence per episode.

    python human_post.py --raw /content/human_raw/dinov3 --run /content/results/human_v1 \
        [--episodes 0-29] [--depth sam|moge] [--lam-t 20] [--lam-r 4] [--lam-p 4]

Per episode (reads human_vm.py output and the Step 3 cache):
1. Identity: one `scales` [68] and `shape` [45], the mean over frames where the actor is fully
   visible (mask area >= 70% of its 90th percentile and the box off the image border).
2. Translation: with the identity fixed, each frame's translation is re-solved (linear least squares)
   so the 22 scored joints project where SAM 3D Body's own per-frame joints did.
   `--depth moge` then rescales it along the camera ray so the body's depth follows the stabilized
   MoGe depth of the actor mask (MoGe's per-episode scale divided out: the human keeps its size).
3. Smoothing: robust Whittaker smoother (second differences) on translation, root rotation
   (quaternions) and body pose (unwrapped Euler params), weighted by visibility; frames whose joints
   end up far from their per-frame estimate (flips, occlusion) are down-weighted and re-solved.
4. Checks: k_moge = MoGe depth / body depth (metric scale of the Step 3 depth relative to the body),
   k_floor = the same ratio implied by the feet touching the Step 3 floor plane, actor stature,
   joint acceleration before and after smoothing.

World frame = the static camera's OpenCV frame (x right, y down, z forward, metres). SAM 3D Body's body
is in the Track 1 frame (diag(1,-1,-1) MHR / 100) placed at t_cam, so pose[0:3] = 10 diag(1,-1,-1) t_cam
and world_to_camera is the identity.

Writes <run>/episodes/episode_XXXXXX.npz (submission format, object = 2 cm placeholder cube until
Step 5), episode_XXXXXX_object.obj, episode_XXXXXX_camera.npz (for proxy_eval reprojection),
<run>/human_meta/episode_XXXXXX.json (checks above) and <run>/overlays/episode_XXXXXX.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from human_vm import parse_episodes, scored_frames, video_path  # noqa: E402
from proxy_eval import MHRBody  # noqa: E402
from v2dlb.mhr_metrics import MHR_TABLE3_BODY_JOINT_INDICES as BODY22  # noqa: E402

FLIP = np.array([1.0, -1.0, -1.0])
IMAGE_SIZE = (1536, 1152)
ROBUST_M = 0.08  # m; mean joint distance at which a frame's weight halves
WORLD_TO_CAMERA = np.eye(4)  # world = camera frame


# --------------------------------------------------------------------------------------- MHR helpers
def native(body: MHRBody, params: np.ndarray, shape: np.ndarray):
    """MHR vertices and joints in its native frame, metres (the frame its Euler params rotate in)."""
    v, j = body(params, shape)
    return v * FLIP, j * FLIP


def posed(body: MHRBody, params: np.ndarray, shape: np.ndarray):
    """Vertices and joints in the Track 1 frame = camera axes; camera frame = this + translation."""
    return body(params, shape)


def euler_convention(body: MHRBody) -> str:
    """scipy convention of MHR's root rotation params [3:6], found by fitting a rotation to the mesh."""
    rest, _ = native(body, np.zeros((1, 204)), np.zeros(45))
    e = np.array([0.3, -0.5, 0.7])
    p = np.zeros((1, 204))
    p[0, 3:6] = e
    moved, _ = native(body, p, np.zeros(45))
    a, b = rest[0] - rest[0].mean(0), moved[0] - moved[0].mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    R = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    for conv in ("xyz", "XYZ", "zyx", "ZYX"):
        if np.allclose(Rotation.from_euler(conv, e).as_matrix(), R, atol=1e-3):
            return conv
    raise RuntimeError("MHR root rotation matches no Euler convention")


# ------------------------------------------------------------------------------------------ smoothing
def whittaker(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """argmin_z sum_t w_t |z_t - y_t|^2 + lam sum_t |z_{t+1} - 2 z_t + z_{t-1}|^2, every column of y."""
    n = len(y)
    if n < 3 or lam <= 0:
        return y.copy()
    d = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    a = sparse.diags(np.maximum(w, 1e-6)) + lam * (d.T @ d)
    return splu(a.tocsc()).solve(w[:, None] * y)


def continuous_quats(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    return q


def smooth_sequence(t, rot_euler, body_pose, w, lam, conv):
    """Smoothed (t [N,3], root euler [N,3], body pose [N,130]) with per-frame weights w."""
    t_s = whittaker(t, w, lam["t"])
    q = whittaker(continuous_quats(Rotation.from_euler(conv, rot_euler).as_matrix()), w, lam["r"])
    rot_s = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_euler(conv)
    unwrapped = np.unwrap(body_pose, axis=0)
    pose_s = whittaker(unwrapped, w, lam["p"])
    pose_s = (pose_s - unwrapped) + body_pose  # same branch as the input values
    return t_s, rot_s, pose_s


# ----------------------------------------------------------------------------------------- geometry
def solve_translation(j_body: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per frame t with (j_body + t) projecting onto targets' projections (normalized coords, linear LS).

    x (Z + tz) = X + tx  ->  tx - x tz = x Z - X  (same for y), x = targets' X/Z.
    """
    x = targets[..., 0] / targets[..., 2]
    y = targets[..., 1] / targets[..., 2]
    X, Y, Z = j_body[..., 0], j_body[..., 1], j_body[..., 2]
    n, m = x.shape
    A = np.zeros((n, 2 * m, 3))
    b = np.zeros((n, 2 * m))
    A[:, :m, 0], A[:, :m, 2], b[:, :m] = 1, -x, x * Z - X
    A[:, m:, 1], A[:, m:, 2], b[:, m:] = 1, -y, y * Z - Y
    return np.linalg.solve(np.einsum("nki,nkj->nij", A, A), np.einsum("nki,nk->ni", A, b)[..., None])[..., 0]


def render_depth(verts: np.ndarray, K: np.ndarray, shape: tuple[int, int], step: int) -> np.ndarray:
    """Point-splat z-buffer of camera-frame vertices on a 1/step grid (inf = empty)."""
    h, w = shape
    z = verts[:, 2]
    u = ((verts[:, 0] / z) * K[0, 0] + K[0, 2]) / step
    v = ((verts[:, 1] / z) * K[1, 1] + K[1, 2]) / step
    ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(h * w, np.inf)
    np.minimum.at(out, v[ok].astype(int) * w + u[ok].astype(int), z[ok])
    return out.reshape(h, w)


def moge_ratio(verts, K, depth, human_mask) -> float:
    """median(MoGe depth / body depth) over pixels where both see the actor."""
    rend = render_depth(verts[::3], K, depth.shape, IMAGE_SIZE[0] // depth.shape[1])
    m = human_mask[::2, ::2] if human_mask.shape != depth.shape else human_mask
    ok = m & np.isfinite(rend) & (depth > 0)
    return float(np.median(depth[ok] / rend[ok])) if ok.sum() > 30 else np.nan


def stretch_acc(joints: np.ndarray, frames: np.ndarray) -> float:
    """Mean |second difference| (cm/frame^2) within runs of consecutive frames."""
    accs = []
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    for run in np.split(np.arange(len(frames)), breaks):
        if len(run) >= 3:
            j = joints[run]
            accs.append(np.linalg.norm(j[2:] - 2 * j[1:-1] + j[:-2], axis=-1).mean(-1))
    return float(np.concatenate(accs).mean() * 100) if accs else np.nan


# ------------------------------------------------------------------------------------------ outputs
def cube_obj(path: Path, size=0.02):
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size / 2
    faces = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    lines = [f"v {a:.4f} {b:.4f} {c:.4f}" for a, b, c in corners]
    lines += [f"f {a} {b} {c}\nf {a} {c} {d}" for a, b, c, d in faces]
    path.write_text("\n".join(lines) + "\n")


def overlay(episode, data, frames, verts_by_frame, mask_human, out: Path, picks, tile=(320, 400)):
    """Projected body (orange points) and the actor mask outline (red), cropped to the actor."""
    tiles = []
    cap = cv2.VideoCapture(str(video_path(data, episode)))
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, (768, 576))
        verts, K = verts_by_frame(f)
        z = verts[:, 2]
        u = (verts[:, 0] / z * K[0, 0] + K[0, 2]) / 2
        v = (verts[:, 1] / z * K[1, 1] + K[1, 2]) / 2
        ok = (u >= 0) & (u < 768) & (v >= 0) & (v < 576)
        layer = img.copy()
        layer[v[ok].astype(int), u[ok].astype(int)] = (255, 160, 40)
        img = cv2.addWeighted(img, 0.4, layer, 0.6, 0)
        m = mask_human[f].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (60, 60, 255), 2)
        ys, xs = np.nonzero(m)
        pts_x, pts_y = np.r_[xs, u[ok]], np.r_[ys, v[ok]]
        cx, cy = np.median(pts_x), np.median(pts_y)
        half = max(np.ptp(pts_y) * 0.6, np.ptp(pts_x) * 0.6 * tile[1] / tile[0], 40)
        x0, y0 = int(max(cx - half * tile[0] / tile[1], 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half * tile[0] / tile[1], 768)), int(min(cy + half, 576))
        crop = cv2.resize(img[y0:y1, x0:x1], tile)
        cv2.putText(crop, f"ep{episode} f{f}", (8, 24), 0, 0.7, (255, 255, 255), 2)
        tiles.append(crop)
    cap.release()
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


# --------------------------------------------------------------------------------------------- main
def process(episode: int, args, body: MHRBody, conv: str, scored: np.ndarray) -> dict:
    raw = dict(np.load(args.raw / f"episode_{episode:06d}.npz"))
    folder = args.cache / f"episode_{episode:06d}"
    meta = json.loads((folder / "meta.json").read_text())
    masks = np.load(folder / "masks.npz")["human"]
    K = raw["K"].astype(np.float64)
    frames = raw["frames"]
    n, total = len(frames), meta["frames"]
    params = raw["mhr_model_params"].astype(np.float64)

    # visibility weights: mask area relative to its 90th percentile, halved when the box is cut off
    area = raw["mask_area"].astype(float)
    vis = np.clip(area / max(np.percentile(area, 90), 1), 0, 1)
    box = raw["bbox"]
    cut = (box[:, 0] <= 3) | (box[:, 1] <= 3) | (box[:, 2] >= IMAGE_SIZE[0] - 3) | (box[:, 3] >= IMAGE_SIZE[1] - 3)
    vis = np.where(cut, vis * 0.5, vis)

    # 1. identity
    full = (vis >= 0.7) & ~cut
    if full.sum() < 10:
        full = vis >= np.median(vis)
    scales = params[full, 136:].mean(0)
    shape = raw["shape"][full].astype(np.float64).mean(0)

    # 2. translation for the fixed identity
    body_params = params.copy()
    body_params[:, :3] = 0
    body_params[:, 136:] = scales
    _, j_new = posed(body, body_params, shape)
    j_old = raw["joints"].astype(np.float64) + raw["cam_t"][:, None].astype(np.float64)
    idx = list(BODY22)
    t = solve_translation(j_new[:, idx], j_old[:, idx])

    stats = {"episode": episode, "frames_processed": n, "identity_frames": int(full.sum()),
             "depth_mode": args.depth, "lambda": dict(t=args.lam_t, r=args.lam_r, p=args.lam_p),
             "euler": conv, "visibility_median": round(float(np.median(vis)), 3)}

    # MoGe vs body depth (always measured; applied with --depth moge)
    depth_file = folder / "depth.npz"
    ratio = np.full(n, np.nan)
    if depth_file.exists():
        depth = np.load(depth_file)["depth"]
        verts_raw, _ = posed(body, body_params, shape)
        for i, f in enumerate(frames):
            if vis[i] > 0.3:
                ratio[i] = moge_ratio(verts_raw[i] + t[i], K, depth[f].astype(np.float32), masks[f])
        del verts_raw, depth
    k_moge = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else np.nan
    stats["k_moge"] = round(k_moge, 4) if np.isfinite(k_moge) else None
    stats["k_moge_p10_p90"] = (np.round(np.nanpercentile(ratio, [10, 90]), 4).tolist()
                               if np.isfinite(ratio).any() else None)
    if args.depth == "moge":
        if not np.isfinite(k_moge):
            raise RuntimeError("--depth moge needs depth.npz")
        rel = np.where(np.isfinite(ratio), ratio / k_moge, 1.0)
        t = t * rel[:, None]

    # 3. robust smoothing
    lam = {"t": args.lam_t, "r": args.lam_r, "p": args.lam_p}
    j_target = j_new + t[:, None]
    w = np.maximum(vis, 0.05)
    for _ in range(args.robust_iters + 1):
        t_s, rot_s, pose_s = smooth_sequence(t, params[:, 3:6], params[:, 6:136], w, lam, conv)
        smooth_params = body_params.copy()
        smooth_params[:, 3:6], smooth_params[:, 6:136] = rot_s, pose_s
        _, j_s = posed(body, smooth_params, shape)
        err = np.linalg.norm(j_s[:, idx] + t_s[:, None] - j_target[:, idx], axis=-1).mean(1)
        w = np.maximum(vis, 0.05) / (1 + (err / ROBUST_M) ** 2)
    stats["downweighted_frames"] = int((w < 0.5 * np.maximum(vis, 0.05)).sum())
    stats["joint_dev_cm_median_p95"] = np.round(np.percentile(err * 100, [50, 95]), 2).tolist()

    # 4. checks
    in_scored = np.isin(frames, scored)
    fs = frames[in_scored]
    stats["acc_raw_per_frame_identity"] = round(stretch_acc(j_old[in_scored][:, idx], fs), 4)
    stats["acc_raw"] = round(stretch_acc(j_target[in_scored][:, idx], fs), 4)
    stats["acc_smooth"] = round(stretch_acc((j_s + t_s[:, None])[in_scored][:, idx], fs), 4)
    rest, _ = posed(body, np.r_[np.zeros(136), scales][None], shape)
    stats["stature_m"] = round(float(np.ptp(rest[0][:, 1])), 3)
    n_up = np.asarray(meta["floor_normal"], float)
    d_cam = meta["camera_height_m"]
    verts_s, _ = posed(body, smooth_params, shape)
    lowest = (verts_s + t_s[:, None]) @ n_up  # n.p of each vertex; height above floor = n.p + d
    low = lowest.min(1)
    good = vis > 0.7
    stats["k_floor"] = round(float(np.median(-d_cam / low[good])), 4) if good.any() else None
    if np.isfinite(k_moge):
        stats["foot_height_cm_median"] = round(float(np.median(low[good] + d_cam / k_moge) * 100), 1)

    # submission arrays over every video frame (unprocessed frames copy the nearest processed one)
    near = np.clip(np.searchsorted(frames, np.arange(total)), 0, n - 1)
    left = np.clip(near - 1, 0, n - 1)
    near = np.where(np.abs(frames[left] - np.arange(total)) < np.abs(frames[near] - np.arange(total)), left, near)
    pose = np.zeros((total, 136), np.float32)
    pose[:, 0:3] = 10 * FLIP * t_s[near]
    pose[:, 3:6] = rot_s[near]
    pose[:, 6:136] = pose_s[near]
    pelvis_world = j_s[:, 1] + t_s
    episodes_dir = args.run / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    np.savez(episodes_dir / f"episode_{episode:06d}.npz", pose=pose, scales=scales.astype(np.float32),
             shape=shape.astype(np.float32),
             object_rotation=np.broadcast_to(np.eye(3, dtype=np.float32), (total, 3, 3)).copy(),
             object_translation=np.broadcast_to(pelvis_world.mean(0).astype(np.float32), (total, 3)).copy(),
             object_scale=np.float32(1.0))
    cube_obj(episodes_dir / f"episode_{episode:06d}_object.obj")
    # depth_scale: Step 3 depth x this = world (body-scale) metres, for proxy_eval and Step 5
    np.savez(episodes_dir / f"episode_{episode:06d}_camera.npz", K=K, world_to_camera=WORLD_TO_CAMERA,
             image_size=np.array(IMAGE_SIZE), depth_scale=1 / k_moge if np.isfinite(k_moge) else 1.0)
    np.savez_compressed(args.run / "human_meta" / f"episode_{episode:06d}_series.npz", frames=frames,
                        t_raw=t, t_smooth=t_s, weight=w, vis=vis, moge_ratio=ratio, joint_dev=err)

    picks = np.unique(scored[np.linspace(0, len(scored) - 1, 8).astype(int)])
    lookup = {int(f): i for i, f in enumerate(frames)}
    overlay(episode, args.data, frames, lambda f: (verts_s[lookup[int(f)]] + t_s[lookup[int(f)]], K),
            masks, args.run / "overlays" / f"episode_{episode:06d}.jpg", picks)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="human_vm.py output folder")
    parser.add_argument("--run", type=Path, required=True, help="results folder to write")
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--depth", choices=["sam", "moge"], default="sam")
    parser.add_argument("--lam-t", type=float, default=20.0)
    parser.add_argument("--lam-r", type=float, default=4.0)
    parser.add_argument("--lam-p", type=float, default=4.0)
    parser.add_argument("--robust-iters", type=int, default=2)
    parser.add_argument("--cache", type=Path, default=REPO / "cache")
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--sample", type=Path,
                        default=REPO / "data/v2d_submission_kit/data/track_1_sample_submission.parquet")
    args = parser.parse_args()
    (args.run / "human_meta").mkdir(parents=True, exist_ok=True)
    body = MHRBody()
    conv = euler_convention(body)
    scored = scored_frames(args.sample)
    for e in parse_episodes(args.episodes):
        if not (args.raw / f"episode_{e:06d}.npz").exists():
            print(f"ep {e}: no raw output, skipped", flush=True)
            continue
        stats = process(e, args, body, conv, scored[e])
        (args.run / "human_meta" / f"episode_{e:06d}.json").write_text(json.dumps(stats, indent=1))
        print("ep " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()

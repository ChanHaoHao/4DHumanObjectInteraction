#!/usr/bin/env python3
"""Render a run as videos of the reconstructed person and object (runs on the Colab VM, GPU).

    python render_video.py --run <repo>/results/<run> [--episodes 0-29] [--cache <repo>/cache/d4rt]
        [--data /content/v2d/track_1] [--out <run>/videos] [--width 768] [--side-deg 60] [--side-dist 0.7]
        [--side-lift 0.4] [--max-frames N]

Per episode, over the fitted span (scored frames +- 15, as Steps 4-5), one mp4 at the video's 30 fps with two panels:
  left   the input frame with the human (blue) and object (orange) meshes shaded on top, through the run's K
  right  the same scene from a virtual camera orbited --side-deg about the floor normal around the actor (at
         --side-dist of the real camera's distance, raised --side-lift), over a floor grid (Step 3 floor plane x depth_scale, the body's metric scale)
Geometry is exactly what gets scored: MHR from the submission npz, the object mesh after the kit's face budget.
Meshes are drawn as dense surface samples through a GPU z-buffer (no rasterizer dependency), shaded by the
normal against the view ray; one-pixel holes are filled from their neighbours.
Writes <out>/episode_XXXXXX.mp4 (H.264 via ffmpeg).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_vm import episode_range, parse_episodes, scored_frames, video_path  # noqa: E402
from proxy_eval import DEFAULT_SAMPLE, MHRBody, load_episode, mesh_budget, surface_samples  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (1536, 1152)
HUMAN_RGB = np.array([110, 165, 235]) / 255
OBJECT_RGB = np.array([245, 150, 50]) / 255
ALPHA = 0.7
EMPTY = torch.iinfo(torch.int64).max


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """[T,V,3] -> unit area-weighted vertex normals."""
    a, b, c = (verts[:, faces[:, k]] for k in range(3))
    fn = torch.cross(b - a, c - a, dim=-1)
    vn = torch.zeros_like(verts).index_add_(1, faces.reshape(-1), fn.repeat_interleave(3, dim=1))
    return F.normalize(vn, dim=-1)


class Surface:
    """Fixed barycentric samples on a mesh; points/normals for any vertex positions."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, count: int, seed: int, device: str):
        chosen, w = surface_samples(vertices, faces, count, seed)
        self.faces = torch.as_tensor(faces, dtype=torch.long, device=device)
        self.tri = self.faces[torch.as_tensor(chosen, device=device)]  # [P,3]
        self.w = torch.as_tensor(w, dtype=torch.float32, device=device)

    def at(self, verts: torch.Tensor, normals: torch.Tensor):
        """verts, normals [V,3] -> points, normals [P,3]."""
        p = (verts[self.tri] * self.w[..., None]).sum(1)
        n = F.normalize((normals[self.tri] * self.w[..., None]).sum(1), dim=-1)
        return p, n


def splat(points, normals, colors, K, R, t, size):
    """Z-buffer of camera-frame samples. Returns (rgb [h,w,3] float, mask [h,w] bool) on the GPU."""
    w, h = size
    cam = points @ R.T + t
    n_cam = normals @ R.T
    z = cam[:, 2]
    front = z > 0.05
    cam, n_cam, col, z = cam[front], n_cam[front], colors[front], z[front]
    u = (cam[:, 0] / z * K[0, 0] + K[0, 2]).floor().long()
    v = (cam[:, 1] / z * K[1, 1] + K[1, 2]).floor().long()
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, cam, n_cam, col = u[inside], v[inside], z[inside], cam[inside], n_cam[inside], col[inside]
    idx = torch.arange(len(z), device=z.device)
    key = ((z * 1e4).long() << 22) | idx  # nearest sample wins; < 4M samples
    buf = torch.full((h * w,), EMPTY, dtype=torch.long, device=z.device)
    buf.scatter_reduce_(0, v * w + u, key, reduce="amin")
    hit = buf != EMPTY
    win = buf[hit] & ((1 << 22) - 1)
    ray = F.normalize(cam[win], dim=-1)
    shade = 0.3 + 0.7 * (n_cam[win] * ray).sum(-1).abs()
    rgb = torch.zeros((h * w, 3), device=z.device)
    rgb[hit] = col[win] * shade[:, None]
    rgb, mask = rgb.view(h, w, 3), hit.view(h, w)
    # fill pinholes: empty pixels with >= 5 of 8 neighbours drawn take their mean colour
    m = mask.float()[None, None]
    count = F.avg_pool2d(m, 3, 1, 1) * 9
    mean = F.avg_pool2d(rgb.permute(2, 0, 1)[None] * m, 3, 1, 1) * 9 / count.clamp(min=1)
    fill = (~mask) & (count[0, 0] >= 5)
    rgb[fill] = mean[0].permute(1, 2, 0)[fill]
    return rgb, mask | fill


def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray):
    """OpenCV camera (x right, y down, z forward) at position looking at target; returns R, t (world->camera)."""
    f = target - position
    f /= np.linalg.norm(f)
    down = -up - np.dot(-up, f) * f
    down /= np.linalg.norm(down)
    R = np.stack([np.cross(down, f), down, f])
    return R, -R @ position


def rotate_about(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    k = axis / np.linalg.norm(axis)
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * np.dot(k, v) * (1 - np.cos(a))


def floor_grid(img, K, R, t, n_up, h, center, half=3.0, step=0.5, color=(170, 170, 170)):
    """Lines of the floor plane n.p + h = 0 around center's foot point, projected and clipped to z > 0.1."""
    foot = center - (np.dot(n_up, center) + h) * n_up
    e1 = np.cross(n_up, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_up, e1)
    ticks = np.arange(-half, half + 1e-6, step)
    segments = [(foot + a * e1 - half * e2, foot + a * e1 + half * e2) for a in ticks]
    segments += [(foot - half * e1 + a * e2, foot + half * e1 + a * e2) for a in ticks]
    for p, q in segments:
        pts = np.linspace(p, q, 40) @ R.T + t
        ok = pts[:, 2] > 0.1
        uv = pts[:, :2] / np.maximum(pts[:, 2:], 1e-6) @ K[:2, :2].T + K[:2, 2]
        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[i + 1]).astype(int)),
                         color, 1, cv2.LINE_AA)


def label(img, text, y=24):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def render_episode(episode: int, args, body: MHRBody, budget, scored: np.ndarray):
    name = f"episode_{episode:06d}"
    episodes = args.run / "episodes"
    cam = np.load(episodes / f"{name}_camera.npz")
    K_full = cam["K"].astype(np.float64)
    depth_scale = float(cam["depth_scale"]) if "depth_scale" in cam else 1.0
    meta = json.loads((args.cache / name / "meta.json").read_text())
    total = meta["frames"]
    frames = episode_range(scored, total)
    if args.max_frames:
        frames = frames[:args.max_frames]
    ep = load_episode(episodes, episode, frames, budget, body)

    W = args.width
    H = round(W * IMAGE_SIZE[1] / IMAGE_SIZE[0])
    K = K_full.copy()
    K[:2] *= W / IMAGE_SIZE[0]
    dev = "cuda"
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)

    # side camera: orbit the original camera about the floor normal through the actor's median position
    n_up = np.asarray(meta["floor_normal"], float)
    n_up /= np.linalg.norm(n_up)
    h_floor = meta["camera_height_m"] * depth_scale
    center = np.median(ep["verts"].mean(1), 0)
    pos = center + args.side_dist * rotate_about(-center, n_up, args.side_deg) + args.side_lift * n_up
    R_side, t_side = look_at(pos, center, n_up)
    views = [(np.eye(3), np.zeros(3)), (R_side, t_side)]
    views_t = [(torch.as_tensor(R, dtype=torch.float32, device=dev), torch.as_tensor(t, dtype=torch.float32, device=dev))
               for R, t in views]

    faces_h = torch.as_tensor(body.faces.astype(np.int64), device=dev)
    human = Surface(ep["verts"][0], body.faces, args.human_samples, 0, dev)
    mv, mf = ep["mesh_vertices"], ep["mesh_faces"].astype(np.int64)
    obj = Surface(mv, mf, args.object_samples, 1, dev)
    mv_t = torch.as_tensor(mv, dtype=torch.float32, device=dev)
    mn_t = vertex_normals(mv_t[None], torch.as_tensor(mf, device=dev))[0]
    colors = torch.cat([torch.as_tensor(HUMAN_RGB, dtype=torch.float32, device=dev).expand(args.human_samples, 3),
                        torch.as_tensor(OBJECT_RGB, dtype=torch.float32, device=dev).expand(args.object_samples, 3)])

    side_bg = np.full((H, W, 3), 245, np.uint8)
    floor_grid(side_bg, K, R_side, t_side, n_up, h_floor, center)
    scored_set = set(int(f) for f in scored)

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{name}.mp4"
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{2 * W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                              stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(video_path(args.data, episode)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    chunk = 64
    for a in range(0, len(frames), chunk):
        verts = torch.as_tensor(ep["verts"][a:a + chunk], dtype=torch.float32, device=dev)
        normals = vertex_normals(verts, faces_h)
        for i in range(len(verts)):
            f = int(frames[a + i])
            ok, bgr = cap.read()
            rgb_in = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA) if ok \
                else np.zeros((H, W, 3), np.uint8)
            hp, hn = human.at(verts[i], normals[i])
            Ro = torch.as_tensor(ep["rotation"][a + i], dtype=torch.float32, device=dev)
            to = torch.as_tensor(ep["translation"][a + i], dtype=torch.float32, device=dev)
            op, on = obj.at(ep["scale"] * mv_t @ Ro.T + to, mn_t @ Ro.T)
            pts, nrm = torch.cat([hp, op]), torch.cat([hn, on])
            panels = []
            for (R, t), bg, blend in zip(views_t, (rgb_in, side_bg), (ALPHA, 1.0)):
                col, mask = splat(pts, nrm, colors, Kt, R, t, (W, H))
                base = torch.as_tensor(bg, dtype=torch.float32, device=dev) / 255
                m = mask[..., None].float() * blend
                panels.append(((base * (1 - m) + col * m) * 255).clamp(0, 255).byte().cpu().numpy())
            label(panels[0], f"ep {episode}  frame {f}" + ("" if f in scored_set else "  (unscored)"))
            label(panels[0], args.run.name, 46)
            label(panels[1], f"side view, {args.side_deg:+.0f} deg about the floor normal")
            ffmpeg.stdin.write(np.ascontiguousarray(np.concatenate(panels, 1)).tobytes())
    cap.release()
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {name}")
    print(f"ep {episode}: {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--cache", type=Path, default=REPO / "cache" / "d4rt", help="for meta.json (frames, floor)")
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/videos")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--width", type=int, default=768, help="per panel")
    parser.add_argument("--side-deg", type=float, default=60.0)
    parser.add_argument("--side-dist", type=float, default=0.7, help="side camera distance / the real camera's")
    parser.add_argument("--side-lift", type=float, default=0.4, help="m, side camera raised along the floor normal")
    parser.add_argument("--human-samples", type=int, default=400_000)
    parser.add_argument("--object-samples", type=int, default=200_000)
    parser.add_argument("--max-frames", type=int, default=0, help="test: only the first N frames")
    args = parser.parse_args()
    args.out = args.out or args.run / "videos"
    body = MHRBody(device="cuda")
    budget = mesh_budget(args.sample)
    scored = scored_frames(args.sample)
    for e in parse_episodes(args.episodes):
        with torch.no_grad():
            render_episode(e, args, body, budget, scored[e])
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

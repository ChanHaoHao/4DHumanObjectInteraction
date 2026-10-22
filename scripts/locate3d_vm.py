#!/usr/bin/env python3
"""Ground a phrase in the D4RT scene with Meta's Locate 3D. Runs on the Colab VM, in the Locate 3D venv.

    cd /content/locate-3d && /content/l3d_env/bin/python locate3d_vm.py --cache DIR --videos DIR --episodes 19
        --phrase "paper roller" --out OUT [--stride 10] [--scale 2] [--max-depth 10] [--points 30000]

Every --stride-th frame is lifted to a featurized point cloud the way Locate 3D's preprocessing does it
(SAM-H masks + CLIP ViT-L/14 per mask, DINOv2-B per pixel, 5 cm voxels), from the D4RT cache:
depth [T, 288, 384] upsampled --scale x (nearest) with the video frame resized to match, K from camera.npz.
The camera is static, so cam_to_world only puts the floor at z = 0 with z up (ScanNet convention), from
meta.json floor_normal and camera_height_m. Each keyframe is a single-view scene; no fusion over time
because the object moves.
Writes to OUT:
  <ep>_locate3d.npz   frames (F,), boxes (F, 6) world xyzxyz (nan = no instance), conf (F,),
                      centroid_cam (F, 3) mean of the best instance's mask points in camera coords, cam_to_world
  <ep>_locate3d.json  phrase, per-keyframe instances (text, confidence, bbox), timing
  <ep>_locate3d.mp4   keyframes with the best box projected (green, confidence), half resolution
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch


class HarmonicEmbedding(torch.nn.Module):
    """pytorch3d's HarmonicEmbedding (defaults: logspace, omega_0 1, append_input), the only pytorch3d piece used."""

    def __init__(self, n_harmonic_functions: int = 6):
        super().__init__()
        self.register_buffer("_frequencies", 2.0 ** torch.arange(n_harmonic_functions, dtype=torch.float32),
                             persistent=False)

    def forward(self, x):
        embed = (x[..., None] * self._frequencies).reshape(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos(), x), dim=-1)


def stub_imports():
    # shortcut: pytorch3d needs a 20+ min source build and open3d is imported but unused; install them if a later
    # Locate 3D code path needs more than HarmonicEmbedding.
    for name in ("pytorch3d", "pytorch3d.renderer", "pytorch3d.renderer.implicit"):
        sys.modules[name] = types.ModuleType(name)
    mod = types.ModuleType("pytorch3d.renderer.implicit.harmonic_embedding")
    mod.HarmonicEmbedding = HarmonicEmbedding
    sys.modules[mod.__name__] = mod
    sys.modules["open3d"] = types.ModuleType("open3d")


def world_frame(floor_normal, height: float) -> np.ndarray:
    """cam_to_world (4x4) with z along the floor normal (up), x from the camera x, origin on the floor below."""
    z = np.asarray(floor_normal, float)
    z /= np.linalg.norm(z)
    if z[1] > 0:  # camera y points down, so up must have negative y
        z = -z
    x = np.array([1.0, 0, 0]) - z[0] * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z])
    T[2, 3] = height
    return T


def box_corners(b: np.ndarray) -> np.ndarray:
    return np.array([[b[0 + 3 * i], b[1 + 3 * j], b[2 + 3 * k]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def draw_box(img, box_world, T, K, color, label):
    c = (box_corners(box_world) - T[:3, 3]) @ T[:3, :3]  # world -> camera
    if (c[:, 2] <= 0.05).any():
        return
    uv = (c @ K.T)[:, :2] / c[:, 2:]
    for a, b in EDGES:
        cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), color, 2)
    cv2.putText(img, label, tuple(int(v) for v in uv.min(0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="D4RT cache root with episode_XXXXXX/{depth,camera}.npz, meta.json")
    p.add_argument("--videos", required=True)
    p.add_argument("--episodes", required=True, help="comma list")
    p.add_argument("--phrase", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--points", type=int, default=30000)
    args = p.parse_args()

    stub_imports()
    from omegaconf import OmegaConf
    from models.locate_3d import Locate3D, downsample
    from preprocessing.pointcloud_featurizer import FeatureLifter3D

    lifters = {}
    for kind in ("clip", "dino"):
        cfg = OmegaConf.load(f"preprocessing/config/{kind}.yaml")
        cfg.feature_slam.max_depth = args.max_depth  # default 4 m is for indoor scans
        lifters[kind] = FeatureLifter3D(cfg)
    model = Locate3D.from_pretrained("facebook/locate-3d")
    model.eval()  # its train() override returns None, so no chaining

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for e in [int(x) for x in args.episodes.split(",")]:
        n = f"episode_{e:06d}"
        src = Path(args.cache) / n
        dz, cz = np.load(src / "depth.npz"), np.load(src / "camera.npz")
        meta = json.loads((src / "meta.json").read_text())
        T = world_frame(meta["floor_normal"], meta["camera_height_m"])
        frames_all = dz["frames"]
        keep = np.arange(0, len(frames_all), args.stride)
        depth = dz["depth"][keep].astype(np.float32)
        h, w = depth.shape[1] * args.scale, depth.shape[2] * args.scale
        K = cz["K"].astype(np.float64).copy()
        K[:2] *= w / cz["image_size"].max()  # frames are landscape
        assert abs(K[0, 2] * 2 - w) < 2 and abs(K[1, 2] * 2 - h) < 2, (K, w, h)

        wanted = set(frames_all[keep].tolist())
        cap, rgb, f = cv2.VideoCapture(f"{args.videos}/{n}.mp4"), {}, -1
        while len(rgb) < len(wanted):
            ok, im = cap.read()
            if not ok:
                break
            f += 1
            if f in wanted:
                rgb[f] = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

        rows, boxes, conf, cent, vis = [], [], [], [], []
        t0 = time.time()
        for i, f in enumerate(frames_all[keep].tolist()):
            d = torch.from_numpy(cv2.resize(depth[i], (w, h), interpolation=cv2.INTER_NEAREST))[None]
            views = {
                "rgb": torch.from_numpy(cv2.cvtColor(rgb[f], cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255,
                "depth_zbuffer": d,
                "cam_to_world": torch.from_numpy(T).float()[None],
                "cam_K": torch.from_numpy(K).float()[None],
            }
            clip_pc = lifters["clip"].lift_frames(views)
            torch.manual_seed(0)
            dino_pc = lifters["dino"].lift_frames(views)
            pc = {"points": dino_pc["points_reduced"], "rgb": dino_pc["rgb_reduced"],
                  "features_clip": clip_pc["features_reduced"], "features_dino": dino_pc["features_reduced"]}
            torch.manual_seed(0)
            pc = downsample(pc, args.points)
            inst = sorted(model.inference(pc, args.phrase), key=lambda r: -float(r["confidence"]))
            rows.append({"frame": f, "points": int(len(pc["points"])), "instances": [
                {"text": r["text"], "confidence": round(float(r["confidence"]), 4),
                 "bbox": [round(float(v), 4) for v in r["bbox"].tolist()]} for r in inst]})
            img = rgb[f].copy()
            if inst:
                best = inst[0]
                b = best["bbox"].float().cpu().numpy()
                pts = pc["points"][best["mask"] > 0.5].cpu().numpy()
                boxes.append(b), conf.append(float(best["confidence"]))
                cent.append((pts.mean(0) - T[:3, 3]) @ T[:3, :3] if len(pts) else np.full(3, np.nan))
                for r in inst[1:3]:
                    draw_box(img, r["bbox"].float().cpu().numpy(), T, K, (0, 165, 255), f"{float(r['confidence']):.2f}")
                draw_box(img, b, T, K, (0, 255, 0), f"{conf[-1]:.2f}")
            else:
                boxes.append(np.full(6, np.nan)), conf.append(0.0), cent.append(np.full(3, np.nan))
            cv2.putText(img, f"{f} {args.phrase}" + ("" if inst else " (none)"), (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0) if inst else (0, 0, 255), 2)
            vis.append(img)
            print(f"ep {e} frame {f} instances {len(inst)} conf {conf[-1]:.3f} "
                  f"{(time.time() - t0) / (i + 1):.1f}s/frame", flush=True)

        raw = out / f"{n}_raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 3, (w, h))
        for img in vis:
            writer.write(img)
        writer.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "26", str(out / f"{n}_locate3d.mp4")], check=True)
        raw.unlink()
        np.savez_compressed(out / f"{n}_locate3d.npz", frames=frames_all[keep], boxes=np.array(boxes, np.float32),
                            conf=np.array(conf, np.float32), centroid_cam=np.array(cent, np.float32), cam_to_world=T)
        found = np.array(conf) > 0
        summary = {"phrase": args.phrase, "stride": args.stride, "size": [w, h], "keyframes": len(rows),
                   "keyframes_with_instance": int(found.sum()),
                   "median_conf": round(float(np.median(np.array(conf)[found])), 4) if found.any() else None,
                   "sec_per_frame": round((time.time() - t0) / max(len(rows), 1), 2), "frames": rows}
        (out / f"{n}_locate3d.json").write_text(json.dumps(summary, indent=1))
        print("done", n, json.dumps({k: v for k, v in summary.items() if k != "frames"}), flush=True)


if __name__ == "__main__":
    assert torch.allclose(HarmonicEmbedding(2)(torch.tensor([[1.0]])),
                          torch.tensor([[math.sin(1), math.sin(2), math.cos(1), math.cos(2), 1.0]]))
    assert np.allclose(world_frame([0, -1, 0], 1.5) @ [0, 1.5, 3, 1], [0, 3, 0, 1])  # floor point 1.5 m below, 3 m ahead
    main()

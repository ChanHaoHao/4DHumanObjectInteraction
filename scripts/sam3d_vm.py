#!/usr/bin/env python3
"""SAM 3D Objects meshes + layouts for the object crops (e2e_vm.py's mesh stage with --sam3d; A100, >= 32 GB).

    python sam3d_vm.py --stage inputs --episodes 0,1 --data D --cache C --human H --raw R --out IN
    python sam3d_vm.py --stage infer  --episodes 0,1 --inputs IN --out S3D        # SAM 3D's venv (/content/s3denv)
    python sam3d_vm.py --stage import --episodes 0,1 --s3d S3D --raw R --layout L

inputs: for each crop in R/crops/episode_E.json (object_vm.py --stage crops), IN/episode_E_k.{jpg, _mask.png, npz}:
the RGB frame, the object mask (full res) and the body-scale pointmap (OpenCV camera, 1/4 res, cache depth x the
human stage's depth_scale, as object_fit.py sees it).
infer: S3D/episode_E_k.glb and .json: R, t, s with x_cam = s R v + t for the vertices v of
trimesh.load(glb, force="mesh") (metres at the body's scale), plus checks: the glb-vs-gaussian axis fit and how well
the posed mesh projects into the mask.
import: R/meshes/episode_E_k.obj (+ _hi.ply) cleaned like object_vm.py's meshes, and L/episode_E.json:
[{k, frame, R, t, s}] re-expressed for the normalized mesh (x = s R v + t with v = L p + c -> s L, R, t + s R c),
object_fit.py --layout's start poses.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SAM3DO = Path("/content/sam3do")
SCRIPTS = Path(__file__).resolve().parent
# the 24 proper axis rotations: the glb's local frame vs the gaussians' (layout is defined on the latter)
AXES = [m for m in (np.diag(s) @ np.eye(3)[list(p)] for p in
                    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                    for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3))
        if np.linalg.det(m) > 0]


def umeyama(src, dst):
    """s, R, t minimizing |dst - (s R src + t)| for corresponding rows."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (a ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def stage_inputs(args, episodes):
    sys.path.insert(0, str(SCRIPTS))
    from object_vm import read_frame, video_path

    args.out.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        name = f"episode_{e:06d}"
        crops = json.loads((args.raw / "crops" / f"{name}.json").read_text())
        cam = np.load(args.human / f"{name}_camera.npz")
        K, ds = cam["K"], float(cam["depth_scale"])
        m, d = np.load(args.cache / name / "masks.npz"), np.load(args.cache / name / "depth.npz")
        mf, df = list(m["frames"]), list(d["frames"])
        obj, depth = m["object"], d["depth"]
        for c in crops:
            f, k = c["frame"], c["k"]
            bgr = read_frame(video_path(args.data, e), f)
            H, W = bgr.shape[:2]
            mask = cv2.resize(obj[mf.index(f)].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            z = depth[df.index(f)].astype(np.float32) * ds
            h4, w4 = z.shape
            u, v = np.meshgrid((np.arange(w4) + 0.5) * W / w4, (np.arange(h4) + 0.5) * H / h4)
            pts = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
            pts[z <= 0] = np.nan
            stem = args.out / f"{name}_{k}"
            cv2.imwrite(f"{stem}.jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{stem}_mask.png", mask * 255)
            np.savez_compressed(f"{stem}.npz", pointmap=pts.astype(np.float32), K=K, depth_scale=ds, frame=f)
            print(e, k, f, int(mask.sum()), round(float(np.nanmedian(z[cv2.resize(mask, (w4, h4)) > 0])), 2), flush=True)


def stage_infer(args, episodes):
    import torch
    import trimesh
    from scipy.spatial import cKDTree

    sys.path.insert(0, str(SAM3DO / "notebook"))
    from inference import Inference
    from pytorch3d.transforms import Transform3d
    from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
    from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer

    args.out.mkdir(parents=True, exist_ok=True)
    stems = [p.stem for e in episodes for p in sorted(args.inputs.glob(f"episode_{e:06d}_*.jpg"))]
    inf = Inference(str(SAM3DO / "checkpoints/hf/pipeline.yaml"), compile=False)
    conv = Transform3d().rotate(camera_to_pytorch3d_camera(device="cuda").rotation).to("cuda")  # OpenCV -> p3d
    for stem in stems:
        if (args.out / f"{stem}.json").exists():
            continue
        t0 = time.time()
        rgb = cv2.cvtColor(cv2.imread(str(args.inputs / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(args.inputs / f"{stem}_mask.png"), 0) > 127
        z = np.load(args.inputs / f"{stem}.npz")
        K = z["K"]
        pm = torch.from_numpy(z["pointmap"]).float().cuda()
        pm = conv.transform_points(pm.reshape(-1, 3)).reshape(pm.shape)
        o = inf(rgb, mask, seed=42, pointmap=pm)

        def posed(local):  # local (N, 3) -> OpenCV camera, the way make_scene places the gaussians
            p = SceneVisualizer.object_pointcloud(points_local=torch.as_tensor(local, dtype=torch.float32, device="cuda")[None],
                                                  quat_l2c=o["rotation"], trans_l2c=o["translation"], scale_l2c=o["scale"])
            return conv.inverse().transform_points(p.points_list()[0]).cpu().numpy()

        o["glb"].export(args.out / f"{stem}.glb")
        mesh = trimesh.load(args.out / f"{stem}.glb", force="mesh")
        V = np.asarray(mesh.vertices)
        g = posed(o["gaussian"][0].get_xyz.detach().cpu().numpy())
        tree = cKDTree(g)
        size = float(np.ptp(g, 0).max())
        sub = V[np.random.default_rng(0).choice(len(V), min(len(V), 5000), replace=False)]
        fits = [(float(tree.query(posed(sub @ A.T))[0].mean()) / size, i) for i, A in enumerate(AXES)]
        err, best = min(fits)
        A = AXES[best]
        s, R, t = umeyama(V, posed(V @ A.T))  # exact: posed() is a similarity
        x = s * V @ R.T + t
        uv = x[:, :2] / x[:, 2:] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        H, W = mask.shape
        ok = (x[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        ui = uv[ok].astype(int)
        near = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        splat = np.zeros_like(mask, np.uint8)
        splat[ui[:, 1], ui[:, 0]] = 1
        splat = cv2.dilate(splat, np.ones((9, 9), np.uint8)) > 0
        rec = {"stem": stem, "frame": int(z["frame"]), "R": R.tolist(), "t": t.tolist(), "s": s,
               "glb_axes": A.tolist(), "axes_err_rel": round(err, 4),
               "axes_runner_up_rel": round(sorted(fits)[1][0], 4),
               "size_m": round(size, 4), "depth_m": round(float(np.median(x[:, 2])), 3),
               "in_mask": round(float(near[ui[:, 1], ui[:, 0]].mean()) if len(ui) else 0.0, 3),
               "mask_covered": round(float(splat[mask].mean()), 3),
               "seconds": round(time.time() - t0, 1)}
        (args.out / f"{stem}.json").write_text(json.dumps(rec))
        print(json.dumps({k: v for k, v in rec.items() if k not in ("R", "t", "glb_axes")}), flush=True)


def stage_import(args, episodes):
    import trimesh

    sys.path.insert(0, str(SCRIPTS))
    from object_vm import HI_FACES, MAX_FACES, decimate, main_part, normalize

    (args.raw / "meshes").mkdir(parents=True, exist_ok=True)
    args.layout.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        index = args.raw / "crops" / f"episode_{e:06d}.json"
        crops = json.loads(index.read_text()) if index.exists() else []
        ks, layout = [], []
        for rec_path in sorted(args.s3d.glob(f"episode_{e:06d}_*.json")):
            rec = json.loads(rec_path.read_text())
            k = int(rec_path.stem.split("_")[-1])
            mesh = trimesh.load(rec_path.with_suffix(".glb"), force="mesh")
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            mesh = main_part(mesh)
            lo, hi = mesh.bounds
            c, L = (lo + hi) / 2, float((hi - lo).max())
            mesh = normalize(mesh)
            decimate(mesh, HI_FACES).export(args.raw / "meshes" / f"episode_{e:06d}_{k}_hi.ply")
            decimate(mesh, MAX_FACES).export(args.raw / "meshes" / f"episode_{e:06d}_{k}.obj")
            R, t, s = np.asarray(rec["R"]), np.asarray(rec["t"]), float(rec["s"])
            layout.append({"k": k, "frame": rec["frame"], "R": R.tolist(), "t": (t + s * R @ c).tolist(), "s": s * L,
                           "in_mask": rec["in_mask"], "mask_covered": rec["mask_covered"]})
            ks.append(k)
            print(e, k, rec["frame"], round(s * L, 3), rec["in_mask"], rec["mask_covered"], flush=True)
        if not ks:  # no SAM 3D output: leave the episode unmeshed, so e2e_vm records it as failed
            continue
        index.write_text(json.dumps([cr for cr in crops if cr["k"] in ks] or [{"k": k} for k in ks]))
        (args.layout / f"episode_{e:06d}.json").write_text(json.dumps(layout))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["inputs", "infer", "import"], required=True)
    p.add_argument("--episodes", required=True, help="comma-separated")
    p.add_argument("--data", type=Path, help="inputs: dataset root (videos)")
    p.add_argument("--cache", type=Path, help="inputs: cache with masks.npz, depth.npz")
    p.add_argument("--human", type=Path, help="inputs: human stage episodes (episode_E_camera.npz)")
    p.add_argument("--raw", type=Path, help="inputs/import: object_raw (crops/, meshes/)")
    p.add_argument("--inputs", type=Path, help="infer: the inputs stage's --out")
    p.add_argument("--s3d", type=Path, help="import: the infer stage's --out")
    p.add_argument("--layout", type=Path, help="import: where episode_E.json layouts go")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    episodes = [int(e) for e in args.episodes.split(",")]
    {"inputs": stage_inputs, "infer": stage_infer, "import": stage_import}[args.stage](args, episodes)


if __name__ == "__main__":
    main()

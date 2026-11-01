"""Laptop (geometry only): carry the SAM 3D layouts of lay's meshes over to the web-demo meshes (sam_3d_mesh/).

Both meshes are normalized like import_sam3d.py / sam3d_vm.py --stage import (merge_vertices, main_part, normalize). A similarity
u = a Q w + c maps the normalized demo mesh w onto lay's normalized mesh u (ICP from the 24 axis rotations, best
chamfer), so the layout x = s R u + t becomes x = (s a) (R Q) w + (s R c + t).

    python align_demo.py <out_layout_dir>        # writes episode_E.json like sam3d_vm.py --stage import, plus align stats
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path("/home/byebyehello/Desktop/SP/4DHumanObjectInteraction")
AXES = [m for m in (np.diag(s) @ np.eye(3)[list(p)] for p in
                    [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                    for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3))
        if np.linalg.det(m) > 0]


def main_part(mesh, margin: float = 0.2):  # object_vm.main_part (its module imports pull in torch/pandas)
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    main = max(parts, key=lambda m: m.area)
    lo, hi = main.bounds
    pad = margin * (hi - lo).max()
    keep = [m for m in parts if m is main or (np.all(m.bounds[0] >= lo - pad) and np.all(m.bounds[1] <= hi + pad))]
    return mesh if len(keep) == len(parts) else trimesh.util.concatenate(keep)


def normalize(mesh):
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    mesh.apply_scale(1.0 / float((hi - lo).max()))
    return mesh


def align(w_mesh, u_mesh, n=3000):
    """Best similarity (4x4, u = M w) over ICP runs from the 24 axis rotations; (M, cost, runner-up cost).
    Point clouds and KD-trees (mesh closest-point queries took ~7 min per crop)."""
    w, u = w_mesh.sample(n, seed=0), u_mesh.sample(n, seed=1)
    tw = cKDTree(w)
    runs = []
    for A in AXES:
        init = np.eye(4)
        init[:3, :3] = A
        M, moved, cost = trimesh.registration.icp(w, u, initial=init, scale=True, reflection=False, max_iterations=60)
        a = np.cbrt(np.linalg.det(M[:3, :3]))
        back = tw.query(trimesh.transform_points(u, np.linalg.inv(M)))[0] * a  # in u's units
        runs.append((float(cost) + float(back.mean()), M))  # both directions: a shrunk w hiding inside u is not a fit
    runs.sort(key=lambda r: r[0])
    return runs[0][1], runs[0][0], runs[1][0]


def demo_mesh(e, k):
    mesh = trimesh.load(ROOT / f"sam_3d_mesh/object_{e:06d}_{k}.glb", force="mesh")
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return normalize(main_part(mesh))


def self_check():
    u = trimesh.util.concatenate([trimesh.creation.box(extents=[1.0, 0.5, 0.2]),  # asymmetric: one corner block
                                  trimesh.creation.box(extents=[0.2, 0.2, 0.2], transform=trimesh.transformations.translation_matrix([0.4, 0.2, 0.15]))])
    true = np.eye(4)
    true[:3, :3] = 0.8 * trimesh.transformations.rotation_matrix(0.3, [0, 0, 1])[:3, :3] @ AXES[7]
    true[:3, 3] = [0.05, -0.02, 0.01]
    w = u.copy().apply_transform(np.linalg.inv(true))
    M, cost, _ = align(w, u)
    assert np.allclose(trimesh.transform_points(w.vertices, M), u.vertices, atol=0.02), cost


def main():
    self_check()
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for e in range(30):
        layout = json.loads((ROOT / f"results/lay/layout/episode_{e:06d}.json").read_text())
        new = []
        for c in layout:
            k = c["k"]
            if not (ROOT / f"sam_3d_mesh/object_{e:06d}_{k}.glb").exists():
                continue
            u = trimesh.load(ROOT / f"results/lay/object_raw/meshes/episode_{e:06d}_{k}.obj", force="mesh")
            M, cost, second = align(demo_mesh(e, k), u)
            a = float(np.cbrt(np.linalg.det(M[:3, :3])))
            assert a > 0, (e, k, a)  # a mirrored fit would give the layout a negative scale
            Q, cvec = M[:3, :3] / a, M[:3, 3]
            R, t, s = np.asarray(c["R"]), np.asarray(c["t"]), float(c["s"])
            new.append({"k": k, "frame": c["frame"], "R": (R @ Q).tolist(), "t": (s * R @ cvec + t).tolist(), "s": s * a,
                        "align_cost": round(cost, 4), "align_runner_up": round(second, 4)})
            stats.append((e, k, round(cost, 4), round(second, 4), round(a, 3)))
            print(e, k, "cost", round(cost, 4), "runner-up", round(second, 4), "scale", round(a, 3), flush=True)
        (out / f"episode_{e:06d}.json").write_text(json.dumps(new))
    costs = [s[2] for s in stats]
    print("crops", len(stats), "median cost", round(float(np.median(costs)), 4),
          "ambiguous (runner-up < 1.2x)", sum(s[3] < 1.2 * s[2] for s in stats))


if __name__ == "__main__":
    main()

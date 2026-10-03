#!/usr/bin/env python3
"""Step 4 GPU stage (runs on the Colab VM): per-frame SAM 3D Body on the actor, native MHR output.

    python human_vm.py --stage infer [--episodes 0-29] [--model dinov3] [--no-mask] [--tag NAME]
    python human_vm.py --stage check --episodes 6      # SAM 3D Body vs the official MHR forward

The actor's box and mask come from the Step 3 cache (<repo>/cache/episode_XXXXXX/masks.npz, human =
the chosen actor track) and the intrinsics from its camera.npz, so no detector or FOV estimator runs.
Frames from PAD before the first scored frame to PAD after the last one are processed; the rest of
the clip is unscored (post-processing pads it with the nearest frame).

Several workers can run at once (the GPU is mostly idle in one): each claims an episode with
<out>/<tag>/episode_XXXXXX.lock (delete stale locks after a crash).
Writes <out>/<tag>/episode_XXXXXX.npz (skipped when present, so reruns resume), per processed frame:
  frames [N]                  video frame indices
  mhr_model_params [N,204]    MHR model params (translation slot 0:3 is zero; the body sits at cam_t)
  shape [N,45], expr [N,72]   identity shape and face expression (expression is dropped later)
  scale_params [N,28]         PCA coefficients behind mhr_model_params[136:204]
  cam_t [N,3]                 metres, OpenCV camera frame: vertices_cam = diag(1,-1,-1) MHR(...)/100 + cam_t
                              (SAM 3D Body flips y, z of MHR's native output: its body is in the Track 1 frame)
  kp2d [N,70,2], kp3d [N,70,3]   MHR70 keypoints (pixels; Track 1 frame, metres, before cam_t)
  joints [N,127,3]            MHR skeleton joints (Track 1 frame, metres, before cam_t)
  bbox, lhand_bbox, rhand_bbox [N,4]   prompt box and the hand decoder's boxes (pixels)
  mask_area [N]               actor mask pixels at 1/2 resolution (0 = no mask; box interpolated)
  K [3,3], model, use_mask

The MHR inside SAM 3D Body is loaded from the official v1.0.1 TorchScript model (the one the host
scores with), passed as mhr_path, so its vertices are the evaluator's. `--stage check` verifies that
diag(1,-1,-1) MHR(shape, params with [0:3] = 10 diag(1,-1,-1) cam_t, expr) / 100 = pred_vertices + cam_t,
i.e. with world = camera frame, the submission's pose[0:3] is 10 diag(1,-1,-1) cam_t.
Setup on the VM: clone facebookresearch/sam-3d-body to /content/sam-3d-body and pip install its
dependencies (not detectron2, not the `mhr` package: without it the TorchScript model is used).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
EPISODE_FRAME = 999999  # row_id frame of the per-episode constants (v2dlb.mhr_submission)
PAD = 15  # frames processed beyond the scored span on each side (smoothing context)
MODELS = {"dinov3": "facebook/sam-3d-body-dinov3", "vith": "facebook/sam-3d-body-vith"}
MASK_STEP = 2  # cache masks are at 1/2 resolution
FLIP = np.array([1.0, -1.0, -1.0], np.float32)  # MHR native <-> Track 1 / SAM 3D Body frame
KEYS = ["mhr_model_params", "shape", "expr", "scale_params", "cam_t", "kp2d", "kp3d", "joints",
        "bbox", "lhand_bbox", "rhand_bbox"]


def parse_episodes(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def scored_frames(sample: Path) -> dict[int, np.ndarray]:
    parts = pd.read_parquet(sample, columns=["row_id"]).row_id.str.split("_", expand=True)
    episodes, frames = parts[1].astype(int), parts[2].astype(int)
    keep = frames != EPISODE_FRAME
    return {int(e): np.unique(g.to_numpy()) for e, g in frames[keep].groupby(episodes[keep])}


def video_path(data: Path, episode: int) -> Path:
    return data / "videos/chunk-000/observation.images.exo_camera" / f"episode_{episode:06d}.mp4"


def save(path: Path, **arrays):
    """savez_compressed via a temporary name, so a concurrent download never sees a partial file."""
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_estimator(model: str, mhr_path: Path):
    sys.path.insert(0, "/content/sam-3d-body")
    from huggingface_hub import snapshot_download
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    local = Path(snapshot_download(MODELS[model]))
    net, cfg = load_sam_3d_body(str(local / "model.ckpt"), device="cuda", mhr_path=str(mhr_path))
    return SAM3DBodyEstimator(net, cfg)


def actor_boxes(human: np.ndarray, frames: np.ndarray, size=(1536, 1152)) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution xyxy box of the actor mask per frame; frames without a mask get the nearest box."""
    boxes = np.full((len(frames), 4), np.nan)
    area = np.zeros(len(frames), np.int64)
    for i, f in enumerate(frames):
        m = human[f]
        area[i] = m.sum()
        if area[i] < 20:
            continue
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        boxes[i] = np.array([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1]) * MASK_STEP
    ok = np.flatnonzero(~np.isnan(boxes[:, 0]))
    if not ok.size:
        raise RuntimeError("no actor mask in the processed range")
    for i in np.flatnonzero(np.isnan(boxes[:, 0])):
        boxes[i] = boxes[ok[np.abs(ok - i).argmin()]]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, size[0])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, size[1])
    return boxes.astype(np.float32), area


def episode_range(scored: np.ndarray, num_frames: int) -> np.ndarray:
    return np.arange(max(0, scored[0] - PAD), min(num_frames - 1, scored[-1] + PAD) + 1)


def estimate(estimator, rgb, box, mask, cam_int) -> dict:
    """One person; SAM 3D Body prints a few lines per call, which would flood the job log."""
    with contextlib.redirect_stdout(io.StringIO()):
        return estimator.process_one_image(rgb, bboxes=box[None], masks=mask, cam_int=cam_int)[0]


def run_frames(estimator, video: Path, frames: np.ndarray, boxes, human, K, use_mask: bool, log_every=100):
    """SAM 3D Body on each wanted frame of the video, prompted with the actor's box (and mask)."""
    out = {k: [] for k in KEYS}
    wanted = {int(f): i for i, f in enumerate(frames)}
    cam_int = torch.as_tensor(K, dtype=torch.float32)[None]
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    f, t0, done = int(frames[0]), time.time(), 0
    while done < len(frames):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"video ended at frame {f}, {len(frames) - done} frames short")
        if f in wanted:
            i = wanted[f]
            rgb = bgr[:, :, ::-1].copy()
            mask = None
            if use_mask and human[f].any():
                mask = human[f].repeat(MASK_STEP, 0).repeat(MASK_STEP, 1)[None].astype(np.uint8)
            r = estimate(estimator, rgb, boxes[i], mask, cam_int)
            out["mhr_model_params"].append(r["mhr_model_params"])
            out["shape"].append(r["shape_params"])
            out["expr"].append(r["expr_params"])
            out["scale_params"].append(r["scale_params"])
            out["cam_t"].append(r["pred_cam_t"])
            out["kp2d"].append(r["pred_keypoints_2d"])
            out["kp3d"].append(r["pred_keypoints_3d"])
            out["joints"].append(r["pred_joint_coords"])
            out["bbox"].append(r["bbox"])
            out["lhand_bbox"].append(r.get("lhand_bbox", np.full(4, np.nan)))
            out["rhand_bbox"].append(r.get("rhand_bbox", np.full(4, np.nan)))
            done += 1
            if done % log_every == 0:
                print(f"  {done}/{len(frames)} frames, {done / (time.time() - t0):.2f} fps", flush=True)
        f += 1
    cap.release()
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


def stage_infer(args, episodes, scored):
    estimator = load_estimator(args.model, args.mhr)
    # process_one_image empties the CUDA cache on every call, which stalls the GPU (~1.1 fps on an L4)
    torch.cuda.empty_cache = lambda: None
    dst = args.out / args.tag
    dst.mkdir(parents=True, exist_ok=True)
    for e in episodes:
        path = dst / f"episode_{e:06d}.npz"
        lock = path.with_suffix(".lock")  # several workers share the GPU; each episode goes to one
        if path.exists():
            continue
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        try:
            t0 = time.time()
            folder = args.cache / f"episode_{e:06d}"
            human = np.load(folder / "masks.npz")["human"]
            K = np.load(folder / "camera.npz")["K"]
            frames = episode_range(scored[e], len(human))
            boxes, area = actor_boxes(human, frames)
            out = run_frames(estimator, video_path(args.data, e), frames, boxes, human, K, not args.no_mask)
            save(path, frames=frames, mask_area=area, K=K, model=args.model, use_mask=not args.no_mask, **out)
            print(f"ep {e}: {len(frames)} frames, {len(frames) / (time.time() - t0):.2f} fps, "
                  f"cam_t z {np.median(out['cam_t'][:, 2]):.2f} m", flush=True)
        except Exception:
            print(f"ep {e} FAILED", flush=True)
            traceback.print_exc()
            lock.unlink(missing_ok=True)
        torch.cuda.empty_cache()


def stage_check(args, episodes, scored):
    """Does the official MHR reproduce SAM 3D Body's vertices from the saved parameters?"""
    estimator = load_estimator(args.model, args.mhr)
    mhr = torch.jit.load(str(args.mhr), map_location="cuda").eval()
    report = {}
    for e in episodes:
        folder = args.cache / f"episode_{e:06d}"
        human = np.load(folder / "masks.npz")["human"]
        K = np.load(folder / "camera.npz")["K"]
        picks = scored[e][np.linspace(0, len(scored[e]) - 1, 3).astype(int)]
        boxes, _ = actor_boxes(human, picks)
        cap = cv2.VideoCapture(str(video_path(args.data, e)))
        for i, f in enumerate(picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            rgb = cap.read()[1][:, :, ::-1].copy()
            mask = human[f].repeat(MASK_STEP, 0).repeat(MASK_STEP, 1)[None].astype(np.uint8)
            r = estimate(estimator, rgb, boxes[i], mask, torch.as_tensor(K, dtype=torch.float32)[None])
            params = r["mhr_model_params"].astype(np.float32).copy()
            trans_slot = params[:3].copy()
            params[:3] = 10 * FLIP * r["pred_cam_t"]
            target = r["pred_vertices"] + r["pred_cam_t"]
            row = {"trans_slot": trans_slot.tolist()}
            dev = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None], device="cuda")  # noqa: E731
            for name, expr in (("expr", r["expr_params"]), ("expr0", np.zeros(72))):
                for corr in (True, False):
                    with torch.no_grad():
                        v, skel = mhr(dev(r["shape_params"]), dev(params), dev(expr), corr)
                    err = np.linalg.norm(FLIP * v[0].cpu().numpy() / 100 - target, axis=-1)
                    row[f"{name}_corr{int(corr)}_mm"] = [round(float(err.mean() * 1000), 4),
                                                         round(float(err.max() * 1000), 4)]
            proj = (target @ K.T)[:, :2] / target[:, 2:]
            px = np.clip((proj / MASK_STEP).astype(int), 0, [human.shape[2] - 1, human.shape[1] - 1])
            row["verts_in_mask"] = round(float(human[f][px[:, 1], px[:, 0]].mean()), 3)
            row["cam_t"] = np.round(r["pred_cam_t"], 3).tolist()
            report[f"ep{e}_f{f}"] = row
            print(f"ep {e} f {f}: {row}", flush=True)
    (args.out / "check.json").write_text(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["infer", "check"], required=True)
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--model", choices=sorted(MODELS), default="dinov3")
    parser.add_argument("--no-mask", action="store_true", help="prompt with the box only")
    parser.add_argument("--tag", default=None, help="output subfolder (default: model name, +_nomask)")
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--cache", type=Path, default=REPO / "cache")
    parser.add_argument("--sample", type=Path,
                        default=REPO / "data/v2d_submission_kit/data/track_1_sample_submission.parquet")
    parser.add_argument("--mhr", type=Path, default=REPO / "body_models/mhr/mhr_model.pt")
    parser.add_argument("--out", type=Path, default=Path("/content/human_raw"))
    args = parser.parse_args()
    args.tag = args.tag or args.model + ("_nomask" if args.no_mask else "")
    token = Path("/content/.hf_token")
    if token.exists():
        os.environ.setdefault("HF_TOKEN", token.read_text().strip())
    args.out.mkdir(parents=True, exist_ok=True)
    scored = scored_frames(args.sample)
    {"infer": stage_infer, "check": stage_check}[args.stage](args, parse_episodes(args.episodes), scored)


if __name__ == "__main__":
    main()

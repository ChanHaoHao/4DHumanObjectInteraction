#!/usr/bin/env python3
"""Step 3 GPU stage (runs on the Colab VM): raw SAM 3 tracks and MoGe-2 depth per episode.

    python preprocess_vm.py --stage sam   [--episodes 0-29] [--data /content/v2d/track_1] [--out /content/raw]
    python preprocess_vm.py --stage depth [--episodes 0-29] ...
    python preprocess_vm.py --stage sam --episodes 9 --object-prompt "table"   # extra object candidates
    python preprocess_vm.py --stage fallback --episodes 9   # point-prompted track from the depth foreground

Writes, per episode, into <out>/episode_XXXXXX/ (skipped when present, so reruns resume):
  sam_person.npz, sam_object.npz   every track SAM 3 returns for the text prompt
  (sam_object_<text>.npz with --object-prompt):
      obj_ids [K], probs [K,T] float16 (0 = absent), masks [K,T,H/2,W/8] uint8 (np.packbits of the
      half-resolution [K,T,576,768] bool masks along the last axis), prompt, image_size (W, H)
  depth_raw.npz                    MoGe-2 (Ruicheng/moge-2-vitl-normal) at one fixed FOV per episode:
      depth [T,288,384] float16 metres (0 = invalid, nearest-sampled from full resolution),
      K [3,3] at video resolution, fov_x (deg), fov_x_samples [16] (free per-frame estimates)

The fallback stage needs depth_raw.npz and sam_person.npz in the same folder (run it on the VM that
builds) and writes sam_object_fallback.npz, one more candidate for build_cache.
Track selection, depth stabilization and the final cache are local (scripts/build_cache.py), so
they can be redone without a GPU. Setup on the VM: clone facebookresearch/sam3 and microsoft/MoGe,
`pip install -e` both, download the dataset to --data, write the HF token to /content/.hf_token.
Run from a directory other than /content (there, `import sam3` picks up the repo folder).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

# Short noun phrases work better with SAM 3 than the dataset's descriptive prompts
OBJECT_PROMPTS = {
    "hula_hoop": "hula hoop",
    "big_red_bowl": "red bowl",
    "iron": "clothes iron",
    "white_desk": "white desk",
    "black_pan": "frying pan",
    "foam_grass_block": "foam block",
    "paint_roller": "paint roller",
    "pink_foam_roll": "pink foam roller",
    "short_wood_stool": "wooden stool",
    "white_laptop_cart": "white cart",
}
MASK_STEP = 2  # masks kept at 1/2 resolution
DEPTH_STEP = 4  # depth kept at 1/4 resolution
FOV_SAMPLES = 16


def parse_episodes(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def video_path(data: Path, episode: int) -> Path:
    return data / "videos/chunk-000/observation.images.exo_camera" / f"episode_{episode:06d}.mp4"


def read_frames(path: Path, indices=None):
    """Yield (index, RGB uint8 frame) for every frame, or only `indices`."""
    cap = cv2.VideoCapture(str(path))
    wanted = None if indices is None else set(int(i) for i in indices)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if wanted is None or i in wanted:
            yield i, frame[:, :, ::-1]
        i += 1
    cap.release()


def save(path: Path, **arrays):
    """savez_compressed via a temporary name, so a concurrent download never sees a partial file."""
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ---------------------------------------------------------------------------------------------- SAM 3
def run_sam(predictor, video: Path, prompt: str, num_frames: int, offload_state: bool = False) -> dict:
    session = predictor.handle_request(dict(type="start_session", resource_path=str(video),
                                            offload_video_to_cpu=True,
                                            offload_state_to_cpu=offload_state))["session_id"]
    predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=0, text=prompt))
    tracks: dict[int, dict] = {}
    size = None
    request = dict(type="propagate_in_video", session_id=session, evict_cached_frame_outputs=True)
    for response in predictor.handle_stream_request(request):
        f, out = response["frame_index"], response["outputs"]
        masks = out["out_binary_masks"]
        masks = masks.cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        size = (masks.shape[2], masks.shape[1]) if masks.ndim == 3 and masks.shape[0] else size
        for obj_id, prob, mask in zip(np.asarray(out["out_obj_ids"]), np.asarray(out["out_probs"]), masks):
            t = tracks.setdefault(int(obj_id), {"probs": np.zeros(num_frames, np.float16), "masks": {}})
            t["probs"][f] = max(float(prob), 1e-3)
            t["masks"][f] = np.packbits(mask[::MASK_STEP, ::MASK_STEP].astype(bool), axis=-1)
    predictor.handle_request(dict(type="close_session", session_id=session))
    size = size or (1536, 1152)
    h, wb = size[1] // MASK_STEP, (size[0] // MASK_STEP + 7) // 8
    ids = sorted(tracks)
    packed = np.zeros((len(ids), num_frames, h, wb), np.uint8)
    for k, obj_id in enumerate(ids):
        for f, m in tracks[obj_id]["masks"].items():
            packed[k, f] = m
    probs = np.stack([tracks[i]["probs"] for i in ids]) if ids else np.zeros((0, num_frames), np.float16)
    return dict(obj_ids=np.array(ids, np.int64), probs=probs, masks=packed, prompt=prompt,
                image_size=np.array(size))


def stage_sam(args, episodes, meta):
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor()
    for episode in episodes:
        folder = args.out / f"episode_{episode:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        video = video_path(args.data, episode)
        n = frame_count(video)
        jobs = [("person", "person"), ("object", OBJECT_PROMPTS[meta[episode]["object"]])]
        if args.object_prompt:  # extra candidates for build_cache, which pools every sam_object*.npz
            jobs = [("object_" + args.object_prompt.replace(" ", "_"), args.object_prompt)]
        for name, prompt in jobs:
            path = folder / f"sam_{name}.npz"
            if path.exists():
                continue
            t0 = time.time()
            try:
                result = run_sam(predictor, video, prompt, n, args.offload_state)
            except Exception:
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
            save(path, **result)
            present = (result["probs"] > 0).sum(1)
            print(f"ep {episode} {name} '{prompt}': {n} frames, {time.time() - t0:.0f}s, tracks "
                  f"{dict(zip(result['obj_ids'].tolist(), present.tolist()))}, "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------- MoGe-2
def stage_depth(args, episodes, meta):
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
    to_tensor = lambda frames: torch.from_numpy(np.stack(frames)).cuda().permute(0, 3, 1, 2).float() / 255  # noqa: E731
    for episode in episodes:
        folder = args.out / f"episode_{episode:06d}"
        path = folder / "depth_raw.npz"
        if path.exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        video = video_path(args.data, episode)
        n = frame_count(video)
        t0 = time.time()
        # One FOV per episode (static camera): the median of free estimates on evenly spaced frames
        fovs = []
        for _, frame in read_frames(video, np.linspace(0, n - 1, FOV_SAMPLES).astype(int)):
            K = model.infer(to_tensor([frame]), use_fp16=True)["intrinsics"][0].cpu().numpy()
            fovs.append(math.degrees(2 * math.atan(0.5 / K[0, 0])))
        fov = float(np.median(fovs))
        depth, batch, K = [], [], None
        h = w = None

        def flush():
            nonlocal K
            out = model.infer(to_tensor(batch), fov_x=fov, use_fp16=True)
            d = out["depth"].float()
            d = torch.where(out["mask"].bool() & torch.isfinite(d), d, torch.zeros_like(d))
            off = DEPTH_STEP // 2
            depth.append(d[:, off::DEPTH_STEP, off::DEPTH_STEP].cpu().numpy().astype(np.float16))
            K = out["intrinsics"][0].cpu().numpy()
            batch.clear()

        for _, frame in read_frames(video):
            h, w = frame.shape[:2]
            batch.append(frame)
            if len(batch) == args.batch:
                flush()
        if batch:
            flush()
        K_pix = np.diag([w, h, 1.0]) @ K.astype(np.float64)
        save(path, depth=np.concatenate(depth), K=K_pix, fov_x=fov,
                            fov_x_samples=np.array(fovs), image_size=np.array([w, h]))
        print(f"ep {episode} depth: {n} frames, {time.time() - t0:.0f}s, fov_x {fov:.2f} "
              f"(spread {np.ptp(fovs):.2f}), fx {K_pix[0, 0]:.1f}", flush=True)


# ------------------------------------------------------------------------------------------ fallback
def foreground_seed(raw: Path, scored: np.ndarray, gap: float = 0.2, min_area: int = 150):
    """Where the object probably is when no prompt finds it: the largest depth-foreground blob (closer
    than the static background by `gap` m, not a person, at least `min_area` px at 1/4 resolution)
    touching the actor on a scored frame.
    Returns (frame, points [k,2] in pixels at video resolution) or None."""
    from scipy import ndimage

    from build_cache import load_tracks

    depth = np.load(raw / "depth_raw.npz")["depth"].astype(np.float32)  # [T,288,384]
    people = load_tracks(raw / "sam_person.npz")
    persons = people["masks"][:, :, ::2, ::2]  # [K,T,288,384]
    anyone = np.stack([ndimage.binary_dilation(m, iterations=3) for m in persons.any(0)])
    sample = np.arange(0, len(depth), 5)
    background = np.nanmedian(np.where((depth[sample] > 0) & ~anyone[sample], depth[sample], np.nan), axis=0)
    actor = int(np.argmax(np.median(persons[:, scored[::2]].sum((2, 3)), axis=1)))
    best = (0, None, None)
    for f in scored[::3]:
        body = persons[actor, f]
        if body.sum() < 30:
            continue
        z = np.median(depth[f][body & (depth[f] > 0)]) if (body & (depth[f] > 0)).any() else np.nan
        fg = (depth[f] > 0) & (depth[f] < background - gap) & ~anyone[f] & (np.abs(depth[f] - z) < 1.0)
        labels, n = ndimage.label(fg)
        if not n:
            continue
        near = ndimage.binary_dilation(body, iterations=15)
        for k in np.unique(labels[near & fg]):
            blob = labels == k
            if blob.sum() > best[0]:
                best = (int(blob.sum()), int(f), blob)
    area, frame, blob = best
    if area < min_area:
        return None
    dist = ndimage.distance_transform_edt(blob)
    points = []
    for _ in range(3):  # up to 3 well-separated interior points
        y, x = np.unravel_index(dist.argmax(), dist.shape)
        if dist[y, x] < 2:
            break
        points.append((x * 4 + 2, y * 4 + 2))
        yy, xx = np.ogrid[: dist.shape[0], : dist.shape[1]]
        dist[(yy - y) ** 2 + (xx - x) ** 2 < (2 * dist[y, x]) ** 2] = 0
    return frame, np.array(points, float)


def stage_fallback(args, episodes, meta):
    """Point-prompted track seeded by foreground_seed -> sam_object_fallback.npz. Uses SAM 3's tracker
    (the SAM 2-style interface): the text-prompt predictor ignores points on a session without objects."""
    from build_cache import scored_frames
    from sam3.model_builder import build_sam3_video_model

    scored = scored_frames(args.sample)
    model = build_sam3_video_model()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    for episode in episodes:
        folder = args.out / f"episode_{episode:06d}"
        path = folder / "sam_object_fallback.npz"
        if path.exists():
            continue
        seed = foreground_seed(folder, scored[episode], args.fg_gap, args.fg_min_area)
        if seed is None:
            print(f"ep {episode} fallback: no foreground blob touches the actor", flush=True)
            continue
        frame, points = seed
        video = video_path(args.data, episode)
        n = frame_count(video)
        t0 = time.time()
        state = tracker.init_state(video_path=str(video))
        tracker.clear_all_points_in_video(state)
        rel = torch.tensor(points / np.array([1536.0, 1152.0]), dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tracker.add_new_points(inference_state=state, frame_idx=frame, obj_id=1, points=rel,
                                   labels=torch.ones(len(rel), dtype=torch.int32), clear_old_points=False)
            probs = np.zeros(n, np.float16)
            packed = np.zeros((1, n, 576, 96), np.uint8)
            for reverse in (False, True):
                for f, obj_ids, _, video_masks, scores in tracker.propagate_in_video(
                        state, start_frame_idx=frame, max_frame_num_to_track=n, reverse=reverse,
                        propagate_preflight=True):
                    mask = (video_masks[0] > 0).reshape(video_masks.shape[-2:]).cpu().numpy()
                    if mask.any():
                        probs[f] = max(float(torch.sigmoid(torch.as_tensor(scores).float().flatten()[0])), 1e-3)
                        packed[0, f] = np.packbits(mask[::MASK_STEP, ::MASK_STEP], axis=-1)
        save(path, obj_ids=np.array([1]), probs=probs[None], masks=packed, image_size=np.array([1536, 1152]),
             prompt=f"fallback: points {points.astype(int).tolist()} at frame {frame}")
        print(f"ep {episode} fallback: seed frame {frame}, {len(points)} points, {(probs > 0).sum()} frames, "
              f"{time.time() - t0:.0f}s", flush=True)
        del state
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["sam", "depth", "fallback"], required=True)
    parser.add_argument("--sample", type=Path, default=Path("/content/work/track_1_sample_submission.parquet"))
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--out", type=Path, default=Path("/content/raw"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--fg-gap", type=float, default=0.2, help="fallback: m in front of the background")
    parser.add_argument("--fg-min-area", type=int, default=150, help="fallback: min blob px at 1/4 resolution")
    parser.add_argument("--offload-state", action="store_true",
                        help="keep SAM's tracking state on the CPU (long clips with many instances; slower)")
    parser.add_argument("--object-prompt", help="also track this text, into sam_object_<text>.npz")
    args = parser.parse_args()
    token = Path("/content/.hf_token")
    if token.exists():
        os.environ.setdefault("HF_TOKEN", token.read_text().strip())
    meta = {}
    with open(args.data / "meta/episodes_metadata.jsonl") as f:
        for line in f:
            d = json.loads(line)
            meta[d["episode_index"]] = d
    episodes = parse_episodes(args.episodes)
    with torch.inference_mode():
        {"sam": stage_sam, "depth": stage_depth, "fallback": stage_fallback}[args.stage](args, episodes, meta)


if __name__ == "__main__":
    main()

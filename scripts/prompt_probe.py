#!/usr/bin/env python3
"""Probe SAM 3 (image model) text prompts per object on mid-contact frames (runs on the Colab VM).

    python prompt_probe.py --sample track_1_sample_submission.parquet [--out /content/cache_check/prompt_probe.jpg]

One row per (episode, frame): the middle scored frame of one or two episodes per object. One column per
candidate prompt, with every detection above 0.5 drawn and its score. Used to choose OBJECT_PROMPTS.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from build_cache import scored_frames

CANDIDATES = {
    "hula_hoop": ["hula hoop", "hoop", "ring"],
    "big_red_bowl": ["red bowl", "bowl"],
    "iron": ["clothes iron", "iron", "steam iron"],
    "white_desk": ["white desk", "desk", "table", "white table", "folding table"],
    "black_pan": ["frying pan", "pan"],
    "foam_grass_block": ["foam block", "block", "box", "green block"],
    "paint_roller": ["paint roller", "roller"],
    "pink_foam_roll": ["pink foam roller", "foam roller", "pink cylinder"],
    "short_wood_stool": ["wooden stool", "stool"],
    "white_laptop_cart": ["white cart", "cart", "laptop cart", "rolling cart"],
}
EPISODES = [0, 3, 6, 9, 11, 12, 15, 18, 21, 24, 27, 28]
TILE = (384, 288)


def frame(data: Path, episode: int, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(data / "videos/chunk-000/observation.images.exo_camera" / f"episode_{episode:06d}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    return img[:, :, ::-1].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/content/v2d/track_1"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("/content/cache_check/prompt_probe.jpg"))
    args = parser.parse_args()
    token = Path("/content/.hf_token")
    if token.exists():
        os.environ.setdefault("HF_TOKEN", token.read_text().strip())
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    processor = Sam3Processor(build_sam3_image_model())
    meta = {}
    with open(args.data / "meta/episodes_metadata.jsonl") as f:
        for line in f:
            d = json.loads(line)
            meta[d["episode_index"]] = d
    scored = scored_frames(args.sample)
    width = max(len(v) for v in CANDIDATES.values())
    rows, report = [], {}
    for episode in EPISODES:
        name = meta[episode]["object"]
        f = int(scored[episode][len(scored[episode]) // 2])
        rgb = frame(args.data, episode, f)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(rgb))
            tiles = []
            for prompt in CANDIDATES[name]:
                out = processor.set_text_prompt(state=state, prompt=prompt)
                scores = out["scores"].float().cpu().numpy()
                masks = out["masks"].cpu().numpy().reshape(len(scores), *rgb.shape[:2]) if len(scores) else []
                img = rgb[:, :, ::-1].copy()
                for m, s in zip(masks, scores):
                    m = m.astype(bool)
                    img[m] = (0.4 * img[m] + 0.6 * np.array([60, 255, 60])).astype(np.uint8)
                    ys, xs = np.nonzero(m)
                    if len(xs):
                        cv2.putText(img, f"{s:.2f}", (int(xs.mean()), int(ys.min())), 0, 1.6, (255, 255, 255), 4)
                img = cv2.resize(img, TILE)
                cv2.putText(img, f"ep{episode} f{f} '{prompt}' n={len(scores)}", (6, 22), 0, 0.6, (0, 255, 255), 2)
                tiles.append(img)
                report[f"{episode}:{prompt}"] = np.round(scores, 2).tolist()
            tiles += [np.zeros_like(tiles[0])] * (width - len(tiles))
            rows.append(np.concatenate(tiles, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(json.dumps(report, indent=0))


if __name__ == "__main__":
    main()

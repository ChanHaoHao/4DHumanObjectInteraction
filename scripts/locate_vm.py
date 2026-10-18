#!/usr/bin/env python3
"""Ground one object phrase in every frame with NVIDIA LocateAnything-3B. Runs on the Colab VM.

    python locate_vm.py --videos DIR --episode 19 --phrase "paper towel roll" --out OUT [--mode hybrid] [--stride 1 | --keyframes 10]

Uses the card's single-instance template. Writes to OUT:
  <ep>_locate.npz   boxes: (F, 5) rows [x1, y1, x2, y2, found] in pixels; frames: (F,) frame indices
  <ep>_locate.json  phrase, mode, timing, frames with a box, median box, raw answers of the first frames
  <ep>_locate.mp4   overlay at half resolution
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = "nvidia/LocateAnything-3B"
BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--phrase", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="hybrid", choices=("fast", "slow", "hybrid"))
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--keyframes", type=int, help="run about this many evenly spaced frames (overrides --stride)")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    question = f"Locate a single instance that matches the following description: {args.phrase}."

    @torch.no_grad()
    def ground(img: Image.Image) -> str:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
        text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = proc.process_vision_info(msgs)
        x = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        r = model.generate(pixel_values=x["pixel_values"].to(torch.bfloat16), input_ids=x["input_ids"],
                           attention_mask=x["attention_mask"], image_grid_hws=x.get("image_grid_hws"), tokenizer=tok,
                           max_new_tokens=8192, use_cache=True, generation_mode=args.mode, temperature=0.7,
                           do_sample=True, top_p=0.9, repetition_penalty=1.1, verbose=False)  # card's settings
        return r[0] if isinstance(r, tuple) else r

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = f"episode_{args.episode:06d}"
    cap = cv2.VideoCapture(f"{args.videos}/{n}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if args.keyframes:
        args.stride = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // args.keyframes)
    raw = out / f"{n}_raw.mp4"
    writer, rows, frames, answers = None, [], [], {}
    torch.manual_seed(0)
    t0, f = time.time(), -1
    while True:
        ok, im = cap.read()
        if not ok:
            break
        f += 1
        if f % args.stride:
            continue
        h, w = im.shape[:2]
        ans = ground(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if len(answers) < 20:
            answers[f] = ans
        m = BOX.search(ans)
        box = [int(g) / 1000 * s for g, s in zip(m.groups(), (w, h, w, h))] + [1.0] if m else [0.0] * 5
        rows.append(box)
        frames.append(f)
        vis = im.copy()
        if m:
            cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
        cv2.putText(vis, f"{f} {args.phrase}" + ("" if m else " (none)"), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 0) if m else (0, 0, 255), 3)
        vis = cv2.resize(vis, (w // 2, h // 2))
        if writer is None:
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (w // 2, h // 2))
        writer.write(vis)
        if len(frames) % 25 == 0:
            print(f"frame {f} {(time.time() - t0) / len(frames):.2f}s/frame", flush=True)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", str(out / f"{n}_locate.mp4")], check=True)
    raw.unlink()
    B = np.array(rows, np.float32).reshape(-1, 5)
    np.savez_compressed(out / f"{n}_locate.npz", boxes=B, frames=np.array(frames))
    found = B[:, 4] > 0
    summary = {
        "phrase": args.phrase, "mode": args.mode, "stride": args.stride, "frames_run": len(frames),
        "frames_with_box": int(found.sum()), "sec_per_frame": round((time.time() - t0) / max(len(frames), 1), 3),
        "median_box": np.median(B[found, :4], 0).round(1).tolist() if found.any() else None,
        "first_answers": answers,
    }
    (out / f"{n}_locate.json").write_text(json.dumps(summary, indent=1))
    print("done", json.dumps({k: v for k, v in summary.items() if k != "first_answers"}), flush=True)


if __name__ == "__main__":
    main()

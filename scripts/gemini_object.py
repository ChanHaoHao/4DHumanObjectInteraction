#!/usr/bin/env python3
"""Ask Gemini Flash what object the human interacts with in each episode video. Runs locally (stdlib only).

    python3 scripts/gemini_object.py --episodes 0-4            # exo camera videos of track_1
    python3 scripts/gemini_object.py --video path/to/clip.mp4  # any video
    python3 scripts/gemini_object.py --episodes 0 --force --model gemini-flash-latest

The key comes from GEMINI_API_KEY (environment or .env.local). Each video goes up through the Files API (inline
requests cap at 20 MB), is answered with PROMPT, then deleted. Answers are merged into
results/gemini_object/answers.json keyed by video name; already answered videos are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API = "https://generativelanguage.googleapis.com"
VIDEOS = ROOT / "data/video_to_data_challenge/track_1/videos/chunk-000/observation.images.exo_camera"
OUT = ROOT / "results/gemini_object/answers.json"
PROMPT = "What is the human interacting with in the video, just give me the object, don't give me a sentence"


def object_name(answer: str) -> str:
    """Gemini's answer as a short SAM 3 prompt: 'Hula hoop.' -> 'hula hoop'."""
    return " ".join(answer.replace("_", " ").strip().strip(".").lower().split())


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    env = ROOT / ".env.local"
    if not key and env.exists():
        key = dict(line.split("=", 1) for line in env.read_text().split() if "=" in line).get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set (export it or add it to .env.local)")
    return key


def request(url: str, key: str, method: str = "GET", body: bytes | None = None, headers: dict | None = None,
            retries: int = 4):
    """Send one request; retry on 429/5xx. Returns (json or None, response headers)."""
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method=method, headers={"x-goog-api-key": key, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                raw = r.read()
                return (json.loads(raw) if raw else None), r.headers
        except urllib.error.HTTPError as err:
            msg = err.read().decode(errors="replace")[:500]
            if err.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"{method} {url.split('?')[0]} -> {err.code}: {msg}") from None
        except (urllib.error.URLError, OSError) as err:  # dropped connection, SSL EOF, timeout
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"{method} {url.split('?')[0]} -> {err}") from None


def upload(path: Path, key: str) -> dict:
    data = path.read_bytes()
    _, headers = request(f"{API}/upload/v1beta/files", key, "POST", json.dumps({"file": {"display_name": path.name}}).encode(),
                         {"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                          "X-Goog-Upload-Header-Content-Length": str(len(data)),
                          "X-Goog-Upload-Header-Content-Type": "video/mp4", "Content-Type": "application/json"})
    info, _ = request(headers["x-goog-upload-url"], key, "POST", data,
                      {"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"})
    f = info["file"]
    while f["state"] == "PROCESSING":  # videos must be ACTIVE before they can be referenced
        time.sleep(3)
        f, _ = request(f"{API}/v1beta/{f['name']}", key)
    if f["state"] != "ACTIVE":
        raise RuntimeError(f"{path.name}: file state {f['state']}")
    return f


def ask(path: Path, key: str, model: str, prompt: str, temperature: float | None) -> str:
    f = upload(path, key)
    try:
        body = {"contents": [{"parts": [{"file_data": {"mime_type": f["mimeType"], "file_uri": f["uri"]}}, {"text": prompt}]}]}
        if temperature is not None:
            body["generationConfig"] = {"temperature": temperature}
        out, _ = request(f"{API}/v1beta/models/{model}:generateContent", key, "POST", json.dumps(body).encode(),
                         {"Content-Type": "application/json"})
    finally:
        request(f"{API}/v1beta/{f['name']}", key, "DELETE")
    try:
        return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"]).strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"{path.name}: no answer: {json.dumps(out)[:500]}") from None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--episodes", help="e.g. 0-4,7 (exo camera videos)")
    src.add_argument("--video", nargs="+", type=Path)
    p.add_argument("--model", default="gemini-flash-latest")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--temperature", type=float, default=None, help="API default if unset")
    p.add_argument("--force", action="store_true", help="re-ask videos already in answers.json")
    p.add_argument("--workers", type=int, default=1, help="videos asked in parallel")
    args = p.parse_args()

    key = api_key()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from step4_colab import parse_episodes

    videos = args.video or [VIDEOS / f"episode_{e:06d}.mp4" for e in parse_episodes(args.episodes)]
    answers = json.loads(OUT.read_text()) if OUT.exists() else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    todo = []
    for v in videos:
        if v.stem in answers and not args.force:
            print(f"{v.stem}: {answers[v.stem]['answer']} (cached)")
        else:
            todo.append(v)
    # requests run in parallel (each waits minutes on Gemini); answers are written here, one thread
    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        jobs = {pool.submit(ask, v, key, args.model, args.prompt, args.temperature): v for v in todo}
        for job in as_completed(jobs):
            v = jobs[job]
            try:
                answer = job.result()
            except Exception as err:
                print(f"{v.stem}: ERROR {err}", file=sys.stderr, flush=True)
                continue
            answers[v.stem] = {"answer": answer, "model": args.model, "prompt": args.prompt}
            OUT.write_text(json.dumps(dict(sorted(answers.items())), indent=1))  # after every video
            print(f"{v.stem}: {answer}", flush=True)


if __name__ == "__main__":
    main()

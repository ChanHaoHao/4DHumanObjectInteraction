#!/usr/bin/env python3
"""Drive LocateAnything-3B (locate_vm.py) on a Colab VM. All compute is remote.

    colab new -s la --gpu L4
    python3 scripts/locate_colab.py setup SESSION --episodes 19   # pip deps, fetch weights, upload videos
    python3 scripts/locate_colab.py run SESSION --episodes 19 [-- locate_vm.py args, e.g. --mode fast]
    python3 scripts/locate_colab.py pull SESSION --episodes 19

The phrase is the Gemini object name from gemini_results.json. Outputs: results/locate/ (npz, json, overlay mp4, log).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d4rt_steps_colab import job, upload_tar  # noqa: E402
from step3_colab import colab, keepalive, launch, ls, remote  # noqa: E402
from step4_colab import parse_episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VM = "/content/repo"
OUT = f"{VM}/results/locate"
VIDEOS = "videos/chunk-000/observation.images.exo_camera"
SETUP = r"""
set -e
trap 'echo FAILED' ERR
pip install -q transformers==4.57.1 decord==0.6.0 lmdb peft
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/LocateAnything-3B')"
echo ok > /content/setup_locate_done
"""


def setup(session: str, episodes: list[int]):
    remote(session, f"mkdir -p {VM}/scripts /content/logs")
    if "setup_locate.txt" not in ls(session, "/content/logs"):
        launch(session, "setup_locate", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    data = ROOT / "data/video_to_data_challenge/track_1"
    members = [(data / VIDEOS / f"episode_{e:06d}.mp4", f"v2d/track_1/{VIDEOS}/episode_{e:06d}.mp4") for e in episodes]
    upload_tar(session, f"locate_{session}", members, "/content")
    while "setup_locate_done" not in ls(session, "/content"):
        log = remote(session, "tail -3 /content/logs/setup_locate.txt")
        if "Error" in log or "FAILED" in log:
            stop.set()
            raise RuntimeError(f"[{session}] setup failed:\n{log}")
        time.sleep(20)
    stop.set()
    print(remote(session, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))


def run(session: str, episodes: list[int], extra: str):
    colab("upload", "-s", session, str(ROOT / "scripts/locate_vm.py"), f"{VM}/scripts/locate_vm.py")
    names = json.loads((ROOT / "gemini_results.json").read_text())
    cmds = [f"python {VM}/scripts/locate_vm.py --videos /content/v2d/track_1/{VIDEOS} --episode {e} "
            f"--phrase {json.dumps(names[str(e)])} --out {OUT} {extra}" for e in episodes]
    job(session, f"locate_{session}", f"mkdir -p {OUT} && ({'; '.join(cmds)}) > {OUT}/log_{session}.txt 2>&1")


def pull(session: str, episodes: list[int]):
    dst = ROOT / "results" / "locate"
    dst.mkdir(parents=True, exist_ok=True)
    have = set(ls(session, OUT))
    want = {f"episode_{e:06d}_locate.{x}" for e in episodes for x in ("npz", "json", "mp4")}
    for f in have:
        if f in want or f.endswith(".txt"):
            colab("download", "-s", session, f"{OUT}/{f}", str(dst / f), timeout=1800)
    print(f"[{session}] pulled into {dst}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "run", "pull"):
        p = sub.add_parser(name)
        p.add_argument("session"), p.add_argument("--episodes", required=True)
    args, rest = parser.parse_known_args()
    extra = " ".join(a for a in rest if a != "--")
    eps = parse_episodes(args.episodes)
    if args.cmd == "setup":
        setup(args.session, eps)
    elif args.cmd == "run":
        run(args.session, eps, extra)
    else:
        pull(args.session, eps)


if __name__ == "__main__":
    sys.exit(main())

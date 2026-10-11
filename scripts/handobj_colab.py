#!/usr/bin/env python3
"""Drive the 100DOH hand-object detector (handobj_vm.py) on a Colab VM. All compute is remote.

    python3 scripts/handobj_colab.py setup SESSION --episodes 0   # clone repo, fetch the 100K+ego checkpoint, upload videos
    python3 scripts/handobj_colab.py run SESSION --episodes 0 [-- handobj_vm.py args]
    python3 scripts/handobj_colab.py pull SESSION --episodes 0

Outputs: results/handobj/ (npz, summary json, overlay mp4, log).
"""

from __future__ import annotations

import argparse
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
OUT = f"{VM}/results/handobj"
VIDEOS = "videos/chunk-000/observation.images.exo_camera"
CKPT = "/content/faster_rcnn_1_8_132028.pth"  # handobj_100K+ego
# The repo's Google Drive links 404 (checked 2026-10-09); this is the copy bundled with FrankMocap's hand module
CKPT_URL = ("https://huggingface.co/ragamounibatchu/frankmocap-hand-detector-weights/resolve/main/"
            "faster_rcnn_1_8_132028.pth")
SETUP = r"""
set -e
trap 'echo FAILED' ERR
mkdir -p /content/logs
cd /content
[ -d hand_object_detector ] || git clone -q https://github.com/ddshan/hand_object_detector.git
pip install -q easydict
[ -s CKPT ] || curl -sfL -o CKPT URL
echo ok > /content/setup_done
""".replace("CKPT", CKPT).replace("URL", CKPT_URL)


def setup(session: str, episodes: list[int]):
    remote(session, f"mkdir -p {VM}/scripts /content/logs")
    if "setup_handobj.txt" not in ls(session, "/content/logs"):
        launch(session, "setup_handobj", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    colab("upload", "-s", session, str(ROOT / "scripts/handobj_vm.py"), f"{VM}/scripts/handobj_vm.py")
    data = ROOT / "data/video_to_data_challenge/track_1"
    members = [(data / VIDEOS / f"episode_{e:06d}.mp4", f"v2d/track_1/{VIDEOS}/episode_{e:06d}.mp4") for e in episodes]
    upload_tar(session, f"handobj_{session}", members, "/content")
    print(f"[{session}] uploaded {len(episodes)} episodes", flush=True)
    while "setup_done" not in ls(session, "/content"):
        log = remote(session, "tail -3 /content/logs/setup_handobj.txt")
        if "Error" in log or "FAILED" in log:
            stop.set()
            raise RuntimeError(f"[{session}] setup failed:\n{log}")
        time.sleep(20)
    stop.set()
    print(remote(session, f"ls -la {CKPT}; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))


def run(session: str, episodes: list[int], extra: str):
    colab("upload", "-s", session, str(ROOT / "scripts/handobj_vm.py"), f"{VM}/scripts/handobj_vm.py")
    eps = ",".join(map(str, episodes))
    job(session, f"handobj_{session}", f"mkdir -p {OUT} && python {VM}/scripts/handobj_vm.py "
                                       f"--repo /content/hand_object_detector --ckpt {CKPT} "
                                       f"--videos /content/v2d/track_1/{VIDEOS} --episodes {eps} --out {OUT} {extra} "
                                       f"> {OUT}/log_{session}.txt 2>&1")


def pull(session: str, episodes: list[int]):
    dst = ROOT / "results" / "handobj"
    dst.mkdir(parents=True, exist_ok=True)
    have = set(ls(session, OUT))
    for e in episodes:
        n = f"episode_{e:06d}"
        for f in (f"{n}_summary.json", f"{n}_handobj.npz", f"{n}_handobj.mp4"):
            if f in have:
                colab("download", "-s", session, f"{OUT}/{f}", str(dst / f), timeout=1800)
    for f in have:
        if f.endswith(".txt"):
            colab("download", "-s", session, f"{OUT}/{f}", str(dst / f))
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

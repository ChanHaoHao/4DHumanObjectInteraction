#!/usr/bin/env python3
"""Drive D4RT point tracking (track_vm.py) on a Colab VM. All compute is remote.

    python3 scripts/track_colab.py setup SESSION --episodes 0     # OpenD4RT + checkpoint, the episode videos, D4RT cache
    python3 scripts/track_colab.py run SESSION --episodes 0 [-- track_vm.py args]
    python3 scripts/track_colab.py pull SESSION --episodes 0

VM layout mirrors the repo at /content/repo (scripts/, cache/d4rt/<ep>/ with masks, camera, depth, meta).
Videos are uploaded from data/video_to_data_challenge to /content/v2d/track_1 (no full dataset download).
Outputs: results/track/ (tracks npz, summary json, overlay mp4, log).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d4rt_colab import CKPT  # noqa: E402
from d4rt_steps_colab import job, upload_tar  # noqa: E402
from step3_colab import colab, keepalive, launch, ls, remote  # noqa: E402
from step4_colab import parse_episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VM = "/content/repo"
OUT = f"{VM}/results/track"
VIDEOS = "videos/chunk-000/observation.images.exo_camera"
SCRIPTS = ["track_vm.py", "d4rt_vm.py"]
SETUP = r"""
set -e
mkdir -p /content/logs
cd /content
[ -d Open-d4rt ] || git clone -q https://github.com/Lijiaxin0111/Open-d4rt.git
pip install -q PyYAML opencv-python-headless imageio lz4 scipy
export HF_TOKEN=$(cat /content/.hf_token)
python -c "
from huggingface_hub import hf_hub_download
for f in ('model.yaml', 'opend4rt.ckpt'):
    print(hf_hub_download('Lijiaxin0111/OpenD4RT', 'CKPT/' + f, local_dir='/content/Open-d4rt'))
"
echo ok > /content/setup_done
""".replace("CKPT", CKPT)


def upload_scripts(session: str):
    for name in SCRIPTS:
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")


def setup(session: str, episodes: list[int]):
    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p {VM}/scripts /content/logs && printf %s {env['HF_TOKEN']} > /content/.hf_token")
    if "setup_track.txt" not in ls(session, "/content/logs"):
        launch(session, "setup_track", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    upload_scripts(session)
    data = ROOT / "data/video_to_data_challenge/track_1"
    members = []
    for e in episodes:
        n = f"episode_{e:06d}"
        members.append((data / VIDEOS / f"{n}.mp4", f"v2d/track_1/{VIDEOS}/{n}.mp4"))
        members += [(ROOT / "cache/d4rt" / n / f, f"repo/cache/d4rt/{n}/{f}")
                    for f in ("masks.npz", "camera.npz", "depth.npz", "meta.json")]
    upload_tar(session, f"track_{session}", members, "/content")
    print(f"[{session}] uploaded {len(episodes)} episodes", flush=True)
    while "setup_done" not in ls(session, "/content"):
        log = remote(session, "tail -3 /content/logs/setup_track.txt")
        if "Error" in log or "FAILED" in log:
            stop.set()
            raise RuntimeError(f"[{session}] setup failed:\n{log}")
        time.sleep(30)
    stop.set()
    print(remote(session, f"tail -2 /content/logs/setup_track.txt; ls -la /content/Open-d4rt/{CKPT}; "
                          "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))


def run(session: str, episodes: list[int], extra: str):
    upload_scripts(session)
    eps = ",".join(map(str, episodes))
    job(session, f"track_{session}", f"mkdir -p {OUT} && cd /content && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                                     f"&& python {VM}/scripts/track_vm.py --episodes {eps} --out {OUT} {extra} "
                                     f"> {OUT}/log_{session}.txt 2>&1")


def pull(session: str, episodes: list[int]):
    dst = ROOT / "results" / "track"
    dst.mkdir(parents=True, exist_ok=True)
    have = set(ls(session, OUT))
    for e in episodes:
        n = f"episode_{e:06d}"
        for f in (f"{n}_summary.json", f"{n}_tracks.mp4", f"{n}_check.jpg", f"{n}_tracks.npz"):  # npz last: largest
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

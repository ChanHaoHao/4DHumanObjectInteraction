#!/usr/bin/env python3
"""Drive Meta's Locate 3D (locate3d_vm.py) on the D4RT scene, on a Colab VM. All compute is remote.

    colab new -s l3d --gpu L4
    python3 scripts/locate3d_colab.py setup SESSION --episodes 18-20   # py3.10 venv, repo, SAM-H, uploads
    python3 scripts/locate3d_colab.py run SESSION --episodes 18-20 --phrase "paper roller" [-- --stride 10]
    python3 scripts/locate3d_colab.py pull SESSION --episodes 18-20

Colab's torch (2.11/cu130, py3.13) has no spconv/pyg wheels, so Locate 3D gets its own venv pinned like the
repo's environment_frozen.yml. Inputs: cache/d4rt/<ep>/{depth,camera}.npz, meta.json and the videos.
Outputs: results/locate3d/ (npz, json, overlay mp4, log).
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
OUT = f"{VM}/results/locate3d"
VIDEOS = "videos/chunk-000/observation.images.exo_camera"
PY = "/content/l3d_env/bin/python"
SETUP = rf"""
set -e
trap 'echo FAILED' ERR
cd /content
command -v uv || pip install -q uv
[ -d locate-3d ] || git clone -q --depth 1 https://github.com/facebookresearch/locate-3d.git
[ -x {PY} ] || uv venv -q --python 3.10 /content/l3d_env
uv pip install -q --python {PY} torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -q --python {PY} "torch-scatter==2.1.2+pt23cu121" "torch-cluster==1.6.3+pt23cu121" \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
uv pip install -q --python {PY} numpy==1.26.4 "setuptools<70" spconv-cu120 torch-geometric==2.6.1 timm transformers==4.52.3 \
    hydra-core omegaconf addict opencv-python-headless huggingface_hub safetensors ftfy regex \
    git+https://github.com/facebookresearch/segment-anything.git git+https://github.com/openai/CLIP.git \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
mkdir -p locate-3d/preprocessing/weights
wget -q -nc -P locate-3d/preprocessing/weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
{PY} -c "import spconv.pytorch, torch_scatter, torch_cluster, flash_attn, clip, segment_anything; print('imports ok')"
echo ok > /content/setup_l3d_done
"""


def setup(session: str, episodes: list[int]):
    remote(session, f"mkdir -p {VM}/scripts /content/logs")
    if "setup_l3d.txt" not in ls(session, "/content/logs"):
        launch(session, "setup_l3d", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    data = ROOT / "data/video_to_data_challenge/track_1"
    members = [(data / VIDEOS / f"episode_{e:06d}.mp4", f"v2d/track_1/{VIDEOS}/episode_{e:06d}.mp4") for e in episodes]
    members += [(ROOT / "cache/d4rt" / f"episode_{e:06d}" / f, f"cache_d4rt/episode_{e:06d}/{f}")
                for e in episodes for f in ("depth.npz", "camera.npz", "meta.json")]
    upload_tar(session, f"l3d_{session}", members, "/content")
    while "setup_l3d_done" not in ls(session, "/content"):
        log = remote(session, "tail -3 /content/logs/setup_l3d.txt")
        if "FAILED" in log:
            stop.set()
            raise RuntimeError(f"[{session}] setup failed:\n{log}")
        time.sleep(20)
    stop.set()
    print(remote(session, "tail -2 /content/logs/setup_l3d.txt"))


def run(session: str, episodes: list[int], phrase: str, extra: str):
    colab("upload", "-s", session, str(ROOT / "scripts/locate3d_vm.py"), f"{VM}/scripts/locate3d_vm.py")
    eps = ",".join(map(str, episodes))
    job(session, f"l3d_{session}",
        f"mkdir -p {OUT} && cd /content/locate-3d && PYTHONPATH=. {PY} {VM}/scripts/locate3d_vm.py "
        f"--cache /content/cache_d4rt --videos /content/v2d/track_1/{VIDEOS} --episodes {eps} "
        f"--phrase '{phrase}' --out {OUT} {extra} > {OUT}/log_{session}.txt 2>&1")


def pull(session: str, episodes: list[int]):
    dst = ROOT / "results" / "locate3d"
    dst.mkdir(parents=True, exist_ok=True)
    want = {f"episode_{e:06d}_locate3d.{x}" for e in episodes for x in ("npz", "json", "mp4")}
    for f in ls(session, OUT):
        if f in want or f.endswith(".txt"):
            colab("download", "-s", session, f"{OUT}/{f}", str(dst / f), timeout=1800)
    print(f"[{session}] pulled into {dst}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "run", "pull"):
        p = sub.add_parser(name)
        p.add_argument("session"), p.add_argument("--episodes", required=True)
        if name == "run":
            p.add_argument("--phrase", required=True)
    args, rest = parser.parse_known_args()
    extra = " ".join(a for a in rest if a != "--")
    eps = parse_episodes(args.episodes)
    if args.cmd == "setup":
        setup(args.session, eps)
    elif args.cmd == "run":
        run(args.session, eps, args.phrase, extra)
    else:
        pull(args.session, eps)


if __name__ == "__main__":
    sys.exit(main())

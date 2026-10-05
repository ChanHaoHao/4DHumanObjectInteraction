#!/usr/bin/env python3
"""Drive the Step 3 D4RT revision on a Colab VM (all compute remote; this uploads, launches and pulls).

    python3 scripts/d4rt_colab.py setup SESSION [--episodes 0]   # deps, OpenD4RT + checkpoint, data, MHR, slim uploads
    python3 scripts/d4rt_colab.py upload SESSION EPISODES [--full]  # slim: masks/camera/meta + background.npz;
                                                                 # --full: MoGe depth.npz + human_v1 files (d4rt_eval)
    python3 scripts/step3_colab.py launch SESSION NAME "cd /content/repo && python scripts/d4rt_vm.py --episodes 0"
    python3 scripts/d4rt_colab.py sync SESSION... --episodes 1-29  # keep kernels alive, pull each finished episode
    python3 scripts/d4rt_colab.py pull SESSION EPISODES [--eval DIR]  # cache/d4rt/<ep>, check images, eval outputs

VM layout mirrors the repo at /content/repo (scripts/, cache/, cache/d4rt/, data/v2d_submission_kit/,
body_models/mhr/, results/human_v1/episodes/), so d4rt_eval.py and proxy_eval.py run from there.
OpenD4RT at /content/Open-d4rt with its 48-frame checkpoint (14 GB release; d4rt_vm.py keeps a weights-only
copy next to it). Videos at /content/v2d/track_1. Locally: cache/d4rt/ and results/d4rt_check/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_colab import colab, keepalive, launch, ls, remote  # noqa: E402
from step4_colab import parse_episodes, upload_tar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VM = "/content/repo"
SCRIPTS = ["d4rt_vm.py", "d4rt_eval.py", "proxy_eval.py", "human_post.py", "human_vm.py", "step3_colab.py"]
CKPT = "checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"
SETUP = r"""
set -e
mkdir -p /content/logs
cd /content
[ -d Open-d4rt ] || git clone -q https://github.com/Lijiaxin0111/Open-d4rt.git
# keep Colab's torch; OpenD4RT only needs these on top
pip install -q PyYAML opencv-python-headless imageio lz4 trimesh numba fast_simplification pyarrow pandas scipy
export HF_TOKEN=$(cat /content/.hf_token)
python -c "
from huggingface_hub import hf_hub_download
for f in ('model.yaml', 'opend4rt.ckpt'):
    print(hf_hub_download('Lijiaxin0111/OpenD4RT', 'CKPT/' + f, local_dir='/content/Open-d4rt'))
"
if [ ! -f /content/repo/body_models/mhr/mhr_model.pt ]; then
  curl -sL -o /content/mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  unzip -q -j -o /content/mhr_assets.zip assets/mhr_model.pt -d /content/repo/body_models/mhr/
fi
echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  /content/repo/body_models/mhr/mhr_model.pt" | sha256sum -c
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset', allow_patterns=['track_1/*'], local_dir='/content/v2d')"
ls /content/v2d/track_1/videos/chunk-000/*/ | wc -l > /content/setup_done
""".replace("CKPT", CKPT)


def upload_scripts(session: str):
    for name in SCRIPTS:
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")


def upload_episodes(session: str, episodes: list[int], full: bool = False):
    """Slim: what d4rt_vm.py needs (masks, camera, meta, and background.npz = the MoGe depth.npz's frames +
    background, 0.2 MB instead of 57 MB). Full: the MoGe depth.npz and human_v1 files, for d4rt_eval.py."""
    SCRATCH = ROOT / "results" / "colab_jobs"
    for e in episodes:
        n = f"episode_{e:06d}"
        src = ROOT / "cache" / n
        if full:
            members = [(src / "depth.npz", f"cache/{n}/depth.npz")]
            members += [(p, f"results/human_v1/episodes/{p.name}")
                        for p in sorted((ROOT / "results/human_v1/episodes").glob(f"{n}*"))]
        else:
            bg = SCRATCH / f"{n}_background.npz"
            with np.load(src / "depth.npz") as dz:  # lazy: reads only these two arrays
                np.savez_compressed(bg, frames=dz["frames"], background=dz["background"])
            members = [(src / f, f"cache/{n}/{f}") for f in ("masks.npz", "camera.npz", "meta.json")]
            members.append((bg, f"cache/{n}/background.npz"))
        upload_tar(session, f"{session}_ep{e}{'_full' if full else ''}", members, VM)
        if not full:
            bg.unlink()
        print(f"[{session}] uploaded {n} ({'full' if full else 'slim'})", flush=True)


def setup(session: str, episodes: list[int]):
    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p {VM}/scripts {VM}/body_models/mhr {VM}/data /content/logs && "
                    f"printf %s {env['HF_TOKEN']} > /content/.hf_token")
    upload_tar(session, f"kit_{session}", [(ROOT / "data" / "v2d_submission_kit", "v2d_submission_kit")], f"{VM}/data")
    upload_scripts(session)
    launch(session, "setup_d4rt", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    upload_episodes(session, episodes)
    while "setup_done" not in remote(session, "ls /content"):
        log = remote(session, "tail -3 /content/logs/setup_d4rt.txt")
        if "Error" in log or "FAILED" in log:
            print(log)
            break
        time.sleep(30)
    stop.set()
    print(remote(session, f"tail -4 /content/logs/setup_d4rt.txt; ls -la /content/Open-d4rt/{CKPT}; "
                          "cat /content/setup_done; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
                          "free -g | head -2; df -h /content | tail -1"))


def pull_episode(session: str, n: str):
    """depth.npz + meta.json of one finished episode and its check image; masks/camera are Step 3v1's, copied locally."""
    dst = ROOT / "cache" / "d4rt" / n
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta.json").unlink(missing_ok=True)
    for f in ("masks.npz", "camera.npz"):
        shutil.copy(ROOT / "cache" / n / f, dst / f)
    for f in ("depth.npz", "meta.json"):  # meta last: marks the copy complete
        colab("download", "-s", session, f"{VM}/cache/d4rt/{n}/{f}", str(dst / f), timeout=1800)
    check = ROOT / "results" / "d4rt_check"
    check.mkdir(parents=True, exist_ok=True)
    colab("download", "-s", session, f"{VM}/cache/d4rt/check/{n}.jpg", str(check / f"{n}.jpg"))
    d = json.loads((dst / "meta.json").read_text())["d4rt"]
    keys = ["depth_scale_std", "window_scale_cv", "flicker_d4rt", "actor_acc_d4rt_cm", "cam_rot_deg_median_max",
            "cam_trans_rel_median_max", "fov_x", "seconds"]
    print(f"[{session}] pulled {n}: " + ", ".join(f"{k}={d.get(k)}" for k in keys), flush=True)


def pull(session: str, episodes: list[int], eval_dir: str | None):
    for e in episodes:
        n = f"episode_{e:06d}"
        if "meta.json" not in ls(session, f"{VM}/cache/d4rt/{n}"):
            print(f"{n}: no D4RT cache on the VM yet", flush=True)
            continue
        pull_episode(session, n)
    if eval_dir:  # d4rt_eval.py --out results/<eval_dir>
        out = ROOT / "results" / eval_dir
        for rel in ["summary.json", "cache/proxy_eval.csv", "d4rt/proxy_eval.csv"]:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            colab("download", "-s", session, f"{VM}/results/{eval_dir}/{rel}", str(out / rel))
        print((out / "summary.json").read_text())


def sync(sessions: list[str], episodes: list[int], every: int = 120):
    """Keep every kernel alive and pull each episode as soon as its meta.json exists on any VM."""
    threading.Thread(target=keepalive, args=(sessions,), daemon=True).start()
    want = {f"episode_{e:06d}" for e in episodes}
    while True:
        todo = {n for n in want if not (ROOT / "cache" / "d4rt" / n / "meta.json").exists()}
        if not todo:
            print("all episodes pulled", flush=True)
            return
        for s in sessions:
            try:
                for n in sorted(todo & set(ls(s, f"{VM}/cache/d4rt"))):
                    if "meta.json" in ls(s, f"{VM}/cache/d4rt/{n}"):
                        pull_episode(s, n)
            except RuntimeError as err:
                print(f"sync {s} error (will retry): {err}", flush=True)
        time.sleep(every)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("session"), p.add_argument("--episodes", default="0")
    p = sub.add_parser("upload")
    p.add_argument("session"), p.add_argument("episodes"), p.add_argument("--full", action="store_true")
    p = sub.add_parser("sync")
    p.add_argument("sessions", nargs="+"), p.add_argument("--episodes", required=True)
    p = sub.add_parser("scripts")
    p.add_argument("session")
    p = sub.add_parser("pull")
    p.add_argument("session"), p.add_argument("episodes"), p.add_argument("--eval", default=None, metavar="DIR")
    args = parser.parse_args()
    if args.cmd == "setup":
        setup(args.session, parse_episodes(args.episodes))
    elif args.cmd == "upload":
        upload_episodes(args.session, parse_episodes(args.episodes), args.full)
    elif args.cmd == "sync":
        sync(args.sessions, parse_episodes(args.episodes))
    elif args.cmd == "scripts":
        upload_scripts(args.session)
    elif args.cmd == "pull":
        pull(args.session, parse_episodes(args.episodes), args.eval)


if __name__ == "__main__":
    sys.exit(main())

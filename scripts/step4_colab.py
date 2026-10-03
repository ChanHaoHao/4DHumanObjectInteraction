#!/usr/bin/env python3
"""Drive Step 4 (human baseline) on a Colab VM. All compute is remote; this uploads, launches and pulls.

    python3 scripts/step4_colab.py setup SESSION [--depth 0,3,6]     # deps, data, repo mirror, cache upload
    python3 scripts/step4_colab.py depth SESSION EPISODES            # upload more depth.npz (58 MB each)
    python3 scripts/step4_colab.py check SESSION [--episodes 6]      # SAM 3D Body vs official MHR
    python3 scripts/step4_colab.py infer SESSION [--episodes 0-29] [--model dinov3] [--no-mask]
    python3 scripts/step4_colab.py sync SESSION TAG [--episodes 0-29]  # pull raw outputs as they finish
    python3 scripts/step4_colab.py post SESSION TAG RUN [-- human_post.py args]   # post + proxy eval, pull
    python3 scripts/step4_colab.py upload-raw SESSION TAG            # restore raw outputs on a new VM

Layout on the VM mirrors the repo at /content/repo (scripts/, cache/, data/v2d_submission_kit/,
body_models/mhr/mhr_model.pt), so human_post.py and proxy_eval.py find everything by relative path.
Videos at /content/v2d/track_1. Raw SAM 3D Body output: /content/human_raw/<tag>/; runs:
/content/repo/results/<run>/. Locally: results/human_raw/<tag>/ and results/<run>/ (laptop = store).
Keep `sync` (or step3_colab.py keepalive) running while the VM is up: Colab reclaims it ~45 min
after the last kernel interaction even mid-job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_colab import colab, keepalive, launch, ls, remote  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "results" / "colab_jobs"
VM = "/content/repo"
SCRIPTS = ["human_vm.py", "human_post.py", "proxy_eval.py", "step3_colab.py"]
SETUP = r"""
set -e
mkdir -p /content/logs /content/human_raw
cd /content
[ -d sam-3d-body ] || git clone -q https://github.com/facebookresearch/sam-3d-body.git
# keep Colab's torch: pin the installed versions while adding SAM 3D Body's dependencies
python -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" > /content/pins.txt
pip install -q -c /content/pins.txt pytorch-lightning yacs roma einops timm dill hydra-core pyrootutils \
    webdataset braceexpand omegaconf loguru optree fvcore jsonlines pycocotools scikit-image rich \
    trimesh numba fast_simplification pyarrow pandas scipy
if [ ! -f /content/repo/body_models/mhr/mhr_model.pt ]; then
  curl -sL -o /content/mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  unzip -q -j -o /content/mhr_assets.zip assets/mhr_model.pt -d /content/repo/body_models/mhr/
fi
echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  /content/repo/body_models/mhr/mhr_model.pt" | sha256sum -c
export HF_TOKEN=$(cat /content/.hf_token)
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset', allow_patterns=['track_1/*'], local_dir='/content/v2d')"
python -c "
from huggingface_hub import snapshot_download
for repo in ('facebook/sam-3d-body-dinov3', 'facebook/sam-3d-body-vith'):
    try:
        snapshot_download(repo); print('checkpoint ok', repo)
    except Exception as err:
        print('checkpoint MISSING', repo, type(err).__name__)
"
# DINOv3 code comes from torch.hub at model build; fetch it now so a later failure is a setup failure
python -c "import torch; torch.hub.list('facebookresearch/dinov3', trust_repo=True)" > /dev/null
ls /content/v2d/track_1/videos/chunk-000/*/ | wc -l > /content/setup_done
"""


def parse_episodes(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def upload_tar(session: str, name: str, members: list[tuple[Path, str]], dest: str):
    """Pack local files (path, arcname) into one tar, upload it, unpack under dest on the VM."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    local = SCRATCH / f"{name}.tar"
    with tarfile.open(local, "w") as tar:
        for path, arcname in members:
            tar.add(path, arcname=arcname)
    colab("upload", "-s", session, str(local), f"/content/{name}.tar", timeout=1800)
    remote(session, f"mkdir -p {dest} && tar -xf /content/{name}.tar -C {dest} && rm /content/{name}.tar")
    local.unlink()


def upload_scripts(session: str):
    for name in SCRIPTS:
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")


def upload_depth(session: str, episodes: list[int]):
    for e in episodes:
        n = f"episode_{e:06d}"
        colab("upload", "-s", session, str(ROOT / "cache" / n / "depth.npz"), f"{VM}/cache/{n}/depth.npz",
              timeout=1800)
        print(f"uploaded depth {n}", flush=True)


def setup(session: str, depth: list[int]):
    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p {VM}/scripts {VM}/body_models/mhr {VM}/data && "
                    f"printf %s {env['HF_TOKEN']} > /content/.hf_token")
    kit = ROOT / "data" / "v2d_submission_kit"
    upload_tar(session, "kit", [(kit, "v2d_submission_kit")], f"{VM}/data")
    cache = [(p, f"{p.parent.name}/{p.name}") for e in range(30)
             for p in sorted((ROOT / "cache" / f"episode_{e:06d}").glob("*"))
             if p.name in ("masks.npz", "camera.npz", "meta.json")]
    upload_tar(session, "cache", cache, f"{VM}/cache")
    upload_scripts(session)
    launch(session, "setup4", SETUP)  # fetches MHR v1.0.1 itself (faster than uploading it)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    upload_depth(session, depth)
    while "setup_done" not in remote(session, "ls /content"):
        log = remote(session, "tail -2 /content/logs/setup4.txt")
        if "Error" in log or "FAILED" in log:
            print(log)
            break
        time.sleep(20)
    stop.set()
    print(remote(session, "grep -a 'checkpoint\\|OK\\|FAILED' /content/logs/setup4.txt; cat /content/setup_done"))


def sync(session: str, tag: str, episodes: list[int], every: int = 90):
    """Pull each finished raw episode until all are local (and keep the kernel alive meanwhile)."""
    threading.Thread(target=keepalive, args=([session],), daemon=True).start()
    dst = ROOT / "results" / "human_raw" / tag
    dst.mkdir(parents=True, exist_ok=True)
    want = {f"episode_{e:06d}.npz" for e in episodes}
    while True:
        have = {p.name for p in dst.glob("episode_*.npz")}
        for f in sorted(set(ls(session, f"/content/human_raw/{tag}")) & (want - have)):
            colab("download", "-s", session, f"/content/human_raw/{tag}/{f}", str(dst / f), timeout=900)
            print(f"pulled {tag}/{f}", flush=True)
        have = {p.name for p in dst.glob("episode_*.npz")}
        if want <= have:
            print("all raw episodes pulled", flush=True)
            return
        log = remote(session, f"grep -a -A8 'FAI''LED' /content/logs/infer_{tag}_*.txt | tail -30")
        if "FAILED" in log:
            print(log[-1500:], flush=True)
        time.sleep(every)


def pull_run(session: str, run: str):
    dst = ROOT / "results" / run
    for sub in ("episodes", "human_meta", "overlays"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for f in ls(session, f"{VM}/results/{run}/{sub}"):
            colab("download", "-s", session, f"{VM}/results/{run}/{sub}/{f}", str(dst / sub / f), timeout=900)
    for f in ("proxy_eval.csv", "proxy_eval.txt", "post.txt"):
        try:
            colab("download", "-s", session, f"{VM}/results/{run}/{f}", str(dst / f))
        except RuntimeError:
            print(f"no {f}")
    print(f"pulled results/{run}", flush=True)


def wait_job(session: str, name: str):
    """Until the job writes /content/logs/<name>.done (checked via the contents API: console output
    echoes the command, which would contain the file name)."""
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    while f"{name}.done" not in ls(session, "/content/logs"):
        time.sleep(30)
    stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("session"), p.add_argument("--depth", default="")
    p = sub.add_parser("depth")
    p.add_argument("session"), p.add_argument("episodes")
    sub.add_parser("scripts").add_argument("session")
    p = sub.add_parser("check")
    p.add_argument("session"), p.add_argument("--episodes", default="6"), p.add_argument("--model", default="dinov3")
    p = sub.add_parser("infer")
    p.add_argument("session"), p.add_argument("--episodes", default="0-29")
    p.add_argument("--model", default="dinov3"), p.add_argument("--no-mask", action="store_true")
    p.add_argument("--workers", type=int, default=3, help="processes sharing the GPU")
    p = sub.add_parser("sync")
    p.add_argument("session"), p.add_argument("tag"), p.add_argument("--episodes", default="0-29")
    p = sub.add_parser("post")
    p.add_argument("session"), p.add_argument("tag"), p.add_argument("run")
    p.add_argument("--episodes", default="0-29")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="after --: more human_post.py arguments")
    p = sub.add_parser("pull-run")
    p.add_argument("session"), p.add_argument("run")
    p = sub.add_parser("upload-raw")
    p.add_argument("session"), p.add_argument("tag")
    args = parser.parse_args()

    if args.cmd == "setup":
        setup(args.session, parse_episodes(args.depth) if args.depth else [])
    elif args.cmd == "depth":
        upload_depth(args.session, parse_episodes(args.episodes))
    elif args.cmd == "scripts":
        upload_scripts(args.session)
    elif args.cmd == "check":
        upload_scripts(args.session)
        remote(args.session, "rm -f /content/logs/check.done")
        launch(args.session, "check", f"cd {VM}/scripts && python human_vm.py --stage check "
                                      f"--episodes {args.episodes} --model {args.model}; "
                                      f"echo done > /content/logs/check.done")
        wait_job(args.session, "check")
        print(remote(args.session, "grep -a '^ep \\|Error\\|Traceback' /content/logs/check.txt | tail -20"))
        (ROOT / "results" / "human_raw").mkdir(parents=True, exist_ok=True)
        colab("download", "-s", args.session, "/content/human_raw/check.json",
              str(ROOT / "results" / "human_raw" / "check.json"))
    elif args.cmd == "infer":
        upload_scripts(args.session)
        tag = args.model + ("_nomask" if args.no_mask else "")
        flags = " --no-mask" if args.no_mask else ""
        order = parse_episodes(args.episodes)
        for w in range(args.workers):  # each worker walks the list from a different offset
            mine = order[w * len(order) // args.workers:] + order[:w * len(order) // args.workers]
            launch(args.session, f"infer_{tag}_{w}",
                   f"cd {VM}/scripts && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                   f"python human_vm.py --stage infer --episodes {','.join(map(str, mine))} "
                   f"--model {args.model}{flags}")
            time.sleep(60)  # stagger model loading
    elif args.cmd == "sync":
        sync(args.session, args.tag, parse_episodes(args.episodes))
    elif args.cmd == "post":
        upload_scripts(args.session)
        extra = " ".join(a for a in args.extra if a != "--")
        run = f"{VM}/results/{args.run}"
        remote(args.session, f"rm -f /content/logs/post_{args.run}.done")
        launch(args.session, f"post_{args.run}",
               f"mkdir -p {run} && cd {VM}/scripts && "
               f"python human_post.py --raw /content/human_raw/{args.tag} --run {run} "
               f"--episodes {args.episodes} {extra} 2>&1 | tee {run}/post.txt; "
               f"python proxy_eval.py {run}/episodes --episodes "
               f"{','.join(map(str, parse_episodes(args.episodes)))} --csv {run}/proxy_eval.csv "
               f"2>&1 | tee {run}/proxy_eval.txt; echo done > /content/logs/post_{args.run}.done")
        wait_job(args.session, f"post_{args.run}")
        pull_run(args.session, args.run)
        print((ROOT / "results" / args.run / "proxy_eval.txt").read_text()
              if (ROOT / "results" / args.run / "proxy_eval.txt").exists() else "no proxy_eval output")
    elif args.cmd == "pull-run":
        pull_run(args.session, args.run)
    elif args.cmd == "upload-raw":
        local = sorted((ROOT / "results" / "human_raw" / args.tag).glob("episode_*.npz"))
        upload_tar(args.session, f"raw_{args.tag}", [(p, p.name) for p in local], f"/content/human_raw/{args.tag}")


if __name__ == "__main__":
    sys.exit(main())

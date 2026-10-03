#!/usr/bin/env python3
"""Drive Step 5 (object baseline) on a Colab VM. All compute is remote; this uploads, launches and pulls.

    python3 scripts/step5_colab.py setup SESSION [--human human_v1]   # deps, data, cache + human run upload
    python3 scripts/step5_colab.py crops SESSION [--episodes 0-29]    # pick views, pull crops/ (+ sheet.jpg)
    python3 scripts/step5_colab.py mesh SESSION [--episodes 0-29]     # Hunyuan3D-2.1, pulls meshes as they finish
    python3 scripts/step5_colab.py fit SESSION RUN [--episodes 0-29] [--workers 2] [-- object_fit.py args]
    python3 scripts/step5_colab.py upload-raw SESSION                 # restore crops/meshes on a new VM

Layout on the VM mirrors the repo at /content/repo (scripts/, cache/ with depth, results/<human run>/,
data/v2d_submission_kit/, body_models/mhr/). Videos at /content/v2d/track_1. Crops and meshes:
/content/object_raw/{crops,meshes}; locally results/object_raw/ (laptop = store). Runs: results/<RUN>/.
A keep-alive thread pings the kernel while any command here waits (Colab reclaims idle VMs ~45 min after
the last kernel interaction, even mid-job); between commands run step3_colab.py keepalive SESSION.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_colab import colab, keepalive, launch, ls, remote  # noqa: E402
from step4_colab import parse_episodes, upload_depth, upload_tar, wait_job  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VM = "/content/repo"
RAW = "/content/object_raw"
LOCAL_RAW = ROOT / "results" / "object_raw"
SCRIPTS = ["object_vm.py", "object_fit.py", "human_vm.py", "proxy_eval.py", "step3_colab.py"]
SETUP = r"""
set -e
mkdir -p /content/logs /content/object_raw
cd /content
[ -d Hunyuan3D-2.1 ] || git clone -q https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
python -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" > /content/pins.txt
pip install -q -c /content/pins.txt diffusers transformers accelerate einops omegaconf pymeshlab trimesh \
    fast_simplification pyarrow pandas scipy opencv-python-headless numba pytorch-lightning
if [ ! -f /content/repo/body_models/mhr/mhr_model.pt ]; then
  curl -sL -o /content/mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  unzip -q -j -o /content/mhr_assets.zip assets/mhr_model.pt -d /content/repo/body_models/mhr/
fi
echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  /content/repo/body_models/mhr/mhr_model.pt" | sha256sum -c
export HF_TOKEN=$(cat /content/.hf_token)
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset', allow_patterns=['track_1/*'], local_dir='/content/v2d')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', allow_patterns=['hunyuan3d-dit-v2-1/*'])"
cd /content/Hunyuan3D-2.1/hy3dshape && python -c "from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline; print('hy3dshape import OK')"
ls /content/v2d/track_1/videos/chunk-000/*/ | wc -l > /content/setup_done
"""


def upload_scripts(session: str):
    for name in SCRIPTS:
        if not (ROOT / "scripts" / name).exists():
            continue
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")


def setup(session: str, human: str):
    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p {VM}/scripts {VM}/body_models/mhr {VM}/data {VM}/results && "
                    f"printf %s {env['HF_TOKEN']} > /content/.hf_token")
    upload_tar(session, "kit", [(ROOT / "data" / "v2d_submission_kit", "v2d_submission_kit")], f"{VM}/data")
    cache = [(p, f"{p.parent.name}/{p.name}") for e in range(30)
             for p in sorted((ROOT / "cache" / f"episode_{e:06d}").glob("*"))
             if p.name in ("masks.npz", "camera.npz", "meta.json")]
    upload_tar(session, "cache", cache, f"{VM}/cache")
    upload_tar(session, "human", [(ROOT / "results" / human / "episodes", f"{human}/episodes")], f"{VM}/results")
    upload_scripts(session)
    launch(session, "setup5", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    upload_depth(session, list(range(30)))
    while "setup_done" not in remote(session, "ls /content"):
        log = remote(session, "tail -3 /content/logs/setup5.txt")
        if "Error" in log or "FAILED" in log:
            print(log)
            break
        time.sleep(20)
    stop.set()
    print(remote(session, "grep -a 'OK\\|FAILED\\|Error' /content/logs/setup5.txt | tail; cat /content/setup_done"))


def pull_dir(session: str, remote_dir: str, local: Path, names: set[str] | None = None) -> list[str]:
    local.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in local.iterdir()}
    got = []
    for f in ls(session, remote_dir):
        if f not in have and (names is None or f in names) and ".partial" not in f:
            colab("download", "-s", session, f"{remote_dir}/{f}", str(local / f), timeout=900)
            got.append(f)
    return got


def run_job(session: str, name: str, cmd: str):
    remote(session, f"rm -f /content/logs/{name}.done")
    launch(session, name, f"{cmd}; echo done > /content/logs/{name}.done")


def mesh_sync(session: str, episodes: list[int], every: int = 60):
    """Pull meshes as the job writes them, until it ends (keeps the kernel alive meanwhile)."""
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    while True:
        done = "mesh.done" in ls(session, "/content/logs")
        for f in pull_dir(session, f"{RAW}/meshes", LOCAL_RAW / "meshes"):
            print(f"pulled {f}", flush=True)
        if done:
            break
        time.sleep(every)
    stop.set()
    print(remote(session, "grep -a '^ep \\|FAILED\\|Error' /content/logs/mesh.txt | tail -100"))


def pull_run(session: str, run: str):
    dst = ROOT / "results" / run
    for sub in ("episodes", "object_meta", "overlays"):
        pull_dir(session, f"{VM}/results/{run}/{sub}", dst / sub)
    for f in ("fit.txt", "proxy_eval.csv", "proxy_eval.txt"):
        try:
            colab("download", "-s", session, f"{VM}/results/{run}/{f}", str(dst / f))
        except RuntimeError:
            print(f"no {f}")
    print(f"pulled results/{run}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("session"), p.add_argument("--human", default="human_v1")
    sub.add_parser("scripts").add_argument("session")
    for name in ("crops", "mesh"):
        p = sub.add_parser(name)
        p.add_argument("session"), p.add_argument("--episodes", default="0-29")
    p = sub.add_parser("fit")
    p.add_argument("session"), p.add_argument("run"), p.add_argument("--episodes", default="0-29")
    p.add_argument("--human", default="human_v1"), p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("pull-run")
    p.add_argument("session"), p.add_argument("run")
    sub.add_parser("upload-raw").add_argument("session")
    args, rest = parser.parse_known_args()  # unknown arguments (after an optional --) go to the VM script
    extra = " ".join(a for a in rest if a != "--")
    if args.cmd == "setup":
        setup(args.session, args.human)
    elif args.cmd == "scripts":
        upload_scripts(args.session)
    elif args.cmd == "crops":
        upload_scripts(args.session)
        run_job(args.session, "crops", f"cd {VM}/scripts && python object_vm.py --stage crops "
                                       f"--episodes {args.episodes} {extra}")
        wait_job(args.session, "crops")
        print(remote(args.session, "grep -a '^ep \\|Error\\|Traceback' /content/logs/crops.txt | tail -40"))
        pull_dir(args.session, f"{RAW}/crops", LOCAL_RAW / "crops")
        print(f"pulled crops to {LOCAL_RAW / 'crops'}")
    elif args.cmd == "mesh":
        upload_scripts(args.session)
        run_job(args.session, "mesh", f"cd {VM}/scripts && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                                      f"python object_vm.py --stage mesh --episodes {args.episodes} {extra}")
        mesh_sync(args.session, parse_episodes(args.episodes))
    elif args.cmd == "fit":
        upload_scripts(args.session)
        run = f"{VM}/results/{args.run}"
        eps = ",".join(map(str, parse_episodes(args.episodes)))
        fit = (f"python object_fit.py --human {VM}/results/{args.human}/episodes --run {run} "
               f"--episodes {args.episodes} {extra}")
        # workers share the GPU (episodes claimed by lock files); proxy eval once all have finished
        workers = " & ".join(f"{fit} > {run}/fit_{w}.txt 2>&1" for w in range(args.workers))
        run_job(args.session, f"fit_{args.run}",
                f"mkdir -p {run} && cd {VM}/scripts && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ({workers} & wait); cat {run}/fit_*.txt > {run}/fit.txt; "
                f"python proxy_eval.py {run}/episodes --episodes {eps} --csv {run}/proxy_eval.csv "
                f"2>&1 | tee {run}/proxy_eval.txt")
        wait_job(args.session, f"fit_{args.run}")
        pull_run(args.session, args.run)
        out = ROOT / "results" / args.run / "proxy_eval.txt"
        print(out.read_text() if out.exists() else "no proxy_eval output")
    elif args.cmd == "pull-run":
        pull_run(args.session, args.run)
    elif args.cmd == "upload-raw":
        members = [(p, f"{p.parent.name}/{p.name}") for d in ("crops", "meshes", "primitives")
                   for p in sorted((LOCAL_RAW / d).glob("*"))]
        upload_tar(args.session, "object_raw", members, RAW)


if __name__ == "__main__":
    sys.exit(main())

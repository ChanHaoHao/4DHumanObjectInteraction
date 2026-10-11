#!/usr/bin/env python3
"""Drive the name-free object finder test (motion_vm.py) on one or two Colab VMs. All compute is remote.

    python3 scripts/motion_colab.py setup SESSION --episodes 0,5,7,8     # D4RT + SAM 3, data, slim cache, person tracks
    python3 scripts/motion_colab.py run SESSION --episodes 0,5,7,8 [--stages motion,seed,text] [-- motion_vm.py args]
    python3 scripts/motion_colab.py pull SESSION --episodes 0,5,7,8

VM layout mirrors the repo at /content/repo (scripts/, cache/ slim: masks, camera, meta; cache/raw/<ep>/sam_person.npz;
data/v2d_submission_kit/). Outputs: results/motion2/ (motion npz kept on the VM; candidate tracks, json and check images
pulled). OpenD4RT at /content/Open-d4rt, SAM 3 at /content/sam3, videos at /content/v2d/track_1.
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
OUT = f"{VM}/results/motion2"
SCRIPTS = ["motion_vm.py", "d4rt_vm.py", "build_cache.py", "preprocess_vm.py"]
SETUP = r"""
set -e
mkdir -p /content/logs
cd /content
[ -d sam3 ] || git clone -q https://github.com/facebookresearch/sam3.git
[ -d Open-d4rt ] || git clone -q https://github.com/Lijiaxin0111/Open-d4rt.git
pip install -q -e ./sam3
pip install -q einops pycocotools psutil decord PyYAML opencv-python-headless imageio lz4 pyarrow pandas scipy
export HF_TOKEN=$(cat /content/.hf_token)
python -c "
from huggingface_hub import hf_hub_download
for f in ('model.yaml', 'opend4rt.ckpt'):
    print(hf_hub_download('Lijiaxin0111/OpenD4RT', 'CKPT/' + f, local_dir='/content/Open-d4rt'))
"
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset', allow_patterns=['track_1/*'], local_dir='/content/v2d')"
cd /content/work 2>/dev/null || mkdir -p /content/work && cd /content/work
python -c "import sam3, numpy, cv2, torch; print('sam3 ok, numpy', numpy.__version__, 'cv2', cv2.__version__, 'torch', torch.__version__)"
ls /content/v2d/track_1/videos/chunk-000/*/ | wc -l > /content/setup_done
""".replace("CKPT", CKPT)


def upload_scripts(session: str):
    for name in SCRIPTS:
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")


def setup(session: str, episodes: list[int]):
    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p {VM}/scripts {VM}/data /content/logs && printf %s {env['HF_TOKEN']} > /content/.hf_token")
    if "setup_motion.txt" not in ls(session, "/content/logs"):
        launch(session, "setup_motion", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    upload_scripts(session)
    sample = ROOT / "data/v2d_submission_kit/data/track_1_sample_submission.parquet"
    members = [(sample, "data/v2d_submission_kit/data/track_1_sample_submission.parquet")]
    for e in episodes:
        n = f"episode_{e:06d}"
        members += [(ROOT / "cache" / n / f, f"cache/{n}/{f}") for f in ("masks.npz", "camera.npz", "meta.json")]
        members.append((ROOT / "cache/raw" / n / "sam_person.npz", f"cache/raw/{n}/sam_person.npz"))
    upload_tar(session, f"motion_{session}", members, VM)
    print(f"[{session}] uploaded {len(episodes)} episodes", flush=True)
    while "setup_done" not in ls(session, "/content"):
        log = remote(session, "tail -3 /content/logs/setup_motion.txt")
        if "Error" in log or "FAILED" in log:
            stop.set()
            raise RuntimeError(f"[{session}] setup failed:\n{log}")
        time.sleep(30)
    stop.set()
    print(remote(session, "tail -2 /content/logs/setup_motion.txt; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))


def run(session: str, episodes: list[int], stages: str, extra: str):
    upload_scripts(session)
    eps = ",".join(map(str, episodes))
    cmds = [f"python motion_vm.py --stage {s} --episodes {eps} --out {OUT} {extra} > {OUT}/{s}_{session}.txt 2>&1"
            for s in stages.split(",")]
    job(session, f"motion2_{session}", f"mkdir -p {OUT} && cd /content/work && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                                       f"&& export PYTHONPATH={VM}/scripts && " + " ; ".join(c.replace("python motion_vm.py",
                                                                                                   f"python {VM}/scripts/motion_vm.py") for c in cmds))


def pull(session: str, episodes: list[int]):
    dst = ROOT / "results" / "motion2"
    dst.mkdir(parents=True, exist_ok=True)
    have = set(ls(session, OUT))
    for e in episodes:
        n = f"episode_{e:06d}"
        for f in (f"{n}_cand.npz", f"{n}_text.npz", f"{n}_seed.json", f"{n}_seed.jpg", f"{n}_motion_small.npz"):
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
        if name == "run":
            p.add_argument("--stages", default="motion,seed,text")
    args, rest = parser.parse_known_args()
    extra = " ".join(a for a in rest if a != "--")
    eps = parse_episodes(args.episodes)
    if args.cmd == "setup":
        setup(args.session, eps)
    elif args.cmd == "run":
        run(args.session, eps, args.stages, extra)
    else:
        pull(args.session, eps)


if __name__ == "__main__":
    sys.exit(main())

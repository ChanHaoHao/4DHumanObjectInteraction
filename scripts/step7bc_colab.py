#!/usr/bin/env python3
"""Drive Steps 7b/7c (joint_fit.py) on one or two Colab VMs. All compute is remote; this uploads, launches, pulls.

    python3 scripts/step7bc_colab.py setup SESSION --episodes 0-14
    python3 scripts/step7bc_colab.py fit SESSION RUN --human human_v1 --episodes 0-14 [-- joint_fit.py args]
    python3 scripts/step7bc_colab.py render SESSION RUN --episodes 0,12
    python3 scripts/step7bc_colab.py merge RUN

setup   deps, MHR, dataset, kit, scripts; the MoGe cache (masks, depth, meta) of these episodes and the D4RT cache
        of the hoop episodes; the human runs (human_v1, human_d4dep) and the object inits (object_v3_w5d03 = MoGe
        Step 5 + 7a, object_d4_push = the same on D4RT).
fit     joint_fit.py (default init object_v3_w5d03 + MoGe cache; the hoop from object_d4_push + D4RT cache)
        -> results/RUN_j, then 7a's pushout.py on the non-hoop episodes -> results/RUN (--no-push: joint_fit
        writes RUN directly), then proxy_eval.py against the MoGe cache; pulls episodes/, joint_meta/ and the tables (named by session).
merge   joins the per-session proxy tables of a run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d4rt_steps_colab import job, pull, pull_files, upload_tar  # noqa: E402
from step3_colab import colab, ls, remote  # noqa: E402
from step4_colab import parse_episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VM = "/content/repo"
HOOP = [0, 1, 2]
HUMANS = ["human_v1", "human_d4dep"]
INIT, INIT_D4 = "object_v3_w5d03", "object_d4_push"
SCRIPTS = ["human_vm.py", "human_post.py", "object_fit.py", "pushout.py", "proxy_eval.py", "joint_fit.py",
           "render_video.py"]
SETUP = r"""
set -e
mkdir -p /content/logs
python -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" > /content/pins.txt
pip install -q -c /content/pins.txt trimesh fast_simplification pyarrow pandas scipy numba opencv-python-headless
if [ ! -f /content/repo/body_models/mhr/mhr_model.pt ]; then
  curl -sL -o /content/mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  unzip -q -j -o /content/mhr_assets.zip assets/mhr_model.pt -d /content/repo/body_models/mhr/
fi
echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  /content/repo/body_models/mhr/mhr_model.pt" | sha256sum -c
export HF_TOKEN=$(cat /content/.hf_token)
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset', allow_patterns=['track_1/*'], local_dir='/content/v2d')"
ls /content/v2d/track_1/videos/chunk-000/*/ | wc -l > /content/setup_done
"""


def eps_arg(episodes: list[int]) -> str:
    return ",".join(map(str, episodes))


def upload_scripts(session: str):
    for name in SCRIPTS:
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")


def setup(session: str, episodes: list[int]):
    import threading
    import time

    from step3_colab import keepalive, launch

    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p {VM}/scripts {VM}/body_models/mhr {VM}/data {VM}/results /content/logs && "
                    f"printf %s {env['HF_TOKEN']} > /content/.hf_token")
    if "setup_7bc.txt" not in ls(session, "/content/logs"):
        launch(session, "setup_7bc", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    upload_scripts(session)
    if not ls(session, f"{VM}/data/v2d_submission_kit"):
        upload_tar(session, f"kit_{session}", [(ROOT / "data" / "v2d_submission_kit", "v2d_submission_kit")], f"{VM}/data")
    if not ls(session, f"{VM}/results/{INIT_D4}/episodes"):
        upload_tar(session, f"runs_{session}", [(ROOT / "results" / r / "episodes", f"{r}/episodes")
                                                for r in HUMANS + [INIT, INIT_D4]], f"{VM}/results")
    for e in episodes:
        n = f"episode_{e:06d}"
        members = []
        if "depth.npz" not in ls(session, f"{VM}/cache/{n}"):
            members += [(ROOT / "cache" / n / f, f"{n}/{f}") for f in ("masks.npz", "camera.npz", "meta.json", "depth.npz")]
        if e in HOOP and "depth.npz" not in ls(session, f"{VM}/cache/d4rt/{n}"):
            members += [(ROOT / "cache" / "d4rt" / n / f, f"d4rt/{n}/{f}")
                        for f in ("masks.npz", "camera.npz", "meta.json", "depth.npz")]
        if members:
            upload_tar(session, f"{session}_c{e}", members, f"{VM}/cache")
        print(f"[{session}] cache {n} on the VM", flush=True)
    while "setup_done" not in remote(session, "ls /content"):
        log = remote(session, "tail -3 /content/logs/setup_7bc.txt")
        if "Error" in log or "FAILED" in log:
            stop.set()
            raise RuntimeError(f"[{session}] setup failed:\n{log}")
        time.sleep(20)
    stop.set()
    print(remote(session, f"cd {VM}/scripts && python -c 'import joint_fit, proxy_eval, render_video' && echo imports OK; "
                          "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"), flush=True)


def fit(session: str, run: str, human: str, episodes: list[int], extra: str, push: bool = True):
    """joint_fit.py -> <run>_j, then (push) 7a's pushout.py (w_pen 5, w_dep 0.3) -> <run> on the non-hoop episodes
    (hoop episodes copied: no penetration there, and a free re-fit costs the thin ring IoU); proxy on <run>."""
    upload_scripts(session)
    r = f"{VM}/results/{run}"
    rj = f"{r}_j" if push else r
    alt = f"--alt 0-2:{VM}/results/{INIT_D4}/episodes:{VM}/cache/d4rt" if set(HOOP) & set(episodes) else ""
    cmd = (f"mkdir -p {r}/episodes {rj} && cd {VM}/scripts && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
           f"python joint_fit.py --human {VM}/results/{human}/episodes --init {VM}/results/{INIT}/episodes "
           f"--run {rj} --episodes {eps_arg(episodes)} --cache {VM}/cache {alt} {extra} > {r}/fit_{session}.txt 2>&1; ")
    if push:
        rest = [e for e in episodes if e not in HOOP]
        hoop = [e for e in episodes if e in HOOP]
        if rest:
            cmd += (f"python pushout.py --src {rj}/episodes --run {r} --episodes {eps_arg(rest)} --cache {VM}/cache "
                    f"--w-pen 5 --w-dep 0.3 > {r}/pushout_{session}.txt 2>&1; cp {r}/pushout.json {r}/pushout_{session}.json; ")
        for e in hoop:
            cmd += f"cp {rj}/episodes/episode_{e:06d}* {r}/episodes/; "
    cmd += (f"python proxy_eval.py {r}/episodes --cache {VM}/cache --episodes {eps_arg(episodes)} "
            f"--csv {r}/proxy_eval_{session}.csv > {r}/proxy_eval_{session}.txt 2>&1")
    job(session, f"fit_{run}_{session}", cmd)  # per-session name: launch() stages the script locally by name
    dst = ROOT / "results" / run
    pull(session, f"{r}/episodes", dst / "episodes")
    pull(session, f"{rj}/joint_meta", dst / "joint_meta")
    pull_files(session, run, [f"fit_{session}.txt", f"proxy_eval_{session}.csv", f"proxy_eval_{session}.txt",
                              f"pushout_{session}.txt", f"pushout_{session}.json"])
    print((dst / f"proxy_eval_{session}.txt").read_text() if (dst / f"proxy_eval_{session}.txt").exists() else "no table")


def render(session: str, run: str, episodes: list[int]):
    upload_scripts(session)
    job(session, f"render_{run}_{session}", f"cd {VM}/scripts && python render_video.py --run {VM}/results/{run} "
                                  f"--episodes {eps_arg(episodes)} --cache {VM}/cache > {VM}/results/{run}/render_{session}.txt 2>&1")
    pull(session, f"{VM}/results/{run}/videos", ROOT / "results" / run / "videos", suffixes=[".mp4"])


def merge(run: str):
    folder = ROOT / "results" / run
    parts = sorted(folder.glob("proxy_eval_*.csv"))
    rows = sorted((r for p in parts for r in csv.DictReader(p.open())), key=lambda r: int(r["episode"]))
    with (folder / "proxy_eval.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{run}/proxy_eval.csv: {len(rows)} episodes from {len(parts)} parts")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("session"), p.add_argument("--episodes", required=True)
    p = sub.add_parser("fit")
    p.add_argument("session"), p.add_argument("run"), p.add_argument("--human", default="human_v1")
    p.add_argument("--episodes", required=True)
    p.add_argument("--no-push", dest="push", action="store_false", help="skip the 7a push-out pass")
    p = sub.add_parser("render")
    p.add_argument("session"), p.add_argument("run"), p.add_argument("--episodes", required=True)
    sub.add_parser("merge").add_argument("run")
    args, rest = parser.parse_known_args()
    extra = " ".join(a for a in rest if a != "--")
    if args.cmd == "setup":
        setup(args.session, parse_episodes(args.episodes))
    elif args.cmd == "fit":
        fit(args.session, args.run, args.human, parse_episodes(args.episodes), extra, args.push)
    elif args.cmd == "render":
        render(args.session, args.run, parse_episodes(args.episodes))
    elif args.cmd == "merge":
        merge(args.run)


if __name__ == "__main__":
    sys.exit(main())

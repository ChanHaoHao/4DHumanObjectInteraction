#!/usr/bin/env python3
"""Rerun Steps 4, 5 and 7a on the D4RT cache (cache/d4rt) and render videos, split over one or two Colab VMs.

    python3 scripts/d4rt_steps_colab.py pipeline SESSION --episodes 0-14     # one VM per half; nothing is shared
    python3 scripts/d4rt_steps_colab.py merge                                  # combine both VMs' tables

Per VM, for its episodes (pipeline = these in order; each is also its own command):
  setup     deps, MHR, dataset, kit, scripts; cache/d4rt and the MoGe cache (masks + depth on the frames
            proxy_eval scores, for cross-scoring); raw SAM 3D Body outputs, object meshes, human_v1 and
            object_v3_w5d03 episodes (baselines). Uploads skip what is already there, so a rerun resumes.
  step4     human_post.py on cache/d4rt -> results/human_d4 (+ --depth moge variant results/human_d4dep,
            i.e. translation following the D4RT actor depth); proxy_eval vs cache/d4rt
  step5     object_fit.py (hoop: torus primitive, as object_v2) -> results/object_d4
  step7     pushout.py --w-pen 5 --w-dep 0.3 (object_v3_w5d03's settings) -> results/object_d4_push
  cross     proxy_eval of object_d4_push against the MoGe cache (with human_v1's depth_scale) and of
            object_v3_w5d03 against cache/d4rt (with human_d4's), so both runs are scored on both depths
  render    render_video.py -> results/object_d4_push/videos/episode_XXXXXX.mp4
Tables written by a VM carry its session name (proxy_eval_<session>.csv, ...); `merge` joins them.
Layout on the VM mirrors the repo at /content/repo; videos at /content/v2d/track_1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_colab import colab, keepalive, launch, ls, remote  # noqa: E402
from step4_colab import parse_episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VM = "/content/repo"
D4 = f"{VM}/cache/d4rt"
HUMAN, HUMAN_DEP, OBJECT, PUSH = "human_d4", "human_d4dep", "object_d4", "object_d4_push"
BASE_HUMAN, BASE_OBJECT = "human_v1", "object_v3_w5d03"
HOOP = [0, 1, 2]
TORUS = "/content/object_raw/primitives/hula_hoop_torus.obj"
SCRIPTS = ["human_vm.py", "human_post.py", "object_fit.py", "pushout.py", "proxy_eval.py", "render_video.py"]
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
which ffmpeg
ls /content/v2d/track_1/videos/chunk-000/*/ | wc -l > /content/setup_done
"""


# ------------------------------------------------------------------------------------------- helpers
PART = 40 << 20  # uploads of ~80 MB fail with a server error; send tars in 40 MB pieces


def upload_tar(session: str, name: str, members: list[tuple[Path, str]], dest: str):
    """Pack (path, arcname) into one tar, upload it in PART-sized pieces, join and unpack under dest on the VM."""
    scratch = ROOT / "results" / "colab_jobs"
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch / f"{name}.tar"
    with tarfile.open(local, "w") as tar:
        for path, arcname in members:
            tar.add(path, arcname=arcname)
    data = local.read_bytes()
    local.unlink()
    pieces = []
    for i in range(0, max(len(data), 1), PART):
        piece = scratch / f"{name}.part{i // PART:03d}"
        piece.write_bytes(data[i:i + PART])
        colab("upload", "-s", session, str(piece), f"/content/{piece.name}", timeout=1800)
        piece.unlink()
        pieces.append(f"/content/{piece.name}")
    remote(session, f"mkdir -p {dest} && cat {' '.join(pieces)} > /content/{name}.tar && rm {' '.join(pieces)} && "
                    f"tar -xf /content/{name}.tar -C {dest} && rm /content/{name}.tar")
    out = remote(session, f"ls {dest} | head -3")
    if "No such file" in out:
        raise RuntimeError(f"[{session}] unpacking {name} failed: {out[-300:]}")


def job(session: str, name: str, cmd: str, every: int = 30):
    """Launch cmd on the VM and block until it ends, keeping the kernel alive."""
    remote(session, f"rm -f /content/logs/{name}.done")
    launch(session, name, f"{cmd}; echo done > /content/logs/{name}.done")
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    while f"{name}.done" not in ls(session, "/content/logs"):
        time.sleep(every)
    stop.set()
    print(f"[{session}] {name} finished", flush=True)


def pull(session: str, remote_dir: str, local: Path, suffixes=None) -> list[str]:
    local.mkdir(parents=True, exist_ok=True)
    got = []
    for f in ls(session, remote_dir):
        if ".partial" in f or (suffixes and not f.endswith(tuple(suffixes))):
            continue
        if not (local / f).exists():
            colab("download", "-s", session, f"{remote_dir}/{f}", str(local / f), timeout=1800)
            got.append(f)
    return got


def pull_files(session: str, run: str, names: list[str]):
    for f in names:
        try:
            colab("download", "-s", session, f"{VM}/results/{run}/{f}", str(ROOT / "results" / run / f))
        except RuntimeError:
            print(f"[{session}] no {run}/{f}", flush=True)


def eps_arg(episodes: list[int]) -> str:
    return ",".join(map(str, episodes))


def proxy(run: str, episodes: list[int], cache: str, out: str, extra: str = "") -> str:
    return (f"python proxy_eval.py {VM}/results/{run}/episodes --cache {cache} --episodes {eps_arg(episodes)} "
            f"--csv {VM}/results/{run}/{out}.csv {extra} > {VM}/results/{run}/{out}.txt 2>&1")


# -------------------------------------------------------------------------------------------- stages
def slim_moge(e: int) -> Path:
    """The MoGe depth.npz on the frames proxy_eval scores (every 5th scored frame): ~1/6 of the upload."""
    scored = json.loads((ROOT / "results" / "colab_jobs" / "scored_frames.json").read_text())[str(e)]
    out = ROOT / "results" / "colab_jobs" / f"moge_slim_{e}.npz"
    with np.load(ROOT / "cache" / f"episode_{e:06d}" / "depth.npz") as dz:
        frames = dz["frames"]
        keep = np.isin(frames, scored[::5])
        np.savez_compressed(out, frames=frames[keep], depth=dz["depth"][keep])
    return out


def setup(session: str, episodes: list[int]):
    """Uploads skip what the VM already has, so a restarted driver resumes."""
    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p {VM}/scripts {VM}/body_models/mhr {VM}/data {VM}/results /content/logs && "
                    f"printf %s {env['HF_TOKEN']} > /content/.hf_token")
    if "setup_steps.txt" not in ls(session, "/content/logs"):  # a restarted driver must not run it twice
        launch(session, "setup_steps", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    for name in SCRIPTS:
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")
    if not ls(session, f"{VM}/data/v2d_submission_kit"):
        upload_tar(session, f"kit_{session}", [(ROOT / "data" / "v2d_submission_kit", "v2d_submission_kit")], f"{VM}/data")
    if not ls(session, "/content/human_raw/dinov3"):
        upload_tar(session, f"hraw_{session}", [(ROOT / "results" / "human_raw" / "dinov3", "human_raw/dinov3")], "/content")
    if not ls(session, "/content/object_raw/meshes"):
        upload_tar(session, f"oraw_{session}", [(ROOT / "results" / "object_raw" / d, f"object_raw/{d}")
                                                for d in ("meshes", "primitives")], "/content")
    if not ls(session, f"{VM}/results/{BASE_OBJECT}/episodes"):
        upload_tar(session, f"base_{session}", [(ROOT / "results" / r / "episodes", f"{r}/episodes")
                                                for r in (BASE_HUMAN, BASE_OBJECT)], f"{VM}/results")
    # this VM's episodes: the D4RT cache, and the MoGe masks + slim depth (cross-scoring)
    for e in episodes:
        n = f"episode_{e:06d}"
        members = []
        if "meta.json" not in ls(session, f"{D4}/{n}"):
            members += [(ROOT / "cache" / "d4rt" / n / f, f"d4rt/{n}/{f}")
                        for f in ("masks.npz", "camera.npz", "depth.npz", "meta.json")]
        if e in episodes and "depth.npz" not in ls(session, f"{VM}/cache/{n}"):
            members += [(ROOT / "cache" / n / f, f"{n}/{f}") for f in ("masks.npz", "meta.json")]
            members.append((slim_moge(e), f"{n}/depth.npz"))
        if members:
            upload_tar(session, f"{session}_c{e}", members, f"{VM}/cache")
        (ROOT / "results" / "colab_jobs" / f"moge_slim_{e}.npz").unlink(missing_ok=True)
        print(f"[{session}] cache {n} on the VM", flush=True)
    while "setup_done" not in remote(session, "ls /content"):
        log = remote(session, "tail -3 /content/logs/setup_steps.txt")
        if "Error" in log or "FAILED" in log:
            stop.set()
            raise RuntimeError(f"[{session}] setup failed:\n{log}")
        time.sleep(20)
    stop.set()
    print(remote(session, f"cd {VM}/scripts && python -c 'import proxy_eval, object_fit, pushout, human_post, render_video' "
                          "&& echo imports OK"), flush=True)
    print(remote(session, "tail -4 /content/logs/setup_steps.txt; cat /content/setup_done; "
                          "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"), flush=True)


def step4(session: str, episodes: list[int]):
    """Main run and the depth-following variant side by side (CPU-bound post-processing + MHR)."""
    cmds = []
    for run, flags in ((HUMAN, ""), (HUMAN_DEP, "--depth moge")):
        r = f"{VM}/results/{run}"
        cmds.append(f"(mkdir -p {r} && python human_post.py --raw /content/human_raw/dinov3 --run {r} "
                    f"--episodes {eps_arg(episodes)} --cache {D4} {flags} > {r}/post_{session}.txt 2>&1; "
                    + proxy(run, episodes, D4, f"proxy_eval_{session}") + ")")
    job(session, "step4", f"cd {VM}/scripts && ({' & '.join(cmds)} & wait)")
    for run in (HUMAN, HUMAN_DEP):
        dst = ROOT / "results" / run
        for sub in ("episodes", "human_meta", "overlays"):
            pull(session, f"{VM}/results/{run}/{sub}", dst / sub)
        pull_files(session, run, [f"post_{session}.txt", f"proxy_eval_{session}.csv", f"proxy_eval_{session}.txt"])


def step5(session: str, episodes: list[int], workers: int = 2):
    r = f"{VM}/results/{OBJECT}"
    base = f"python object_fit.py --human {VM}/results/{HUMAN}/episodes --run {r} --cache {D4}"
    hoop, rest = [e for e in episodes if e in HOOP], [e for e in episodes if e not in HOOP]
    fits = []
    for w in range(workers):
        cmd = f"{base} --episodes {eps_arg(rest)}" if rest else "true"
        if w == 0 and hoop:
            cmd = f"{base} --episodes {eps_arg(hoop)} --mesh {TORUS}; {cmd}"
        fits.append(f"({cmd}) > {r}/fit_{session}_{w}.txt 2>&1")
    job(session, "step5", f"mkdir -p {r} && cd {VM}/scripts && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
                          f"({' & '.join(fits)} & wait); cat {r}/fit_{session}_*.txt > {r}/fit_{session}.txt; "
                          + proxy(OBJECT, episodes, D4, f"proxy_eval_{session}"))
    dst = ROOT / "results" / OBJECT
    for sub in ("episodes", "object_meta", "overlays"):
        pull(session, f"{r}/{sub}", dst / sub)
    pull_files(session, OBJECT, [f"fit_{session}.txt", f"proxy_eval_{session}.csv", f"proxy_eval_{session}.txt"])


def step7(session: str, episodes: list[int]):
    r = f"{VM}/results/{PUSH}"
    job(session, "step7", f"mkdir -p {r} && cd {VM}/scripts && python pushout.py --src {VM}/results/{OBJECT}/episodes "
                          f"--run {r} --episodes {eps_arg(episodes)} --cache {D4} --w-pen 5 --w-dep 0.3 "
                          f"> {r}/pushout_{session}.txt 2>&1; cp {r}/pushout.json {r}/pushout_{session}.json; "
                          + proxy(PUSH, episodes, D4, f"proxy_eval_{session}"))
    pull(session, f"{r}/episodes", ROOT / "results" / PUSH / "episodes")
    pull_files(session, PUSH, [f"pushout_{session}.txt", f"pushout_{session}.json",
                               f"proxy_eval_{session}.csv", f"proxy_eval_{session}.txt"])


def cross(session: str, episodes: list[int]):
    """Each run scored on the other cache, with the depth_scale of the Step 4 run made from that cache."""
    job(session, "cross", f"cd {VM}/scripts && "
        + proxy(PUSH, episodes, f"{VM}/cache", f"proxy_moge_{session}",
                f"--depth-scale-from {VM}/results/{BASE_HUMAN}/episodes") + "; "
        + proxy(BASE_OBJECT, episodes, D4, f"proxy_d4rt_{session}",
                f"--depth-scale-from {VM}/results/{HUMAN}/episodes"))
    pull_files(session, PUSH, [f"proxy_moge_{session}.csv", f"proxy_moge_{session}.txt"])
    pull_files(session, BASE_OBJECT, [f"proxy_d4rt_{session}.csv", f"proxy_d4rt_{session}.txt"])


def render(session: str, episodes: list[int]):
    job(session, "render", f"cd {VM}/scripts && python render_video.py --run {VM}/results/{PUSH} "
                           f"--episodes {eps_arg(episodes)} --cache {D4} > {VM}/results/{PUSH}/render_{session}.txt 2>&1")
    pull(session, f"{VM}/results/{PUSH}/videos", ROOT / "results" / PUSH / "videos", suffixes=[".mp4"])
    pull_files(session, PUSH, [f"render_{session}.txt"])


def pipeline(session: str, episodes: list[int], workers: int):
    setup(session, episodes)
    step4(session, episodes)
    step5(session, episodes, workers)
    step7(session, episodes)
    cross(session, episodes)
    render(session, episodes)
    print(f"[{session}] pipeline done", flush=True)


def merge():
    """Join the per-VM tables: <run>/<name>_<session>.csv -> <run>/<name>.csv, pushout_*.json -> pushout.json."""
    for run, names in ((HUMAN, ["proxy_eval"]), (HUMAN_DEP, ["proxy_eval"]), (OBJECT, ["proxy_eval"]), (PUSH, ["proxy_eval", "proxy_moge"]), (BASE_OBJECT, ["proxy_d4rt"])):
        folder = ROOT / "results" / run
        for name in names:
            parts = sorted(folder.glob(f"{name}_*.csv"))
            if parts:
                rows = [r for p in parts for r in csv.DictReader(p.open())]
                rows.sort(key=lambda r: int(r["episode"]))
                with (folder / f"{name}.csv").open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{run}/{name}.csv: {len(rows)} episodes from {len(parts)} parts")
    merged = {}
    for p in sorted((ROOT / "results" / PUSH).glob("pushout_*.json")):
        merged.update(json.loads(p.read_text()))
    if merged:
        (ROOT / "results" / PUSH / "pushout.json").write_text(json.dumps(merged, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pipeline")
    p.add_argument("session"), p.add_argument("--episodes", required=True)
    p.add_argument("--workers", type=int, default=2)
    for name in ("setup", "step4", "step5", "step7", "cross", "render"):
        p = sub.add_parser(name)
        p.add_argument("session"), p.add_argument("--episodes", required=True)
    sub.add_parser("merge")
    args = parser.parse_args()
    eps = parse_episodes(args.episodes) if getattr(args, "episodes", None) else []
    if args.cmd == "pipeline":
        pipeline(args.session, eps, args.workers)
    elif args.cmd == "setup":
        setup(args.session, eps)
    elif args.cmd == "step4":
        step4(args.session, eps)
    elif args.cmd == "step5":
        step5(args.session, eps)
    elif args.cmd == "step7":
        step7(args.session, eps)
    elif args.cmd == "cross":
        cross(args.session, eps)
    elif args.cmd == "render":
        render(args.session, eps)
    elif args.cmd == "merge":
        merge()


if __name__ == "__main__":
    sys.exit(main())

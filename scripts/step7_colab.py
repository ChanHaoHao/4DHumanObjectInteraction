#!/usr/bin/env python3
"""Drive Step 7 on a Colab VM. All compute is remote; this uploads, launches and pulls.

    python3 scripts/step7_colab.py setup SESSION [--src object_v2] [--no-depth]   # deps, kit, MHR, cache, source run
    python3 scripts/step7_colab.py pushout SESSION RUN [--src object_v2] [--episodes 0-29] [-- pushout.py args]
    python3 scripts/step7_colab.py pull-run SESSION RUN

Layout on the VM mirrors the repo at /content/repo (scripts/, cache/, results/<run>/, data/v2d_submission_kit/,
body_models/mhr/). `pushout` runs pushout.py then proxy_eval.py over the same episodes and pulls
results/<RUN>/ (episodes/, pushout.json, proxy_eval.{csv,txt}). Between commands run
step3_colab.py keepalive SESSION (Colab reclaims VMs ~45 min after the last kernel interaction).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_colab import colab, keepalive, launch, remote  # noqa: E402
from step4_colab import parse_episodes, upload_depth, upload_tar, wait_job  # noqa: E402
from step5_colab import pull_dir, run_job  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VM = "/content/repo"
SCRIPTS = ["proxy_eval.py", "pushout.py", "object_fit.py", "human_vm.py"]
SETUP = r"""
set -e
pip install -q trimesh fast_simplification pyarrow pandas scipy numba
if [ ! -f /content/repo/body_models/mhr/mhr_model.pt ]; then
  curl -sL -o /content/mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  unzip -q -j -o /content/mhr_assets.zip assets/mhr_model.pt -d /content/repo/body_models/mhr/
fi
echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  /content/repo/body_models/mhr/mhr_model.pt" | sha256sum -c
cd /content/repo/scripts && python -c "import proxy_eval, pushout; print('imports OK')"
echo ok > /content/setup_done
"""


def upload_scripts(session: str):
    for name in SCRIPTS:
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")


def setup(session: str, src: str, depth: bool):
    remote(session, f"mkdir -p /content/work {VM}/scripts {VM}/body_models/mhr {VM}/data {VM}/results")
    upload_tar(session, "kit", [(ROOT / "data" / "v2d_submission_kit", "v2d_submission_kit")], f"{VM}/data")
    cache = [(p, f"{p.parent.name}/{p.name}") for e in range(30)
             for p in sorted((ROOT / "cache" / f"episode_{e:06d}").glob("*"))
             if p.name in ("masks.npz", "camera.npz", "meta.json")]
    upload_tar(session, "cache", cache, f"{VM}/cache")
    upload_tar(session, "src", [(ROOT / "results" / src / "episodes", f"{src}/episodes")], f"{VM}/results")
    upload_scripts(session)
    launch(session, "setup7", SETUP)
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    if depth:
        upload_depth(session, list(range(30)))
    while "setup_done" not in remote(session, "ls /content"):
        log = remote(session, "tail -3 /content/logs/setup7.txt")
        if "Error" in log or "FAILED" in log:
            print(log)
            break
        time.sleep(15)
    stop.set()
    print(remote(session, "tail -3 /content/logs/setup7.txt"))


def pull_run(session: str, run: str):
    dst = ROOT / "results" / run
    pull_dir(session, f"{VM}/results/{run}/episodes", dst / "episodes")
    for f in ("pushout.json", "pushout.txt", "proxy_eval.csv", "proxy_eval.txt"):
        try:
            colab("download", "-s", session, f"{VM}/results/{run}/{f}", str(dst / f))
        except RuntimeError:
            print(f"no {f}")
    print(f"pulled results/{run}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("session"), p.add_argument("--src", default="object_v2")
    p.add_argument("--no-depth", action="store_true", help="skip depth.npz (proxy reprojection without dep_*)")
    sub.add_parser("scripts").add_argument("session")
    p = sub.add_parser("pushout")
    p.add_argument("session"), p.add_argument("run"), p.add_argument("--src", default="object_v2")
    p.add_argument("--episodes", default="0-29")
    p = sub.add_parser("pull-run")
    p.add_argument("session"), p.add_argument("run")
    args, rest = parser.parse_known_args()  # unknown arguments (after an optional --) go to the VM script
    extra = " ".join(a for a in rest if a != "--")
    if args.cmd == "setup":
        setup(args.session, args.src, not args.no_depth)
    elif args.cmd == "scripts":
        upload_scripts(args.session)
    elif args.cmd == "pushout":
        upload_scripts(args.session)
        run = f"{VM}/results/{args.run}"
        eps = ",".join(map(str, parse_episodes(args.episodes)))
        run_job(args.session, f"pushout_{args.run}",
                f"mkdir -p {run} && cd {VM}/scripts && "
                f"python pushout.py --src {VM}/results/{args.src}/episodes --run {run} --episodes {args.episodes} "
                f"{extra} 2>&1 | tee {run}/pushout.txt; "
                f"python proxy_eval.py {run}/episodes --episodes {eps} --csv {run}/proxy_eval.csv "
                f"2>&1 | tee {run}/proxy_eval.txt")
        wait_job(args.session, f"pushout_{args.run}")
        pull_run(args.session, args.run)
        for name in ("pushout.txt", "proxy_eval.txt"):
            out = ROOT / "results" / args.run / name
            print(out.read_text() if out.exists() else f"no {name}")
    elif args.cmd == "pull-run":
        pull_run(args.session, args.run)


if __name__ == "__main__":
    sys.exit(main())

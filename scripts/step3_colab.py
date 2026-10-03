#!/usr/bin/env python3
"""Drive Step 3 preprocessing on Colab VMs (all compute remote; this only uploads, launches and pulls).

    python3 scripts/step3_colab.py setup  SESSION            # install SAM 3 + MoGe, fetch data, upload scripts
    python3 scripts/step3_colab.py launch SESSION NAME CMD   # run CMD detached in /content/work, log /content/logs/NAME.txt
    python3 scripts/step3_colab.py status SESSION...         # last log lines of every job
    python3 scripts/step3_colab.py sync --sam SESSION --build SESSION   # loop: relay SAM tracks, pull finished cache
    python3 scripts/step3_colab.py rebuild --sam S --build B EPISODE...   # after extra prompts / fallback
    python3 scripts/step3_colab.py pull SESSION EPISODE...   # re-pull rebuilt episodes

Layout on a VM: /content/v2d/track_1 (data), /content/raw/episode_XXXXXX (preprocess_vm.py output),
/content/cache/episode_XXXXXX + /content/cache_check (build_cache.py output). VMs are reclaimed when the
kernel sees no interaction for ~45 min, so `sync` pings each kernel every 4 min and pulls every finished
episode right away into cache/ and results/cache_check/.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SAM = ("sam_person.npz", "sam_object.npz")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][A-Za-z0-9]|\x1b[=>]|\r")
SAMPLE = ROOT / "data/v2d_submission_kit/data/track_1_sample_submission.parquet"
SETUP = r"""
set -e
mkdir -p /content/work /content/logs
cd /content
[ -d sam3 ] || git clone -q https://github.com/facebookresearch/sam3.git
[ -d MoGe ] || git clone -q https://github.com/microsoft/MoGe.git
# separately: sam3 pins numpy<2 and MoGe pulls numpy 2; both run fine on the result
pip install -q -e ./sam3
pip install -q -e ./MoGe
pip install -q einops pycocotools psutil decord  # decord: the SAM 3 tracker reads mp4 with it
export HF_TOKEN=$(cat /content/.hf_token)
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset', allow_patterns=['track_1/*'], local_dir='/content/v2d')"
ls /content/v2d/track_1/videos/chunk-000/*/ | wc -l > /content/setup_done
"""
# Builds every episode whose SAM tracks and depth are present and whose cache is missing
BUILD_LOOP = r"""
cd /content/work
while true; do
  for d in /content/raw/episode_*; do
    e=$((10#${d##*_}))
    if [ -f $d/sam_person.npz ] && [ -f $d/sam_object.npz ] && [ -f $d/depth_raw.npz ] \
       && [ ! -f /content/cache/$(basename $d)/meta.json ]; then
      python build_cache.py --raw /content/raw --out /content/cache --data /content/v2d/track_1 \
        --sample /content/work/track_1_sample_submission.parquet --episodes $e --overlay /content/cache_check
    fi
  done
  [ -f /content/build_stop ] && break
  sleep 30
done
"""


def colab(*args, stdin: str | None = None, timeout: int = 300, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            p = subprocess.run(["colab", *args], input=stdin, capture_output=True, text=True, timeout=timeout)
            if p.returncode == 0 and "Connection was lost" not in p.stderr + p.stdout:
                return p.stdout
            err = (p.stderr or p.stdout)[-400:]
        except subprocess.TimeoutExpired:
            err = "timeout"
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"colab {' '.join(args[:3])} failed: {err}")


def remote(session: str, bash: str, timeout: int = 300) -> str:
    """Run a short bash snippet on the VM through `colab console` and return its output.

    The console is a tmux shell, independent of the Jupyter kernel, so this works while a kernel cell
    is busy. Output is cut between two markers (spelled split in the command so its echo never matches).
    """
    script = f"printf '%s\\n' '__ST''ART__'\n{bash}\nprintf '%s\\n' '__E''ND__'\nexit\n"
    out = ANSI.sub("", colab("console", "-s", session, stdin=script, timeout=timeout))
    lines = [line.rstrip() for line in out.splitlines()]
    try:
        start = max(i for i, line in enumerate(lines) if "__START__" in line and "printf" not in line)
        end = max(i for i, line in enumerate(lines) if "__END__" in line and "printf" not in line)
    except ValueError:
        return out
    return "\n".join(lines[start + 1:end])


def launch(session: str, name: str, cmd: str):
    """Run `cmd` detached in /content/work (only the job is backgrounded, with its own stdio)."""
    local = ROOT / "results" / "colab_jobs" / f"{name}.sh"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(cmd.strip() + "\n")
    remote(session, "mkdir -p /content/logs")
    colab("upload", "-s", session, str(local), f"/content/logs/{name}.sh")
    remote(session, f"cd /content/work; setsid nohup bash /content/logs/{name}.sh "
                    f"> /content/logs/{name}.txt 2>&1 < /dev/null &")
    print(f"[{session}] launched {name}")


def setup(session: str):
    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p /content/work && printf %s {env['HF_TOKEN']} > /content/.hf_token")  # noqa: E501
    for path in ["scripts/preprocess_vm.py", "scripts/build_cache.py"]:
        colab("upload", "-s", session, str(ROOT / path), f"/content/work/{Path(path).name}")
    colab("upload", "-s", session, str(SAMPLE), f"/content/work/{SAMPLE.name}")
    launch(session, "setup", SETUP)
    while "setup_done" not in remote(session, "ls /content; tail -3 /content/logs/setup.txt"):
        time.sleep(20)
    print(f"[{session}] setup done:", remote(session, "cat /content/setup_done").strip(), "videos")


def status(sessions: list[str]):
    for s in sessions:
        print(f"== {s}")
        print(remote(s, "for f in /content/logs/*.txt; do echo \"-- $f\"; grep -a '^ep \\|Error\\|Traceback' $f | tail -4; "
                        "done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"))


def ls(session: str, path: str) -> list[str]:
    """Names in a VM directory, via the Jupyter contents API (works while the kernel is busy)."""
    try:
        out = colab("ls", "-s", session, path, timeout=120, retries=2)
    except RuntimeError:
        return []
    return [line.strip().rstrip("/") for line in out.splitlines() if line.strip() and "[colab]" not in line]


def keepalive(sessions: list[str], every: int = 240, stop: threading.Event | None = None):
    """Ping each kernel with a trivial exec. Colab reclaims a VM ~45 min after the last kernel
    interaction even while a detached job keeps its GPU busy (ls/upload/download/console don't count)."""
    stop = stop or threading.Event()
    while not stop.is_set():
        for s in sessions:
            try:
                colab("exec", "-s", s, stdin="1", timeout=120, retries=2)
            except RuntimeError as err:
                print(f"keepalive {s} failed: {err}", flush=True)
        stop.wait(every)


def pull(build: str, n: str):
    """Download one finished episode's cache and check images from the build VM."""
    dst = ROOT / "cache" / n
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta.json").unlink(missing_ok=True)
    for f in ["masks.npz", "depth.npz", "camera.npz", "meta.json"]:  # meta last: marks the copy complete
        colab("download", "-s", build, f"/content/cache/{n}/{f}", str(dst / f), timeout=900)
    check = ROOT / "results/cache_check"
    check.mkdir(parents=True, exist_ok=True)
    for img in (f"{n}.jpg", f"{n}_tracks.jpg"):
        colab("download", "-s", build, f"/content/cache_check/{img}", str(check / img))
    m = json.loads((dst / "meta.json").read_text())
    keys = ["warning", "contact", "human_ids", "object_ids", "object_coverage_scored", "fov_x",
            "depth_scale_std", "camera_height_m", "actor_top_m_p90"]
    print(f"pulled {n}: " + ", ".join(f"{k}={m.get(k)}" for k in keys), flush=True)


def relay(sam: str, build: str, n: str) -> list[str]:
    """Copy every sam_*.npz of one episode from `sam` to `build` that `build` lacks; returns the names."""
    have = set(ls(build, f"/content/raw/{n}"))
    moved = []
    for f in ls(sam, f"/content/raw/{n}"):
        if f.startswith("sam_") and f.endswith(".npz") and ".partial" not in f and f not in have:
            local = ROOT / "cache/raw" / n / f
            local.parent.mkdir(parents=True, exist_ok=True)
            colab("download", "-s", sam, f"/content/raw/{n}/{f}", str(local))
            colab("upload", "-s", build, str(local), f"/content/raw/{n}/{f}")
            moved.append(f)
    return moved


def rebuild(sam: str, build: str, episodes: list[int]):
    """Relay new SAM candidates (extra prompts), rebuild on `build`, and pull the result."""
    names = [f"episode_{e:06d}" for e in episodes]
    for n in names:
        print(n, "relayed", relay(sam, build, n) if sam != build else [], flush=True)
    remote(build, "rm -f " + " ".join(f"/content/cache/{n}/meta.json" for n in names))
    for n in names:
        while "meta.json" not in ls(build, f"/content/cache/{n}"):
            time.sleep(20)
        pull(build, n)


def summary():
    """Markdown table of every pulled episode's meta.json."""
    rows = ["| ep | object prompt(s) | chosen object track | contact | obj cov | fov_x | scale std | cam h (m) | actor top (m) | warning |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for path in sorted((ROOT / "cache").glob("episode_*/meta.json")):
        m = json.loads(path.read_text())
        prompts = m["prompts"]["object"].split(" | ")
        files = ["base"] + [p.split(":")[0] if p.startswith("fallback") else p for p in prompts[1:]]
        chosen = sorted({files[int(i.split(":")[0])] if ":" in i else "base" for i in m["object_ids"]})
        rows.append(f"| {m['episode']} | {', '.join(p if not p.startswith('fallback') else 'fallback' for p in prompts)} "
                    f"| {', '.join(chosen) or '-'} | {m['contact']:.2f} | {m['object_coverage_scored']:.2f} "
                    f"| {m.get('fov_x', '')} | {m.get('depth_scale_std', '')} | {m.get('camera_height_m', '')} "
                    f"| {m.get('actor_top_m_p90', '')} | {m.get('warning') or ''} |")
    print("\n".join(rows))


def sync(sam: str, build: str, episodes: int = 30, every: int = 90):
    """Relay SAM tracks from `sam` to `build`, and pull each finished episode, until all are local."""
    threading.Thread(target=keepalive, args=(sorted({sam, build}),), daemon=True).start()
    names = [f"episode_{e:06d}" for e in range(episodes)]
    pulled = {n for n in names if (ROOT / "cache" / n / "meta.json").exists()}
    # a local copy means it was already relayed (sync downloads, then uploads); episodes already pulled
    # are skipped, so to relay a later file (extra prompt, fallback) delete that episode's local meta.json
    if sam != build:
        relayed = {(n, p.name) for n in names for p in (ROOT / "cache/raw" / n).glob("sam_*.npz")}
    else:
        relayed = {(n, f) for n in names for f in BASE_SAM}
    if sam != build:  # upload targets must exist before depth reaches the episode
        remote(build, "mkdir -p " + " ".join(f"/content/raw/{n}" for n in names))
    while len(pulled) < episodes:
        try:
            started = set(ls(sam, "/content/raw"))
            for n in names:
                if n not in started or (n in pulled and all((n, f) in relayed for f in BASE_SAM)):
                    continue
                for f in ls(sam, f"/content/raw/{n}"):
                    if not (f.startswith("sam_") and f.endswith(".npz")) or ".partial" in f or (n, f) in relayed:
                        continue
                    local = ROOT / "cache/raw" / n / f
                    local.parent.mkdir(parents=True, exist_ok=True)
                    colab("download", "-s", sam, f"/content/raw/{n}/{f}", str(local))
                    if sam != build:
                        colab("upload", "-s", build, str(local), f"/content/raw/{n}/{f}")
                    relayed.add((n, f))
                    print(f"relayed {n}/{f}", flush=True)
            built = set(ls(build, "/content/cache"))
            for n in names:
                if n in pulled or n not in built or "meta.json" not in ls(build, f"/content/cache/{n}"):
                    continue
                pull(build, n)
                pulled.add(n)
        except RuntimeError as err:
            print(f"sync error (will retry): {err}", flush=True)
        time.sleep(every)
    print("all episodes pulled", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").add_argument("session")
    p = sub.add_parser("launch")
    p.add_argument("session"), p.add_argument("name"), p.add_argument("command")
    sub.add_parser("status").add_argument("sessions", nargs="+")
    p = sub.add_parser("sync")
    p.add_argument("--sam", required=True), p.add_argument("--build", required=True)
    p.add_argument("--episodes", type=int, default=30)
    sub.add_parser("build-loop").add_argument("session")
    sub.add_parser("keepalive").add_argument("sessions", nargs="+")
    sub.add_parser("summary")
    p = sub.add_parser("rebuild")
    p.add_argument("--sam", required=True), p.add_argument("--build", required=True)
    p.add_argument("episodes", nargs="+", type=int)
    p = sub.add_parser("pull")
    p.add_argument("session"), p.add_argument("episodes", nargs="+", type=int)
    args = parser.parse_args()
    if args.cmd == "setup":
        setup(args.session)
    elif args.cmd == "launch":
        launch(args.session, args.name, args.command)
    elif args.cmd == "status":
        status(args.sessions)
    elif args.cmd == "sync":
        sync(args.sam, args.build, args.episodes)
    elif args.cmd == "summary":
        summary()
    elif args.cmd == "rebuild":
        rebuild(args.sam, args.build, args.episodes)
    elif args.cmd == "pull":
        for e in args.episodes:
            pull(args.session, f"episode_{e:06d}")
    elif args.cmd == "keepalive":
        keepalive(args.sessions)
    elif args.cmd == "build-loop":
        launch(args.session, "build", BUILD_LOOP)


if __name__ == "__main__":
    sys.exit(main())

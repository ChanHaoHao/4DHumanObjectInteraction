#!/usr/bin/env python3
"""End-to-end Track 1 run: object name from Gemini (local), every other stage on one Colab VM (e2e_vm.py).

    colab new -s e2e --gpu L4
    python3 scripts/e2e_colab.py run e2e --episodes 0 [--run e2e_v1] [--from human]   # name, upload, launch, follow
    python3 scripts/e2e_colab.py follow e2e --run e2e_v1                               # re-attach: keepalive + pulls
    python3 scripts/e2e_colab.py status e2e --run e2e_v1

`run`: uploads the scripts, the submission kit, the Gemini key and the names already in
results/gemini_object/answers.json, launches e2e_vm.py detached and follows it. On the VM, the videos come from the
HF dataset and the name stage asks Gemini for the episodes not named yet (answers merged back into answers.json). `follow` pings the kernel every 4 min (Colab reclaims a VM ~45 min after the last kernel
interaction) and pulls each stage's outputs when it finishes, into results/<run>/ (the laptop is the store):
  cache/episode_X/          masks, depth, camera, meta (separate from the repo's cache/)
  cache_check/, object_raw/ track overlays, crops + meshes
  stages/<stage>/           episodes/ (+ meta) of the intermediate runs: human, obj, p1, j
  episodes/                 the final submission-format episodes; proxy_eval.{csv,txt}; videos/
  logs/, status.json
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemini_object  # noqa: E402
from d4rt_steps_colab import upload_tar  # noqa: E402
from step3_colab import colab, keepalive, launch, ls, remote  # noqa: E402
from step4_colab import parse_episodes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VM = "/content/repo"
SCRIPTS = ["e2e_vm.py", "gemini_object.py", "preprocess_vm.py", "build_cache.py", "human_vm.py", "human_post.py", "object_vm.py",
           "object_fit.py", "pushout.py", "joint_fit.py", "proxy_eval.py", "render_video.py",
           "track_check.py", "motion_vm.py", "d4rt_vm.py"]
VIDEOS = "videos/chunk-000/observation.images.exo_camera"
STAGES = ["name", "sam", "depth", "cache", "d4rt", "human", "mesh", "object", "push1", "joint", "push2", "eval"]


def known_names(episodes: list[int]) -> dict[int, str]:
    """Names already in results/gemini_object/answers.json; the VM's name stage asks Gemini for the rest."""
    answers = json.loads(gemini_object.OUT.read_text()) if gemini_object.OUT.exists() else {}
    names = {e: gemini_object.object_name(answers[f"episode_{e:06d}"]["answer"])
             for e in episodes if f"episode_{e:06d}" in answers}
    print(f"names known for {sorted(names)}; Gemini on the VM for the other {len(episodes) - len(names)}", flush=True)
    return names


def merge_answers(session: str, run: str):
    """Fold the VM's Gemini answers into results/gemini_object/answers.json (two sessions may do this at once)."""
    local = ROOT / "results" / run / f"answers_{session}.json"
    if "answers.json" not in ls(session, f"/content/e2e/{run}"):  # every name was known, or Gemini failed on all
        return
    colab("download", "-s", session, f"/content/e2e/{run}/answers.json", str(local), timeout=120)
    lock = gemini_object.OUT.with_suffix(".lock")
    gemini_object.OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        answers = json.loads(gemini_object.OUT.read_text()) if gemini_object.OUT.exists() else {}
        answers.update(json.loads(local.read_text()))
        gemini_object.OUT.write_text(json.dumps(dict(sorted(answers.items())), indent=1))
        names_path = ROOT / "results" / run / "names.json"
        names = json.loads(names_path.read_text()) if names_path.exists() else {}
        names.update({str(int(k.split("_")[1])): gemini_object.object_name(v["answer"]) for k, v in answers.items()})
        names_path.write_text(json.dumps(names, indent=1, sort_keys=True))


def upload(session: str, run: str, episodes: list[int], names: dict[int, str], key_var: str = "GEMINI_API_KEY"):
    env = dict(line.split("=", 1) for line in (ROOT / ".env.local").read_text().split() if "=" in line)
    remote(session, f"mkdir -p {VM}/scripts {VM}/data /content/e2e/{run} /content/logs /content/work && "
                    f"printf %s {env['HF_TOKEN']} > /content/.hf_token && "
                    f"printf %s {env[key_var]} > /content/.gemini_key")
    for name in SCRIPTS:
        colab("upload", "-s", session, str(ROOT / "scripts" / name), f"{VM}/scripts/{name}")
    local = ROOT / "results" / run
    local.mkdir(parents=True, exist_ok=True)
    merged = json.loads((local / "names.json").read_text()) if (local / "names.json").exists() else {}
    merged.update({str(e): v for e, v in names.items()})
    (local / "names.json").write_text(json.dumps(merged, indent=1, sort_keys=True))
    (local / f"names_{session}.json").write_text(json.dumps(names, indent=1))
    colab("upload", "-s", session, str(local / f"names_{session}.json"), f"/content/e2e/{run}/names.json")
    if not ls(session, f"{VM}/data/v2d_submission_kit"):  # videos + meta: the VM's data install (HF)
        upload_tar(session, f"e2e_{session}", [(ROOT / "data/v2d_submission_kit", "repo/data/v2d_submission_kit")],
                   "/content")
    print(f"[{session}] uploaded scripts, names, kit", flush=True)


def read_status(session: str, run: str) -> dict | None:
    local = ROOT / "results" / run / f"status_{session}.json"
    try:
        colab("download", "-s", session, f"/content/e2e/{run}/status.json", str(local), timeout=120, retries=2)
    except RuntimeError:
        return None
    return json.loads(local.read_text())


def pull_dir(session: str, remote_dir: str, local: Path, prefixes: tuple[str, ...] = ("",), force: bool = False):
    """Download files of one VM directory (not recursive) whose names start with one of `prefixes`."""
    local.mkdir(parents=True, exist_ok=True)
    for f in ls(session, remote_dir):
        if f.startswith(prefixes) and "." in f and ".partial" not in f and not f.endswith(".lock"):
            if force or not (local / f).exists():
                colab("download", "-s", session, f"{remote_dir}/{f}", str(local / f), timeout=1800)


def pull_stage(session: str, run: str, stage: str, episodes: list[int]):
    dst = ROOT / "results" / run
    names = tuple(f"episode_{e:06d}" for e in episodes)
    res = f"{VM}/results"
    if stage == "name":
        merge_answers(session, run)
    elif stage == "cache":
        for n in names:
            pull_dir(session, f"{VM}/cache/{n}", dst / "cache" / n, force=True)
        pull_dir(session, f"{res}/{run}/cache_check", dst / "cache_check", names, force=True)
        if ls(session, f"{res}/{run}/cache_check_motion"):
            pull_dir(session, f"{res}/{run}/cache_check_motion", dst / "cache_check_motion", names, force=True)
        if ls(session, "/content/motion"):  # motion fallback: seed report + check image per flagged episode
            pull_dir(session, "/content/motion", dst / "motion", tuple(f"{n}_seed" for n in names), force=True)
    elif stage == "d4rt":  # the swapped cache files (depth.npz is now D4RT's)
        for n in names:
            (dst / "cache" / n).mkdir(parents=True, exist_ok=True)
            for f in ("depth.npz", "meta.json"):
                colab("download", "-s", session, f"{VM}/cache/{n}/{f}", str(dst / "cache" / n / f), timeout=1800)
    elif stage == "human":
        for sub in ("episodes", "human_meta", "overlays"):
            pull_dir(session, f"{res}/{run}_human/{sub}", dst / "stages/human" / sub, names, force=True)
    elif stage == "mesh":
        for sub in ("crops", "meshes"):
            pull_dir(session, f"/content/object_raw/{sub}", dst / "object_raw" / sub, names, force=True)
    elif stage in ("object", "push1", "joint"):
        src = {"object": f"{run}_obj", "push1": f"{run}_p1", "joint": f"{run}_j"}[stage]
        for sub in ("episodes", "object_meta", "joint_meta"):
            pull_dir(session, f"{res}/{src}/{sub}", dst / "stages" / stage / sub, names, force=True)
    elif stage == "push2":
        pull_dir(session, f"{res}/{run}/episodes", dst / "episodes", names, force=True)
    elif stage == "eval":
        for f in ("proxy_eval.csv", "proxy_eval.txt"):  # per session: two VMs can share a run
            colab("download", "-s", session, f"{res}/{run}/{f}", str(dst / f.replace(".", f"_{session}.")), timeout=600)
        pull_dir(session, f"{res}/{run}/videos", dst / "videos", names, force=True)
    pull_dir(session, f"/content/e2e/{run}/logs", dst / "logs" / session, force=True)


def pull_logs(session: str, run: str):
    pull_dir(session, f"/content/e2e/{run}/logs", ROOT / "results" / run / "logs" / session, force=True)


def poll(session: str, run: str, seen: dict) -> int | None:
    """One look at the VM: report and pull finished stages. Returns an exit code when the run is over."""
    st = read_status(session, run)
    if (st is None or not st.get("current")) and f"e2e_{run}.txt" in ls(session, "/content/logs"):
        tail = remote(session, f"tail -15 /content/logs/e2e_{run}.txt")
        if "Traceback" in tail:
            pull_logs(session, run)
            print("runner exited:\n" + tail, flush=True)
            return 1
    if st is None:
        return None
    for stage, info in st["stages"].items():
        if seen.get(stage) == info["state"]:
            continue
        extra = f" in {info['sec']} s" if "sec" in info else ""
        if info.get("failed"):
            extra += f", FAILED episodes {info['failed']} ({(info.get('error') or '')[:200]})"
        print(f"[{time.strftime('%H:%M')}] [{session}] {stage}: {info['state']}{extra}", flush=True)
        if info["state"] == "done":
            failed = set(info.get("failed") or [])
            pull_stage(session, run, stage, [e for e in info.get("episodes", st["episodes"]) if e not in failed])
            if failed:
                pull_logs(session, run)
        elif info["state"] == "failed":
            pull_logs(session, run)
            log = ROOT / "results" / run / "logs" / session / f"{stage}.txt"
            print(info.get("error"), "\n--- last log lines ---\n",
                  "\n".join(log.read_text(errors="replace").splitlines()[-25:]) if log.exists() else "", flush=True)
            return 1
        seen[stage] = info["state"]  # only once its pull went through; a failed pull is retried next time
    if st.get("finished"):
        print(f"finished: results/{run}/", flush=True)
        table = ROOT / "results" / run / f"proxy_eval_{session}.txt"
        if table.exists():
            print(table.read_text())
        return 0
    return None


def follow(session: str, run: str, every: int = 60, skip: tuple[str, ...] = ()):
    """Report stage changes and pull each finished stage; `skip`: stages already pulled (before --from).
    Errors on the laptop side (a failed download, a dropped console) are retried: the keepalive thread must outlive
    them, or Colab reclaims the VM ~45 min later with unpulled work on it."""
    stop = threading.Event()
    threading.Thread(target=keepalive, args=([session], 240, stop), daemon=True).start()
    seen = {stage: "done" for stage in skip}
    try:
        while True:
            try:
                code = poll(session, run, seen)
            except Exception as err:
                print(f"[{time.strftime('%H:%M')}] [{session}] follow error, retrying: {str(err)[:300]}", flush=True)
                code = None
            if code is not None:
                return code
            time.sleep(every)
    finally:
        stop.set()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "follow", "status"):
        s = sub.add_parser(name)
        s.add_argument("session")
        s.add_argument("--run", default="e2e_v1")
        if name == "run":
            s.add_argument("--episodes", required=True)
            s.add_argument("--from", dest="start", choices=STAGES, default="name")
            s.add_argument("--key-var", default="GEMINI_API_KEY", help=".env.local entry holding the Gemini key")
            s.add_argument("--fallback-args", default="", help="forwarded to e2e_vm.py")
    args = p.parse_args()
    if args.cmd == "status":
        print(json.dumps(read_status(args.session, args.run), indent=1))
        return 0
    if args.cmd == "run":
        episodes = parse_episodes(args.episodes)
        names = known_names(episodes)
        upload(args.session, args.run, episodes, names, args.key_var)
        remote(args.session, f"rm -f /content/logs/e2e_{args.run}.txt")
        launch(args.session, f"e2e_{args.run}",
               f"cd {VM}/scripts && python e2e_vm.py --run {args.run} --names /content/e2e/{args.run}/names.json "
               f"--episodes {','.join(map(str, episodes))} --from {args.start} "
               f"--fallback-args '{args.fallback_args}'")
    skip = tuple(STAGES[:STAGES.index(args.start)]) if args.cmd == "run" else ()
    return follow(args.session, args.run, skip=skip)


if __name__ == "__main__":
    sys.exit(main())

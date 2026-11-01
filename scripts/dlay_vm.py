"""Fit VM `dlay` (L4): web-demo meshes (sam_3d_mesh/, imported like s3dall) + SAM 3D start poses carried over by
align_demo.py; e2e object, push1, joint; pull the joint stage; stop the VM. render_fit runs on the laptop afterwards.

    python dlay_vm.py setup     # VM, uploads, installs (no layouts needed)
    python dlay_vm.py run       # layouts, import, object..joint, pull, stop
    python dlay_vm.py resume    # same without the import, for the episodes still missing
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/byebyehello/Desktop/SP/4DHumanObjectInteraction")
sys.path.insert(0, str(ROOT / "scripts"))
from d4rt_steps_colab import upload_tar  # noqa: E402
from step3_colab import colab, launch, ls, remote  # noqa: E402

S, VM, HERE = "dlay", "/content/repo", Path(__file__).resolve().parent


def done(job: str, marker: str) -> bool:
    """Wait for /content/<marker>; False if the job ended without it."""
    while marker not in ls(S, "/content"):
        if "NOJOB" in remote(S, f'pgrep -f [{job[0]}]{job[1:]}.sh >/dev/null || echo "NO""JOB"') and marker not in ls(S, "/content"):
            print(remote(S, f"tail -c 2000 /content/logs/{job}.txt"), flush=True)
            return False
        time.sleep(60)
    return True


def setup():
    subprocess.run(["colab", "new", "-s", S, "--gpu", "L4"], check=True)
    tok = HERE / "hf_token_dlay"
    tok.write_text(dict(l.split("=", 1) for l in (ROOT / ".env.local").read_text().split() if "=" in l)["HF_TOKEN"])
    os.chmod(tok, 0o600)
    try:
        remote(S, f"mkdir -p {VM}/scripts {VM}/results /content/logs /content/work /content/e2e/dlay")
        colab("upload", "-s", S, str(tok), "/content/.hf_token")  # a file upload: the token never shows in console output
    finally:
        tok.unlink()
    colab("upload", "-s", S, str(ROOT / "results/e2e_box/names.json"), "/content/e2e_names.json")
    colab("upload", "-s", S, str(HERE / "import_sam3d.py"), "/content/import_sam3d.py")
    upload_tar(S, "dlay_kit", [(ROOT / "scripts", "scripts"), (ROOT / "data/v2d_submission_kit", "data/v2d_submission_kit"),
                               (ROOT / "results/e2e_box/stages/human", "results/dlay_human"),
                               (ROOT / "sam_3d_mesh", "sam_3d_mesh")], VM)
    upload_tar(S, "dlay_crops", [(ROOT / "results/e2e_box/object_raw/crops", "object_raw/crops")], "/content")
    launch(S, "dlay_install", """cd /content/repo/scripts && python -c "
import e2e_vm as m; from pathlib import Path
r = m.Runner('dlay', {}, list(range(30)), Path('/content/e2e_names.json'))
[r.install(g, 'object') for g in ('data', 'fit')]
" && touch /content/dlay_installed""")
    upload_tar(S, "dlay_cache", [(ROOT / "results/e2e_box/cache", "cache")], VM)
    print("setup uploads done; install", "ok" if done("dlay_install", "dlay_installed") else "FAILED", flush=True)


def run(resume: bool = False):
    """resume: meshes already imported; stages skip the episodes whose outputs exist."""
    if resume:
        remote(S, "rm -f /content/dlay_done")
    upload_tar(S, "dlay_layout", [(ROOT / "results/lay/layout_demo", "layout")], "/content")
    eps = ",".join(map(str, range(30)))
    imp = "" if resume else "cd /content && python /content/import_sam3d.py > /content/logs/dlay_import.txt 2>&1 && "
    launch(S, "dlay_run", f"""{imp}cd /content/repo/scripts && python e2e_vm.py --run dlay --names /content/e2e_names.json --episodes {eps} --from object \
--only object,push1,joint --object-args '--layout /content/layout' && touch /content/dlay_done""")
    ok = done("dlay_run", "dlay_done")
    for rd, ld in {f"{VM}/results/dlay_obj/object_meta": "results/dlay/obj/object_meta",
                   f"{VM}/results/dlay_j/episodes": "results/dlay/j/episodes",
                   f"{VM}/results/dlay_j/joint_meta": "results/dlay/j/joint_meta",
                   "/content/object_raw/meshes": "results/dlay/object_raw/meshes",
                   "/content/e2e/dlay/logs": "results/dlay/logs"}.items():
        local = ROOT / ld
        local.mkdir(parents=True, exist_ok=True)
        for f in ls(S, rd):
            if "." in f and not f.endswith(".lock") and not (local / f).exists():
                colab("download", "-s", S, f"{rd}/{f}", str(local / f), timeout=1800)
    n = len(list((ROOT / "results/dlay/j/episodes").glob("episode_*[0-9].npz")))
    print("run", "ok" if ok else "FAILED", "joint episodes pulled", n, flush=True)
    if ok and n == 30:
        subprocess.run(["colab", "stop", "-s", S])
        print("VM STOPPED", flush=True)


if __name__ == "__main__":
    {"setup": setup, "run": run, "resume": lambda: run(True)}[sys.argv[1]]()

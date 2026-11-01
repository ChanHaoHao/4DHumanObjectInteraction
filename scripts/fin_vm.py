"""Finalize a run on a short L4 like s3dall_rf_fin, pull, stop the VM. With render, merge the render_fit episodes
(joint pose where render_fit has none) first.

    python fin_vm.py <src run dir> <out name> [render]    # e.g. results/dlay dlay_rf_fin render
"""
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/byebyehello/Desktop/SP/4DHumanObjectInteraction")
sys.path.insert(0, str(ROOT / "scripts"))
from d4rt_steps_colab import upload_tar  # noqa: E402
from e2e_vm import INSTALL  # noqa: E402
from step3_colab import colab, keepalive, launch, ls, remote  # noqa: E402

lay, OUT = ROOT / sys.argv[1], sys.argv[2]
src = lay / "rj/episodes" if sys.argv[3:] == ["render"] else lay / "j/episodes"
src.mkdir(parents=True, exist_ok=True)
for e in range(30) if sys.argv[3:] == ["render"] else ():
    n = f"episode_{e:06d}"
    d = lay / "r/episodes" if (lay / f"r/render_meta/{n}.json").exists() else lay / "j/episodes"
    print(e, d.parent.name, flush=True)
    for f in d.glob(f"{n}*"):
        shutil.copy(f, src / f.name)

S = OUT.replace("_", "")[:12]
subprocess.run(["colab", "new", "-s", S, "--gpu", "L4"], check=True)
stop = threading.Event()
threading.Thread(target=keepalive, args=([S],), kwargs={"stop": stop}, daemon=True).start()
try:
    upload_tar(S, f"{S}_kit", [(ROOT / "scripts", "scripts"), (ROOT / "data/v2d_submission_kit", "data/v2d_submission_kit"),
                                 (src, "results/src/episodes")], "/content/repo")
    launch(S, S, INSTALL["fit"] + f"""
cd /content/repo/scripts && python finalize.py --src /content/repo/results/src/episodes --run /content/repo/results/{OUT} \
    --episodes 0-29 --sample /content/repo/data/v2d_submission_kit/data/track_1_sample_submission.parquet \
    --lam-o 100000 --lam-h 1000,600,600 && touch /content/{S}_done
""")
    while f"{S}_done" not in ls(S, "/content"):
        time.sleep(60)
        if "NOJOB" in remote(S, f'pgrep -f [{S[0]}]{S[1:]}.sh >/dev/null || echo "NO""JOB"') and f"{S}_done" not in ls(S, "/content"):
            raise RuntimeError(remote(S, f"tail -c 1500 /content/logs/{S}.txt"))
    out = ROOT / f"results/{OUT}/episodes"
    out.mkdir(parents=True, exist_ok=True)
    fs = [f for f in ls(S, f"/content/repo/results/{OUT}/episodes") if "." in f]
    for f in fs:
        colab("download", "-s", S, f"/content/repo/results/{OUT}/episodes/{f}", str(out / f), timeout=1800)
    colab("download", "-s", S, f"/content/logs/{S}.txt", str(ROOT / f"results/{OUT}/finalize_log.txt"), timeout=300)
    print("pulled", len(fs), flush=True)
finally:
    stop.set()
    subprocess.run(["colab", "stop", "-s", S])  # outputs are pulled above; on failure there is nothing but the log
    print("VM STOPPED", flush=True)

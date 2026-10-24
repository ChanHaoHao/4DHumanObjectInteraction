#!/usr/bin/env python3
"""End-to-end Track 1 run on one Colab VM: video + object name -> submission episode. Driven by e2e_colab.py.

    python e2e_vm.py --run e2e_v1 --names /content/e2e/names.json [--episodes 0] [--from human]
        [--only sam,depth,cache,mesh] [--no-fallback]

names.json maps episode -> object name (the Gemini answer); nothing else about the object is used, and every
episode goes through the same stages with the same settings (no per-object rules; depth: whole-frame D4RT on
MoGe-2's metric scale):

  name    Gemini Flash names the object from the video (missing names only)  gemini_object.py
  locate  LocateAnything-3B boxes for the name on ~11 keyframes          locate_vm.py --keyframes 10

  sam     SAM 3 video tracks for "person" and each named object          preprocess_vm.py --stage sam
  depth   MoGe-2 depth, one FOV per clip                                preprocess_vm.py --stage depth
  cache   actor/object track selection, stabilized depth, floor         build_cache.py
          box first: SAM 3 tracker tracks from the locate boxes, voted on by the boxes, still ones lose to a moving
          one (motion_vm.py --stage box, pick()); the winner alone is rebuilt into the cache if it touches the actor.
          Otherwise the text-prompt cache goes through the track check (track_check.py): no object, coverage < 0.5 or a still track (furniture the hand rests
          on) -> motion candidates (RGB change seeds -> SAM 3 tracker, motion_vm.py --stage change/seed) built
          into a second cache; it replaces the Gemini one only if it passes the same check. Still nothing ->
          the depth-foreground fallback track, then rebuilt. Decision in cache/<ep>/check.json
  d4rt    OpenD4RT depth (48-frame windows, scaled to the MoGe background) swapped into the cache;
          MoGe's kept as depth_moge.npz. /content/cache_d4rt/<ep> already there = reused  d4rt_vm.py
  human   SAM 3D Body per frame -> one smoothed MHR sequence            human_vm.py, human_post.py
  mesh    3 unoccluded object crops -> Hunyuan3D-2.1 meshes             object_vm.py
  object  mesh choice, scale, pose track (episode fitted alone)         object_fit.py --per-episode
  push1   hand-penetration push-out (7a)                                pushout.py --w-pen 5 --w-dep 0.3
  joint   contact / floor / per-frame depth placement (7b/7c)           joint_fit.py --per-episode
  render  (--render only) render-and-compare refinement, no depth        render_fit.py
  push2   push-out again on the placed track (or on render's)           pushout.py --w-pen 5 --w-dep 0.3
  final   mesh origin at its centroid, smoothed translation, hands out  finalize.py
  eval    proxy metrics and the overlay | side-view video               proxy_eval.py, render_video.py

Stages whose outputs exist are skipped, so a rerun resumes. Installs happen just before the first stage that needs
them (marker /content/installed_<group>). Layout: /content/repo mirrors the repo (scripts, cache, results, kit,
body_models), videos + meta under /content/v2d/track_1. Runs: results/<run>_human, <run>_obj, <run>_p1, <run>_j,
<run>_p2, <run> (final). Progress: /content/e2e/<run>/status.json, stage logs in /content/e2e/<run>/logs/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/content/repo")
SCRIPTS = REPO / "scripts"
DATA = Path("/content/v2d/track_1")
CACHE = REPO / "cache"
SAMPLE = REPO / "data/v2d_submission_kit/data/track_1_sample_submission.parquet"
RAW, HUMAN_RAW, OBJECT_RAW = Path("/content/raw"), Path("/content/human_raw"), Path("/content/object_raw")
MOTION, CACHE_MOTION = Path("/content/motion"), Path("/content/cache_motion")
CACHE_D4RT = Path("/content/cache_d4rt")
LOCATE, CACHE_BOX = Path("/content/locate"), Path("/content/cache_box")
STILL, MIN_COVER = 0.85, 0.5  # track check, read off eps 0-29 of e2e_v1 (notes/preprocess.md)
SAME_OBJECT = 0.35  # motion track this much inside the Gemini track = a moving part of the named object (desk legs)
STAGES = ["name", "locate", "sam", "depth", "cache", "d4rt", "human", "mesh", "object", "push1", "joint", "render", "push2",
          "final", "eval"]
PINS = "python -c \"import torch, torchvision; print(f'torch=={torch.__version__}'); " \
       "print(f'torchvision=={torchvision.__version__}')\" > /content/pins.txt"
MHR = r"""
if [ ! -f /content/repo/body_models/mhr/mhr_model.pt ]; then
  mkdir -p /content/repo/body_models/mhr
  curl -sL -o /content/mhr_assets.zip https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip
  unzip -q -j -o /content/mhr_assets.zip assets/mhr_model.pt -d /content/repo/body_models/mhr/
fi
echo "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc  /content/repo/body_models/mhr/mhr_model.pt" | sha256sum -c
"""
INSTALL = {
    # videos + meta straight from the dataset (the laptop's uplink is slow)
    "data": r"""
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/video_to_data_challenge', repo_type='dataset', allow_patterns=['track_1/*'], local_dir='/content/v2d')"
""",
    # sam3 pins numpy<2 and MoGe pulls numpy 2; both run fine on the result (Step 3 setup)
    # LocateAnything's remote code wants transformers 4.57: kept in its own folder, first on PYTHONPATH for that
    # stage only (like the mesh stage's Hunyuan pins), so the other stages keep Colab's transformers
    "locate": r"""
pip install -q decord lmdb peft
pip install -q --no-deps --target /content/locate_pkgs transformers==4.57.1 tokenizers==0.22.1 huggingface-hub==0.36.2
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/LocateAnything-3B')"
PYTHONPATH=/content/locate_pkgs python -c "import transformers; assert transformers.__version__ == '4.57.1'"
""",
    "sam": r"""
cd /content
[ -d sam3 ] || git clone -q https://github.com/facebookresearch/sam3.git
[ -d MoGe ] || git clone -q https://github.com/microsoft/MoGe.git
pip install -q -e ./sam3
pip install -q -e ./MoGe
pip install -q einops pycocotools psutil decord pandas pyarrow scipy
""",
    "body": PINS + r"""
cd /content
[ -d sam-3d-body ] || git clone -q https://github.com/facebookresearch/sam-3d-body.git
pip install -q -c /content/pins.txt pytorch-lightning yacs roma einops timm dill hydra-core pyrootutils \
    webdataset braceexpand omegaconf loguru optree fvcore jsonlines pycocotools scikit-image rich \
    trimesh numba fast_simplification pyarrow pandas scipy
""" + MHR + r"""
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam-3d-body-dinov3')"
python -c "import torch; torch.hub.list('facebookresearch/dinov3', trust_repo=True)" > /dev/null
""",
    "mesh": PINS + r"""
cd /content
[ -d Hunyuan3D-2.1 ] || git clone -q https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
pip install -q -c /content/pins.txt diffusers transformers accelerate einops omegaconf pymeshlab trimesh \
    fast_simplification pyarrow pandas scipy opencv-python-headless numba pytorch-lightning
# Hunyuan's DINOv2 checkpoint keys follow transformers 4.x (5.x renamed the attention weights): the mesh stage
# runs with Hunyuan's pins from MESH_PKGS first on PYTHONPATH (Colab's python has no ensurepip, so no venv);
# nothing else sees the downgrade
rm -rf /content/venv_mesh
pip install -q --no-deps --target /content/mesh_pkgs transformers==4.46.0 tokenizers==0.20.3 \
    huggingface-hub==0.30.2 diffusers==0.30.0 accelerate==1.1.1
export PYTHONPATH=/content/mesh_pkgs
python -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', allow_patterns=['hunyuan3d-dit-v2-1/*'])"
cd /content/Hunyuan3D-2.1/hy3dshape && python -c "import transformers; assert transformers.__version__ == '4.46.0'; from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline"
""",
    # OpenD4RT + its 14 GB checkpoint (Step 3 D4RT setup, d4rt_colab.py)
    "d4rt": r"""
cd /content
[ -d Open-d4rt ] || git clone -q https://github.com/Lijiaxin0111/Open-d4rt.git
pip install -q PyYAML opencv-python-headless imageio lz4
python -c "
from huggingface_hub import hf_hub_download
for f in ('model.yaml', 'opend4rt.ckpt'):
    print(hf_hub_download('Lijiaxin0111/OpenD4RT', 'checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/' + f, local_dir='/content/Open-d4rt'))
"
""",
    "fit": PINS + r"""
pip install -q -c /content/pins.txt trimesh fast_simplification pyarrow pandas scipy numba opencv-python-headless
""" + MHR,
    # nvdiffrast 0.4 builds its CUDA plugin at install (0.3.3 left none: "No module named nvdiffrast_plugin"); no OpenGL
    "render": PINS + r"""
pip install -q -c /content/pins.txt trimesh fast_simplification pyarrow pandas scipy numba opencv-python-headless ninja
pip install -q --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0
python -c "import nvdiffrast.torch as dr, torch; dr.RasterizeCudaContext(); print('nvdiffrast ok')"
""" + MHR,
}
NEEDS = {"name": "data", "locate": "locate", "sam": "sam", "depth": "sam", "cache": "sam", "d4rt": "data", "human": "body", "mesh": "mesh",
         "object": "fit", "push1": "fit", "joint": "fit", "render": "render", "push2": "fit", "final": "fit",
         "eval": "fit"}
PUSH = "--w-pen 5 --w-dep 0.3"  # Step 7a settings, every episode
FINAL = ""  # finalize.py defaults


def ep(e: int) -> str:
    return f"episode_{e:06d}"


class Runner:
    def __init__(self, run: str, names: dict[int, str], episodes: list[int], names_path: Path):
        self.run, self.names, self.episodes, self.names_path = run, names, episodes, names_path
        self.home = Path("/content/e2e") / run
        self.logs = self.home / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.res = REPO / "results"
        self.status = {"run": run, "episodes": episodes, "names": names, "stages": {}}
        status = self.home / "status.json"
        if status.exists():
            self.status["stages"] = json.loads(status.read_text()).get("stages", {})

    def save(self):
        (self.home / "status.json").write_text(json.dumps(self.status, indent=1))

    def sh(self, stage: str, cmd: str, cwd: Path = SCRIPTS):
        """Run one shell command, appending to the stage log; raise on failure."""
        with open(self.logs / f"{stage}.txt", "a") as log:
            log.write(f"\n$ {cmd}\n")
            log.flush()
            env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            code = subprocess.run(["bash", "-c", "set -eo pipefail\n" + cmd], cwd=cwd, stdout=log,
                                  stderr=subprocess.STDOUT, env=env).returncode
        if code:
            raise RuntimeError(f"{stage}: exit {code} (see logs/{stage}.txt)")

    def install(self, group: str, stage: str):
        marker = Path(f"/content/installed_{group}")
        if not marker.exists():
            self.sh(stage, INSTALL[group], cwd=Path("/content"))
            marker.touch()

    # ------------------------------------------------------------------ stages; each returns when its outputs exist
    def todo(self, check) -> list[int]:
        return [e for e in self.episodes if not check(e)]

    def meshed(self, e: int) -> bool:
        index = OBJECT_RAW / "crops" / f"{ep(e)}.json"
        if not index.exists():
            return False
        crops = json.loads(index.read_text())
        return bool(crops) and all((OBJECT_RAW / "meshes" / f"{ep(e)}_{c['k']}.obj").exists() for c in crops)

    def outputs(self, stage: str, e: int) -> bool:
        """True when `stage` has produced what the next stage needs for episode e."""
        n, r = ep(e), self.res
        return {
            "name": lambda: bool(self.names.get(e)),
            # a failed episode leaves a marker: the cache stage then keeps the text-prompt path
            "locate": lambda: any((LOCATE / f"{n}_locate{x}").exists() for x in (".npz", ".failed")),
            "sam": lambda: all((RAW / n / f).exists() for f in ["sam_person.npz", "sam_object.npz"] + [
                f"sam_object_{x.replace(' ', '_')}.npz" for x in prompt_parts(self.names[e])[1:]]),
            "depth": lambda: (RAW / n / "depth_raw.npz").exists(),
            "cache": lambda: (CACHE / n / "check.json").exists(),
            "d4rt": lambda: json.loads((CACHE / n / "meta.json").read_text()).get("depth_source") == "d4rt",
            "human": lambda: (r / f"{self.run}_human/episodes/{n}.npz").exists(),
            "mesh": lambda: self.meshed(e),
            "object": lambda: (r / f"{self.run}_obj/episodes/{n}.npz").exists(),
            "push1": lambda: (r / f"{self.run}_p1/episodes/{n}.npz").exists(),
            "joint": lambda: (r / f"{self.run}_j/episodes/{n}.npz").exists(),
            "render": lambda: (r / f"{self.run}_r/render_meta/{n}.json").exists(),
            "push2": lambda: (r / f"{self.run}_p2/episodes/{n}.npz").exists(),
            "final": lambda: (r / f"{self.run}/episodes/{n}.npz").exists(),
            "eval": lambda: (r / f"{self.run}/videos/{n}.mp4").exists(),
        }[stage]()

    def stage_name(self):
        """Gemini Flash names the object from the video (gemini_object.py); answers.json keeps the raw answers."""
        sys.path.insert(0, str(SCRIPTS))
        import gemini_object as g

        key = Path("/content/.gemini_key").read_text().strip()
        answers_path = self.home / "answers.json"
        answers = json.loads(answers_path.read_text()) if answers_path.exists() else {}
        for e in self.todo(lambda e: bool(self.names.get(e))):
            video = DATA / "videos/chunk-000/observation.images.exo_camera" / f"{ep(e)}.mp4"
            try:
                answer = g.ask(video, key, "gemini-flash-latest", g.PROMPT, None)
            except Exception as err:
                with open(self.logs / "name.txt", "a") as log:
                    log.write(f"{ep(e)}: ERROR {err}\n")
                continue
            answers[ep(e)] = {"answer": answer, "model": "gemini-flash-latest", "prompt": g.PROMPT}
            self.names[e] = g.object_name(answer)
            answers_path.write_text(json.dumps(answers, indent=1))
            self.names_path.write_text(json.dumps({str(k): v for k, v in self.names.items()}, indent=1))
            self.status["names"] = self.names
            with open(self.logs / "name.txt", "a") as log:
                log.write(f"{ep(e)}: {answer!r} -> {self.names[e]!r}\n")

    def stage_locate(self):
        LOCATE.mkdir(exist_ok=True)
        for e in self.todo(lambda e: self.outputs("locate", e)):
            try:
                self.sh("locate", f"PYTHONPATH=/content/locate_pkgs python {SCRIPTS}/locate_vm.py --videos "
                                  f"{DATA}/videos/chunk-000/observation.images.exo_camera --episode {e} "
                                  f"--phrase {shlex.quote(self.names[e])} --out {LOCATE} --keyframes 10",
                        cwd=Path("/content/work"))
            except RuntimeError as err:
                (LOCATE / f"{ep(e)}_locate.failed").write_text(str(err))

    def stage_sam(self):
        """SAM 3 for "person" and each object Gemini named ("table and bowl" -> "table", "bowl"): the first part goes
        to sam_object.npz, the others to sam_object_<part>.npz; build_cache pools them and keeps the track that
        touches the actor."""
        Path("/content/work").mkdir(exist_ok=True)
        for e in self.todo(lambda e: self.outputs("sam", e)):
            parts = prompt_parts(self.names[e])
            jobs = [f'--object-name "{parts[0]}"'] + [f'--object-prompt "{x}"' for x in parts[1:]]
            for job in jobs:  # each call skips the files that exist
                # from /content/work: run from /content, `import sam3` would pick up the repo folder
                try:
                    self.sh("sam", f"python {SCRIPTS}/preprocess_vm.py --stage sam --episodes {e} --data {DATA} "
                                   f"--out {RAW} --sample {SAMPLE} {job}", cwd=Path("/content/work"))
                except RuntimeError as err:  # recorded by go() as a failed episode
                    print(err, flush=True)

    def stage_depth(self):
        eps = self.todo(lambda e: (RAW / ep(e) / "depth_raw.npz").exists())
        if eps:
            self.sh("depth", f"python {SCRIPTS}/preprocess_vm.py --stage depth --episodes {csv(eps)} --data {DATA} "
                             f"--out {RAW} --sample {SAMPLE}", cwd=Path("/content/work"))

    def stage_cache(self):
        build = (f"python build_cache.py --raw {RAW} --data {DATA} --sample {SAMPLE} --episodes")
        check = self.res / self.run / "cache_check"
        for e in self.todo(lambda e: (CACHE / ep(e) / "meta.json").exists()):
            self.sh("cache", f"{build} {e} --out {CACHE} --overlay {check}")
        for e in self.todo(lambda e: (CACHE / ep(e) / "check.json").exists() or not (CACHE / ep(e) / "meta.json").exists()):
            n = ep(e)
            decision = {"gemini": suspect(CACHE / n)}
            try:
                decision["box"] = self.box_candidate(e, build)
            except RuntimeError as err:  # no boxes or no track: the text-prompt path below
                decision["box"] = str(err)
            if decision["box"] is None:
                shutil.rmtree(CACHE / n)
                shutil.copytree(CACHE_BOX / n, CACHE / n)
                shutil.copy(self.res / self.run / "cache_check_box" / f"{n}.jpg", check / f"{n}.jpg")
                decision["used"] = "box"
                (CACHE / n / "check.json").write_text(json.dumps(decision, indent=1))
                continue
            if self.no_fallback:  # the named object's SAM 3 track as is, flagged or not
                decision["used"] = "gemini"
                (CACHE / n / "check.json").write_text(json.dumps(decision, indent=1))
                continue
            if decision["gemini"]:
                try:
                    decision["motion"] = self.motion_candidate(e, build)
                except RuntimeError as err:  # no motion candidate: keep the Gemini track
                    decision["motion"] = str(err)
                if decision["motion"] is None and not decision["gemini"].startswith("no object"):
                    inside = overlap(CACHE_MOTION / n, CACHE / n)
                    if inside >= SAME_OBJECT:  # motion confirms the named object; keep its full mask
                        decision["motion"] = f"part of the Gemini object (inside {inside:.2f})"
                if decision["motion"] is None:
                    shutil.rmtree(CACHE / n)
                    shutil.copytree(CACHE_MOTION / n, CACHE / n)
                    shutil.copy(self.res / self.run / "cache_check_motion" / f"{n}.jpg", check / f"{n}.jpg")
            meta = json.loads((CACHE / n / "meta.json").read_text())
            if meta.get("warning") and not (RAW / n / "sam_object_fallback.npz").exists():
                self.sh("cache", f"python preprocess_vm.py --stage fallback --episodes {e} --data {DATA} "
                                 f"--out {RAW} --sample {SAMPLE} {self.fallback_args}")
                (CACHE / n / "meta.json").unlink()
                self.sh("cache", f"{build} {e} --out {CACHE} --overlay {check}")
                decision["depth_fallback"] = suspect(CACHE / n)
            decision["used"] = ("motion" if decision.get("motion") is None and decision["gemini"] else
                                "depth_fallback" if "depth_fallback" in decision else "gemini")
            (CACHE / n / "check.json").write_text(json.dumps(decision, indent=1))

    def box_candidate(self, e: int, build: str) -> str | None:
        """The box stage's winning track alone, built into CACHE_BOX (the text-prompt cache gives the actor masks its
        stillness needs); None when it touches the actor, else why not."""
        n = ep(e)
        if not (LOCATE / f"{n}_locate.npz").exists():
            return "no LocateAnything boxes"
        work = Path("/content/work")  # from /content, `import sam3` would pick up the repo folder
        work.mkdir(exist_ok=True)
        self.sh("cache", f"python {SCRIPTS}/motion_vm.py --stage box --episodes {e} --data {DATA} --cache {CACHE} "
                         f"--locate {LOCATE} --out {MOTION}", cwd=work)
        import numpy as np

        data = dict(np.load(MOTION / f"{n}_box.npz"))
        w = int(data["winner"])
        np.savez_compressed(RAW / n / "box_cand.npz", **{k: v[w:w + 1] if k in ("obj_ids", "probs", "masks") else v
                                                        for k, v in data.items()})
        self.sh("cache", f"{build} {e} --out {CACHE_BOX} --objects box_cand.npz "
                         f"--overlay {self.res / self.run / 'cache_check_box'}")
        return json.loads((CACHE_BOX / n / "meta.json").read_text()).get("warning")

    def motion_candidate(self, e: int, build: str) -> str | None:
        """Name-free candidate cache in CACHE_MOTION; None when it passes the track check, else why not."""
        n = ep(e)
        work = Path("/content/work")  # from /content, `import sam3` would pick up the repo folder
        work.mkdir(exist_ok=True)
        common = f"--episodes {e} --data {DATA} --cache /content --out {MOTION} --tau 0.1"
        self.sh("cache", f"python {SCRIPTS}/motion_vm.py --stage change {common}", cwd=work)
        self.sh("cache", f"python {SCRIPTS}/motion_vm.py --stage seed {common}", cwd=work)
        cand = MOTION / f"{n}_cand.npz"
        import numpy as np

        if not np.load(cand)["obj_ids"].size:
            return "no moving object touches the actor"
        # a seed can grow onto the furniture the object moves on (ep 18: the tabletop); still tracks are dropped here,
        # or build_cache's contact choice prefers the big furniture track
        sys.path.insert(0, str(SCRIPTS))
        from build_cache import load_tracks
        from track_check import stillness

        tracks = load_tracks(cand)
        people = load_tracks(RAW / n / "sam_person.npz")["masks"].any(0)[:, ::4, ::4]
        still = [stillness(m[:, ::4, ::4], people[:m.shape[0]])[0] for m in tracks["masks"]]
        with open(self.logs / "cache.txt", "a") as log:
            log.write(f"{n} motion candidates stillness {[round(x, 2) for x in still]}\n")
        keep = [k for k, x in enumerate(still) if x < STILL]
        if not keep:
            return f"every motion candidate is still ({min(still):.2f})"
        data = dict(np.load(cand))
        np.savez_compressed(RAW / n / "motion_cand.npz", **{k: v[keep] if k in ("obj_ids", "probs", "masks") else v
                                                           for k, v in data.items()})
        self.sh("cache", f"{build} {e} --out {CACHE_MOTION} --objects motion_cand.npz "
                         f"--overlay {self.res / self.run / 'cache_check_motion'}")
        return suspect(CACHE_MOTION / n)

    def stage_d4rt(self):
        eps = self.todo(lambda e: (CACHE_D4RT / ep(e) / "meta.json").exists())
        if eps:
            self.install("d4rt", "d4rt")  # here, not in go(): reused episodes need no 14 GB checkpoint
            self.sh("d4rt", f"python d4rt_vm.py --episodes {csv(eps)} --data {DATA} --cache {CACHE} --out {CACHE_D4RT}")
        for e in self.todo(lambda e: self.outputs("d4rt", e)):
            n = ep(e)
            if not (CACHE_D4RT / n / "meta.json").exists():
                continue  # failed above; go() records it
            if (CACHE / n / "depth.npz").exists():
                os.replace(CACHE / n / "depth.npz", CACHE / n / "depth_moge.npz")
            shutil.copy(CACHE_D4RT / n / "depth.npz", CACHE / n / "depth.npz")
            meta = json.loads((CACHE / n / "meta.json").read_text())
            d4 = json.loads((CACHE_D4RT / n / "meta.json").read_text())
            meta.update({k: d4[k] for k in ("depth_scale_std", "depth_scale_p5_p95", "d4rt") if k in d4})
            meta["depth_source"] = "d4rt"  # last: outputs("d4rt") reads it
            (CACHE / n / "meta.json").write_text(json.dumps(meta, indent=1))

    def stage_human(self):
        run = self.res / f"{self.run}_human"
        eps = self.todo(lambda e: (run / "episodes" / f"{ep(e)}.npz").exists())
        if eps:
            self.sh("human", f"python human_vm.py --stage infer --episodes {csv(eps)} --data {DATA} --cache {CACHE} "
                             f"--sample {SAMPLE} --out {HUMAN_RAW}")
            self.sh("human", f"python human_post.py --raw {HUMAN_RAW}/dinov3 --run {run} --episodes {csv(eps)} "
                             f"--cache {CACHE} --data {DATA} --sample {SAMPLE}")

    def stage_mesh(self):
        eps = self.todo(lambda e: (OBJECT_RAW / "crops" / f"{ep(e)}.json").exists())
        if eps:
            self.sh("mesh", f"python object_vm.py --stage crops --episodes {csv(eps)} --data {DATA} --cache {CACHE} "
                            f"--out {OBJECT_RAW}")
        eps = [e for e in self.episodes if (OBJECT_RAW / "crops" / f"{ep(e)}.json").exists() and not self.meshed(e)]
        if eps:
            self.sh("mesh", f"PYTHONPATH=/content/mesh_pkgs python object_vm.py --stage mesh --episodes {csv(eps)} --data {DATA} --cache {CACHE} "
                            f"--out {OBJECT_RAW}")

    def stage_object(self):
        run = self.res / f"{self.run}_obj"
        eps = self.todo(lambda e: (run / "object_meta" / f"{ep(e)}.json").exists())
        if eps:
            self.sh("object", f"python object_fit.py --human {self.res / self.run}_human/episodes --run {run} "
                              f"--episodes {csv(eps)} --raw {OBJECT_RAW} --cache {CACHE} --data {DATA} "
                              f"--sample {SAMPLE} --per-episode {self.object_args}")

    def push(self, stage: str, src: str, dst: str):
        run = self.res / dst
        eps = self.todo(lambda e: (run / "episodes" / f"{ep(e)}.npz").exists())
        if eps:
            self.sh(stage, f"python pushout.py --src {self.res / src}/episodes --run {run} --episodes {csv(eps)} "
                           f"--cache {CACHE} --sample {SAMPLE} {PUSH}")

    def stage_push1(self):
        self.push("push1", f"{self.run}_obj", f"{self.run}_p1")

    def stage_joint(self):
        run = self.res / f"{self.run}_j"
        eps = self.todo(lambda e: (run / "joint_meta" / f"{ep(e)}.json").exists())
        if eps:
            self.sh("joint", f"python joint_fit.py --human {self.res / self.run}_human/episodes "
                             f"--init {self.res / self.run}_p1/episodes --run {run} --episodes {csv(eps)} "
                             f"--cache {CACHE} --data {DATA} --sample {SAMPLE} --per-episode {self.joint_args}")

    def stage_render(self):
        run = self.res / f"{self.run}_r"
        eps = self.todo(lambda e: self.outputs("render", e))
        if eps:
            self.sh("render", f"python render_fit.py --human {self.res / self.run}_human/episodes "
                              f"--init {self.res / self.run}_j/episodes --run {run} --episodes {csv(eps)} "
                              f"--cache {CACHE} --data {DATA} --sample {SAMPLE} {self.render_args}")

    def stage_push2(self):
        self.push("push2", f"{self.run}_r" if self.render else f"{self.run}_j", f"{self.run}_p2")

    def stage_final(self):
        run = self.res / self.run
        eps = self.todo(lambda e: (run / "episodes" / f"{ep(e)}.npz").exists())
        if eps:
            self.sh("final", f"python finalize.py --src {self.res / self.run}_p2/episodes --run {run} "
                             f"--episodes {csv(eps)} --sample {SAMPLE} {FINAL}")

    def stage_eval(self):
        run = self.res / self.run
        self.sh("eval", f"python proxy_eval.py {run}/episodes --cache {CACHE} --episodes {csv(self.episodes)} "
                        f"--sample {SAMPLE} --csv {run}/proxy_eval.csv > {run}/proxy_eval.txt")
        eps = self.todo(lambda e: (run / "videos" / f"{ep(e)}.mp4").exists())
        if eps:
            self.sh("eval", f"python render_video.py --run {run} --episodes {csv(eps)} --cache {CACHE} --data {DATA} "
                            f"--sample {SAMPLE}")

    def go(self, start: str, only: list[str]):
        os.environ.setdefault("HF_TOKEN", Path("/content/.hf_token").read_text().strip())
        self.status["failed_episodes"] = {}
        stages = [s for s in STAGES[STAGES.index(start):] if (not only or s in only) and (s != "render" or self.render)]
        for stage in stages:  # a rerun's stale "done" would make the driver pull too early
            self.status["stages"][stage] = {"state": "pending"}
        self.install("data", stages[0])  # every stage reads the videos; --only may skip the name stage that installs them
        for stage in stages:
            self.status["stages"][stage] = {"state": "running", "start": time.time(), "episodes": self.episodes}
            self.status["current"] = stage
            self.save()
            t0, error = time.time(), None
            try:
                self.install(NEEDS[stage], stage)
                getattr(self, f"stage_{stage}")()
            except Exception as err:  # a multi-episode call can fail on one episode; outputs decide below
                error = str(err)
            failed = [e for e in self.episodes if not self.outputs(stage, e)]
            for e in failed:  # later stages skip it; the others go on
                self.status["failed_episodes"][str(e)] = stage
            self.episodes = [e for e in self.episodes if e not in failed]
            self.status["stages"][stage] = {"state": "done" if self.episodes else "failed", "sec": round(time.time() - t0),
                                            "failed": failed, "error": error}
            self.save()
            if not self.episodes:
                self.status["current"] = None
                self.save()
                raise RuntimeError(f"{stage}: every episode failed ({error})")
        self.status["current"] = None
        self.status["finished"] = True
        self.save()


def suspect(folder: Path) -> str | None:
    """Why a cached object track is probably not the handled object (None = looks right)."""
    sys.path.insert(0, str(SCRIPTS))
    import numpy as np
    from track_check import stillness

    meta = json.loads((folder / "meta.json").read_text())
    if meta.get("warning"):
        return meta["warning"]
    m = np.load(folder / "masks.npz")
    still, cover = stillness(m["object"][:, ::4, ::4], m["human"][:, ::4, ::4])
    if cover < MIN_COVER:
        return f"coverage {cover:.2f}"
    return f"still {still:.2f}" if still >= STILL else None


def overlap(motion: Path, gemini: Path) -> float:
    """Fraction of the motion track's pixels (all frames) inside the Gemini track."""
    import numpy as np

    m = np.load(motion / "masks.npz")["object"][:, ::2, ::2]
    g = np.load(gemini / "masks.npz")["object"][:, ::2, ::2]
    t = min(len(m), len(g))
    return float((m[:t] & g[:t]).sum() / max(m[:t].sum(), 1))


def prompt_parts(name: str) -> list[str]:
    """One SAM 3 prompt per object in a Gemini answer: 'table and bowl' -> ['table', 'bowl']."""
    parts = [x.strip() for x in re.split(r",|;|&|/|\+|\band\b", name.replace('"', ""))]
    return list(dict.fromkeys(x for x in parts if x)) or [name]


def csv(eps) -> str:
    return ",".join(map(str, eps))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--names", type=Path, required=True,
                   help="JSON: episode -> object name; the name stage asks Gemini for missing ones and writes them here")
    p.add_argument("--episodes", required=True, help="comma-separated")
    p.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0],
                   help="first stage to (re)check; earlier stages are assumed done")
    p.add_argument("--only", default="", help="comma-separated stages to run (the rest are assumed done)")
    p.add_argument("--no-fallback", action="store_true", help="keep the named object's track: no motion/depth fallback")
    p.add_argument("--fallback-args", default="", help='extra preprocess_vm.py fallback flags, e.g. "--fg-gap 0.1"')
    p.add_argument("--object-args", default="", help='extra object_fit.py flags, e.g. "--layout /content/layout"')
    p.add_argument("--joint-args", default="", help='extra joint_fit.py flags, e.g. "--refine full"')
    p.add_argument("--render", action="store_true", help="run the render stage (render_fit.py) before push2")
    p.add_argument("--render-args", default="", help="extra render_fit.py flags")
    args = p.parse_args()
    names = {int(k): v for k, v in json.loads(args.names.read_text()).items()} if args.names.exists() else {}
    episodes = [int(e) for e in args.episodes.split(",")]
    runner = Runner(args.run, names, episodes, args.names)
    runner.fallback_args, runner.no_fallback = args.fallback_args, args.no_fallback
    runner.joint_args, runner.object_args = args.joint_args, args.object_args
    runner.render, runner.render_args = args.render, args.render_args
    only = [x for x in args.only.split(",") if x]
    assert set(only) <= set(STAGES), f"unknown stages {set(only) - set(STAGES)}"
    runner.go(args.start, only)


if __name__ == "__main__":
    main()

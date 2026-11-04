# 4D Human–Object Reconstruction from a Single Video

Team **ByeByeHelloHaHaHa**'s entry to the [V2D Challenge 2026](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/),
Track 1 (Reconstruction). From one static third-person RGB video, the pipeline reconstructs an MHR human sequence
and a textureless object mesh with a per-frame similarity pose. Every clip goes through the same stages with the
same settings; the only object-specific input is the object's name, which Gemini reads from the video.

Technical report: [`report/v2d_track1_report.pdf`](report/v2d_track1_report.pdf) (source in `report/`).

## Pipeline

| stage | what | script |
|---|---|---|
| name | Gemini Flash names the manipulated object | `gemini_object.py` |
| locate | LocateAnything-3B boxes for the name on ~11 keyframes | `locate_vm.py` |
| sam, cache | SAM 3 tracks; the actor is the foreground person touching the object track | `preprocess_vm.py`, `build_cache.py`, `motion_vm.py` |
| depth, d4rt | MoGe-2 intrinsics and metric background; OpenD4RT depth on MoGe's scale | `preprocess_vm.py`, `d4rt_vm.py` |
| human | SAM 3D Body per frame, then identity averaging and smoothing | `human_vm.py`, `human_post.py` |
| mesh | SAM 3D Objects meshes + layouts on 3 unoccluded crops (`--sam3d`) | `object_vm.py`, `sam3d_vm.py` |
| object | render-free point fit, with the SAM 3D layouts as extra start poses | `object_fit.py` |
| push1, joint | hand penetration push-out; contact, floor and depth placement | `pushout.py`, `joint_fit.py` |
| render | optional render-and-compare refinement (`--render`) | `render_fit.py` |
| final | centroid origin, smoothed object translation, human smoothing | `finalize.py` |

`scripts/e2e_vm.py` runs the stages on one Colab VM; `scripts/e2e_colab.py` drives it from your machine, keeps the VM
alive and pulls each stage's outputs as it finishes. Stages skip work whose outputs exist, so a rerun resumes.

## Reproducing the reference run (`lay_jfin`)

Requirements on your machine: Python 3, [`uv`](https://docs.astral.sh/uv/), the
[Colab CLI](https://pypi.org/project/google-colab-cli/) logged in, and:

- `data/v2d_submission_kit/`: the [submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip), unzipped;
- `.env.local` with `HF_TOKEN` (access to the challenge dataset and the gated SAM 3, SAM 3D Body and SAM 3D Objects
  repos) and `GEMINI_API_KEY`.

Then, on an A100 session (SAM 3D Objects needs at least 32 GB):

```bash
colab new -s e2e --gpu A100
python3 scripts/e2e_colab.py run e2e --run lay --episodes 0-29 --sam3d
```

The submission episodes land in `results/lay/episodes/`. Pack and check them with the kit's packer:

```bash
uv run --with numpy --with pandas --with pyarrow --with trimesh --with fast_simplification \
    scripts/validate_submission.py results/lay/episodes --pack results/submissions/lay_track1.parquet \
    --commit https://github.com/ChanHaoHao/4DHumanObjectInteraction/commit/<sha>
```

## Results

Public leaderboard scores of the reference run (lower is better):

| CD-H (cm) | CD-O (cm) | ACC-H | ACC-O | PEN (cm) |
|---|---|---|---|---|
| 14.24 | 26.93 | 0.1375 | 0.0732 | 0.0015 |

The report's ablation table covers the variants behind it. The scripts for the web-demo-mesh variants
(`import_sam3d.py`, `align_demo.py`, `dlay_vm.py`, `fin_vm.py`) are in `scripts/` and keep their original paths.

## External models

Gemini Flash, LocateAnything-3B, SAM 3, MoGe-2, OpenD4RT, SAM 3D Body, SAM 3D Objects. No training.

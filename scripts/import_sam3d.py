"""On the VM: SAM 3D web-demo .glb (sam_3d_mesh/object_EEEEEE_k.glb) -> object_raw/meshes/episode_EEEEEE_k.obj (+ _hi.ply),
cleaned like object_vm.py's Hunyuan meshes, and crops/episode_EEEEEE.json listing them (so e2e_vm's meshed() passes)."""
import json, sys
from pathlib import Path
import trimesh
sys.path.insert(0, "/content/repo/scripts")
from object_vm import HI_FACES, MAX_FACES, decimate, main_part, normalize

src, raw = Path("/content/repo/sam_3d_mesh"), Path("/content/object_raw")
(raw / "meshes").mkdir(parents=True, exist_ok=True)
for e in range(30):
    index = raw / "crops" / f"episode_{e:06d}.json"
    crops = json.loads(index.read_text()) if index.exists() else []
    ks = []
    for glb in sorted(src.glob(f"object_{e:06d}_*.glb")):
        k = int(glb.stem.split("_")[-1])
        mesh = trimesh.load(glb, force="mesh")
        mesh.merge_vertices(merge_tex=True, merge_norm=True)  # the demo glb is split at UV seams (36-56 pieces)
        mesh = normalize(main_part(mesh))
        decimate(mesh, HI_FACES).export(raw / "meshes" / f"episode_{e:06d}_{k}_hi.ply")
        low = decimate(mesh, MAX_FACES)
        low.export(raw / "meshes" / f"episode_{e:06d}_{k}.obj")
        ks.append(k)
        print(e, k, len(low.faces), low.extents.round(3).tolist(), flush=True)
    crops = [c for c in crops if c["k"] in ks] or [{"k": k} for k in ks]
    index.write_text(json.dumps(crops))

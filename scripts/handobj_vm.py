#!/usr/bin/env python3
"""Run the 100DOH hand-object detector (Shan et al., CVPR 2020) on episode videos. Runs on the Colab VM.

    python handobj_vm.py --repo /content/hand_object_detector --ckpt /content/faster_rcnn_1_8_132028.pth \
        --videos /content/v2d/track_1/videos/chunk-000/observation.images.exo_camera --episodes 0 --out OUT

The repo's compiled ops (model._C: nms, roi_align, roi_pool) are replaced by torchvision.ops, so nothing is
built. Per episode it writes to OUT:
  <ep>_handobj.npz   hands, objects: (M, 11) rows [frame, x1, y1, x2, y2, score, state, mag, dx, dy, side]
                     contact: (H, 6) rows [frame, hand row, x1, y1, x2, y2] = object each in-contact hand points to
  <ep>_summary.json  per-state / per-side counts, frames with a contacted object, union of contacted object boxes
  <ep>_handobj.mp4   overlay at half resolution
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tvo


def install_ops(lib: Path):
    sys.path.insert(0, str(lib))
    import model  # lib/model package

    c = types.ModuleType("model._C")
    c.nms = lambda boxes, scores, thresh: tvo.nms(boxes.float(), scores.float(), float(thresh))
    c.roi_align_forward = lambda x, rois, scale, oh, ow, ratio: tvo.roi_align(x, rois, (oh, ow), scale, ratio, aligned=False)
    c.roi_pool_forward = lambda x, rois, scale, oh, ow: (tvo.roi_pool(x, rois, (oh, ow), scale), None)
    model._C = c
    sys.modules["model._C"] = c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--videos", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--thresh_hand", type=float, default=0.5)
    p.add_argument("--thresh_obj", type=float, default=0.5)
    args = p.parse_args()

    repo = Path(args.repo)
    install_ops(repo / "lib")
    from model.faster_rcnn.resnet import resnet
    from model.roi_layers import nms
    from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
    from model.utils.blob import im_list_to_blob
    from model.utils.config import cfg, cfg_from_file, cfg_from_list
    from model.utils.net_utils import filter_object, vis_detections_filtered_objects_PIL
    from model.utils.viz_hand_obj import side_map2, state_map

    cfg_from_file(str(repo / "cfgs/res101.yml"))
    cfg_from_list(["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"])
    cfg.USE_GPU_NMS = True
    cfg.CUDA = True
    classes = np.asarray(["__background__", "targetobject", "hand"])
    net = resnet(classes, 101, pretrained=False, class_agnostic=False)
    net.create_architecture()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)  # third-party mirror: no pickled code
    net.load_state_dict(ckpt["model"])
    if "pooling_mode" in ckpt:
        cfg.POOLING_MODE = ckpt["pooling_mode"]
    net.cuda().eval()
    stds = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS, device="cuda")
    means = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS, device="cuda")
    font = str(repo / "lib/model/utils/times_b.ttf")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def detect(im):
        x = im.astype(np.float32) - cfg.PIXEL_MEANS
        s = float(cfg.TEST.SCALES[0]) / min(x.shape[:2])
        if np.round(s * max(x.shape[:2])) > cfg.TEST.MAX_SIZE:
            s = float(cfg.TEST.MAX_SIZE) / max(x.shape[:2])
        blob = im_list_to_blob([cv2.resize(x, None, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)])
        im_data = torch.from_numpy(blob).permute(0, 3, 1, 2).contiguous().cuda()
        im_info = torch.tensor([[blob.shape[1], blob.shape[2], s]], dtype=torch.float32, device="cuda")
        zeros5 = torch.zeros(1, 1, 5, device="cuda")
        rois, cls_prob, bbox_pred, *_, extra = net(im_data, im_info, zeros5, torch.zeros(1, dtype=torch.long, device="cuda"), zeros5)
        scores = cls_prob[0]
        contact = extra[0][0].argmax(2).squeeze(0).unsqueeze(-1).float()
        offset = extra[1][0].squeeze(0)
        side = (torch.sigmoid(extra[2][0]) > 0.5).squeeze(0).float()
        deltas = (bbox_pred.view(-1, 4) * stds + means).view(1, -1, 4 * len(classes))
        boxes = clip_boxes(bbox_transform_inv(rois[:, :, 1:5], deltas, 1), im_info, 1)[0] / s
        dets = {}
        for j, name in ((1, "obj"), (2, "hand")):
            inds = torch.nonzero(scores[:, j] > (args.thresh_hand if name == "hand" else args.thresh_obj)).view(-1)
            if inds.numel() == 0:
                dets[name] = None
                continue
            sc = scores[inds, j]
            order = torch.sort(sc, 0, True)[1]
            bx = boxes[inds][:, j * 4:(j + 1) * 4]
            d = torch.cat((bx, sc[:, None], contact[inds], offset[inds], side[inds]), 1)[order]
            keep = nms(bx[order], sc[order], cfg.TEST.NMS)
            dets[name] = d[keep.view(-1).long()].cpu().numpy()
        return dets["obj"], dets["hand"]

    for e in (int(v) for v in args.episodes.split(",")):
        n = f"episode_{e:06d}"
        cap = cv2.VideoCapture(f"{args.videos}/{n}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        raw = out / f"{n}_raw.mp4"
        writer = None
        hands, objs, contact = [], [], []
        t0, f = time.time(), 0
        with torch.no_grad():
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                obj_d, hand_d = detect(im)
                if hand_d is not None:
                    hands += [[f, *r] for r in hand_d]
                if obj_d is not None:
                    objs += [[f, *r] for r in obj_d]
                if hand_d is not None and obj_d is not None:
                    match = filter_object(obj_d, hand_d)
                    contact += [[f, i, *obj_d[k, :4]] for i, k in enumerate(match) if hand_d[i, 5] > 0 and k >= 0]
                vis = np.asarray(vis_detections_filtered_objects_PIL(im.copy(), obj_d, hand_d, args.thresh_hand,
                                                                     args.thresh_obj, font_path=font).convert("RGB"))
                vis = cv2.resize(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), (im.shape[1] // 2, im.shape[0] // 2))
                if writer is None:
                    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, vis.shape[1::-1])
                writer.write(vis)
                f += 1
                if f % 100 == 0:
                    print(f"ep {e} frame {f} {(time.time() - t0) / f:.3f}s/frame", flush=True)
        writer.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "26", str(out / f"{n}_handobj.mp4")], check=True)
        raw.unlink()
        H = np.array(hands, np.float32).reshape(-1, 11)
        O = np.array(objs, np.float32).reshape(-1, 11)
        C = np.array(contact, np.float32).reshape(-1, 6)
        np.savez_compressed(out / f"{n}_handobj.npz", hands=H, objects=O, contact=C)
        summary = {
            "frames": f, "sec_per_frame": round((time.time() - t0) / max(f, 1), 3),
            "frames_with_hand": int(len(np.unique(H[:, 0]))), "frames_with_object": int(len(np.unique(O[:, 0]))),
            "frames_with_contacted_object": int(len(np.unique(C[:, 0]))),
            "hand_states": {state_map[k]: int((H[:, 6] == k).sum()) for k in state_map},
            "hand_sides": {side_map2[k]: int((H[:, 10] == k).sum()) for k in side_map2},
            "contacted_object_box_median": np.median(C[:, 2:], 0).round(1).tolist() if len(C) else None,
            "contacted_object_box_union": [*C[:, 2:4].min(0).round(1).tolist(), *C[:, 4:6].max(0).round(1).tolist()] if len(C) else None,
        }
        (out / f"{n}_summary.json").write_text(json.dumps(summary, indent=1))
        print(f"ep {e} done", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

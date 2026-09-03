#!/usr/bin/env python3
"""Find WorldPose frames whose pose matches a reference photo.

Two stages, two environments:

  1. HMR2 on the photo (run in ~/venvs/hmr2):
       ~/venvs/hmr2/bin/python pose_retrieval.py estimate photo.jpg --out ref_hmr2.npz
     Uses YOLO11-pose (base env) keypoints if present as <out>.kpts.npy, otherwise the full image
     as the person box. Writes body_pose (23,3,3) and global_orient (3,3).

  2. Retrieval (run in the `worldpose` conda env):
       python pose_retrieval.py scan  --cache joints_scan.npz [--stride 4]      # once, ~25 min CPU
       python pose_retrieval.py match ref_hmr2.npz --cache joints_scan.npz --top 12 [--render out_dir]
     Every WorldPose frame is converted to root-relative 3D joints; both sides are put in the SMPL
     canonical frame (undo global_orient), scaled by torso length, and compared with a weighted joint
     distance (ankles/knees/wrists weigh most). A left/right-mirrored reference is also tried, so a
     right-footed photo can match a left-footed frame (`mirrored=True` -> render with --mirror).
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

BODY_MODELS = "/mnt/HDD12TB-1/ding_2026/KiTQA/WorldPose_vis_code/body_models"
DATASET = "/mnt/HDD12TB-1/ding_2026/SportsMesh/datasets/WorldPose_FIFA_full"
RENDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worldpose_lowpoly.py")

# SMPL joint order: 0 pelvis 1 lhip 2 rhip 3 spine1 4 lknee 5 rknee 6 spine2 7 lankle 8 rankle 9 spine3
# 10 lfoot 11 rfoot 12 neck 13 lcollar 14 rcollar 15 head 16 lsho 17 rsho 18 lelb 19 relb 20 lwri 21 rwri 22 lhand 23 rhand
WEIGHTS = np.array([0, 1, 1, .5, 2, 2, .5, 3, 3, .5, 1.5, 1.5, .5, .3, .3, 1, 1, 1, 1.5, 1.5, 2, 2, .5, .5], float)
SWAP = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]


def cmd_estimate(a):
    import cv2
    import torch
    _orig = torch.load
    torch.load = lambda *x, **k: _orig(*x, **{**k, "weights_only": False})
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    from hmr2.datasets.vitdet_dataset import ViTDetDataset
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.cuda().eval()
    img = cv2.imread(a.photo)
    kp = a.out + ".kpts.npy"
    if os.path.exists(kp):
        k = np.load(kp)
        pts = k[k[:, 2] > 0.3, :2]
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        mx, my = 0.2 * (x1 - x0), 0.15 * (y1 - y0)
        box = [[x0 - mx, y0 - my - 20, x1 + mx, y1 + my]]
    else:
        box = [[0, 0, img.shape[1], img.shape[0]]]
    ds = ViTDetDataset(cfg, img, np.array(box, np.float32))
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch)
    np.savez(a.out, body_pose=out["pred_smpl_params"]["body_pose"][0].cpu().numpy(),
             global_orient=out["pred_smpl_params"]["global_orient"][0, 0].cpu().numpy())
    print("saved", a.out)


def cmd_scan(a):
    import smplx
    import torch
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    seqs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DATASET}/poses/*.npz"))
    J_all, GO_all, meta = [], [], []
    t0 = time.time()
    for si, seq in enumerate(seqs):
        z = np.load(f"{DATASET}/poses/{seq}.npz")
        N, T = z["body_pose"].shape[:2]
        for n in range(N):
            if np.isnan(z["betas"][n]).any():
                continue
            fr = np.arange(0, T, a.stride)
            bp, go, tr = z["body_pose"][n, fr], z["global_orient"][n, fr], z["transl"][n, fr]
            ok = ~(np.isnan(bp).any(1) | np.isnan(go).any(1) | np.isnan(tr).any(1))
            if ok.sum() == 0:
                continue
            fr, bp, go, tr = fr[ok], bp[ok], go[ok], tr[ok]
            m = smplx.create(BODY_MODELS, model_type="smpl", gender="neutral", batch_size=len(fr))
            with torch.no_grad():
                J = m(betas=torch.tensor(np.repeat(z["betas"][n:n + 1], len(fr), 0)), body_pose=torch.tensor(bp),
                      global_orient=torch.tensor(go), transl=torch.tensor(tr)).joints[:, :24].numpy()
            J_all.append((J - J[:, 0:1]).astype(np.float32))
            GO_all.append(go.astype(np.float32))
            meta += [(seq, n, int(f)) for f in fr]
        print(f"[{si + 1}/{len(seqs)}] {seq} {time.time() - t0:.0f}s", flush=True)
    np.savez(a.cache, J=np.concatenate(J_all), GO=np.concatenate(GO_all), meta=np.array(meta, dtype=object))
    print("saved", len(meta), "frames ->", a.cache)


def cmd_match(a):
    import smplx
    import torch
    from scipy.spatial.transform import Rotation as Rot
    z = np.load(a.ref)
    bp = Rot.from_matrix(z["body_pose"]).as_rotvec().reshape(1, 69).astype(np.float32)
    go = Rot.from_matrix(z["global_orient"])
    m = smplx.create(BODY_MODELS, model_type="smpl", gender="neutral", batch_size=1)
    with torch.no_grad():
        Jr = m(body_pose=torch.tensor(bp),
               global_orient=torch.tensor(go.as_rotvec().reshape(1, 3), dtype=torch.float32)).joints[0, :24].numpy()
    Jr = go.inv().apply(Jr - Jr[0])

    d = np.load(a.cache, allow_pickle=True)
    J, GO, meta = d["J"], d["GO"], d["meta"]
    Rw = Rot.from_rotvec(GO)
    Jc = np.stack([Rw.inv().apply(J[:, j]) for j in range(24)], 1)
    torso = lambda X: np.linalg.norm(X[..., 15, :] - X[..., 0, :], axis=-1)
    Jc = Jc / torso(Jc)[:, None, None]
    Jr = Jr / torso(Jr)
    w = WEIGHTS / WEIGHTS.sum()
    dist = lambda ref: np.sqrt((np.sum((Jc - ref[None]) ** 2, 2) * w).sum(1))
    ref_m = Jr[SWAP].copy(); ref_m[:, 0] *= -1
    d1, d2 = dist(Jr), dist(ref_m)
    mir = d2 < d1
    dd = np.where(mir, d2, d1)
    picked = []
    for i in np.argsort(dd):
        s, p, f = meta[i]
        if any(s == q[1] and p == q[2] and abs(f - q[3]) < a.min_gap for q in picked):
            continue
        picked.append((float(dd[i]), s, int(p), int(f), bool(mir[i])))
        if len(picked) == a.top:
            break
    for k, r in enumerate(picked):
        print(f"{k}: --seq {r[1]} --player {r[2]} --frame {r[3]}{' --mirror' if r[4] else ''}   dist={r[0]:.3f}")
    if a.render:
        os.makedirs(a.render, exist_ok=True)
        for k, r in enumerate(picked):
            cmd = [sys.executable, RENDER, "--seq", r[1], "--player", str(r[2]), "--frame", str(r[3]),
                   "--out", os.path.join(a.render, f"match{k}.svg"), "--face", "-3", "--dissolve", "1", "--cell", "0.04"]
            if r[4]:
                cmd.append("--mirror")
            subprocess.run(cmd, check=True, capture_output=True)
        print("rendered to", a.render)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate"); e.add_argument("photo"); e.add_argument("--out", default="ref_hmr2.npz")
    s = sub.add_parser("scan"); s.add_argument("--cache", default="joints_scan.npz"); s.add_argument("--stride", type=int, default=4)
    mt = sub.add_parser("match"); mt.add_argument("ref"); mt.add_argument("--cache", default="joints_scan.npz")
    mt.add_argument("--top", type=int, default=12); mt.add_argument("--min-gap", type=int, default=45)
    mt.add_argument("--render", default=None, help="directory to render the matches as SVG")
    a = ap.parse_args()
    {"estimate": cmd_estimate, "scan": cmd_scan, "match": cmd_match}[a.cmd](a)


if __name__ == "__main__":
    main()

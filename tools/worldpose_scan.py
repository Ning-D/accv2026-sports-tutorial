#!/usr/bin/env python3
"""Scan WorldPose for goalkeeper-dive and shooting-kick poses (run in the `worldpose` env).

    python worldpose_scan.py --stride 5 --out scan.npz

Scores per (sequence, player, frame):
  dive : torso far from vertical, pelvis low, arms wide / raised   -> goalkeeper saves
  kick : one foot high while the other is planted, leg swung forward -> shots / clearances
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import smplx

BODY_MODELS = "/mnt/HDD12TB-1/ding_2026/KiTQA/WorldPose_vis_code/body_models"
DATASET = "/mnt/HDD12TB-1/ding_2026/SportsMesh/datasets/WorldPose_FIFA_full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="scan.npz")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    rows = []
    t0 = time.time()
    seqs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DATASET}/poses/*.npz"))
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
                J = m(betas=torch.tensor(np.repeat(z["betas"][n:n + 1], len(fr), 0)),
                      body_pose=torch.tensor(bp), global_orient=torch.tensor(go),
                      transl=torch.tensor(tr)).joints[:, :24].numpy()
            pel, spine3, head = J[:, 0], J[:, 9], J[:, 15]
            lank, rank, lkne, rkne = J[:, 7], J[:, 8], J[:, 4], J[:, 5]
            lwr, rwr = J[:, 20], J[:, 21]
            ground = np.minimum(lank[:, 2], rank[:, 2])
            torso = head - pel
            torso_tilt = np.degrees(np.arccos(np.clip(torso[:, 2] / (np.linalg.norm(torso, axis=1) + 1e-9), -1, 1)))
            pel_h = pel[:, 2] - ground
            arms = np.linalg.norm(lwr - rwr, axis=1)
            hands_up = np.maximum(lwr[:, 2], rwr[:, 2]) - head[:, 2]
            dive = (torso_tilt > 45) * (torso_tilt / 90) * (1.2 - np.clip(pel_h, 0, 1.2)) * (0.5 + arms + np.clip(hands_up, 0, 0.6))
            foot_h = np.abs(lank[:, 2] - rank[:, 2])
            planted = np.minimum(lank[:, 2], rank[:, 2]) - ground < 0.12
            # kicking leg: the higher ankle; swung forward = ankle ahead of pelvis along body forward
            hi_l = lank[:, 2] > rank[:, 2]
            kank = np.where(hi_l[:, None], lank, rank)
            kick = foot_h * planted * (1 + np.clip(np.linalg.norm(kank - pel, axis=1) - 0.6, 0, 0.6))
            for i in range(len(fr)):
                rows.append((seq, n, int(fr[i]), float(dive[i]), float(kick[i]), float(torso_tilt[i]),
                             float(pel_h[i]), float(arms[i]), float(foot_h[i])))
        print(f"[{si + 1}/{len(seqs)}] {seq} N={N} T={T} {time.time() - t0:.0f}s", flush=True)
    arr = np.array(rows, dtype=object)
    np.save(a.out.replace(".npz", ".npy"), arr, allow_pickle=True)
    print("saved", len(rows), "rows")


if __name__ == "__main__":
    main()

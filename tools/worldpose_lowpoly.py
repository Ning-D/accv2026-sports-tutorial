#!/usr/bin/env python3
"""Low-poly wireframe athlete SVG from a real WorldPose (FIFA World Cup) SMPL mesh.

Run inside the `worldpose` conda env (needs smplx + torch):

    python worldpose_lowpoly.py --seq BRA_KOR_231503 --player 19 --frame 756 \
        --out athlete.svg [--cell 0.035] [--yaw 0] [--mirror] [--seed 7] \
        [--color "#dff3d3"] [--dissolve 0.62] [--height 1000]

Pipeline: SMPL forward pass -> vertex-clustering decimation (cell size in metres)
-> project with the sequence's real broadcast camera (optional extra yaw about the
vertical axis) -> back-face culling + depth fade -> the top-right part of the body
"dissolves" into fragments drifting away in 3D -> transparent SVG.
"""
import argparse
import math

import numpy as np

BODY_MODELS = "/mnt/HDD12TB-1/ding_2026/KiTQA/WorldPose_vis_code/body_models"
DATASET = "/mnt/HDD12TB-1/ding_2026/SportsMesh/datasets/WorldPose_FIFA_full"


def smpl_vertices(seq, player, frame):
    import smplx
    import torch
    z = np.load(f"{DATASET}/poses/{seq}.npz")
    m = smplx.create(BODY_MODELS, model_type="smpl", gender="neutral", batch_size=1)
    with torch.no_grad():
        out = m(betas=torch.tensor(z["betas"][player:player + 1]),
                body_pose=torch.tensor(z["body_pose"][player, frame:frame + 1]),
                global_orient=torch.tensor(z["global_orient"][player, frame:frame + 1]),
                transl=torch.tensor(z["transl"][player, frame:frame + 1]))
    V = out.vertices[0].numpy().astype(np.float64)
    if np.isnan(V).any():
        raise SystemExit("player not visible in this frame (NaN pose)")
    cam = np.load(f"{DATASET}/cameras/{seq}.npz")
    go = z["global_orient"][player, frame].astype(np.float64)
    return V, m.faces.astype(np.int64), cam["R"][frame], cam["t"][frame], cam["K"][frame], go


def decimate(V, F, cell):
    """Vertex clustering: snap vertices to a 3D grid, merge, drop degenerate faces."""
    keys = np.floor((V - V.min(0)) / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    P = np.zeros((n, 3))
    cnt = np.zeros(n)
    np.add.at(P, inv, V)
    np.add.at(cnt, inv, 1)
    P /= cnt[:, None]
    F2 = inv[F]
    good = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[good]
    # remove duplicate faces (same vertex set)
    F2 = np.unique(np.sort(F2, axis=1), axis=0, return_index=True)[1]
    return P, inv[F][good][F2]


def smpl_from_params(npz_path):
    """Render arbitrary SMPL params (e.g. an HMR2 estimate): body_pose (23,3,3) or (69,),
    global_orient (3,3) or (3,), optional betas. Uses a synthetic frontal camera 6 m away."""
    import smplx
    import torch
    from scipy.spatial.transform import Rotation as Rot
    z = np.load(npz_path)
    bp = z["body_pose"]
    bp = Rot.from_matrix(bp.reshape(-1, 3, 3)).as_rotvec().reshape(1, 69) if bp.ndim == 3 else bp.reshape(1, 69)
    go = z["global_orient"]
    go = Rot.from_matrix(go.reshape(3, 3)).as_rotvec().reshape(1, 3) if go.size == 9 else go.reshape(1, 3)
    betas = z["betas"].reshape(1, 10) if "betas" in z.files else np.zeros((1, 10), np.float32)
    m = smplx.create(BODY_MODELS, model_type="smpl", gender="neutral", batch_size=1)
    with torch.no_grad():
        out = m(betas=torch.tensor(betas, dtype=torch.float32), body_pose=torch.tensor(bp, dtype=torch.float32),
                global_orient=torch.tensor(go, dtype=torch.float32))
    V = out.vertices[0].numpy().astype(np.float64)
    # params are in an OpenCV-style camera frame (y down, z forward): use identity camera, 6 m away
    R = np.eye(3); t = np.array([0.0, 0.0, 6.0]); K = np.array([[1500.0, 0, 0], [0, 1500.0, 0], [0, 0, 1]])
    return V, m.faces.astype(np.int64), R, t, K, go[0].astype(np.float64)


def rotz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq")
    ap.add_argument("--player", type=int)
    ap.add_argument("--frame", type=int)
    ap.add_argument("--params", help="npz with SMPL body_pose/global_orient to render instead of a dataset frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=0.035, help="clustering cell size in metres")
    ap.add_argument("--yaw", type=float, default=0.0, help="extra rotation of the body about the vertical axis (deg)")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--face", type=float, default=None,
                    help="auto-rotate so the body faces the camera; value = extra offset in deg (0 = frontal, 30 = three-quarter)")
    ap.add_argument("--dir", choices=["right", "left", "up"], default="right", help="dissolve/fragment direction (screen space)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--color", default="#dff3d3")
    ap.add_argument("--dissolve", type=float, default=0.62, help="threshold 0..1 along the up-right axis; 1 = no dissolve")
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--stroke", type=float, default=1.0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    V, F, R, t, K, go = smpl_from_params(a.params) if a.params else smpl_vertices(a.seq, a.player, a.frame)
    if a.face is not None and not a.params:
        # SMPL canonical body faces +z (template is y-up); WorldPose world is z-up, so the
        # body's forward axis in world = R_root @ [0, 0, 1] with the y-up->z-up mapping baked in R_root.
        from scipy.spatial.transform import Rotation as Rot
        fwd = Rot.from_rotvec(go).apply([0, 0, 1.0])
        fwd[2] = 0; fwd /= (np.linalg.norm(fwd) + 1e-9)
        # camera viewing direction on the ground plane (world), from body toward camera
        cam_pos = -R.T @ t
        to_cam = cam_pos - V.mean(0); to_cam[2] = 0; to_cam /= (np.linalg.norm(to_cam) + 1e-9)
        ang = math.degrees(math.atan2(to_cam[1], to_cam[0]) - math.atan2(fwd[1], fwd[0]))
        a.yaw = a.yaw + ang + a.face
        print(f"auto-facing: rotating body by {ang:.0f} deg (+{a.face:.0f} offset)")
    if a.yaw:
        c = V.mean(0)
        V = (rotz(a.yaw) @ (V - c).T).T + c
    P, F = decimate(V, F, a.cell)

    # camera-space geometry
    Xc = (R @ P.T).T + t                       # (n,3), camera looks along +z
    tri = Xc[F]                                # (m,3,3)
    cen = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = np.einsum("ij,ij->i", nrm, cen) < 0  # facing the camera
    F, tri, cen = F[front], tri[front], cen[front]

    # dissolve field along "up-right" in camera space (x right, y down)
    d = {"right": np.array([1.0, -1.0, 0.0]), "left": np.array([-1.0, -1.0, 0.0]), "up": np.array([0.0, -1.0, 0.0])}[a.dir]
    d = d / np.linalg.norm(d)
    proj = cen @ d
    lo, hi = proj.min(), proj.max()
    s = np.clip(((proj - lo) / (hi - lo + 1e-9) - a.dissolve) / max(1e-6, 1 - a.dissolve), 0, 1)
    body_h = Xc[:, 1].max() - Xc[:, 1].min()

    kept, frags = [], []
    for T, si in zip(tri, s):
        p_rm = 0.0 if a.dissolve >= 1 else min(0.7, si ** 2.2)
        if rng.random() < p_rm:
            c = T.mean(0)
            dist = rng.uniform(0.02, 0.18) * (0.4 + si) * body_h
            jitter = rng.normal(0, 0.02 * body_h, 3)
            shift = d * dist + jitter
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            ang = math.radians(rng.uniform(-45, 45))
            Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            Rm = np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * Kx @ Kx
            shrink = rng.uniform(0.35, 0.9)
            Tf = ((T - c) * shrink) @ Rm.T + c + shift
            frags.append((Tf, 0.3 + 0.5 * (1 - si)))
        else:
            kept.append(T)
    kept = np.array(kept)
    # far spray
    n_spray = int(len(frags) * 0.5)
    for _ in range(n_spray):
        base = frags[rng.integers(len(frags))][0] if frags else kept[rng.integers(len(kept))]
        c = base.mean(0) + d * rng.uniform(0.03, 0.22) * body_h + rng.normal(0, 0.04 * body_h, 3)
        sz = rng.uniform(0.006, 0.02) * body_h
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        v = np.cross(u, rng.normal(size=3)); v /= np.linalg.norm(v)
        a0 = rng.uniform(0, 2 * math.pi)
        Tf = np.array([c + sz * (math.cos(a0 + k * 2.1) * u + math.sin(a0 + k * 2.1) * v) for k in range(3)])
        frags.append((Tf, rng.uniform(0.2, 0.55)))

    # perspective projection with the real intrinsics
    def project(X):
        uv = (K @ X.reshape(-1, 3).T).T
        uv = uv[:, :2] / uv[:, 2:3]
        return uv.reshape(X.shape[:-1] + (2,))

    all_tris = list(kept) + [f for f, _ in frags]
    uv_all = project(np.array(all_tris))
    if a.mirror:
        uv_all[..., 0] *= -1
    mn, mx = uv_all.reshape(-1, 2).min(0), uv_all.reshape(-1, 2).max(0)
    body_uv = project(kept)
    if a.mirror:
        body_uv[..., 0] *= -1
    bmn, bmx = body_uv.reshape(-1, 2).min(0), body_uv.reshape(-1, 2).max(0)
    scale = a.height / (bmx[1] - bmn[1])
    pad = 0.04 * a.height
    off = np.array([mn[0], mn[1]])
    W = int((mx[0] - mn[0]) * scale + 2 * pad)
    H = int((mx[1] - mn[1]) * scale + 2 * pad)

    def to_px(T):
        return (T - off) * scale + pad

    # depth fade for the body: nearer = brighter
    depth = kept.mean(1)[:, 2]
    dn = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    def poly(T2, op, fill_op=0.0, sw=a.stroke):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in T2)
        fill = f'fill="{a.color}" fill-opacity="{fill_op:.2f}"' if fill_op > 0 else 'fill="none"'
        return f'<polygon points="{pts}" {fill} stroke="{a.color}" stroke-opacity="{op:.2f}" stroke-width="{sw}"/>'

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
             '<g stroke-linejoin="round">']
    for T2, z in zip(uv_all[:len(kept)], dn):
        op = 0.95 - 0.45 * z
        fo = rng.uniform(0.03, 0.12) * (1 - z) if rng.random() < 0.4 else 0.0
        parts.append(poly(to_px(T2), op, fo))
    for T2, (_, op) in zip(uv_all[len(kept):], frags):
        parts.append(poly(to_px(T2), op, rng.uniform(0, 0.08) if rng.random() < 0.5 else 0.0, a.stroke * 0.9))
    parts.append("</g></svg>")
    open(a.out, "w").write("\n".join(parts))
    print(f"svg {W}x{H}: {len(kept)} faces kept, {len(frags)} fragments (from {len(F)} front faces, cell={a.cell}) -> {a.out}")


if __name__ == "__main__":
    main()

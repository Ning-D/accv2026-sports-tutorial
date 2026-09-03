#!/usr/bin/env python3
"""Turn a broadcast frame into a low-poly wireframe athlete SVG.

    python3 lowpoly_athlete.py <frame.png> <out.svg> [--seed N] [--color HEX]
                               [--roi x0,y0,x1,y1] [--preview out.png]

Pipeline: YOLO11 person segmentation -> pick the largest person inside ROI ->
upscale + smooth the mask -> sample boundary + interior points -> Delaunay ->
keep triangles inside the silhouette, "dissolve" the top-right part into
drifting fragments -> write a transparent SVG (and an optional PNG preview on a
dark green ground).
"""
import argparse
import math
import random

import cv2
import numpy as np
from scipy.spatial import Delaunay


def person_mask(frame_path, roi, upscale=1):
    from ultralytics import YOLO
    model = YOLO("yolo11x-seg.pt")
    src = frame_path
    if upscale > 1 and roi:
        img = cv2.imread(frame_path)
        crop = img[roi[1]:roi[3], roi[0]:roi[2]]
        src = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
        roi = [0, 0, src.shape[1], src.shape[0]]
    res = model(src, classes=[0], conf=0.15, verbose=False, retina_masks=True,
                imgsz=640)[0]
    if res.masks is None:
        raise SystemExit("no person found")
    best, best_area = None, 0
    for box, m in zip(res.boxes.xyxy.cpu().numpy(), res.masks.data.cpu().numpy()):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if roi and not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
            continue
        area = m.sum()
        if area > best_area:
            best, best_area = m, area
    if best is None:
        raise SystemExit("no person inside ROI")
    return (best > 0.5).astype(np.uint8)


def prep_mask(mask, target_h=1000, margin=0.12, blur=0.006, close=9):
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h = y1 - y0 + 1
    pad = int(h * margin)
    crop = mask[max(0, y0 - pad): y1 + pad, max(0, x0 - pad): x1 + pad]
    scale = target_h / crop.shape[0]
    big = cv2.resize(crop.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), max(1.0, target_h * blur))
    big = (big > 0.5).astype(np.uint8)
    # close small holes / gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    big = cv2.morphologyEx(big, cv2.MORPH_CLOSE, k)
    return big


def sample_points(mask, rng, boundary_step=13, interior_spacing=24):
    h, w = mask.shape
    pts = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        c = c[:, 0, :]
        if len(c) < 30:
            continue
        # resample boundary at ~boundary_step px, with jitter so edges are not too regular
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        n = max(8, int(d[-1] / boundary_step))
        for t in np.linspace(0, d[-1], n, endpoint=False):
            i = np.searchsorted(d, t)
            p = c[min(i, len(c) - 1)].astype(float) + rng.normal(0, 1.5, 2)
            pts.append(p)
    # interior: Poisson-like rejection sampling, denser near the boundary
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    inside = np.argwhere(mask > 0)
    rng.shuffle(inside)
    grid = {}
    cell = interior_spacing * 0.7

    def ok(p, r):
        gx, gy = int(p[0] // cell), int(p[1] // cell)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for q in grid.get((gx + dx, gy + dy), ()):
                    if np.hypot(*(p - q)) < r:
                        return False
        return True

    for y, x in inside[:60000]:
        dd = dist[y, x]
        if dd < 6:
            continue
        r = interior_spacing * (0.75 + 0.5 * min(dd / 60.0, 1.0))
        p = np.array([x, y], float)
        if ok(p, r):
            grid.setdefault((int(x // cell), int(y // cell)), []).append(p)
            pts.append(p)
    return np.array(pts)


def build_svg(mask, pts, rng, color, out_svg, dissolve=True):
    h, w = mask.shape
    tri = Delaunay(pts)
    tris = pts[tri.simplices]
    cent = tris.mean(axis=1)
    keep = []
    for t, c in zip(tris, cent):
        cx, cy = int(round(c[0])), int(round(c[1]))
        if 0 <= cx < w and 0 <= cy < h and mask[cy, cx]:
            # drop slivers
            d1, d2 = t[1] - t[0], t[2] - t[0]
            a = abs(d1[0] * d2[1] - d1[1] * d2[0]) / 2
            if a > 25:
                keep.append(t)
    keep = np.array(keep)

    # dissolve: the further a triangle is toward the top-right corner of the
    # silhouette, the more likely it is removed and re-emitted as a drifting fragment
    ys, xs = np.where(mask > 0)
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    frags = []
    kept = []
    for t in keep:
        c = t.mean(axis=0)
        u = (c[0] - bx0) / (bx1 - bx0 + 1e-6)  # 0 left .. 1 right
        v = 1 - (c[1] - by0) / (by1 - by0 + 1e-6)  # 0 bottom .. 1 top
        s = max(0.0, (0.55 * u + 0.45 * v) - 0.62) / 0.38  # 0..1 in the top-right band
        p_remove = 0.0 if not dissolve else min(0.85, s ** 1.8)
        if rng.random() < p_remove:
            # emit a fragment drifting up-right
            d = rng.uniform(20, 140) * (0.5 + s)
            ang = math.radians(rng.uniform(-70, -15))
            off = np.array([math.cos(ang), math.sin(ang)]) * d
            shrink = rng.uniform(0.35, 0.9)
            f = (t - c) * shrink + c + off
            rot = math.radians(rng.uniform(-40, 40))
            R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
            f = (f - f.mean(axis=0)) @ R.T + f.mean(axis=0)
            frags.append((f, 0.35 + 0.5 * (1 - s)))
        else:
            kept.append(t)

    # extra far fragments for the "spray"
    for _ in range(int(len(frags) * 0.6)):
        base = frags[rng.integers(len(frags))][0] if frags else kept[rng.integers(len(kept))]
        c = base.mean(axis=0)
        d = rng.uniform(80, 260)
        ang = math.radians(rng.uniform(-75, -10))
        c2 = c + np.array([math.cos(ang), math.sin(ang)]) * d
        sz = rng.uniform(6, 22)
        a0 = rng.uniform(0, 2 * math.pi)
        f = np.array([[c2[0] + sz * math.cos(a0 + k * 2.1 + rng.uniform(-0.3, 0.3)),
                       c2[1] + sz * math.sin(a0 + k * 2.1 + rng.uniform(-0.3, 0.3))] for k in range(3)])
        frags.append((f, rng.uniform(0.25, 0.6)))

    # canvas with headroom for fragments
    pad_top, pad_right = int(h * 0.18), int(w * 0.28)
    W, H = w + pad_right, h + pad_top

    def poly(t, op, sw, fill_op=0.0):
        pts_s = " ".join(f"{x:.1f},{y + pad_top:.1f}" for x, y in t)
        fill = f'fill="{color}" fill-opacity="{fill_op:.2f}"' if fill_op > 0 else 'fill="none"'
        return f'<polygon points="{pts_s}" {fill} stroke="{color}" stroke-opacity="{op:.2f}" stroke-width="{sw}"/>'

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
             '<g stroke-linejoin="round">']
    for t in kept:
        c = t.mean(axis=0)
        # faint facet fill on a random subset for depth
        fo = rng.uniform(0.04, 0.14) if rng.random() < 0.45 else 0.0
        parts.append(poly(t, rng.uniform(0.55, 0.95), 1.1, fo))
    for f, op in frags:
        parts.append(poly(f, op, 1.0, rng.uniform(0.0, 0.10) if rng.random() < 0.5 else 0.0))
    parts.append("</g></svg>")
    open(out_svg, "w").write("\n".join(parts))
    return W, H, len(kept), len(frags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("out_svg")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--color", default="#dff3d3")
    ap.add_argument("--roi", default=None, help="x0,y0,x1,y1 in frame pixels; person center must be inside")
    ap.add_argument("--preview", default=None)
    ap.add_argument("--no-dissolve", action="store_true")
    ap.add_argument("--upscale", type=int, default=1, help="upscale the ROI crop before segmentation")
    ap.add_argument("--blur", type=float, default=0.006, help="gaussian sigma as fraction of height")
    ap.add_argument("--close", type=int, default=9, help="morphological closing kernel px")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    roi = [int(v) for v in a.roi.split(",")] if a.roi else None

    mask = person_mask(a.frame, roi, a.upscale)
    mask = prep_mask(mask, blur=a.blur, close=a.close)
    pts = sample_points(mask, rng)
    W, H, nk, nf = build_svg(mask, pts, rng, a.color, a.out_svg, dissolve=not a.no_dissolve)
    print(f"svg {W}x{H}: {nk} mesh triangles, {nf} fragments -> {a.out_svg}")

    if a.preview:
        import cairosvg
        png = cairosvg.svg2png(url=a.out_svg, output_width=W // 2)
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        bg = np.full((arr.shape[0], arr.shape[1], 3), (28, 61, 31)[::-1], np.uint8)  # dark green BGR
        alpha = arr[:, :, 3:4] / 255.0
        comp = (arr[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
        cv2.imwrite(a.preview, comp)
        print("preview ->", a.preview)


if __name__ == "__main__":
    main()

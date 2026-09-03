#!/usr/bin/env python3
"""Wireframe soccer ball (truncated icosahedron, fan-triangulated) as a transparent SVG.

    python ball_lowpoly.py --out ball.svg [--size 420] [--rot 20,35,0] [--seed 3]
                           [--color "#dff3d3"] [--trail right|left|none] [--spin]

12 pentagons get a faint fill (the classic ball look), every face is fan-triangulated so the
mesh matches the low-poly athletes, back faces are culled, and a few fragments trail behind
the ball to suggest motion.
"""
import argparse
import math

import numpy as np


def truncated_icosahedron():
    phi = (1 + 5 ** 0.5) / 2
    ico = np.array([[0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                    [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                    [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]], float)
    ico /= np.linalg.norm(ico[0])
    # edges = pairs at the minimum distance
    d = np.linalg.norm(ico[:, None] - ico[None], axis=2)
    emin = d[d > 1e-6].min()
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if abs(d[i, j] - emin) < 1e-6]
    # faces = triangles of mutually adjacent vertices
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    faces = sorted({tuple(sorted((i, j, k))) for i in range(12) for j in adj[i] for k in adj[i] & adj[j]})
    # truncation points: 1/3 along each directed edge
    P = {}
    for i, j in edges:
        P[(i, j)] = ico[i] + (ico[j] - ico[i]) / 3
        P[(j, i)] = ico[j] + (ico[i] - ico[j]) / 3
    verts = []
    idx = {}
    for k, v in P.items():
        idx[k] = len(verts); verts.append(v)
    verts = np.array(verts)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)  # project onto sphere

    polys = []  # (list of vertex ids, kind)
    for i in range(12):  # pentagons around each icosahedron vertex
        ring = [idx[(i, j)] for j in adj[i]]
        n = ico[i]
        u = np.cross(n, [1, 0, 0]); u = u if np.linalg.norm(u) > 1e-3 else np.cross(n, [0, 1, 0]); u /= np.linalg.norm(u)
        w = np.cross(n, u)
        ring.sort(key=lambda vi: math.atan2(verts[vi] @ w, verts[vi] @ u))
        polys.append((ring, "pent"))
    for a, b, c in faces:  # hexagons inside each icosahedron face
        ids = [idx[(a, b)], idx[(b, a)], idx[(b, c)], idx[(c, b)], idx[(c, a)], idx[(a, c)]]
        n = verts[ids].mean(0)
        u = np.cross(n, [1, 0, 0]); u = u if np.linalg.norm(u) > 1e-3 else np.cross(n, [0, 1, 0]); u /= np.linalg.norm(u)
        w = np.cross(n, u)
        ids.sort(key=lambda vi: math.atan2(verts[vi] @ w, verts[vi] @ u))
        polys.append((ids, "hex"))
    return verts, polys


def rot(deg):
    ax, ay, az = [math.radians(v) for v in deg]
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=420, help="ball diameter in px")
    ap.add_argument("--rot", default="18,32,0")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--color", default="#dff3d3")
    ap.add_argument("--trail", choices=["right", "left", "none"], default="right", help="side the fragments trail toward")
    ap.add_argument("--stroke", type=float, default=1.1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    V, polys = truncated_icosahedron()
    V = V @ rot([float(x) for x in a.rot.split(",")]).T
    # camera at +z looking toward -z, mild perspective
    cam_d = 4.0

    def proj(p):
        f = cam_d / (cam_d - p[..., 2])
        return np.stack([p[..., 0] * f, -p[..., 1] * f], -1)

    r = a.size / 2
    trail_w = 0 if a.trail == "none" else int(a.size * 0.9)
    W = a.size + trail_w + 20
    H = a.size + 40
    cx = 10 + (trail_w if a.trail == "left" else 0) + r
    cy = H / 2

    def px(p2):
        return p2 * r * 0.98 + np.array([cx, cy])

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
             '<g stroke-linejoin="round" stroke-linecap="round">']
    tris_for_frag = []
    for ids, kind in polys:
        P = V[ids]
        n = P.mean(0)
        if n[2] < 0.02:  # back-face cull (viewer on +z)
            continue
        depth = n[2]  # 0..1, 1 = facing viewer
        op = 0.45 + 0.5 * depth
        pts2 = px(proj(P))
        fill = f'fill="{a.color}" fill-opacity="{0.10 + 0.10 * depth:.2f}"' if kind == "pent" else 'fill="none"'
        parts.append('<polygon points="%s" %s stroke="%s" stroke-opacity="%.2f" stroke-width="%.1f"/>' % (
            " ".join(f"{x:.1f},{y:.1f}" for x, y in pts2), fill, a.color, op, a.stroke * 1.25))
        # fan triangulation from the face centre (lighter strokes)
        c2 = px(proj(n / np.linalg.norm(n)))
        for k in range(len(ids)):
            p, q = pts2[k], pts2[(k + 1) % len(ids)]
            parts.append('<path d="M%.1f,%.1f L%.1f,%.1f" stroke="%s" stroke-opacity="%.2f" stroke-width="%.1f"/>' % (
                c2[0], c2[1], p[0], p[1], a.color, op * 0.55, a.stroke * 0.7))
            tris_for_frag.append(np.array([c2, p, q]))
    # silhouette circle for a clean outline
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * 0.985:.1f}" fill="none" stroke="{a.color}" stroke-opacity="0.9" stroke-width="{a.stroke * 1.4:.1f}"/>')

    if a.trail != "none":
        sgn = 1 if a.trail == "right" else -1
        for _ in range(46):
            base = tris_for_frag[rng.integers(len(tris_for_frag))]
            c = base.mean(0)
            # only shed from the trailing half of the ball
            if (c[0] - cx) * sgn < 0:
                continue
            dist = rng.uniform(0.15, 1.0) * trail_w * rng.uniform(0.5, 1.0)
            shift = np.array([sgn * dist, rng.normal(0, 0.12 * r) - 0.15 * r * (dist / max(trail_w, 1))])
            shrink = rng.uniform(0.22, 0.65)
            ang = rng.uniform(-0.7, 0.7)
            Rm = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            T = ((base - c) * shrink) @ Rm.T + c + shift
            op = rng.uniform(0.2, 0.6) * (1 - 0.5 * dist / max(trail_w, 1))
            fill = f'fill="{a.color}" fill-opacity="{rng.uniform(0, 0.08):.2f}"' if rng.random() < 0.5 else 'fill="none"'
            parts.append('<polygon points="%s" %s stroke="%s" stroke-opacity="%.2f" stroke-width="%.1f"/>' % (
                " ".join(f"{x:.1f},{y:.1f}" for x, y in T), fill, a.color, op, a.stroke * 0.9))
    parts.append("</g></svg>")
    open(a.out, "w").write("\n".join(parts))
    print(f"svg {W}x{H} -> {a.out}")


if __name__ == "__main__":
    main()

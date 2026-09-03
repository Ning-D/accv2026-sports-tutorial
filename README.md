# ACCV 2026 Tutorial website

Static single-page site (no build step): `index.html` + `assets/`.

**Live:** https://ning-d.github.io/accv2026-sports-tutorial/
**Repo:** https://github.com/Ning-D/accv2026-sports-tutorial (GitHub Pages, branch `main`, root)

## Update and publish

```bash
cd "ACCV tutorial/website"
# edit index.html
git add . && git commit -m "Update ..." && git push
```

GitHub Pages rebuilds within about a minute of each push.

## Hero artwork

The header shows a **real player mesh from the WorldPose dataset** (SMPL ground truth of FIFA World Cup 2022
broadcasts; ETH Zurich, ECCV 2024) plus a wireframe ball, all rendered as low-poly wireframes that dissolve
into fragments. Generators live in `tools/`; the athlete ones run in the `worldpose` conda env:

```bash
PY=/home/ding/miniconda/envs/worldpose/bin/python
# volley (CRO vs MOR): nearest WorldPose pose to a reference photo, found with tools/pose_retrieval.py
$PY tools/worldpose_lowpoly.py --seq CRO_MOR_184559 --player 20 --frame 1304 --face -3 --mirror --dissolve 0.7 --dir right --seed 4 --out assets/kicker.svg
# ball: truncated icosahedron with trailing fragments (plain python3)
python3 tools/ball_lowpoly.py --out assets/ball.svg --trail right
```

Flags: `--face <deg>` auto-rotates the body to face the camera (0 frontal, ±30 three-quarter); `--cell` mesh
coarseness in metres; `--dissolve` 0.5 (many fragments) … 1 (none); `--dir left|right|up` fragment direction;
`--mirror`, `--seed`. After regenerating, set the `aspect-ratio` of `.hero-art.kicker/.ball` in
`index.html` to the SVG size printed by the script.

**Pose retrieval from a photo** (`tools/pose_retrieval.py`): estimates the SMPL pose of a reference photo with
HMR2 (`~/venvs/hmr2`), converts every WorldPose frame to canonical root-relative 3D joints (cached in
`joints_scan.npz`, ~25 min once), and returns the nearest frames by weighted joint distance, allowing a
left/right mirror. Other close matches for the current volley: CRO_MOR_193322 11 112, ARG_CRO_223805 10 304,
NET_ARG_222226 18 1036 (all with `--mirror`). `tools/worldpose_scan.py` is the older heuristic scorer.
`tools/lowpoly_athlete.py` is the earlier silhouette-based generator (YOLO on a broadcast frame), kept for reference.

## To update later

- Clock times: edit the `Duration` column in `#schedule` once the program is fixed.
- Slides / reading list / resource sheet: replace the `Coming soon` badges in `#materials` with links.
- Presenter changes: `#schedule` and the hero `author-row`.

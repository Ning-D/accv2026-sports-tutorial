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
# in-stride strike (MOR vs POR), three-quarter view facing the camera
$PY tools/worldpose_lowpoly.py --seq MOR_POR_182352 --player 8 --frame 420 --face -40 --dissolve 0.68 --dir right --seed 5 --out assets/kicker.svg
# ball: truncated icosahedron with trailing fragments (plain python3)
python3 tools/ball_lowpoly.py --out assets/ball.svg --trail right
```

Flags: `--face <deg>` auto-rotates the body to face the camera (0 frontal, ±30 three-quarter); `--cell` mesh
coarseness in metres; `--dissolve` 0.5 (many fragments) … 1 (none); `--dir left|right|up` fragment direction;
`--mirror`, `--seed`. After regenerating, set the `aspect-ratio` of `.hero-art.kicker/.ball` in
`index.html` to the SVG size printed by the script.

`tools/worldpose_scan.py` scores every (sequence, player, frame) for dive-like and kick-like poses; other good
picks (in-stride strikes): FRA_MOR_220401 20 1980, ARG_FRA_181108 8 190, CRO_MOR_183903 5 635.
`tools/lowpoly_athlete.py` is the earlier silhouette-based generator (YOLO on a broadcast frame), kept for reference.

## To update later

- Clock times: edit the `Duration` column in `#schedule` once the program is fixed.
- Slides / reading list / resource sheet: replace the `Coming soon` badges in `#materials` with links.
- Presenter changes: `#schedule` and the hero `author-row`.

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

`assets/athlete.svg` is a low-poly wireframe of a **real player mesh from the WorldPose dataset**
(SMPL ground truth, FIFA World Cup 2022 broadcast; ETH Zurich, ECCV 2024), rendered with the
sequence's real broadcast camera. Generator: `tools/worldpose_lowpoly.py`, run in the `worldpose` conda env:

```bash
/home/ding/miniconda/envs/worldpose/bin/python tools/worldpose_lowpoly.py \
  --seq BRA_KOR_231503 --player 19 --frame 756 --cell 0.035 --out assets/athlete.svg
```

Useful flags: `--cell` (mesh coarseness in metres, 0.03 fine … 0.05 coarse), `--dissolve` (0.5 more
fragments … 1 none), `--yaw`, `--mirror`, `--seed`. After regenerating, set `aspect-ratio` of `.hero-art`
in `index.html` to the SVG's width/height printed by the script.

Other dynamic poses found by the scan (seq / player / frame): ARG_FRA_201902 15 1026 (sprint, arm
extended), ARG_FRA_182345 14 90 (kick), NET_ARG_231259 15 990, FRA_MOR_231753 16 942.

`tools/lowpoly_athlete.py` is the earlier silhouette-based generator (YOLO segmentation of a broadcast
frame → Delaunay), kept for reference.

## To update later

- Clock times: edit the `Duration` column in `#schedule` once the program is fixed.
- Slides / reading list / resource sheet: replace the `Coming soon` badges in `#materials` with links.
- Presenter changes: `#schedule` and the hero `author-row`.

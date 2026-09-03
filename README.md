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

`assets/athlete.svg` is a low-poly wireframe generated from a BFMD broadcast frame with
`tools/lowpoly_athlete.py` (YOLO11 segmentation → Delaunay mesh → SVG). To regenerate from another frame:

```bash
python3 tools/lowpoly_athlete.py frame.png assets/athlete.svg --roi x0,y0,x1,y1 --upscale 4 --blur 0.003 --close 5 --preview preview.png
```

Then update `aspect-ratio` of `.hero-art` in `index.html` to the SVG's width/height.

## To update later

- Clock times: edit the `Duration` column in `#schedule` once the program is fixed.
- Slides / reading list / resource sheet: replace the `Coming soon` badges in `#materials` with links.
- Presenter changes: `#schedule` and the hero `author-row`.

# ACCV 2026 Tutorial website

Static single-page site (no build step): `index.html` + `assets/`.

## Deploy on GitHub Pages

Option A — clean URL like the reference site (`<name>.github.io`):

1. Create a GitHub **organization** (e.g. `sports-video-tutorial-accv26`).
2. In it, create a repo named exactly `<org>.github.io`.
3. Push this folder to the `main` branch. Pages is enabled automatically.
4. URL: `https://<org>.github.io/`

Option B — under your own account:

1. Create a repo, e.g. `Ning-D/accv2026-tutorial`.
2. Push this folder to `main`.
3. Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
4. URL: `https://ning-d.github.io/accv2026-tutorial/`

```bash
cd website
git init -b main
git add .
git commit -m "Initial tutorial website"
git remote add origin git@github.com:<owner>/<repo>.git
git push -u origin main
```

Send the final URL to the ACCV tutorial chairs (they link tutorial pages from mid-September).

## To update later

- Clock times: edit the `Duration` column in `#schedule` once the program is fixed.
- Slides / reading list / resource sheet: replace the `Coming soon` badges in `#materials` with links.
- Presenter changes: `#schedule` and the hero `author-row`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the `website/` subdirectory of the BLS project. See the parent `../CLAUDE.md` for full project context, pipeline commands, data structures, and architecture overview.

## Commands

```bash
# Regenerate all data files (run from this directory after any source CSV or mapping change)
python3 generate_data.py

# Run locally (port 5000 is taken by macOS AirPlay)
flask run --port 5001   # → http://127.0.0.1:5001

# Build static site for GitHub Pages (normally done by CI on push)
python freeze.py   # → website/build/
```

## Key design constraint

`app.py` loads pre-computed JSON at request time — **no numpy/pandas/scipy at runtime**. All computation happens in `generate_data.py`, which writes to `data/`. `table_data.json` is pre-loaded at startup to pass Render's health check.

## Denominator options (`?denom=1`–`6`)

Selected via dropdown on all pages, persisted in URL via `history.pushState`.
- On `index.html`: `applyOpt(opt)` updates cells client-side from the embedded `ROW_OPTS` JS dict
- On `analysis.html`: `renderAnalysis()` re-reads `currentDenom` and rebuilds all table rows
- On `industry.html`: `renderShareCharts(opt)` re-renders Charts 3–8 via `Plotly.react`; Charts 1–2 (log-level) never re-render

## Chart layout (industry.html) — Plotly

4 rows × 2 cols (left = actual vs. trend, right = deviation):
- Row 1: Log-linear **level** (option-independent) — chart IDs `chart-ll-*`
- Row 2: Log-linear **share** — chart IDs `chart-ls-*`
- Row 3: HP filter **share** — chart IDs `chart-hp-*`
- Row 4: Raw-linear **share** — chart IDs `chart-rs-*`

The HP counterfactual is split at March 2020: solid dash = HP trend, dotted = linear extrapolation.

All 8 charts share a dual-handle x-axis zoom slider. Dragging calls `Plotly.relayout` on all charts. `tickformatstops` auto-switches from `%Y` to `%b %Y` when zoomed to < 12-month tick spacing.

## Static files

`static/website_guide.docx` is linked from the navbar as a download. Frozen-Flask auto-includes all static files in the build.

## Static site (freeze.py)

Uses Frozen-Flask. The industry route has a trailing slash (`/<series_id>/`, `strict_slashes=False`) so pages freeze as `build/CES.../index.html`. All templates use `url_for()` for internal links so `FREEZER_RELATIVE_URLS=True` converts them to relative paths. CSV files are not auto-crawled — `serve_data()` generator in `freeze.py` explicitly yields all `*_export.csv` filenames.

## Deployment

- **Render**: reads `Procfile`; `render.yaml` sets `healthCheckPath: /health`; push to `main` triggers auto-redeploy
- **GitHub Pages**: `.github/workflows/deploy.yml` runs `python freeze.py` then deploys `website/build/` to `gh-pages` branch via `peaceiris/actions-gh-pages`

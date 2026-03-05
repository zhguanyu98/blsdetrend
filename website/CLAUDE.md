# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is the `website/` subdirectory of the BLS project. See the parent `../CLAUDE.md` for full project context, pipeline commands, and architecture overview.

## Commands

```bash
# Regenerate all data files (run from this directory after any source CSV or mapping change)
python3 generate_data.py

# Run locally
flask run   # → http://127.0.0.1:5000
```

## Key design constraint

`app.py` loads pre-computed JSON at request time — no numpy/pandas/scipy at runtime. All computation happens in `generate_data.py`, which writes to `data/`.

## Data files (`data/`)

- `table_data.json` — loaded by `/`; contains all 842 rows with 5 denominator options each
- `{series_id}.json` — loaded by `/<series_id>`; contains time-series arrays for 6 charts
- `{series_id}_export.csv` — served by `/download/<series_id>`

## Denominator options (`?denom=1`–`5`)

Selected via dropdown on both pages, persisted in URL via `history.pushState`. On `index.html`, `applyOpt(opt)` updates cells client-side from the embedded `ROW_OPTS` JS dict. On `industry.html`, `renderShareCharts(opt)` re-renders Charts 3–6 via `Plotly.react`; Charts 1–2 (log-level) are option-independent and never re-render.

## Deployment

Render.com reads `Procfile` for the start command: `gunicorn app:app --bind 0.0.0.0:$PORT`. Push to `main` on GitHub triggers auto-redeploy.

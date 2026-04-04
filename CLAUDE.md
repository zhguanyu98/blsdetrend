# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Processes BLS Table B-1a data — 842 seasonally-adjusted employment series — and publishes a website showing employment levels, shares, and detrended measures for each industry.

## Key Commands

### Full data pipeline (run in order when source data changes)
```bash
# 1. Pull latest BLS data → b1a_wide_seriesid.csv
jupyter nbconvert --to notebook --execute pull_data.ipynb

# 2. Rebuild denominator mapping (only if b1a_mapping_with_parent.csv changes)
python3 build_mapping.py

# 3. Regenerate all website data files
cd website && python3 generate_data.py
```

### Run locally
```bash
cd website
flask run --port 5001   # port 5000 is taken by macOS AirPlay
# → http://127.0.0.1:5001
```

### Deploy
```bash
git add website/templates/ website/static/ website/generate_data.py \
        build_mapping.py b1a_mapping_with_denominators.csv b1a_wide_seriesid.csv \
        website/data/   # only needed if data was regenerated
git commit -m "..."
git push origin main    # triggers Render.com redeploy + GitHub Pages CI build
```

### Build static site (GitHub Pages — normally done by CI)
```bash
cd website && python freeze.py   # → website/build/  (843 HTML + 842 CSVs)
```

## Source Files (root directory)

| File | Content |
|---|---|
| `b1a_mapping_with_parent.csv` | 842 rows: series_id, industry_name, display_level (0–7), row_order, supersector_code, parent_series_id |
| `b1a_mapping_with_denominators.csv` | Above + `denominator_opt{1-5}` and `denominator_opt{1-5}_name` — primary input to website |
| `b1a_wide_seriesid.csv` | Raw employment (thousands), ~313 months × 842 series, date-indexed — primary input to website |
| `build_mapping.py` | Reads `b1a_mapping_with_parent.csv` → writes `b1a_mapping_with_denominators.csv` |
| `pull_data.ipynb` | Calls BLS API → writes `b1a_wide_seriesid.csv`. BLS API key embedded in notebook. |

Legacy files are in `legacy/`.

## build_mapping.py — Denominator logic

**Critical**: `parent_series_id` fields mostly point directly to Total private (level 1), skipping intermediate levels. `find_ancestor_at_level` scans **backwards through row_order** (not up the parent chain). It returns the first preceding row at `target_lvl`, stopping if it hits a row with level < target_lvl (out of scope).

Denominator options:
- **Opt 1**: Nearest level-4 ancestor (fallback: level-3, then Goods/Service)
- **Opt 2**: Nearest level-3 ancestor (fallback: Goods/Service)
- **Opt 3**: Nearest level-2 ancestor (fallback: Goods/Service)
- **Opt 4**: Always Goods/Service-providing by supersector code
- **Opt 5**: Total private (private) or Total government (govt)
- **Opt 6**: Always Total nonfarm (`CES0000000001`)

Special cases (applied before option logic): Total nonfarm → itself; Total private → Total nonfarm; Goods/Service-providing → Total private; Govt level 2 → Total nonfarm; Govt level > 2 → Total government.

## Website (`website/`)

### Key design constraint
`app.py` loads pre-computed JSON at request time — **no numpy/pandas/scipy at runtime**. All computation is in `generate_data.py` which writes to `data/`. `table_data.json` is pre-loaded at startup (not per-request) to pass Render's health check.

### Flask routes (`app.py`)
- `/` → `index.html` (main table, pre-loads `table_data.json` at startup)
- `/analysis/` → `analysis.html` (quadrant analysis page)
- `/<series_id>/` → `industry.html` (per-industry Plotly charts)
- `/data/<filename>` → serves raw JSON/CSV from `data/`
- `/download/<series_id>` → attachment download of `{series_id}_export.csv`
- `/health` → health check for Render.com

### `generate_data.py` — all computation
- Fit window: **Jan 2010 – Feb 2020**; values NaN before Jan 2010
- `fit_log_linear`: OLS on log values → (trend_log, resid_log)
- `fit_raw_linear`: OLS on raw share → (trend_raw, resid_pct) where resid_pct = (actual − pred) / pred
- `fit_hp_log_share`: `hpfilter(λ=129600)` on log share, extrapolates post-Feb 2020 using last-24-month slope
- Stored lags: `SNAP_LAGS = [1, 3, 6, 9, 12, 15]` months
- Outputs `data/table_data.json`, `data/{series_id}.json`, `data/{series_id}_export.csv`

### `table_data.json` row structure
Each row contains top-level fields plus an `opts` dict keyed `"1"`–`"6"`:
```
emp_recent, emp_prev, display_level, series_id, industry_name
dev_log_level, dev_log_level_{1,3,6,9,12,15}m, dev_log_level_covid   ← option-independent

opts["1"…"6"]: {
  share, share_pct,
  share_{1,3,6,9,12,15}m, share_covid,
  denom_name,
  dev_log_share, dev_log_share_{n}m, dev_log_share_covid,
  dev_log_share_hp, dev_log_share_hp_{n}m, dev_log_share_hp_covid,
  dev_raw_share_pct, dev_raw_share_pct_{n}m, dev_raw_share_pct_covid,
}
```
`share_covid` = share at the dynamically identified COVID shock month (max |MoM share change| in Mar–May 2020).

### Per-industry JSON structure (`{series_id}.json`)
```
dates[], march2020, emp_level[], trend_ll_level[], resid_ll_level[]   ← option-independent
options["1"…"6"]: {
  emp_share_pct[], trend_ls_share_pct[], resid_ls_share[],   ← log-linear share
  trend_rs_share_pct[], resid_rs_pct[],                      ← raw-linear share
  hp_cf_share_pct[], resid_hp[],                             ← HP filter
  denom_name, denom_id
}
```

### 8 charts per industry page (4 rows × 2 cols) — Plotly
- Row 1: Log-linear **level** — actual vs. trend; log deviation (option-independent)
- Row 2: Log-linear **share** — actual vs. trend; log deviation
- Row 3: HP filter **share** — actual vs. HP/extrapolated; log deviation
- Row 4: Raw-linear **share** — actual vs. trend; % deviation

Charts 2–4 re-render on denominator change via `Plotly.react`. All charts share a dual-handle x-axis zoom slider; the March 2020 vertical line is a Plotly `shape` and stays visible at all zoom levels. `tickformatstops` switches the x-axis from `%Y` to `%b %Y` when zoomed in below 12-month tick spacing.

### Homepage (`index.html`) — column order
Industry | Lvl | Employment T | Employment T-1 | Share (% of denom) | 3M share growth | Dev. Log Share (linear) | Dev. Log Share (HP) | Denominator | → Analysis

- **3M share growth**: `(share[t] − share[t-3]) / share[t-3]`, displayed as %, 2dp
- **→ Analysis**: links to `/analysis/?highlight={series_id}&denom={opt}`; for level ≥ 5 rows adds `&set=1,2,3,4,5,6,7` so the row is visible
- Dev. Log Share (linear) and Dev. Log Share (HP) are sortable; 3M share growth is sortable

### Analysis page (`analysis.html`)
Quadrant categorization: above/below trend (sign of deviation) × increasing/decreasing (sign of `recent_pct`).

**Second derivative formulas** (percentage-based, not raw share differences):
- 3M window: `recent_pct = (s[t]−s[t-3])/s[t-3]`; `past_pct = (s[t-3]−s[t-12])/s[t-12]/3`
- 6M window: `recent_pct = (s[t]−s[t-6])/s[t-6]`; `past_pct = (s[t-6]−s[t-12])/s[t-12]`
- `second_deriv = recent_pct − past_pct`; all three displayed as %, 2dp

**Six patterns** based on signs of R (`recent_pct`), P (`past_pct`), D (`second_deriv`):
Rising faster (R>0,P>0,D>0) · Rising slower (R>0,P>0,D<0) · Falling slower (R<0,P<0,D>0) · Falling faster (R<0,P<0,D<0) · Reversal up (R>0,P<0) · Reversal down (R<0,P>0)

**Columns**: Industry | Lvl | Employment | Share | N-month share growth | Past growth rate (with sub-label showing reference period) | 2nd Deriv. | Pattern | Denominator

**URL params** (all optional; these are the defaults):

| Param | Default | Values |
|---|---|---|
| `set` | *(omitted = 1,2,3,4)* | comma-separated levels 1–7, e.g. `1,2,3,4` |
| `denom` | `6` | 1–6 |
| `method` | `1` | 1–4 |
| `window` | `3` | `3` or `6` only |
| `pattern` | *(omitted = all)* | comma-separated pattern labels |

`deriv` param has been removed. Industry set is checkboxes for levels 1–7 (level 0 excluded). Highlight a specific row on load with `?highlight={series_id}`.

## Deployment
- **Render.com** (live Flask): `https://blsdetrend.onrender.com` — auto-redeploys on push to `main`; free tier sleeps after 15 min idle
- **GitHub Pages** (static): `https://zhguanyu98.github.io/blsdetrend/` — CI runs `freeze.py` on every push to `main`, deploys `website/build/` to `gh-pages` branch
- GitHub repo: `https://github.com/zhguanyu98/blsdetrend`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT` (in `Procfile`)

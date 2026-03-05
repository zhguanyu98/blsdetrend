# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository processes BLS (Bureau of Labor Statistics) Table B-1a data — 842 seasonally-adjusted employment series — and publishes a public website showing employment levels, shares, and detrended measures for each industry.

## Key Commands

### Rebuild denominator mapping (only needed if `b1a_mapping_with_parent.csv` changes)
```bash
cd "/Users/guanyuzhou/Penn Dropbox/Guanyu Zhou/MLP/research/BLS"
python3 build_mapping.py
```
Reads `b1a_mapping_with_parent.csv`, writes `b1a_mapping_with_denominators.csv`.

### Generate website data (run after any pipeline or mapping update)
```bash
cd "/Users/guanyuzhou/Penn Dropbox/Guanyu Zhou/MLP/research/BLS/website"
python3 generate_data.py
```
Reads source CSVs from `../` and writes ~1,685 files into `website/data/`. Takes ~1–2 min.

### Run the website locally
```bash
cd "/Users/guanyuzhou/Penn Dropbox/Guanyu Zhou/MLP/research/BLS/website"
flask run
# → http://127.0.0.1:5000
```

### Deploy (after data or code changes)
```bash
git add website/data/ website/ build_mapping.py b1a_mapping_with_denominators.csv
git commit -m "..."
git push origin main    # Render.com auto-redeploys
```

## Architecture

### Data pipeline

Source CSVs live in the root `BLS/` directory:

| File | Content |
|---|---|
| `b1a_mapping_with_parent.csv` | 842 rows: series_id, industry_name, display_level (0–7), row_order, supersector_code, parent_series_id |
| `b1a_mapping_with_denominators.csv` | Above + 10 columns: `denominator_opt{1-5}` and `denominator_opt{1-5}_name` |
| `b1a_wide_seriesid.csv` | Raw employment (thousands), ~313 months × 842 series, date-indexed |

**`build_mapping.py`** — computes the denominator series_id for each of 5 user-selectable options and writes `b1a_mapping_with_denominators.csv`. Key logic:
- Goods/service classification is by `supersector_code`, NOT by walking the parent chain (parent_series_id fields mostly point directly to Total private, skipping intermediate levels)
- `find_ancestor_at_level(sid, target_lvl)` scans **backwards through row_order** from the current row, returning the first row at `target_lvl`, stopping if it encounters a row with level < target_lvl (out of scope)
- Special cases (Total nonfarm, Total private, Goods/Service-providing, Government) are handled before option-specific logic

Denominator options:
- **Opt 1** (default): Goods/Service if level ≤ 4; else nearest level-4 ancestor (fallback: level-3, then Goods/Service)
- **Opt 2**: Goods/Service if level ≤ 3; else nearest level-3 ancestor
- **Opt 3**: Goods/Service if level ≤ 2; else nearest level-2 ancestor
- **Opt 4**: Always Goods/Service by supersector code
- **Opt 5**: Total private (private industries) or Total government (govt industries)

### Website (`website/`)

**`generate_data.py`** — pre-computes everything; no numpy/pandas at request time:
- Fit window: **Jan 2010 – Feb 2020** for all trend estimation; values NaN before Jan 2010
- `fit_log_linear(log_vals)`: always-linear OLS on log values → returns (trend_log, resid_log)
- `fit_raw_quadratic(raw_vals)`: quadratic OLS with t-test on t² coefficient (p > 0.05 → refit as linear) → returns (trend_raw, resid_pct) where resid_pct = (actual − pred) / pred
- Outputs `data/table_data.json` (842 rows, all 5 denom options each) and `data/{series_id}.json` + `data/{series_id}_export.csv` per industry

**`app.py`** — Flask, 3 routes:
- `/` — loads `table_data.json`, renders main table
- `/<series_id>` — loads `{series_id}.json`, renders 6-chart detail page
- `/download/<series_id>` — serves `{series_id}_export.csv`

**Industry JSON structure** (key fields):
```
dates, march2020, emp_level, trend_ll_level, resid_ll_level   ← option-independent
options: {
  "1": { denom_name, denom_id, emp_share_pct, trend_ls_share_pct,
         resid_ls_share, trend_rs_share_pct, resid_rs_pct },
  "2": ..., "3": ..., "4": ..., "5": ...
}
```

**Charts in `industry.html`** (6 total, 3 rows × 2 columns):
- Row 1 (option-independent): Log-linear level — actual vs. trend; log deviation from trend
- Row 2 (option-dependent): Log-linear share — actual vs. trend; log deviation from trend
- Row 3 (option-dependent): Raw share quadratic/linear — actual vs. trend; % deviation = (actual−trend)/trend

The denominator dropdown (`?denom=1`–`5`) is persisted in the URL via `history.pushState`. Charts 1–2 never re-render; Charts 3–6 use `Plotly.react` on option change. Industry links on the main table carry the `?denom=N` param.

### Deployment
- Hosted on Render.com at `https://blsdetrend.onrender.com`
- GitHub repo: `https://github.com/zhguanyu98/blsdetrend`
- Config: `website/render.yaml` + `website/Procfile` (start: `gunicorn app:app --bind 0.0.0.0:$PORT`)
- Free tier spins down after 15 min idle; first request takes ~30–60s to wake

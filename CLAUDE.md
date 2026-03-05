# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository processes BLS (Bureau of Labor Statistics) Table B-1a data — 842 seasonally-adjusted employment series — and publishes a public website showing employment levels, shares, and detrended measures for each industry.

## Key Commands

### Full pipeline (run in order when source data changes)
```bash
# 1. Pull latest BLS data → b1a_wide_seriesid.csv
jupyter nbconvert --to notebook --execute pull_data.ipynb

# 2. Rebuild denominator mapping (only if b1a_mapping_with_parent.csv changes)
python3 build_mapping.py

# 3. Regenerate website data files
cd website && python3 generate_data.py
```

### Run the website locally
```bash
cd website
flask run   # → http://127.0.0.1:5000
```

### Deploy
```bash
git add website/data/ website/ build_mapping.py b1a_mapping_with_denominators.csv
git commit -m "..."
git push origin main    # Render.com auto-redeploys
```

## Source Files (root directory)

| File | Content | Used by website? |
|---|---|---|
| `b1a_mapping_with_parent.csv` | 842 rows: series_id, industry_name, display_level (0–7), row_order, supersector_code, parent_series_id | via `b1a_mapping_with_denominators.csv` |
| `b1a_mapping_with_denominators.csv` | Above + 10 columns: `denominator_opt{1-5}` and `denominator_opt{1-5}_name` | yes — primary mapping |
| `b1a_wide_seriesid.csv` | Raw employment (thousands), ~313 months × 842 series, date-indexed | yes |
| `b1a_employment_shares.csv` | Pre-computed shares (old pipeline) | no — shares now computed inline in `generate_data.py` |
| `detrended/` | Legacy detrended CSVs (MA60, Hamilton, poly3) | no — detrending now inline in `generate_data.py` |

## Notebooks

| Notebook | Purpose |
|---|---|
| `pull_data.ipynb` | Calls BLS API → writes `b1a_wide_seriesid.csv`. Requires a BLS API key. |
| `create_mapping.ipynb` | Builds `b1a_mapping_with_parent.csv` from BLS dict files in `dict/` |
| `compute_share.ipynb` | Legacy — pre-computed shares before website did it inline |
| `detrend_share.ipynb` | Legacy — old detrending methods (MA60, Hamilton, poly3) |
| `plot.ipynb` | Ad-hoc analysis and matplotlib charts |

## build_mapping.py

Reads `b1a_mapping_with_parent.csv`, computes the denominator series_id for each of 5 options, writes `b1a_mapping_with_denominators.csv`.

**Critical**: `parent_series_id` fields mostly point directly to Total private (level 1), skipping intermediate levels. `find_ancestor_at_level` therefore scans **backwards through row_order** rather than walking the parent chain. It returns the first preceding row at `target_lvl`, stopping early if it hits a row with level < target_lvl (out of scope).

Denominator options:
- **Opt 1** (default): Goods/Service if level ≤ 4; else nearest level-4 ancestor (fallback: level-3, then Goods/Service)
- **Opt 2**: Goods/Service if level ≤ 3; else nearest level-3 ancestor
- **Opt 3**: Goods/Service if level ≤ 2; else nearest level-2 ancestor
- **Opt 4**: Always Goods/Service by supersector code
- **Opt 5**: Total private (private) or Total government (govt)

Special cases applied before option logic: Total nonfarm → itself; Total private → Total nonfarm; Goods/Service-providing → Total private; Govt level 2 → Total nonfarm; Govt level > 2 → Total government.

## Website (`website/`)

See `website/CLAUDE.md` for details. Summary:

**`generate_data.py`** — all computation, no runtime dependencies:
- Fit window: **Jan 2010 – Feb 2020**; values NaN before Jan 2010
- `fit_log_linear`: always-linear OLS on log values → (trend_log, resid_log)
- `fit_raw_linear`: linear OLS on raw share → (trend_raw, resid_pct) where resid_pct = (actual − pred) / pred
- Outputs `data/table_data.json`, `data/{series_id}.json`, `data/{series_id}_export.csv`

**Industry JSON structure:**
```
emp_level, trend_ll_level, resid_ll_level        ← option-independent (log-linear level)
options["1"…"5"]: {
  emp_share_pct, trend_ls_share_pct, resid_ls_share,   ← log-linear share
  trend_rs_share_pct, resid_rs_pct,                    ← raw-linear share
  denom_name, denom_id
}
```

**6 charts per industry page** (3 rows × 2 cols):
- Row 1: Log-linear level — actual vs. trend; log deviation
- Row 2: Log-linear share — actual vs. trend; log deviation  *(option-dependent)*
- Row 3: Raw-linear share — actual vs. trend; % deviation    *(option-dependent)*

## Deployment
- Hosted on Render.com at `https://blsdetrend.onrender.com`
- GitHub: `https://github.com/zhguanyu98/blsdetrend`
- Start command (in `Procfile`): `gunicorn app:app --bind 0.0.0.0:$PORT`
- Free tier sleeps after 15 min idle; first request takes ~30–60s

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two parallel BLS data sites in the same repo, same architecture, separate deployments:

- **Employment** (`employment/`) — BLS Table B-1a, 842 seasonally-adjusted employment series, employment levels/shares/detrended measures
- **Earnings** (`earnings/`) — BLS Table B-3a, ~549 AWP (aggregate weekly payroll, data type 57) series, YoY growth vs benchmark

Both follow the same pattern: a data pipeline writes pre-computed JSON to `website/data/`, Flask serves it with no pandas/numpy at runtime, and Frozen-Flask builds a static site for GitHub Pages.

## Key Commands

### Employment site
```bash
python3 employment/pull_data.py            # snapshot prev vintage + pull BLS data
python3 employment/build_mapping.py                                     # rebuild denominator mapping
cd employment/website && python3 generate_data.py                       # regenerate JSON
cd employment/website && flask run --port 5001
```

### Earnings site
```bash
cd earnings && python pull_data.py          # pull BLS data + CPI-U (~5 min, 22 API batches)
cd earnings/website && python generate_data.py   # regenerate JSON
cd earnings/website && flask run --port 5002
```

### Monthly update (after each BLS Employment Situation release)
```bash
./update_monthly.sh              # both sites: pull → generate_data → freeze, then a summary
./update_monthly.sh employment   # one site only
```
Commits nothing. It pins a Python that actually has Frozen-Flask (bare `python3` on this
machine can resolve to Anaconda 3.9, which does not) and aborts if the pull did not
advance the month — otherwise the revision columns would silently vanish.

### Deploy (both sites)
```bash
git push origin main    # triggers Render.com redeploy + GitHub Actions → GitHub Pages
```

## Source Data Files

### Employment (`employment/`)
| File | Content |
|---|---|
| `b1a_mapping_with_parent.csv` | 842 rows: series_id, industry_name, display_level (0–7), row_order, supersector_code, parent_series_id |
| `b1a_mapping_with_denominators.csv` | Above + `denominator_opt{1-6}` and `denominator_opt{1-6}_name` — primary input to website |
| `b1a_wide_seriesid.csv` | Raw employment (thousands), ~319 months × 842 series, date-indexed (current: through July 2026) |
| `b1a_wide_seriesid_pre_revision_<YYYY_MM>.csv` | Snapshots of `b1a_wide_seriesid.csv` taken before each release, named for the snapshot's own last month — the month still preliminary at that pull (e.g. `..._2026_06` has Total Nonfarm June = 158,984k vs revised 158,881k). `generate_data.py` auto-selects the newest one older than the current data; existing snapshots are never overwritten, so a repeated pull is harmless. |
| `build_mapping.py` | Reads `b1a_mapping_with_parent.csv` → writes `b1a_mapping_with_denominators.csv` |
| `pull_data.py` | Snapshots `b1a_wide_seriesid.csv` → `b1a_wide_seriesid_pre_revision_<its last month>.csv`, then calls BLS API and overwrites `b1a_wide_seriesid.csv` |
| `pull_data.ipynb` | One-cell notebook wrapper: `%run pull_data.py` |

### Earnings (`earnings/`)
| File | Content |
|---|---|
| `b3a_series.csv` | ~549 AWP series metadata from BLS `ce.series` |
| `b3a_mapping.csv` | Above + display_level (from `ce.industry`), benchmark_id, benchmark_name |
| `b3a_wide.csv` | AWP levels ($ thousands), date index × series_id columns, ~245 months (current: through July 2026) |
| `cpiu.csv` | CPI-U index (CUSR0000SA0), date + cpiu columns |
| `pull_data.py` | Single script: fetches `ce.series`, `ce.industry`, all AWP series + CPI-U; writes all four files above |

---

## Employment Website (`employment/website/`)

### Flask routes
- `/` → `index.html` (main table, pre-loads `table_data.json` at startup)
- `/analysis/` → `analysis.html` (quadrant analysis page)
- `/<series_id>/` → `industry.html` (per-industry 8-chart Plotly page)
- `/data/<filename>` → raw JSON/CSV from `data/`
- `/download/<series_id>` → attachment download of `{series_id}_export.csv`
- `/health` → Render.com health check

### `generate_data.py` — computation
- Fit window: **Jan 2010 – Feb 2020**; values NaN before Jan 2010
- `fit_log_linear`: OLS on log values → (trend_log, resid_log)
- `fit_raw_linear`: OLS on raw share → (trend_raw, resid_pct) where resid_pct = (actual − pred) / pred
- `fit_hp_log_share`: `hpfilter(λ=129600)` on log share, extrapolates post-Feb 2020 using last-24-month slope
- Stored lags: `SNAP_LAGS = [1, 3, 6, 9, 12, 15]` months

### `table_data.json` row structure
```
emp_recent, emp_prev, display_level, series_id, industry_name
dev_log_level, dev_log_level_{1,3,6,9,12,15}m, dev_log_level_covid   ← option-independent
{mon}{yyyy}_preliminary, {mon}{yyyy}_revised, revision, revision_pct  ← option-independent
mom, yoy, growth_label, mom_label, yoy_label                          ← option-independent

opts["1"…"6"]: {
  share, share_pct, share_{1,3,6,9,12,15}m, share_covid, denom_name,
  dev_log_share, dev_log_share_{n}m, dev_log_share_covid,
  dev_log_share_hp, dev_log_share_hp_{n}m, dev_log_share_hp_covid,
  dev_raw_share_pct, dev_raw_share_pct_{n}m, dev_raw_share_pct_covid,
}
```
`share_covid` = share at the dynamically identified COVID shock month (max |MoM share change| in Mar–May 2020).

Revision fields: `find_prerevision_snapshot()` picks the newest
`b1a_wide_seriesid_pre_revision_<YYYY_MM>.csv` strictly older than the current data and
diffs its last month against the same month now. `revision = revised − preliminary`;
`revision_pct = revision / preliminary` (null if preliminary is 0 or missing). Row keys
are named for the month (`jun2026_preliminary`); `table_data.json` carries
`revision_month`, `revision_month_label`, `prelim_key`, `revised_key` at the top level,
so **no month is hardcoded anywhere** — the columns roll forward on their own.

M/M and Y/Y: `growth_at()` on the employment level, anchored to *each series' own* last
valid month (only ~175 of 842 report in the newest month, so a June-only series shows
Jun/May and Jun/Jun-25 rather than blank). `mom_label` / `yoy_label` name the two months
compared and become the cell tooltip. Always computed from the current — i.e. revised —
data, never the preliminary snapshot.

### Per-industry JSON (`{series_id}.json`)
```
dates[], march2020, emp_level[], trend_ll_level[], resid_ll_level[]   ← option-independent
options["1"…"6"]: {
  emp_share_pct[], trend_ls_share_pct[], resid_ls_share[],
  trend_rs_share_pct[], resid_rs_pct[],
  hp_cf_share_pct[], resid_hp[],
  denom_name, denom_id
}
```

### Homepage table columns
Static: Industry, Lvl, latest month (000s), prev month (000s), **M/M %**, **Y/Y %**, Share (% of denom), 3M share growth, {revision month} Preliminary (000s), Revision (revised − prelim, color-coded), Revision % (%, color-coded), Denominator, → Analysis.

Sortable: `mom`, `yoy`, `share-growth`, `apr-prelim`, `revision`, `revision-pct` — the `apr-prelim` slug is a historical name and does **not** track the month.
The last column links to the analysis page pre-filtered to that industry.
Removed in May 2026 update: Dev. Log Share (linear) and Dev. Log Share (HP filter) — data still generated in `generate_data.py`, just not displayed on homepage.

### 8 charts per industry page (4 rows × 2 cols)
- Row 1: Log-linear **level** — actual vs. trend; log deviation (option-independent)
- Row 2: Log-linear **share** — actual vs. trend; log deviation
- Row 3: HP filter **share** — actual vs. HP/extrapolated; log deviation
- Row 4: Raw-linear **share** — actual vs. trend; % deviation

Charts 2–4 re-render on denominator change via `Plotly.react`. All charts share a dual-handle x-axis zoom slider. `tickformatstops` switches x-axis from `%Y` to `%b %Y` when zoomed in.

### Analysis page URL params (all optional)
| Param | Default | Values |
|---|---|---|
| `set` | 1,2,3,4 | comma-separated levels 1–7 |
| `denom` | `6` | 1–6 |
| `method` | `1` | 1–4 |
| `window` | `3` | `3` or `6` |
| `pattern` | all | comma-separated pattern labels |
| `highlight` | — | series_id to scroll to |

### `build_mapping.py` — denominator logic
`find_ancestor_at_level` scans **backwards through row_order** (not up the parent chain). Returns the first preceding row at `target_lvl`, stopping if it encounters a row with level < target_lvl.

Denominator options: Opt 1 = nearest level-4 ancestor; Opt 2 = level-3; Opt 3 = level-2; Opt 4 = Goods/Service by supersector; Opt 5 = Total private or Total government; Opt 6 = Total nonfarm.

Special cases (applied first): Total nonfarm → itself; Total private → Total nonfarm; Goods/Service-providing → Total private; Govt level 2 → Total nonfarm; Govt level > 2 → Total government.

---

## Earnings Website (`earnings/website/`)

### Flask routes
- `/` → `index.html` (pre-loads `table_data.json` + `table_data_alt.json` at startup)
- `/analysis/` → `analysis.html` (momentum analysis with 6 pattern categories)
- `/<series_id>/` → `industry.html` (per-industry 4-chart detail page)
- `/data/<filename>` → raw JSON/CSV
- `/download/<series_id>` → attachment download of `{series_id}_export.csv`
- `/health` → Render.com health check

### Benchmark system
Every series has a **default benchmark** (Goods-producing or Private service-providing, assigned in `pull_data.py`) and an **alt benchmark** (always Total Private). The homepage and analysis page carry both datasets (`ROWS_DEFAULT` / `ROWS_ALT`) as inline JS; switching is client-side. Per-industry detail pages load `{series_id}.json` (default) and fetch `{series_id}_alt.json` lazily on first toggle.

**Navbar/title**: The "B-3a" table code was removed from `base.html` in the May 2026 update. The navbar subtitle now reads "Aggregate Weekly Payroll, Private Nonfarm" (no table code prefix); the default page `<title>` is "BLS Aggregate Weekly Payroll".

Benchmark assignment in `pull_data.py`:
- Goods supersectors `{10,15,20,25,30,31,32,35}` → `CES0600000057` (Goods-producing)
- Service supersectors `{40,41,42,43,44,50,55,60,65,70,75,80,85}` → `CES0800000057` (Private service-providing)
- Goods/Service-providing series themselves → `CES0500000057` (Total Private)
- Fallback → Total Private

`display_level` is pulled directly from the BLS `ce.industry` file (joined on `industry_code`) — not derived algorithmically.

**Series title prefix**: BLS titles are formatted as `"Aggregate weekly payrolls of all employees, thousands, <industry>, seasonally adjusted"`. The prefix stripped in `pull_data.py` is `"Aggregate weekly payrolls of all employees, thousands, "` (includes the word "thousands" — omitting it causes every name to start with "Thousands,").

### `generate_data.py` — computation
Core series computed per industry per month:
- `yoy(t)` = `(AWP[t] − AWP[t-12]) / AWP[t-12]`
- `avg_yoy_3m` = mean of yoy(t), yoy(t-1), yoy(t-2)
- `real_yoy` = yoy − cpiu_yoy (BLS methodology)
- `rel_yoy` = yoy − benchmark_yoy
- `avg_rel_yoy_3m` = mean of last 3 rel_yoy values (primary homepage metric)
- `past_rel_yoy` = mean of rel_yoy(t-3)…rel_yoy(t-12) (10 months)
- `second_deriv` = avg_rel_yoy_3m − past_rel_yoy

CPI-U lag: if latest AWP month has no CPI-U yet, substitutes most recent available and logs a warning.

Series with fewer than 15 observations are excluded from `table_data.json`.

### `table_data.json` / `table_data_alt.json` row structure
```json
{
  "series_id", "industry_name", "display_level", "row_order",
  "awp_latest", "awp_prev",
  "avg_yoy_3m", "avg_rel_yoy_3m", "avg_real_yoy_3m",
  "benchmark_id", "benchmark_name",
  "second_deriv", "past_rel_yoy",
  "pattern"
}
```

### Per-industry JSON (`{series_id}.json` / `{series_id}_alt.json`)
```json
{
  "series_id", "benchmark_id", "benchmark_name",
  "dates": ["YYYY-MM", ...],
  "awp_level": [...], "benchmark_level": [...],
  "yoy": [...], "benchmark_yoy": [...],
  "real_yoy": [...], "rel_yoy": [...], "cpiu_yoy": [...]
}
```
All arrays are parallel and date-aligned. `industry_name` is not in the JSON — `app.py` injects it from `_TABLE_DATA` when serving `industry.html`.

### Pattern classification (6 categories, checked in this order)
| Condition | Pattern |
|---|---|
| avg_rel > 0, past_rel ≤ 0 | Reversal Up |
| avg_rel > 0, past_rel > 0, second_deriv ≥ 0 | Accelerating Above |
| avg_rel > 0, past_rel > 0, second_deriv < 0 | Decelerating Above |
| avg_rel < 0, past_rel ≥ 0 | Reversal Down |
| avg_rel < 0, past_rel < 0, second_deriv ≤ 0 | Accelerating Below |
| avg_rel < 0, past_rel < 0, second_deriv > 0 | Decelerating Below |

Pattern colors (used in both buttons and table text): Accelerating Above `#2e7d32` · Decelerating Above `#81c784` · Reversal Up `#00897b` · Accelerating Below `#c62828` · Decelerating Below `#ef9a9a` · Reversal Down `#e65100`.

### Industry detail page (`industry.html`)
4 charts in a 2×2 CSS grid (Plotly), all with linked x-axis zoom via a **dual-handle range slider** (two overlapping `position:absolute` range inputs). Do NOT use `plotly_relayout` event listeners for zoom sync — that causes feedback loops and page freeze. The slider calls `Plotly.relayout` directly on all 4 charts via `applyZoomToAll()`.
1. AWP level ($ thousands) — industry only (no benchmark; benchmark scale is incomparable)
2. Nominal YoY growth (%) — industry vs benchmark
3. Real YoY growth (%) — industry vs CPI-U inflation
4. Relative YoY vs benchmark (%) — with green/red fill above/below zero

### Sorting (all tables)
Click once = ascending (▲), twice = descending (▼), three times = reset to BLS `row_order`. Neutral state shows `⇅`. String columns (industry_name, benchmark_name, pattern) use `localeCompare`; numeric columns use subtraction. Analysis page has independent sort state per panel.

---

## Deployment

### Employment site
- **Render.com**: `https://blsdetrend.onrender.com` — `rootDir: employment/website`
- **GitHub Pages**: `https://zhguanyu98.github.io/blsdetrend/employment/` — built from `employment/website/build/`

### Earnings site
- **Render.com**: separate service — `rootDir: earnings/website`
- **GitHub Pages**: `https://zhguanyu98.github.io/blsdetrend/earnings/` — built from `earnings/website/build/`

GitHub repo: `https://github.com/zhguanyu98/blsdetrend`

Both static builds use Frozen-Flask (`freeze.py` in each `website/`). The GitHub Actions workflow (`.github/workflows/deploy.yml`) runs both build jobs in parallel and deploys to the `gh-pages` branch using `keep_files: true` so neither site overwrites the other.

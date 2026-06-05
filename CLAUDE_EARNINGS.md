# CLAUDE_EARNINGS.md

> **OBSOLETE BUILD SPEC** — This file was the original construction brief used to build the earnings site. The site is now live. For current architecture, data structures, and commands, see `CLAUDE.md` instead.
>
> Key divergences from this spec: the live site uses **Aggregate Weekly Payroll (AWP, data type 57)** not AWE (data type 11); series IDs end in `...057` not `...011`; the benchmark series are `CES0600000057` / `CES0800000057` / `CES0500000057`. Steps 1–9 below describe the original intent; Step 10 captures corrections applied during construction.

This file provides instructions to Claude Code for building the BLS Earnings website (`earnings/`), a new site tracking average weekly earnings (AWE) from BLS Table B-3a. Read this file fully before writing any code.

---

## Project Context

You are building a second website alongside an existing one. The existing project lives in a folder called `employment/` (the B1a employment shares site). You are creating a new folder called `earnings/` at the same level. Both folders sit inside the BLS research directory. **Do not touch anything in `employment/`.**

The new site tracks **average weekly earnings (AWE)** from BLS Table B-3a — seasonally adjusted, all employees, private nonfarm payrolls only. It has three pages: a homepage table, an analysis page, and a per-industry detail page. The architecture mirrors the employment site: a data pipeline writes pre-computed JSON to `earnings/website/data/`, Flask serves it at runtime with no pandas/numpy, and Frozen-Flask builds a static site for GitHub Pages.

---

## Step 1: Repository and Folder Structure

Reorganize the existing repo so both sites coexist cleanly:

```
/
├── employment/          # existing B1a site — move all existing files here
│   ├── website/
│   ├── b1a_mapping_with_parent.csv
│   ├── b1a_mapping_with_denominators.csv
│   ├── b1a_wide_seriesid.csv
│   ├── build_mapping.py
│   └── pull_data.ipynb
├── earnings/            # new B3a site — create from scratch
│   ├── website/
│   │   ├── app.py
│   │   ├── freeze.py
│   │   ├── generate_data.py
│   │   ├── data/
│   │   ├── templates/
│   │   │   ├── index.html
│   │   │   ├── analysis.html
│   │   │   └── industry.html
│   │   └── static/
│   ├── pull_data.py
│   ├── b3a_series.csv
│   ├── b3a_wide.csv
│   └── cpiu.csv
└── README.md
```

Update all deployment configs (Render `Procfile`, GitHub Actions CI) to point to `earnings/website/` for the new site. Keep the employment site's deployment intact.

---

## Step 2: Data Pipeline (`earnings/pull_data.py`)

This is a **single script — no notebooks**. Run it whenever source data needs refreshing. It pulls AWE series, benchmark series, and CPI-U all in one execution.

### 2a. BLS API Key

Retrieve the existing API key from `employment/pull_data.ipynb` and hardcode it into `earnings/pull_data.py`.

### 2b. Series to Pull

**AWE series (data type code 11, seasonally adjusted):**

Pull all CES series where:
- Prefix: `CES`
- Seasonal adjustment: `S` (seasonally adjusted)
- Data type code: `11` (average weekly earnings, all employees)
- Supersector: private only (exclude government supersector codes 90–93)

Use the BLS public metadata file to get the full list of series IDs:
```
https://download.bls.gov/pub/time.series/ce/ce.series
```
Filter for `data_type_code == 11` and `seasonal == S`. This yields roughly 500 series. Save the metadata (series_id, industry_code, series_title, supersector_code, begin_year, end_year) to `earnings/b3a_series.csv`.

**Always pull these benchmark series explicitly (they may already be in the above list, but ensure they are included):**
- `CES0500000011` — Total Private
- `CES0600000011` — Goods-producing
- `CES0800000011` — Private service-providing

**CPI-U series (pull in the same batched API calls):**
- Series ID: `CUSR0000SA0` — CPI-U, all items, seasonally adjusted
- This is from the CU survey. Include it in the BLS API v2 batch requests alongside the CES series.

### 2c. API Call Structure

BLS API v2 allows 50 series per call. Batch all ~500 AWE series plus CPI-U into groups of 50. Pull all available years (start year: 2006, end year: current year). Concatenate results into a wide DataFrame indexed by date in `YYYY-MM` format, one column per series_id. Save to:
- `earnings/b3a_wide.csv` — all AWE series (date index + one column per series_id)
- `earnings/cpiu.csv` — two columns only: `date`, `cpiu`

### 2d. Industry Hierarchy and Mapping

After pulling series metadata, build `earnings/b3a_mapping.csv` with these columns:

| Column | Description |
|---|---|
| `series_id` | e.g. `CES3200000011` |
| `industry_code` | 8-digit code from ce.series |
| `industry_name` | parsed from `series_title` — strip the prefix `"Average weekly earnings of all employees, "` |
| `display_level` | derived from industry_code — see logic below |
| `supersector_code` | first 2 digits of industry_code |
| `row_order` | preserve BLS table order from ce.series file |
| `benchmark_id` | assigned by the benchmark logic below |
| `benchmark_name` | human-readable name of the assigned benchmark |

**Display level derivation:**
Count the number of significant (non-zero trailing) digits in the industry_code beyond the supersector prefix. Aggregate industries (all zeros after supersector) are level 1; each additional level of detail adds 1. Total nonfarm (`00000000`) is level 0.

**Benchmark assignment logic:**

| Condition | Assigned benchmark |
|---|---|
| Series is Total Private itself | Total Private (self — omit from relative calc or treat as zero) |
| Series is Goods-producing or Private service-providing | `CES0500000011` (Total Private) |
| Supersector code in `{10, 15, 20, 25, 30, 35, 40}` (goods-producing supersectors) | `CES0600000011` (Goods-producing) |
| Supersector code in `{41, 42, 43, 44, 50, 55, 60, 65, 70, 75, 80, 85}` (service-providing supersectors) | `CES0800000011` (Private service-providing) |
| No clear match | `CES0500000011` (Total Private, fallback) |

---

## Step 3: Data Generation (`earnings/website/generate_data.py`)

This script reads `b3a_wide.csv`, `b3a_mapping.csv`, and `cpiu.csv` and writes pre-computed JSON to `earnings/website/data/`. **No numpy/pandas/scipy may be imported in `app.py` at Flask runtime** — all computation happens here at generation time.

### 3a. Core Computed Quantities

For each series at each month t, compute the following. Use NaN where data is insufficient (first 12 months cannot have YoY; first 14 months cannot have 3-month avg YoY).

**Nominal YoY growth rate:**
```
yoy(t) = (AWE[t] - AWE[t-12]) / AWE[t-12]
```

**Average nominal YoY over past 3 months:**
```
avg_yoy_3m(t) = mean( yoy(t), yoy(t-1), yoy(t-2) )
```

**CPI-U YoY:**
```
cpiu_yoy(t) = (CPI[t] - CPI[t-12]) / CPI[t-12]
```

**Average CPI-U YoY over past 3 months:**
```
avg_cpiu_yoy_3m(t) = mean( cpiu_yoy(t), cpiu_yoy(t-1), cpiu_yoy(t-2) )
```

**Real YoY (approximate, matching BLS methodology):**
```
real_yoy(t) = yoy(t) - cpiu_yoy(t)
```

**Average real YoY over past 3 months:**
```
avg_real_yoy_3m(t) = mean( real_yoy(t), real_yoy(t-1), real_yoy(t-2) )
```

**Relative YoY vs benchmark:**
```
rel_yoy(t) = yoy(t) - benchmark_yoy(t)
```

**Average relative YoY over past 3 months (primary homepage metric):**
```
avg_rel_yoy_3m(t) = mean( rel_yoy(t), rel_yoy(t-1), rel_yoy(t-2) )
```

**Past relative YoY — average over months t-3 through t-12 inclusive (10 months):**
```
past_rel_yoy(t) = mean( rel_yoy(t-3), rel_yoy(t-4), ..., rel_yoy(t-12) )
```

**Second derivative (momentum):**
```
second_deriv(t) = avg_rel_yoy_3m(t) - past_rel_yoy(t)
```

### 3b. CPI-U Lag Handling

When the most recent AWE month has no corresponding CPI-U observation (CPI is released ~10 days after CES), use the most recent available CPI-U month as a substitute. Log a warning message to the console identifying which month was substituted. Do not crash or silently produce wrong values.

### 3c. Pattern Classification

Classify each industry at the latest available month based on the signs of `avg_rel_yoy_3m`, `past_rel_yoy`, and `second_deriv`. Apply Reversal checks first before Accelerating/Decelerating.

**Above benchmark group** (`avg_rel_yoy_3m > 0`):

| Condition | Pattern label |
|---|---|
| `avg_rel_yoy_3m > 0` AND `past_rel_yoy <= 0` | `"Reversal Up"` |
| `avg_rel_yoy_3m > 0` AND `past_rel_yoy > 0` AND `second_deriv >= 0` | `"Accelerating Above"` |
| `avg_rel_yoy_3m > 0` AND `past_rel_yoy > 0` AND `second_deriv < 0` | `"Decelerating Above"` |

**Below benchmark group** (`avg_rel_yoy_3m < 0`):

| Condition | Pattern label |
|---|---|
| `avg_rel_yoy_3m < 0` AND `past_rel_yoy >= 0` | `"Reversal Down"` |
| `avg_rel_yoy_3m < 0` AND `past_rel_yoy < 0` AND `second_deriv <= 0` | `"Accelerating Below"` |
| `avg_rel_yoy_3m < 0` AND `past_rel_yoy < 0` AND `second_deriv > 0` | `"Decelerating Below"` |

Industries where `avg_rel_yoy_3m == 0` exactly: assign to above group with pattern `"Decelerating Above"`.

### 3d. Output Files

**`data/table_data.json`** — array of row objects for the homepage, one per industry, using the default benchmark (Goods-producing or Private service-providing):

```json
[
  {
    "series_id": "CES3200000011",
    "industry_name": "Manufacturing",
    "display_level": 1,
    "row_order": 42,
    "awe_latest": 1443.20,
    "awe_prev": 1398.69,
    "avg_yoy_3m": 0.0321,
    "avg_rel_yoy_3m": 0.0082,
    "avg_real_yoy_3m": 0.0121,
    "benchmark_id": "CES0600000011",
    "benchmark_name": "Goods-producing",
    "second_deriv": 0.0031,
    "past_rel_yoy": 0.0051,
    "pattern": "Accelerating Above"
  }
]
```

**`data/table_data_alt.json`** — identical structure but every industry's benchmark is overridden to Total Private (`CES0500000011`). Recompute all relative/pattern fields accordingly.

**`data/{series_id}.json`** — per-industry time series for the detail page, using default benchmark:

```json
{
  "series_id": "CES3200000011",
  "industry_name": "Manufacturing",
  "benchmark_id": "CES0600000011",
  "benchmark_name": "Goods-producing",
  "dates": ["2006-03", "2006-04", "..."],
  "awe_level": [1100.0, 1102.5, "..."],
  "benchmark_level": [1200.0, 1205.0, "..."],
  "yoy": [null, null, "...", 0.0321, "..."],
  "benchmark_yoy": [null, null, "...", 0.0239, "..."],
  "real_yoy": [null, null, "...", 0.0121, "..."],
  "rel_yoy": [null, null, "...", 0.0082, "..."],
  "cpiu_yoy": [null, null, "...", 0.0200, "..."]
}
```

All arrays are parallel and date-aligned. Use `null` (not 0) for months where values cannot be computed.

**`data/{series_id}_alt.json`** — same structure with Total Private as benchmark.

**`data/{series_id}_export.csv`** — flat CSV for download containing: date, awe_level, benchmark_level, yoy, benchmark_yoy, real_yoy, rel_yoy, cpiu_yoy.

**Exclusion rule:** if a series has fewer than 15 months of data total, exclude it from `table_data.json` and do not generate a detail page for it.

---

## Step 4: Flask App (`earnings/website/app.py`)

### Routes

| Route | Template | Notes |
|---|---|---|
| `/` | `index.html` | Pre-loads `table_data.json` and `table_data_alt.json` at startup |
| `/analysis/` | `analysis.html` | |
| `/<series_id>/` | `industry.html` | |
| `/data/<filename>` | — | Serves raw JSON from `data/` directory |
| `/download/<series_id>` | — | Serves `{series_id}_export.csv` as attachment |
| `/health` | — | Returns 200 OK for Render.com health check |

### Design Constraints
- **No pandas, numpy, or scipy at runtime.** All computation is done in `generate_data.py`.
- Pre-load both `table_data.json` and `table_data_alt.json` into memory at startup (not per-request) to pass Render's health check.
- Local development port: **5002** (5000 is macOS AirPlay, 5001 is the employment site).

---

## Step 5: Homepage (`earnings/website/templates/index.html`)

### Table Columns

Each column has a **main header** and a **sub-header** rendered in smaller, lighter font directly below within the same `<th>` cell.

| Main header | Sub-header | Format | Sortable |
|---|---|---|---|
| Industry | | Text, indented by display_level × 12px | No |
| Lvl | | Integer | No |
| Avg Weekly Earnings | (latest month, $) | `$1,443.20` | No |
| Nominal YoY Growth | (avg past 3 months) | `3.21%` | Yes |
| Relative YoY Growth | (vs benchmark, avg past 3 months) | `0.82%` | Yes |
| Benchmark | | Text | No |
| Real YoY Growth | (CPI-U adjusted, avg past 3 months) | `1.21%` | Yes |
| → Detail | | Link to `/{series_id}/` | No |

### Benchmark Switcher

A toggle control at the top of the page, above the table:

```
Benchmark:  [ Goods / Services ]  [ Total Private ]
```

The active selection is visually highlighted. Clicking switches the entire table between `table_data.json` (default) and `table_data_alt.json` without a page reload. All relative and benchmark columns update instantly via JavaScript.

### Color Coding (all percentage columns)
- Positive: green (`#2e7d32`)
- Negative: red (`#c62828`)
- Absolute value < 0.001: neutral gray

### Level Filter

Checkboxes above the table for display levels 1–7. Default: levels 1–4 checked. Level 0 row (Total Private aggregate, if present) always shown as a non-filterable header row. Filtering is client-side JavaScript — no page reload.

### Sorting

Clicking a sortable column header sorts ascending; clicking again sorts descending. An arrow indicator shows current sort direction. Default order is BLS row order (`row_order` field).

### Missing Values

If a field is `null` for a row, display `—` (em dash). Do not display `0`, `NaN`, or blank.

---

## Step 6: Analysis Page (`earnings/website/templates/analysis.html`)

### Page Controls (above both panels)

- **Benchmark switcher**: same Goods/Services vs Total Private toggle as homepage
- **Level filter**: checkboxes for levels 1–7, default 1–4 checked
- **URL params** (all optional):
  - `denom`: `default` or `alt`
  - `set`: comma-separated levels, e.g. `1,2,3,4`
  - `pattern`: comma-separated pattern labels to pre-filter
  - `highlight`: series_id to highlight on load

### Two Panels

**Panel 1 — Outperforming Benchmark**
Rows where `avg_rel_yoy_3m > 0`. Panel header shows count of industries in this panel.

**Panel 2 — Underperforming Benchmark**
Rows where `avg_rel_yoy_3m < 0`. Panel header shows count of industries in this panel.

Each panel has its own pattern filter pills:

```
[ All ]  [ Accelerating ]  [ Decelerating ]  [ Reversal ]
```

Clicking a pill filters rows within that panel only. "Accelerating" matches both "Accelerating Above" and "Accelerating Below"; "Decelerating" matches both Decelerating variants; "Reversal" matches both Reversal variants.

### Table Columns (same in both panels)

| Main header | Sub-header | Format | Notes |
|---|---|---|---|
| Industry | | Text, indented | |
| Lvl | | Integer | |
| Relative YoY Growth | (avg past 3 months) | Percent, 2dp | Color-coded |
| Past Relative YoY | (avg months t-12 to t-3) | Percent, 2dp | Color-coded |
| 2nd Derivative | (recent minus past) | Percent, 2dp | Color-coded |
| Pattern | | Full label string | e.g. "Accelerating Above" |
| Benchmark | | Text | |
| → Detail | | Link | |

---

## Step 7: Detail Page (`earnings/website/templates/industry.html`)

### Header Section

Display: industry name, benchmark name, latest AWE ($), latest nominal YoY (%), latest real YoY (%). Include the benchmark switcher toggle here as well.

### Benchmark Switcher

Same Goods/Services vs Total Private toggle. Switching reloads chart data from `{series_id}.json` vs `{series_id}_alt.json` and re-renders all 4 plots via `Plotly.react` without a full page reload.

### 4 Plots — Stacked Vertically (Plotly)

All 4 plots share a **linked x-axis range slider** at the bottom of the page. Zooming any one plot zooms all four simultaneously. Use `tickformatstops` to show `%Y` when zoomed out and `%b %Y` when zoomed in below 12-month tick spacing.

---

**Plot 1 — Raw Average Weekly Earnings**
- Y-axis: dollar level, formatted as `$X,XXX`
- Line 1: industry AWE — solid, primary color
- Line 2: benchmark AWE — dashed, secondary color, labeled with benchmark name
- Title: `"Average Weekly Earnings ($)"`

---

**Plot 2 — Nominal Year-over-Year Growth**
- Y-axis: percent, formatted as `X.XX%`
- Line 1: industry nominal YoY — solid, primary color
- Line 2: benchmark nominal YoY — dashed, secondary color
- Zero reference line: gray, dashed, no legend entry
- Title: `"Nominal YoY Growth (%)"`

---

**Plot 3 — Real Year-over-Year Growth**
- Y-axis: percent, formatted as `X.XX%`
- Line 1: industry real YoY — solid, primary color
- Line 2: CPI-U YoY — dashed, orange or muted color, labeled `"CPI-U Inflation"`
- Zero reference line: gray, dashed
- Note in subtitle or annotation: `"Real YoY = Nominal YoY − CPI-U YoY (BLS methodology)"`
- Title: `"Real YoY Growth, CPI-U Adjusted (%)"`

---

**Plot 4 — Relative YoY Growth vs Benchmark**
- Y-axis: percent, formatted as `X.XX%`
- Line 1: industry relative YoY — solid, primary color
- Zero reference line: bold gray, labeled with benchmark name (e.g. `"Goods-producing = 0"`)
- Shaded area between line and zero: green fill when positive, red fill when negative (low opacity, ~0.15)
- No second data line — the zero line IS the benchmark by definition
- Title: `"Relative YoY Growth vs Benchmark (%)"`

---

### Download Link

Place below the plots: `"Download data as CSV"` → `/download/{series_id}`

---

## Step 8: Static Build and Deployment

### `earnings/website/freeze.py`

Mirror the pattern from `employment/website/freeze.py`. Iterates over all series IDs from `table_data.json` to generate one HTML file per industry. Output goes to `earnings/website/build/`.

### Render.com

Create a new Render service pointing to `earnings/website/`. Start command:
```
gunicorn app:app --bind 0.0.0.0:$PORT
```
Add to `Procfile` at repo root (alongside existing employment entry if applicable).

### GitHub Actions CI

Add a second job to the existing workflow that:
1. Runs `cd earnings/website && python generate_data.py`
2. Runs `python freeze.py`
3. Deploys `earnings/website/build/` to the `gh-pages` branch under the subdirectory `earnings/`

The static site will be accessible at:
```
https://zhguanyu98.github.io/blsdetrend/earnings/
```

---

## Step 9: Key Implementation Notes

### CPI-U Lag
When computing real YoY for the most recent AWE month, CPI-U for that same month may not yet be published (CPI releases ~10 days after CES). If the latest CPI-U observation is one month behind the latest AWE observation, use the most recent available CPI-U value as a substitute for the missing month. Log a warning:
```
WARNING: CPI-U for {YYYY-MM} not available. Using {YYYY-MM} as substitute.
```

### Missing and Short Series
- Treat missing individual monthly observations as NaN — do not forward-fill or interpolate.
- Exclude any series with fewer than 15 total monthly observations from `table_data.json`.
- For excluded series, still generate `{series_id}.json` and the detail page if the series appears in `b3a_mapping.csv`, so direct URL access works.

### COVID Composition Spike
AWE spiked sharply in April–May 2020 due to composition effects (lower-wage workers disproportionately lost jobs, mechanically lifting the average). This is real and expected, not a data error. Do not attempt to adjust for it. Ensure chart y-axis scales handle the spike without making the rest of the series unreadable — consider using `autorange` with outlier handling or a note on the chart.

### Missing Values in Frontend
Display `—` (HTML entity `&mdash;`) wherever a field is `null` in the JSON. Never display `0`, `NaN`, `undefined`, or blank string for a missing value.

### Runtime Constraint
`app.py` must not import pandas, numpy, or scipy. All JSON is either pre-loaded at startup or fetched from `data/` per request. This is required for the Render free tier and for the Frozen-Flask static build to work correctly.

### Local Development
```bash
cd earnings/website
flask run --port 5002
# → http://127.0.0.1:5002
```

### Full Pipeline (run in order when source data changes)
```bash
# 1. Pull latest BLS data and CPI-U
cd earnings
python pull_data.py

# 2. Regenerate all website data files
cd website
python generate_data.py

# 3. Run locally to verify
flask run --port 5002

# 4. Deploy (triggers Render redeploy + GitHub Pages CI)
git add .
git commit -m "refresh earnings data"
git push origin main
```

---

## Step 10: Corrections and Additional Requirements

Apply all of the following on top of everything specified above. Where these instructions conflict with earlier ones, these take precedence.

### 10a. Display Level — Pull Directly from BLS

In `pull_data.py`, **delete the `derive_display_level()` function entirely**. Do not compute display level algorithmically from industry_code digits — it is unreliable and will produce wrong levels for many industries.

Instead, download the BLS industry hierarchy file:
```
https://download.bls.gov/pub/time.series/ce/ce.industry
```
This is a tab-separated file. Parse it the same way as `ce.series` (strip all whitespace from column names and values). It contains a `display_level` column and an `industry_code` column. In `fetch_ce_series_metadata()`, after downloading `ce.series`, make a second GET request to `ce.industry`, parse it into a DataFrame, and left-join it onto the metadata DataFrame on `industry_code`. Use the `display_level` column from `ce.industry` directly — do not derive or override it.

Before writing the join code, print the actual column names of `ce.industry` to confirm the exact field names, since they may differ slightly from `display_level` or `industry_code`. Log a warning for any series whose `industry_code` does not appear in `ce.industry` and set their `display_level` to `None`.

### 10b. Industry Names — Capitalized, No "Seasonally Adjusted" Suffix

When parsing `industry_name` from the `series_title` field in `ce.series`:

1. Strip the prefix `"Average weekly earnings of all employees, "` as currently done.
2. Also strip any trailing phrase matching `", seasonally adjusted"` (case-insensitive).
3. Apply Python's `.title()` method to capitalize the result, then manually fix known all-caps abbreviations that `.title()` handles poorly (e.g. `"And"` → `"and"`, `"Of"` → `"of"`, `"For"` → `"for"`— standard English title case rules for prepositions and conjunctions).

The final `industry_name` stored in all CSVs and JSONs must be properly title-cased with no "seasonally adjusted" anywhere.

### 10c. "→ Detail" Column Links to Analysis Page, Not Industry Detail Page

On the **homepage table**, the last column currently labeled `→ Detail` must link to the **analysis page pre-filtered to that industry**, not to a per-industry graph page. The link format is:

```
/analysis/?highlight={series_id}&denom=default&set=1,2,3,4,5,6,7
```

The `set=1,2,3,4,5,6,7` ensures the highlighted row is visible regardless of level. Rename the column header from `→ Detail` to `→ Analysis`.

There is no separate per-industry graph page (`industry.html`). Remove that route from `app.py` and remove `industry.html` from the templates. The 4 charts live on the analysis page, rendered inline when a row is highlighted or selected.

### 10d. Charts are 2×2 Grid, Not 4 Stacked Vertically

Ignore the earlier instruction specifying 4 vertically stacked plots. Charts must be displayed in a **2×2 grid layout**:

```
[ Plot 1: Raw AWE Level    ]  [ Plot 2: Nominal YoY Growth  ]
[ Plot 3: Real YoY Growth  ]  [ Plot 4: Relative YoY Growth ]
```

Each plot occupies 50% of the page width. Use Plotly with `make_subplots(rows=2, cols=2)` or render four independent Plotly divs in a CSS grid. All four plots share a linked x-axis (same date range) but do NOT need a shared range slider — just ensure that when you zoom one plot's x-axis, all four update together. Implement this by listening to Plotly's `plotly_relayout` event on each chart and calling `Plotly.relayout` on the other three.

### 10e. Linked Zoom — Implement Correctly

The existing employment site's linked zoom via a range slider does not work reliably. Do not copy that approach. Instead implement x-axis linking as follows:

```javascript
const plots = ['plot1', 'plot2', 'plot3', 'plot4'];
let isSyncing = false;

plots.forEach(id => {
    document.getElementById(id).on('plotly_relayout', function(eventData) {
        if (isSyncing) return;
        isSyncing = true;
        const update = {};
        if (eventData['xaxis.range[0]']) {
            update['xaxis.range[0]'] = eventData['xaxis.range[0]'];
            update['xaxis.range[1]'] = eventData['xaxis.range[1]'];
        } else if (eventData['xaxis.autorange']) {
            update['xaxis.autorange'] = true;
        }
        plots.filter(p => p !== id).forEach(otherId => {
            Plotly.relayout(document.getElementById(otherId), update);
        });
        isSyncing = false;
    });
});
```

Use the div element IDs directly (not string names) when attaching listeners. Test that double-clicking to reset zoom on one chart resets all four.

### 10f. Analysis Page — 6 Color-Coded Pattern Filter Buttons

Replace the pill filter buttons described earlier with **6 individual buttons**, one per pattern, color-coded as follows:

| Button label | Color |
|---|---|
| Accelerating Above | Green (`#2e7d32`) |
| Decelerating Above | Light green (`#81c784`) |
| Reversal Up | Teal (`#00897b`) |
| Accelerating Below | Red (`#c62828`) |
| Decelerating Below | Light red / salmon (`#ef9a9a`) |
| Reversal Down | Orange (`#e65100`) |

Buttons appear above both panels (not per-panel). Each button toggles on/off independently. When a button is off, rows with that pattern are hidden in both panels. All 6 are on by default. An "All" and "None" shortcut button should appear alongside them for convenience.

Active (on) state: filled background in the pattern color, white text. Inactive (off) state: white background, colored border and text.

### 10g. All Columns Sortable in Both Directions

Every column in every table (homepage, both analysis panels) must be sortable. Clicking a column header once sorts ascending; clicking again sorts descending. An arrow indicator (`↑` or `↓`) appears in the header to show current sort direction and column. Clicking a third time returns to the default BLS row order.

This applies to **all** columns including Industry name, Lvl, Avg Weekly Earnings, Nominal YoY Growth, Relative YoY Growth, Benchmark, Real YoY Growth, Relative YoY (past), 2nd Derivative, and Pattern. The Industry column sorts alphabetically by name when sorted; the Lvl column sorts numerically.

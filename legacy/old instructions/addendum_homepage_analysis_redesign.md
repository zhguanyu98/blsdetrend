# Addendum: Homepage and Analysis Page Redesign
## Instructions for Claude Code

This file covers changes to the homepage table, the analysis page columns and
second derivative definition, industry set selection, and x-axis zoom on charts.

---

## Part 1: Homepage Table Changes

### Columns to keep and add (in order)

| Column | Description |
|---|---|
| Industry | Industry name, linked to industry detail page |
| Display level | Numeric display level |
| Employment (month T) | Most recent month raw employment (thousands) |
| Employment (month T-1) | Previous month raw employment (thousands) |
| Share | Employment share under selected denominator, most recent month, as % |
| 3M share growth (new) | `(share[t] - share[t-3]) / share[t-3]`, displayed as %, 2 decimal places |
| Dev. log share (linear) | Log deviation from linear trend on log share, 3 decimal places |
| Dev. log share (HP) | Log deviation from HP filter on log share, 3 decimal places |
| Denominator | Name of denominator industry |
| → Analysis (new) | Button linking to the analysis page filtered to that industry's row |

### Columns to remove
- Share percentile
- All other deviation columns not listed above

### "→ Analysis" button
Each row has a small button (e.g. an arrow icon or "→") in the last column.
Clicking it navigates to `/analysis/` with the industry pre-highlighted or
scrolled into view. Pass the series_id as a query param: `/analysis/?highlight=CES...`
On the analysis page, scroll to and visually highlight the matching row on load.

### Denominator selector
Keep the existing denominator dropdown at the top of the homepage. It controls
which denominator is used for the Share, 3M share growth, and Dev. columns.

---

## Part 2: Analysis Page — Column Changes

### Columns to keep and add (in order)

| Column | Description |
|---|---|
| Industry | Name, linked to detail page |
| Display level | Numeric |
| Employment | Most recent month's raw employment (thousands). Display as-is — may be NaN for some industries; show NaN or a dash, do not search backwards for a non-NaN value |
| Share | Share under selected denominator, most recent month, as % |
| N-month share growth | `(share[t] - share[t-N]) / share[t-N]` as %, 2 decimal places. Label changes with window: "3M share growth" or "6M share growth" |
| Past growth rate | See definition below. Label: "Avg past 3M growth" (3M window) or "Past 6M growth" (6M window) |
| 2nd Deriv. | `recent_pct_change - past_pct_change`, 2 decimal places |
| Pattern | Badge label from 6-pattern classification (see below) |
| Denominator | Denominator industry name |

### Columns to remove
- Deviation (log diff) column
- "Change from COVID shock" / "Δ Share since COVID shock" column
- "Recent 3M Δ" and "Past 3M Δ" raw columns (replaced by growth rate versions above)

---

## Part 3: Second Derivative — Updated Definition

### Window option: 3 months (default)

```
recent_pct  = (share[t] - share[t-3]) / share[t-3]
past_pct    = (share[t-3] - share[t-12]) / share[t-12] / 3
second_deriv = recent_pct - past_pct
```

`past_pct` is the average 3-month growth rate over the t-12 to t-3 window.
Dividing the total 9-month growth by 3 normalizes it to a per-3-month rate,
making it directly comparable to `recent_pct`.

Column labels:
- "3M share growth" — shows `recent_pct`
- "Avg past 3M growth" — shows `past_pct`
- "2nd Deriv." — shows `second_deriv`

### Window option: 6 months

```
recent_pct  = (share[t] - share[t-6]) / share[t-6]
past_pct    = (share[t-6] - share[t-12]) / share[t-12]
second_deriv = recent_pct - past_pct
```

`past_pct` is the straight 6-month percentage change from t-12 to t-6.
No averaging. Both are symmetric 6-month windows at adjacent periods.

Column labels:
- "6M share growth" — shows `recent_pct`
- "Past 6M growth" — shows `past_pct`
- "2nd Deriv." — shows `second_deriv`

### Remove the A/B sub-option toggle
There is no longer a choice between "vs. 12 months ago" and "vs. 6 months ago"
as separate sub-options. The past reference period is now fully determined by
the window choice (3M or 6M). Remove the `deriv` URL parameter.

### Display
All three values displayed as percentage, 2 decimal places (multiply by 100).
Color code "2nd Deriv." green if > 0, red if < 0.

---

## Part 4: Direction Definition (Increasing / Decreasing)

Keep window options 3M and 6M only (remove 1M and 12M options).

```
direction = (share[t] - share[t-N]) / share[t-N]
```

This is the same as `recent_pct` above — the direction column and the
"N-month share growth" column are identical values. Do not display them twice;
the "N-month share growth" column serves as both the direction indicator and
the reported value. The increasing/decreasing category is determined by its sign.

---

## Part 5: Six Pattern Classification

Based on signs of `recent_pct` (R), `past_pct` (P), and `second_deriv` (D = R - P):

| Pattern label | Condition | Badge color |
|---|---|---|
| "Rising, faster" | R > 0, P > 0, D > 0 | Blue |
| "Rising, slower" | R > 0, P > 0, D < 0 | Light blue |
| "Falling, slower" | R < 0, P < 0, D > 0 | Light orange |
| "Falling, faster" | R < 0, P < 0, D < 0 | Red |
| "Reversal up" | R > 0, P < 0 | Green |
| "Reversal down" | R < 0, P > 0 | Dark red |

Pattern filter (multi-select toggle buttons) remains as before, allowing users
to filter rows to one or more patterns across all four quadrant sections.

---

## Part 6: Industry Set Selection — Checkboxes

Replace the current "Industry Set" dropdown with **checkboxes for display levels 1–7**.
Each checkbox is labeled "Level N". Users can check any combination of levels.

- Default: levels 1–4 checked, levels 5–7 unchecked
- At least one level must remain checked at all times; if a user unchecks the last
  one, re-check it automatically or show a validation message
- Update the URL param: `set` becomes a comma-separated list of active levels,
  e.g. `?set=1,2,3,4`. Default (omitted) = levels 1–4

---

## Part 7: X-Axis Zoom Slider on Industry Charts

Add a date range slider below each chart on the industry detail page, allowing
users to zoom into a specific time window.

### Implementation

Use a dual-handle range slider (two handles: start date and end date) positioned
below each chart. On slide, update the chart's x-axis `min` and `max` values:

```javascript
chart.options.scales.x.min = sliderStartValue;
chart.options.scales.x.max = sliderEndValue;
chart.update();
```

- The slider spans the full date range of the series (earliest to most recent month)
- Default position: full range shown (no zoom)
- Add a "Reset zoom" button next to the slider that restores the full range
- All 8 (or 10) charts on the page share the same slider — moving it zooms all
  charts simultaneously so they remain in sync
- Display the currently selected date range as text next to the slider,
  e.g. "Jan 2015 – Jan 2026"

### Library
Do not add `chartjs-plugin-zoom`. Use a plain HTML `<input type="range">` dual
slider (two overlapping range inputs styled with CSS) and update chart x-axis
bounds directly via Chart.js options. No additional npm dependencies needed.

### Vertical reference line
The existing vertical reference line at March 2020 should remain visible at all
zoom levels.

---

## Part 8: Analysis Page Default Control Values

When the analysis page loads without any URL query parameters, apply these defaults:

| Control | Default |
|---|---|
| Denominator | Option 6 — "Total nonfarm" |
| Detrending method (above/below classification) | Option 1 — "Log-linear trend (share)" |
| Window | 3 months |
| Display levels | Levels 1–4 checked |
| Pattern filter | All patterns shown (no filter active) |

These defaults must not be changed by the homepage/analysis redesign work.

---

## Part 9: Updated URL Parameters

| Param | Values | Default | Description |
|---|---|---|---|
| `set` | comma-separated levels e.g. 1,2,3,4 | 1,2,3,4 | Active display levels (checkboxes) |
| `denom` | 1–6 | 6 | Denominator option |
| `method` | 1–4 | 1 | Detrending method (for above/below classification) |
| `window` | 3,6 | 3 | Look-back window: 3 months or 6 months |
| `pattern` | comma-separated pattern labels | (omitted = all) | Active pattern filter |

Remove `deriv` param (no longer needed). Remove `window=1` and `window=12` as
valid values (only 3 and 6 remain).

---

## Edge Cases

| Situation | Action |
|---|---|
| share[t-3] or share[t-6] is zero or NaN | Show NaN for growth rate and all derived columns |
| share[t-12] is NaN (series too short for 3M past_pct) | Show NaN for past_pct and second_deriv |
| All checkboxes produce empty result set | Show message: "No industries at selected display levels" |
| Zoom slider dragged so start > end | Clamp: prevent handles from crossing |
| highlight param on analysis page matches no industry | Ignore silently |

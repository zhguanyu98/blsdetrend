# Analysis Page: Industry Categorization
## Instructions for Claude Code

---

## Overview

Add a new "Analysis" tab/page to the website. The purpose is to categorize industries
into four quadrants based on where they stand relative to trend (above/below) and
whether that gap is growing or shrinking (increasing/decreasing). The page mirrors
the display structure of the main table but shows only the filtered, categorized subset.

---

## Part 1: User Controls (top of page)

Four controls displayed horizontally at the top, in this order:

### 1. Industry Set
Dropdown. Controls which industries are included in the analysis.

| Option | Description |
|---|---|
| All industries | Every industry in the main table |
| Display level ≤ 4 | Industries at display levels 1, 2, 3, 4 only |
| Display level ≤ 3 | Industries at display levels 1, 2, 3 only |
| Display level ≤ 2 | Industries at display levels 1, 2 only |

Default: Display level ≤ 4.

### 2. Denominator
Dropdown. Controls how employment share is computed. Use the same 5 options defined
in the denominator selection instructions, with the same labels.

Default: Option 4 — "Goods/Service-providing total"

### 3. Detrending Method
Dropdown. Controls which deviation metric is used for categorization and display.

| Option | Label | Series used |
|---|---|---|
| 1 (default) | "Log-linear trend (share)" | Residual from linear trend fit on log share |
| 2 | "HP filter (share)" | Residual from HP filter on log share |
| 3 | "Log-linear trend (employment level)" | Residual from linear trend fit on log employment level |
| 4 | "Linear trend (share, %)" | Deviation from linear trend fit on raw share, reported in percentage points |

Default: Option 1 — log-linear trend on share.

For options 1, 2, and 3 the deviation is a log difference. For option 4 it is in
percentage points. Make sure axis labels and table column headers reflect the units
clearly for whichever method is selected.

### 4. Persist selection in URL
Encode all control values as query parameters so that the URL is shareable and the
back button works. Example: `?set=4&denom=4&method=1`

---

## Part 2: Categorization Logic

Given the selected detrending method and denominator, compute the following for
each industry in the selected set:

### Deviation (current)
The most recent non-NaN value of the deviation series under the selected method.

```
deviation = detrended[most recent month]
```

### Direction (6-month change)
The change in deviation over the past 6 months:

```
direction = deviation[most recent month] - deviation[most recent month - 6]
```

If the deviation 6 months ago is NaN (e.g., series too short), mark direction as NaN
and exclude from categorization.

### Four categories

| Category | Condition |
|---|---|
| Above trend, increasing | deviation > 0 AND direction > 0 |
| Above trend, decreasing | deviation > 0 AND direction < 0 |
| Below trend, increasing | deviation < 0 AND direction > 0 |
| Below trend, decreasing | deviation < 0 AND direction < 0 |

Industries where deviation = 0 or direction = 0 exactly: assign to the "decreasing"
variant (treat zero direction as not improving).

Industries with NaN deviation or NaN direction: show in a separate "Insufficient data"
section at the bottom, outside the four quadrants.

---

## Part 3: Page Layout

Display the four categories as four collapsible sections, in this order:
1. Above trend, increasing
2. Above trend, decreasing
3. Below trend, increasing
4. Below trend, decreasing

Each section has:
- A header showing the category name and count, e.g. "Above trend, increasing (34)"
- Expanded by default
- A sortable table of industries (see columns and sorting below)

---

## Part 4: Table Columns

Each section's table has the following columns:

| Column | Description |
|---|---|
| Industry | Industry name, linked to its detail page (pass current `denom` and `method` as query params) |
| Display level | Numeric display level from the mapping |
| Employment (most recent) | Most recent non-NaN raw employment level (thousands). Label the month in the column header, e.g. "Employment (Jan 2026)" |
| Employment share | Share under the selected denominator, most recent month, as a percentage (e.g. 4.23%) |
| Deviation | Deviation under the selected method, most recent month, 3 decimal places. Units: log difference for methods 1–3, percentage points for method 4 |
| 6-month change | `direction`, 3 decimal places, color-coded (see below) |
| Change from COVID shock | See definition below, 3 decimal places |
| Denominator | Name of the denominator industry for this industry under the selected option |

### 6-month change color coding
Color reflects whether the deviation gap is closing (green) or widening (red),
accounting for the sign of the deviation:

- Above trend + direction < 0 → converging toward trend → **green**
- Above trend + direction > 0 → diverging further above trend → **red**
- Below trend + direction > 0 → converging toward trend → **green**
- Below trend + direction < 0 → diverging further below trend → **red**

### Change from COVID shock definition
```
change_from_covid = deviation[most recent month] - deviation[March 2020]
```

March 2020 is used as the COVID shock reference point — the first month of disruption.
A positive value means the industry is now below where it was relative to trend at the
onset of COVID (deteriorated further since the shock). A negative value means it is
now above where it was at the shock (recovered beyond trend).

If March 2020 deviation is NaN (e.g., series starts after March 2020), display NaN.

---

## Part 5: Sorting

Each table supports column-header sorting on three columns:
**Deviation**, **6-month change**, and **Change from COVID shock**.

### Default sort
Within each category section, sort by **absolute value of deviation, descending** —
most extreme deviation first. For "Above trend" sections this means the most above-trend
industry appears first. For "Below trend" sections the most below-trend industry appears first.

### User-triggered sorting
Clicking a sortable column header sorts that column:
- **Deviation:** sort by raw value, descending by default (most positive first). Toggle to ascending on second click.
- **6-month change:** sort by raw value, descending by default. Toggle on click.
- **Change from COVID shock:** sort by raw value, descending by default. Toggle on click.

Show a sort indicator (▲ / ▼) next to the active sort column header. Sorting applies
**within each section independently** — rows never move across category boundaries.

When the user changes any control (method, denominator, industry set), reset all sort
states back to the default (absolute deviation descending).

---

## Part 6: Summary Statistics

Above the four category sections, show a small summary bar:

```
[All: N]  [Above ↑: N]  [Above ↓: N]  [Below ↑: N]  [Below ↓: N]  [No data: N]
```

These are counts. Clicking any count scrolls to that section.

---

## Part 7: Detail Page Integration

When a user clicks an industry name from the analysis page, navigate to the industry
detail page with query params preserving:
- The selected denominator (`?denom=N`)
- The selected detrending method (`?method=N`)

The detail page should read these params and pre-select the matching denominator and
highlight or default to the matching detrending method's charts.

---

## Part 8: Data Requirements

All values needed for this page should be pre-computed and available at render time
without additional API calls:

- Deviation series for all 4 methods × 5 denominators × all industries
- Scalar snapshots per industry: most recent deviation, deviation 6 months ago,
  deviation at March 2020, most recent employment level, most recent share

If pre-computing all 20 combinations is too expensive, prioritize:
1. All methods × default denominator (Option 4) — needed for the default view
2. All denominators × default method (Option 1) — needed for denominator switching
3. Remaining combinations on demand

---

## Edge Cases

| Situation | Action |
|---|---|
| Series has no data in last 6 months | NaN for direction; put in "Insufficient data" section |
| March 2020 not in series (starts after) | Show NaN for change from COVID shock |
| Industry has NaN deviation for selected method | Put in "Insufficient data" section |
| All industries in "Insufficient data" | Show message: "No industries have sufficient data for this method/denominator combination" |
| Display level filter removes all government industries | Expected under goods/service denominator; handle gracefully with no error |
| Method 4 (share in %) combined with government industries | Government share is relative to total government not total private; label the units clearly |

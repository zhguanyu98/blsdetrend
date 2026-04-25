# Addendum: Three New Features (v2)
## Instructions for Claude Code

This file documents three additions to be implemented on top of the existing
denominator selection and analysis page instructions.

---

## Feature 1: Total Nonfarm as a Denominator Option (Option 6)

### What to add

Add a sixth denominator option:

| Option | Label | Description |
|---|---|---|
| 6 | "Total nonfarm" | Every industry — private and government — divides by Total nonfarm (CES0000000001) |

### Logic

This is the simplest denominator: the same series for every industry, no hierarchy
traversal needed.

```
denominator = CES0000000001   for all industries, all display levels
```

No special cases. Government industries, private industries, and aggregate industries
(Total private, Goods-producing, etc.) all divide by Total nonfarm. This is the only
option where government and private industries are directly on the same share scale.

### What needs to be re-run

1. Add `denominator_opt6 = 'CES0000000001'` and `denominator_opt6_name = 'Total nonfarm'`
   to every row in `b1a_mapping_with_denominators.csv`.

2. Compute `share_opt6[t] = employment[t] / total_nonfarm[t]` for all industries and
   months. Add `share_opt6` to the pre-computed share data.

3. Re-run all detrending methods using `share_opt6` as the share input. Store results
   as `resid_*_opt6` alongside the existing `resid_*_opt1` through `resid_*_opt5`.

4. Add Option 6 to the denominator dropdown on the main table page and analysis page,
   after Option 5.

5. Update the CSV download to include `share_opt6` and its corresponding detrended
   series columns.

### Note on government industries

Under all previous options, government sub-industries divide by Total government.
Under Option 6 they divide by Total nonfarm instead — intentional, as Option 6
is explicitly a "share of all employment" metric that treats government and private
symmetrically.

---

## Feature 2: Configurable Look-Back Window for Increasing/Decreasing

### What to change

Replace the hardcoded 6-month direction window on the analysis page with a
user-selectable control.

### New control

Add a dropdown labeled **"Trend window:"** to the analysis page controls (after
Detrending Method):

| Option | Label |
|---|---|
| 1 | "1 month" |
| 3 | "3 months" |
| 6 (default) | "6 months" |

### Updated direction formula

```
direction = deviation[t] - deviation[t - N]
```

Where N is the selected window (1, 3, or 6) and t is the most recent month.
The categorization logic and color coding are otherwise unchanged.

### Column header

Rename the direction column dynamically: "1-month change", "3-month change",
or "6-month change" based on the selected window.

### URL parameter

Add `window` to the URL query params. Example: `?set=4&denom=4&method=1&window=6`

---

## Feature 3: Second Derivative of Deviation

### Concept

This measures whether the pace of change in the deviation is itself speeding up
or slowing down. It compares the most recent 3-month change in deviation to the
same 3-month change from a reference period in the past.

### Formula

Let:
```
recent_change = deviation[t] - deviation[t-3]
past_change   = deviation[t-K] - deviation[t-K-3]
second_deriv  = recent_change - past_change
```

Where K is the lag to the reference period. Two sub-options:

| Sub-option | K | past_change window |
|---|---|---|
| A (default) | 12 | deviation[t-12] - deviation[t-15] |
| B | 6 | deviation[t-6] - deviation[t-9] |

### Interpretation

The sign and magnitude of `second_deriv` should be interpreted as follows:

**Both recent_change and past_change positive (both periods moving upward):**
- `second_deriv > 0`: upward momentum is accelerating — deviation rising faster now than before
- `second_deriv < 0`: upward momentum is fading — deviation still rising but at a slower pace

**Both recent_change and past_change negative (both periods moving downward):**
- `second_deriv > 0`: downward momentum is decelerating — deviation still falling but slowing down, potential stabilization
- `second_deriv < 0`: downward momentum is accelerating — deviation falling faster now than before

**recent_change positive, past_change negative (reversal upward):**
- `second_deriv > 0` always in this case — signals a genuine turning point from falling to rising; larger value = sharper reversal

**recent_change negative, past_change positive (reversal downward):**
- `second_deriv < 0` always in this case — signals a genuine turning point from rising to falling; more negative value = sharper reversal

The most actionable signal is when recent_change and past_change have **opposite signs**
— this indicates a directional reversal in the deviation's trajectory, not merely a
change in pace.

### Table Columns

Add three columns to the analysis table:

**Column 1: "Recent 3M Δ"**
- Value: `recent_change = deviation[t] - deviation[t-3]`, 3 decimal places

**Column 2: "Past 3M Δ"**
- Value: `past_change = deviation[t-K] - deviation[t-K-3]`, 3 decimal places
- Column header should dynamically reflect the sub-option, e.g.:
  - Sub-option A: "Past 3M Δ (t-12)"
  - Sub-option B: "Past 3M Δ (t-6)"

**Column 3: "2nd Deriv."**
- Value: `second_deriv = recent_change - past_change`, 3 decimal places
- Color code: green if second_deriv > 0, red if second_deriv < 0, neutral if zero

Place the sub-option toggle (A vs. B) near these column headers or in the main
controls row.

Do NOT rename this metric "acceleration" anywhere in the UI — label it
"2nd Deriv." in the table and "Second Derivative" in any tooltips or descriptions.

### Interpretation Column

Add a fourth column **"Pattern"** immediately after "2nd Deriv.". This column
shows a short label classifying each industry's pattern based on the signs of
`recent_change`, `past_change`, and `second_deriv`. There are exactly 6 patterns:

| Pattern label | Condition | Plain-English meaning |
|---|---|---|
| "Rising, faster" | R > 0, P > 0, D > 0 | Upward momentum accelerating |
| "Rising, slower" | R > 0, P > 0, D < 0 | Upward momentum fading |
| "Falling, slower" | R < 0, P < 0, D > 0 | Downward momentum decelerating, potential stabilization |
| "Falling, faster" | R < 0, P < 0, D < 0 | Downward momentum accelerating |
| "Reversal up" | R > 0, P < 0 | Turned from falling to rising (D always > 0) |
| "Reversal down" | R < 0, P > 0 | Turned from rising to falling (D always < 0) |

Where R = recent_change, P = past_change, D = second_deriv.

Show the pattern label as a small colored badge in the cell:
- "Rising, faster" → blue
- "Rising, slower" → light blue
- "Falling, slower" → light orange
- "Falling, faster" → red
- "Reversal up" → green
- "Reversal down" → dark red

If any of R, P, or D is NaN, show no badge (leave cell blank).

### Pattern Filter

Add a **"Pattern filter:"** control to the analysis page (alongside the other
controls at the top). This is a multi-select or set of toggle buttons, one per
pattern, allowing users to show only industries matching one or more specific
patterns across all four category sections.

Options (match the 6 pattern labels exactly):
- "Rising, faster"
- "Rising, slower"
- "Falling, slower"
- "Falling, faster"
- "Reversal up"
- "Reversal down"

Default: all patterns shown (no filter active).

When one or more patterns are selected, hide all rows whose Pattern column does
not match any selected pattern. Section counts in the headers and summary bar
should update to reflect the filtered row count, e.g.
"Above trend, increasing (12 of 34)".

Add `pattern` to the URL query params as a comma-separated list of active pattern
labels, e.g. `?..&pattern=Reversal+up,Reversal+down`. When no filter is active,
omit the param from the URL.

### Edge cases

| Situation | Action |
|---|---|
| Any required lag (t-3, t-6, t-9, t-12, t-15) is NaN | Show NaN for both columns |
| Series has fewer than 16 months of deviation data (sub-option A) | Show NaN |
| Series has fewer than 10 months of deviation data (sub-option B) | Show NaN |

### URL parameter

Add `deriv` to the URL query params: `?..&deriv=A` or `?..&deriv=B`

---

## Summary of All URL Parameters

| Param | Values | Default | Description |
|---|---|---|---|
| `set` | 0,2,3,4 | 4 | Industry display level filter |
| `denom` | 1–6 | 4 | Denominator option (6 = total nonfarm, new) |
| `method` | 1–4 | 1 | Detrending method |
| `window` | 1,3,6 | 6 | Look-back window for direction (new) |
| `deriv` | A,B | A | Second derivative reference period (new) |
| `pattern` | comma-separated pattern labels | (omitted = all) | Active pattern filter (new) |

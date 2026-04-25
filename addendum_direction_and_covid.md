# Addendum: Redefine Direction, Second Derivative, and COVID Shock Column
## Instructions for Claude Code

This file covers two related changes: (1) redefining increasing/decreasing and the
second derivative using share level instead of deviation, and (2) redefining the
COVID shock reference column using share level with a dynamically identified shock month.

---

## Change 1: Direction Now Based on Share Level, Not Deviation

### What to change

Previously, direction (increasing/decreasing) was computed as the change in the
deviation series over N months. Replace this with the change in the **employment
share level** over N months.

### Updated direction formula

```
direction = share[t] - share[t - N]
```

Where:
- `share[t]` is the employment share under the selected denominator at the most recent month
- `N` is the selected look-back window (1, 3, 6, or 12 months — see Change 2 below)

The deviation series is still used for the above/below trend classification and the
deviation column. Only the direction computation changes.

### Rationale

This measures whether the industry is actually gaining or losing share right now,
independent of where it stands relative to trend. An industry can be above trend
but losing share, or below trend but gaining share — this distinction is lost when
using deviation changes.

### Units and display

Share is a decimal (e.g. 0.045 = 4.5%). Display `direction` multiplied by 100 as
percentage points, to 2 decimal places. Rename the column header dynamically:
"1M share Δ", "3M share Δ", "6M share Δ", or "12M share Δ" depending on the
selected window.

---

## Change 2: Add 12-Month Window Option

Extend the look-back window dropdown to include a fourth option:

| Option | Label |
|---|---|
| 1 | "1 month" |
| 3 | "3 months" |
| 6 | "6 months" (default) |
| 12 | "12 months" |

Update the URL param: `window=12` is now valid.

---

## Change 3: Second Derivative Also Based on Share Level

### What to change

The second derivative columns (Recent 3M Δ, Past 3M Δ, 2nd Deriv.) previously used
the deviation series. Change all three to use the **share level** instead.

### Updated formulas

```
recent_change = share[t] - share[t-3]
past_change   = share[t-K] - share[t-K-3]
second_deriv  = recent_change - past_change
```

Where K = 12 (sub-option A) or K = 6 (sub-option B), unchanged from before.

Display all three values multiplied by 100 as percentage points, to 2 decimal places.
Update column header tooltips to read "3-month change in employment share" rather than
"3-month change in deviation".

### Pattern classification

The six pattern labels and logic are **unchanged** — they depend only on the signs of
`recent_change`, `past_change`, and `second_deriv`, which are now derived from share
changes instead of deviation changes. No other changes needed to the pattern filter
or badge colors.

---

## Change 4: Redefine "Change from COVID Shock" Column

### New definition

Replace the previous deviation-based COVID shock column with a share-based measure
that identifies the shock month dynamically per industry.

### Step 1: Identify the COVID shock month

For each industry, find the month of maximum absolute month-over-month share change
within **March, April, and May 2020**:

```
covid_month = argmax |share[t] - share[t-1]|   for t in {Mar 2020, Apr 2020, May 2020}
```

Use the actual (signed) share value at that month as the reference level:

```
covid_share = share[covid_month]
```

### Step 2: Compute the column value

```
change_from_covid = share[most recent month] - covid_share
```

Display multiplied by 100 as percentage points, to 2 decimal places.

A positive value means the industry's share is now **higher** than at the peak
disruption month. A negative value means it is **lower**.

### Step 3: Rename and tooltip

- Rename the column: **"Δ Share since COVID shock"**

### Edge cases

| Situation | Action |
|---|---|
| Industry has no data in Mar–May 2020 | Show NaN |
| Share values in Mar–May 2020 are all NaN | Show NaN |
| All three months have equal absolute change | Default to March 2020 |
| Current share is NaN | Show NaN |

---

## Summary: What Changes vs. What Stays the Same

| Element | Previously based on | Now based on |
|---|---|---|
| Above / below trend classification | Deviation | Deviation — **no change** |
| Deviation column | Deviation | Deviation — **no change** |
| Direction column (N-month change) | Deviation | **Share level** |
| Recent 3M Δ column | Deviation | **Share level** |
| Past 3M Δ column | Deviation | **Share level** |
| 2nd Deriv. column | Deviation | **Share level** |
| Pattern classification logic | Signs of deviation changes | Signs of share changes — logic unchanged |
| COVID shock column | Deviation at March 2020 | **Share level at dynamically identified shock month** |
| Look-back window options | 1, 3, 6 months | **1, 3, 6, 12 months** |

---

## Updated URL Parameter Summary

| Param | Values | Default | Description |
|---|---|---|---|
| `set` | 0,2,3,4 | 4 | Industry display level filter |
| `denom` | 1–6 | 4 | Denominator option |
| `method` | 1–4 | 1 | Detrending method |
| `window` | 1,3,6,12 | 6 | Look-back window for direction |
| `deriv` | A,B | A | Second derivative reference period |
| `pattern` | comma-separated pattern labels | (omitted = all) | Active pattern filter |

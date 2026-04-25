# Employment Share: User-Selectable Denominator Logic
# Instructions for Claude Code

## Overview

Add a user-facing dropdown to select the denominator used when computing employment share.
There are 5 options, each with its own denominator resolution logic. The current behavior
maps to "Display Level 4" (the default). Government industries follow a separate rule in
all options. Total private and Total nonfarm are fixed anchors throughout.

These instructions cover four things in order:
1. Recreate the industry mapping with denominator assignments for all 5 options
2. Pre-compute employment shares for all 5 options and store them
3. Add a denominator selector UI to the main table
4. Reflect the selected option in the industry detail page and detrended graphs

---

## Critical Implementation Note: Goods vs. Service Classification

**Do NOT determine goods/service membership by walking up the parent chain.**
The parent chain skips Goods-producing and Service-providing entirely — all private
industries ultimately link to Total private (display level 1), not to Goods-producing
or Service-providing.

Instead, classify each industry using its `supersector_code`:

```
GOODS_CODES    = {6, 10, 20, 30, 31, 32}
SERVICE_CODES  = {7, 8, 40, 41, 42, 43, 44, 50, 55, 60, 65, 70, 80}
GOVT_CODES     = {90}
SPECIAL_CODES  = {0, 5}   # total nonfarm, total private — handled separately
```

Define the following series IDs as fixed anchors (looked up from the mapping):
```
TOTAL_NONFARM_ID         = 'CES0000000001'
TOTAL_PRIVATE_ID         = 'CES0500000001'
GOODS_PRODUCING_ID       = 'CES0600000001'
SERVICE_PROVIDING_ID     = 'CES0700000001'
TOTAL_GOVERNMENT_ID      = 'CES9000000001'
```

---

## Helper: Find Closest Ancestor at a Given Display Level

Write a function `find_ancestor_at_level(series_id, target_level, mapping)` that:
1. Starts at `series_id`
2. Walks up via `parent_series_id`
3. Returns the first ancestor (inclusive of self) whose `display_level == target_level`
4. Returns `None` if no such ancestor exists before reaching total nonfarm or a cycle

Note: this function is used to find display-level-specific denominators (e.g., the
closest display-level-4 ancestor), not for goods/service classification.

---

## Special Cases Applying to ALL Options

These rules take priority and are evaluated before any option-specific logic:

| Industry | Denominator |
|---|---|
| Total nonfarm (display 0) | itself |
| Total private (display 1, supersector 5) | Total nonfarm |
| Goods-producing (display 1, supersector 6) | Total private |
| Service-providing (display 1, supersector 7) | Total private |
| Private service-providing (display 1, supersector 8) | Total private |
| Government (display 2, supersector 90) | Total nonfarm |
| Any government sub-industry (supersector 90, display > 2) | Total government |

---

## Option 1 (Default): Denominator at Display Level 4

**User label:** "Level 4 parent (default)"

Resolution logic for private non-government industries:

1. If industry's own `display_level < 4` (i.e., display levels 1, 2, 3):
   - Use **Goods-producing** if `supersector_code in GOODS_CODES`
   - Use **Service-providing** if `supersector_code in SERVICE_CODES`

2. If industry's own `display_level == 4`:
   - Use **Goods-producing** if `supersector_code in GOODS_CODES`
   - Use **Service-providing** if `supersector_code in SERVICE_CODES`

3. If industry's own `display_level > 4` (i.e., display levels 5, 6, 7):
   - Find closest ancestor at display level 4 via `find_ancestor_at_level`
   - If found: use that ancestor as denominator
   - If not found: find closest ancestor at display level 3
   - If still not found: use Goods-producing or Service-providing based on supersector code

---

## Option 2: Denominator at Display Level 3

**User label:** "Level 3 parent"

Resolution logic for private non-government industries:

1. If industry's own `display_level <= 3` (display levels 1, 2, 3):
   - Use **Goods-producing** if `supersector_code in GOODS_CODES`
   - Use **Service-providing** if `supersector_code in SERVICE_CODES`

2. If industry's own `display_level > 3` (display levels 4, 5, 6, 7):
   - Find closest ancestor at display level 3 via `find_ancestor_at_level`
   - If found: use that ancestor as denominator
   - If not found: use Goods-producing or Service-providing based on supersector code

---

## Option 3: Denominator at Display Level 2

**User label:** "Level 2 parent"

Resolution logic for private non-government industries:

1. If industry's own `display_level <= 2` (display levels 1, 2):
   - Use **Goods-producing** if `supersector_code in GOODS_CODES`
   - Use **Service-providing** if `supersector_code in SERVICE_CODES`

2. If industry's own `display_level > 2` (display levels 3, 4, 5, 6, 7):
   - Find closest ancestor at display level 2 via `find_ancestor_at_level`
   - If found: use that ancestor as denominator
   - If not found: use Goods-producing or Service-providing based on supersector code

---

## Option 4: Denominator at Display Level 1 (Goods/Service)

**User label:** "Goods/Service-providing total"

Resolution logic for ALL private non-government industries at any display level:

- Use **Goods-producing** if `supersector_code in GOODS_CODES`
- Use **Service-providing** if `supersector_code in SERVICE_CODES`

No ancestor traversal needed. Every private industry is classified directly by supersector code.

---

## Option 5: Total Private / Total Government

**User label:** "Total private / Total government"

Resolution logic:

- All private industries (any display level, any supersector except 90): use **Total private**
- All government industries (supersector 90, display > 2): use **Total government**
- Special cases (Total nonfarm, Total private, Goods/Service-providing, Government at display 2)
  follow the same fixed rules as all other options (see Special Cases section above)

No ancestor traversal needed.

---

## Summary Table

| Industry display level | Option 1 (Level 4) | Option 2 (Level 3) | Option 3 (Level 2) | Option 4 (Goods/Service) | Option 5 (Total private) |
|---|---|---|---|---|---|
| 1 (Total private) | Total nonfarm* | Total nonfarm* | Total nonfarm* | Total nonfarm* | Total nonfarm* |
| 1 (Goods/Service-providing) | Total private* | Total private* | Total private* | Total private* | Total private* |
| 2 (private) | Goods or Service | Goods or Service | Goods or Service | Goods or Service | Total private |
| 3 (private) | Goods or Service | Goods or Service | Display-2 ancestor | Goods or Service | Total private |
| 4 (private) | Goods or Service | Display-3 ancestor | Display-2 ancestor | Goods or Service | Total private |
| 5 (private) | Display-4 ancestor | Display-3 ancestor | Display-2 ancestor | Goods or Service | Total private |
| 6 (private) | Display-4 ancestor | Display-3 ancestor | Display-2 ancestor | Goods or Service | Total private |
| 7 (private) | Display-4 ancestor | Display-3 ancestor | Display-2 ancestor | Goods or Service | Total private |
| Government (display 2) | Total nonfarm* | Total nonfarm* | Total nonfarm* | Total nonfarm* | Total nonfarm* |
| Government (display 3+) | Total government* | Total government* | Total government* | Total government* | Total government* |

*Special case — same across all options.

---

## Part 1: Recreate the Mapping with Denominator Assignments

Write a script `build_mapping.py` that:

1. Loads `b1a_mapping_with_parent.csv` as the base mapping. Do not modify the original columns.

2. For each industry (row), compute the denominator `series_id` under each of the 5 options
   using the logic above. Add 5 new columns to the mapping:

   ```
   denominator_opt1   # Level 4 parent (default)
   denominator_opt2   # Level 3 parent
   denominator_opt3   # Level 2 parent
   denominator_opt4   # Goods/Service-providing total
   denominator_opt5   # Total private / Total government
   ```

3. Also add a human-readable column for each option showing the denominator industry name
   (looked up from the mapping by series_id):

   ```
   denominator_opt1_name
   denominator_opt2_name
   denominator_opt3_name
   denominator_opt4_name
   denominator_opt5_name
   ```

4. Save the enriched mapping as `b1a_mapping_with_denominators.csv`.

5. Print a validation summary:
   - For each option, count how many industries resolve to each denominator
   - Flag any industry where the denominator could not be resolved (should be zero)

Use the `find_ancestor_at_level` helper described above. Pre-build a dict
`{series_id: row}` for O(1) lookups before traversal.

---

## Part 2: Pre-Compute Employment Shares for All 5 Options

Write a script `compute_shares.py` that:

1. Loads `b1a_mapping_with_denominators.csv`
2. Loads the BLS employment time series for all industries (however your existing
   pipeline loads them)
3. For each industry and each option (1–5):
   - Look up the denominator series_id from `denominator_opt{N}` column
   - Compute `share_opt{N}[t] = employment[t] / denominator_employment[t]` for all months
4. Saves a parquet or CSV file per industry (or a single wide file if your pipeline
   supports it) containing columns:
   ```
   date, employment_level,
   share_opt1, share_opt2, share_opt3, share_opt4, share_opt5,
   denominator_opt1_level, denominator_opt2_level, ...,   # denominator employment levels
   ```
5. All downstream detrending (linear trend) must be computed separately
   per option since the share series differs. Pre-compute and store detrended series for
   all 5 options at this stage if computationally feasible, or compute on demand per option.

**Validate:** For Option 5, every private industry's share should be <= 1.0. For Option 4,
the display-level-2 children of Goods-producing should sum to ~1.0 in any given month.

---

## Part 3: Denominator Selector in the Main Table

On the main industry table page:

1. Add a **dropdown selector** above or near the table column headers, labeled:
   **"Employment share denominator:"**

   Options (in order):
   - "Level 4 parent (default)" — Option 1, selected by default
   - "Level 3 parent" — Option 2
   - "Level 2 parent" — Option 3
   - "Goods/Service-providing total" — Option 4
   - "Total private / Total government" — Option 5

2. The dropdown selection controls which share column is shown in the table's
   **Employment Share** column and which share percentile is shown.

3. When the user changes the dropdown:
   - Update the Employment Share column values for all visible rows
   - Update the share percentile column
   - Update the 4 deviation columns (detrended shares for both methods, both options)
   - Do NOT reload the page — update reactively (client-side if shares are pre-loaded,
     or via a lightweight API call if computed server-side)

4. **Persist the selection** in the URL as a query parameter, e.g. `?denom=2`, so that:
   - Sharing a URL preserves the selected denominator
   - The back button restores the previous selection
   - Default (no query param) = Option 1

5. Show a **tooltip or footnote** beneath the dropdown explaining what the selected
   denominator means, e.g.:
   - Option 1: "Share of the closest Level 4 parent industry (e.g., Roofing ÷ Specialty trade contractors)"
   - Option 5: "Share of Total private employment (for private industries)"

---

## Part 4: Industry Detail Page and Detrended Graphs

When a user clicks through to an individual industry's detail page:

1. **Pass the selected denominator option** from the main table to the detail page via
   the URL, e.g. `/industry/CES2023816001?denom=2`. The detail page reads the `denom`
   query parameter on load and defaults to Option 1 if absent.

2. **On the detail page**, show the same dropdown selector so the user can change it
   without navigating back.

3. All 6 detrended charts (2 for log level linear exptrapolating (1 for the counterfactual v.s. actual and 1 for deviation); 2 for log share linear extrapolating; 2 for share quadratic extrapolating(reduce to linear is second order coefficient is not significant)) must reflect the
   selected denominator option for their share-based charts.
   Level-based charts (Charts 1, 2) are unaffected by denominator selection. Basically the detrending method is still the same as before. 

4. When the denominator option changes on the detail page:
   - Re-fetch or recompute the share series for the selected option
   - Replot Charts 3-6 with the new share data
   - Update the chart title/subtitle to indicate the active denominator, e.g.:
     "Employment Share (÷ Specialty trade contractors, Level 4 parent)"

5. The **CSV download** on the detail page should include all 5 share columns and their
   corresponding detrended series, so users can explore all options offline. Label each
   column clearly: `share_opt1`, `share_opt2`, etc., and include a header comment
   explaining each option.

---

## Edge Cases

| Situation | Action |
|---|---|
| Industry is its own display-level-4 ancestor (i.e., is display level 4) | Options 1 & 2: use Goods/Service as denominator, not itself |
| No ancestor found at the target level | Fall back to Goods-producing or Service-providing |
| Industry supersector code not in GOODS_CODES or SERVICE_CODES | Should not occur for private industries; log a warning and fall back to Total private |
| Government industry at display level 2 | Use Total nonfarm for all options |
| Government industry below display level 2 | Use Total government for all options |
| Denominator series has missing months | Propagate NaN for share in those months; do not interpolate |
| URL `denom` param is out of range (not 1–5) | Default to Option 1 silently |

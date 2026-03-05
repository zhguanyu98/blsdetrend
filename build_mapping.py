"""
build_mapping.py — Build b1a_mapping_with_denominators.csv

Computes the denominator series_id for each of 5 options and saves the
enriched mapping. Run from the BLS/ root directory before generate_data.py.
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent

GOODS_CODES   = {6, 10, 20, 30, 31, 32}
SERVICE_CODES = {7, 8, 40, 41, 42, 43, 44, 50, 55, 60, 65, 70, 80}
GOVT_CODES    = {90}

TOTAL_NONFARM_ID     = "CES0000000001"
TOTAL_PRIVATE_ID     = "CES0500000001"
GOODS_PRODUCING_ID   = "CES0600000001"
SERVICE_PROVIDING_ID = "CES0700000001"
TOTAL_GOVERNMENT_ID  = "CES9000000001"

OPT_LABELS = {
    1: "Level 4 parent (default)",
    2: "Level 3 parent",
    3: "Level 2 parent",
    4: "Goods/Service-providing total",
    5: "Total private / Total government",
}

# ── Load mapping ───────────────────────────────────────────────────────────────
mapping    = pd.read_csv(BASE / "b1a_mapping_with_parent.csv")
mapping    = mapping.sort_values("row_order").reset_index(drop=True)
id_to_row  = {row["series_id"]: row for _, row in mapping.iterrows()}
id_to_name = {row["series_id"]: row["industry_name"] for _, row in mapping.iterrows()}
id_to_idx  = {row["series_id"]: i for i, (_, row) in enumerate(mapping.iterrows())}

# Pre-build level array for fast backward scan
_levels = mapping["display_level"].astype(int).values
_sids   = mapping["series_id"].values


# ── Helpers ────────────────────────────────────────────────────────────────────
def goods_or_service(sc: int) -> str:
    if sc in GOODS_CODES:
        return GOODS_PRODUCING_ID
    if sc in SERVICE_CODES:
        return SERVICE_PROVIDING_ID
    print(f"  WARNING: supersector {sc} not in GOODS or SERVICE codes — falling back to Total private")
    return TOTAL_PRIVATE_ID


def find_ancestor_at_level(sid: str, target_lvl: int) -> str | None:
    """
    Scan backwards through the row_order-sorted mapping from sid's position.
    Return the first row whose display_level == target_lvl.
    Stop (return None) if we encounter a row whose display_level < target_lvl,
    since that means we've left the scope where a target-level ancestor exists.

    This is more reliable than walking parent_series_id chains, because BLS
    parent_series_id fields often skip directly to Total private (level 1).
    """
    if sid not in id_to_idx:
        return None
    start = id_to_idx[sid]
    for i in range(start - 1, -1, -1):
        lvl = _levels[i]
        if lvl == target_lvl:
            return _sids[i]
        if lvl < target_lvl:
            return None   # overshot; no ancestor at this level in scope
    return None


def special_case(sid: str, sc: int, lvl: int) -> str | None:
    """Return denominator for industries that are special-cased across all options."""
    if sid == TOTAL_NONFARM_ID:       return TOTAL_NONFARM_ID
    if sid == TOTAL_PRIVATE_ID:       return TOTAL_NONFARM_ID
    if sid == GOODS_PRODUCING_ID:     return TOTAL_PRIVATE_ID
    if sid == SERVICE_PROVIDING_ID:   return TOTAL_PRIVATE_ID
    if lvl == 1 and sc == 8:          return TOTAL_PRIVATE_ID   # Private service-providing
    if sc in GOVT_CODES and lvl == 2: return TOTAL_NONFARM_ID
    if sc in GOVT_CODES and lvl > 2:  return TOTAL_GOVERNMENT_ID
    return None


def resolve(sid: str, opt: int) -> str:
    row = id_to_row[sid]
    sc  = int(row["supersector_code"])
    lvl = int(row["display_level"])

    spec = special_case(sid, sc, lvl)
    if spec is not None:
        return spec

    # Private, non-government industries
    if opt == 1:   # Level 4 parent
        if lvl <= 4:
            return goods_or_service(sc)
        anc = find_ancestor_at_level(sid, 4)
        if anc and anc != sid:
            return anc
        anc = find_ancestor_at_level(sid, 3)
        return anc if anc else goods_or_service(sc)

    elif opt == 2:  # Level 3 parent
        if lvl <= 3:
            return goods_or_service(sc)
        anc = find_ancestor_at_level(sid, 3)
        return anc if anc else goods_or_service(sc)

    elif opt == 3:  # Level 2 parent
        if lvl <= 2:
            return goods_or_service(sc)
        anc = find_ancestor_at_level(sid, 2)
        return anc if anc else goods_or_service(sc)

    elif opt == 4:  # Goods/Service-providing total (all levels)
        return goods_or_service(sc)

    elif opt == 5:  # Total private / Total government
        return TOTAL_PRIVATE_ID

    return TOTAL_PRIVATE_ID


# ── Build enriched mapping ─────────────────────────────────────────────────────
print("Computing denominator assignments…")

for opt in range(1, 6):
    mapping[f"denominator_opt{opt}"]      = mapping["series_id"].apply(lambda s: resolve(s, opt))
    mapping[f"denominator_opt{opt}_name"] = mapping[f"denominator_opt{opt}"].map(id_to_name)

out_path = BASE / "b1a_mapping_with_denominators.csv"
mapping.to_csv(out_path, index=False)
print(f"Saved → {out_path}")

# ── Validation summary ─────────────────────────────────────────────────────────
print()
for opt in range(1, 6):
    col    = f"denominator_opt{opt}"
    nulls  = mapping[col].isna().sum()
    counts = mapping[col].value_counts()
    print(f"Option {opt} — {OPT_LABELS[opt]}: {nulls} unresolved")
    for denom_id, cnt in counts.head(6).items():
        print(f"  {cnt:4d}  {denom_id}  {id_to_name.get(denom_id, '?')}")
    print()

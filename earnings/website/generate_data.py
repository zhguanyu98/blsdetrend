"""
generate_data.py — Pre-compute all JSON data for the BLS B-3a earnings website.

Run from the earnings/website/ directory:
    python generate_data.py

Inputs (from earnings/):
    ../b3a_wide.csv      — wide AWE data (date index × series_id columns)
    ../b3a_mapping.csv   — series metadata + benchmark assignments
    ../cpiu.csv          — CPI-U monthly index

Outputs (to earnings/website/data/):
    table_data.json          — homepage rows (default benchmark)
    table_data_alt.json      — homepage rows (Total Private benchmark for all)
    {series_id}.json         — per-industry time series (default benchmark)
    {series_id}_alt.json     — per-industry time series (Total Private benchmark)
    {series_id}_export.csv   — per-industry downloadable CSV
"""

import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EARNINGS_DIR = HERE.parent
DATA_OUT = HERE / "data"
DATA_OUT.mkdir(exist_ok=True)

BENCHMARK_TOTAL_PRIVATE = "CES0500000011"
BENCHMARK_GOODS = "CES0600000011"
BENCHMARK_SERVICES = "CES0800000011"

MIN_OBSERVATIONS = 15  # exclude series with fewer months


# ── Load source data ───────────────────────────────────────────────────────────

def load_data():
    wide_path = EARNINGS_DIR / "b3a_wide.csv"
    mapping_path = EARNINGS_DIR / "b3a_mapping.csv"
    cpiu_path = EARNINGS_DIR / "cpiu.csv"

    for p in [wide_path, mapping_path, cpiu_path]:
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Run pull_data.py first.")

    awe = pd.read_csv(wide_path, index_col=0)
    awe.index = awe.index.astype(str)
    awe = awe.sort_index()

    mapping = pd.read_csv(mapping_path, dtype=str)
    mapping["display_level"] = pd.to_numeric(mapping["display_level"], errors="coerce").fillna(0).astype(int)
    mapping["row_order"] = mapping["row_order"].astype(int)
    mapping["supersector_code"] = mapping["supersector_code"].astype(int)

    cpiu = pd.read_csv(cpiu_path, dtype={"date": str, "cpiu": float})
    cpiu = cpiu.set_index("date").sort_index()["cpiu"]

    return awe, mapping, cpiu


# ── CPI-U lag handling ─────────────────────────────────────────────────────────

def align_cpiu(awe: pd.DataFrame, cpiu: pd.Series) -> pd.Series:
    """Align CPI-U to AWE dates, substituting latest available if needed."""
    awe_latest = awe.index[-1]
    cpiu_latest = cpiu.index[-1]
    if cpiu_latest < awe_latest:
        print(
            f"WARNING: CPI-U for {awe_latest} not available. "
            f"Using {cpiu_latest} as substitute.",
            file=sys.stderr,
        )
        cpiu = cpiu.copy()
        cpiu[awe_latest] = cpiu[cpiu_latest]
        cpiu = cpiu.sort_index()
    return cpiu


# ── Core calculations ──────────────────────────────────────────────────────────

def yoy(s: pd.Series) -> pd.Series:
    """Year-over-year growth rate."""
    return (s - s.shift(12)) / s.shift(12)


def avg_3m(s: pd.Series) -> pd.Series:
    """Simple 3-month average (t, t-1, t-2)."""
    return (s + s.shift(1) + s.shift(2)) / 3


def past_10m_avg(s: pd.Series) -> pd.Series:
    """Average of months t-3 through t-12 (10 months)."""
    total = sum(s.shift(i) for i in range(3, 13))
    return total / 10


def classify_pattern(avg_rel: Optional[float], past_rel: Optional[float], second_d: Optional[float]) -> str:
    """Classify into one of six momentum patterns."""
    if avg_rel is None or past_rel is None or second_d is None:
        return ""
    if avg_rel >= 0:
        if past_rel <= 0:
            return "Reversal Up"
        return "Accelerating Above" if second_d >= 0 else "Decelerating Above"
    else:
        if past_rel >= 0:
            return "Reversal Down"
        return "Accelerating Below" if second_d <= 0 else "Decelerating Below"


def nan_to_none(v):
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


def series_to_list(s: pd.Series) -> list:
    return [nan_to_none(v) for v in s]


# ── Build per-industry data ────────────────────────────────────────────────────

def compute_industry(
    sid: str,
    awe: pd.DataFrame,
    cpiu_aligned: pd.Series,
    benchmark_id: str,
    benchmark_name: str,
) -> Optional[dict]:
    """Compute all time-series fields for one industry."""
    if sid not in awe.columns:
        return None
    if benchmark_id not in awe.columns:
        return None

    s = awe[sid]
    b = awe[benchmark_id]

    # YoY series
    s_yoy = yoy(s)
    b_yoy = yoy(b)
    cpiu_yoy_s = yoy(cpiu_aligned.reindex(s.index))
    real_yoy_s = s_yoy - cpiu_yoy_s

    # Relative YoY
    if sid == benchmark_id:
        rel_yoy_s = pd.Series(np.nan, index=s.index)
    else:
        rel_yoy_s = s_yoy - b_yoy

    dates = list(awe.index)
    return {
        "series_id": sid,
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_name,
        "dates": dates,
        "awe_level": series_to_list(s.reindex(dates)),
        "benchmark_level": series_to_list(b.reindex(dates)),
        "yoy": series_to_list(s_yoy.reindex(dates)),
        "benchmark_yoy": series_to_list(b_yoy.reindex(dates)),
        "real_yoy": series_to_list(real_yoy_s.reindex(dates)),
        "rel_yoy": series_to_list(rel_yoy_s.reindex(dates)),
        "cpiu_yoy": series_to_list(cpiu_yoy_s.reindex(dates)),
    }


def compute_table_row(
    row: pd.Series,
    awe: pd.DataFrame,
    cpiu_aligned: pd.Series,
    override_benchmark_id: Optional[str] = None,
    override_benchmark_name: Optional[str] = None,
) -> Optional[dict]:
    """Compute one row's snapshot metrics for table_data.json."""
    sid = row["series_id"]
    if sid not in awe.columns:
        return None

    s = awe[sid].dropna()
    if len(s) < MIN_OBSERVATIONS:
        return None

    bmark_id = override_benchmark_id or row["benchmark_id"]
    bmark_name = override_benchmark_name or row["benchmark_name"]

    if bmark_id not in awe.columns:
        return None

    b = awe[bmark_id]

    s_yoy = yoy(awe[sid])
    b_yoy = yoy(b)
    cpiu_yoy_s = yoy(cpiu_aligned.reindex(awe.index))
    real_yoy_s = s_yoy - cpiu_yoy_s

    if sid == bmark_id:
        rel_yoy_s = pd.Series(np.nan, index=awe.index)
    else:
        rel_yoy_s = s_yoy - b_yoy

    avg_rel = avg_3m(rel_yoy_s)
    avg_nom = avg_3m(s_yoy)
    avg_real = avg_3m(real_yoy_s)
    past_rel = past_10m_avg(rel_yoy_s)
    second_d = avg_rel - past_rel

    # Latest non-NaN index
    latest_idx = awe[sid].last_valid_index()
    if latest_idx is None:
        return None
    prev_idx = awe.index[awe.index.get_loc(latest_idx) - 1] if awe.index.get_loc(latest_idx) > 0 else None

    def get(series, idx):
        if idx is None or idx not in series.index:
            return None
        return nan_to_none(series[idx])

    pattern = classify_pattern(
        get(avg_rel, latest_idx),
        get(past_rel, latest_idx),
        get(second_d, latest_idx),
    )

    return {
        "series_id": sid,
        "industry_name": row["industry_name"],
        "display_level": int(row["display_level"]),
        "row_order": int(row["row_order"]),
        "awe_latest": get(awe[sid], latest_idx),
        "awe_prev": get(awe[sid], prev_idx) if prev_idx else None,
        "avg_yoy_3m": get(avg_nom, latest_idx),
        "avg_rel_yoy_3m": get(avg_rel, latest_idx),
        "avg_real_yoy_3m": get(avg_real, latest_idx),
        "benchmark_id": bmark_id,
        "benchmark_name": bmark_name,
        "second_deriv": get(second_d, latest_idx),
        "past_rel_yoy": get(past_rel, latest_idx),
        "pattern": pattern,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading source data…")
    awe, mapping, cpiu = load_data()

    latest_awe_date = awe.index[-1]
    cpiu_aligned = align_cpiu(awe, cpiu)

    # Reindex CPI-U to match AWE dates
    cpiu_full = cpiu_aligned.reindex(awe.index)

    print(f"AWE: {len(awe)} months ({awe.index[0]} – {awe.index[-1]}), {len(awe.columns)} series")
    print(f"CPI-U: {len(cpiu)} months")

    # ── Generate per-industry JSON files ────────────────────────────────────────
    table_rows_default = []
    table_rows_alt = []
    n_detail = 0
    n_skip = 0

    for _, row in mapping.iterrows():
        sid = row["series_id"]
        bmark_id = row["benchmark_id"]
        bmark_name = row["benchmark_name"]

        if sid not in awe.columns:
            n_skip += 1
            continue

        # Default benchmark JSON
        detail_default = compute_industry(sid, awe, cpiu_full, bmark_id, bmark_name)
        if detail_default:
            out_path = DATA_OUT / f"{sid}.json"
            with open(out_path, "w") as f:
                json.dump(detail_default, f, separators=(",", ":"))

        # Alt benchmark JSON (Total Private override)
        detail_alt = compute_industry(sid, awe, cpiu_full, BENCHMARK_TOTAL_PRIVATE, "Total Private")
        if detail_alt:
            out_path = DATA_OUT / f"{sid}_alt.json"
            with open(out_path, "w") as f:
                json.dump(detail_alt, f, separators=(",", ":"))

        # Export CSV
        s_data = awe[sid] if sid in awe.columns else None
        b_data = awe[bmark_id] if bmark_id in awe.columns else None
        if s_data is not None and b_data is not None:
            s_yoy_vals = yoy(awe[sid])
            b_yoy_vals = yoy(b_data)
            cpiu_yoy_vals = yoy(cpiu_full)
            real_yoy_vals = s_yoy_vals - cpiu_yoy_vals
            rel_yoy_vals = s_yoy_vals - b_yoy_vals if sid != bmark_id else pd.Series(np.nan, index=awe.index)

            csv_path = DATA_OUT / f"{sid}_export.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "awe_level", "benchmark_level", "yoy",
                                 "benchmark_yoy", "real_yoy", "rel_yoy", "cpiu_yoy"])
                for d in awe.index:
                    writer.writerow([
                        d,
                        nan_to_none(s_data.get(d)),
                        nan_to_none(b_data.get(d)),
                        nan_to_none(s_yoy_vals.get(d)),
                        nan_to_none(b_yoy_vals.get(d)),
                        nan_to_none(real_yoy_vals.get(d)),
                        nan_to_none(rel_yoy_vals.get(d)) if sid != bmark_id else None,
                        nan_to_none(cpiu_yoy_vals.get(d)),
                    ])

        n_detail += 1

        # Table rows
        tr_default = compute_table_row(row, awe, cpiu_full)
        if tr_default:
            table_rows_default.append(tr_default)

        tr_alt = compute_table_row(
            row, awe, cpiu_full,
            override_benchmark_id=BENCHMARK_TOTAL_PRIVATE,
            override_benchmark_name="Total Private",
        )
        if tr_alt:
            table_rows_alt.append(tr_alt)

    print(f"Generated detail pages for {n_detail} series ({n_skip} skipped — not in AWE data)")

    # Determine latest_label from most recent AWE date
    try:
        import datetime as dt
        d = dt.datetime.strptime(latest_awe_date, "%Y-%m")
        latest_label = d.strftime("%B %Y")
        prev_d = d.replace(day=1) - dt.timedelta(days=1)
        prev_label = prev_d.strftime("%B %Y")
    except Exception:
        latest_label = latest_awe_date
        prev_label = ""

    # ── Write table_data.json ────────────────────────────────────────────────────
    table_default = {"rows": table_rows_default, "latest_label": latest_label, "prev_label": prev_label}
    with open(DATA_OUT / "table_data.json", "w") as f:
        json.dump(table_default, f, separators=(",", ":"))
    print(f"Wrote table_data.json ({len(table_rows_default)} rows)")

    # ── Write table_data_alt.json ────────────────────────────────────────────────
    table_alt = {"rows": table_rows_alt, "latest_label": latest_label, "prev_label": prev_label}
    with open(DATA_OUT / "table_data_alt.json", "w") as f:
        json.dump(table_alt, f, separators=(",", ":"))
    print(f"Wrote table_data_alt.json ({len(table_rows_alt)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

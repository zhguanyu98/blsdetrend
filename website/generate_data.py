"""
generate_data.py — Pre-process BLS B-1a data into JSON files for the website.

Run from the website/ directory:
    python generate_data.py

Outputs:
    data/table_data.json          — All 842 rows for main table
    data/{series_id}.json         — Per-industry chart data
    data/{series_id}_export.csv   — Per-industry downloadable CSV
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent          # BLS/
DATA_OUT = Path(__file__).parent / "data"
DATA_OUT.mkdir(exist_ok=True)

MAPPING_FILE   = BASE / "b1a_mapping_with_parent.csv"
EMP_FILE       = BASE / "b1a_wide_seriesid.csv"
SHARES_FILE    = BASE / "b1a_employment_shares.csv"
DET_MA60_FILE  = BASE / "detrended" / "detrended_share_ma60.csv"
DET_HAM_FILE   = BASE / "detrended" / "detrended_share_hamilton_logshare.csv"
DET_POLY3_FILE = BASE / "detrended" / "detrended_share_poly3_logempdiff.csv"
DET_LOGEMP_FILE= BASE / "detrended" / "detrended_logemp.csv"


# ── Helpers ────────────────────────────────────────────────────────────────────
def pct_of_score(series: pd.Series) -> float:
    """Return percentile of most-recent non-NaN value in full series (0–100)."""
    s = series.dropna()
    if len(s) == 0:
        return None
    return round(float(percentileofscore(s, s.iloc[-1], kind="rank")), 1)


def to_float(x):
    """Convert to Python float, returning None for NaN."""
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 6)
    except (TypeError, ValueError):
        return None


def month_label(dt) -> str:
    return dt.strftime("%b-%y")


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data…")

mapping = pd.read_csv(MAPPING_FILE)
mapping = mapping.sort_values("row_order").reset_index(drop=True)

emp = pd.read_csv(EMP_FILE, index_col=0, parse_dates=True)
shares = pd.read_csv(SHARES_FILE, index_col=0, parse_dates=True)
det_ma60  = pd.read_csv(DET_MA60_FILE,  index_col=0, parse_dates=True)
det_ham   = pd.read_csv(DET_HAM_FILE,   index_col=0, parse_dates=True)
det_poly3 = pd.read_csv(DET_POLY3_FILE, index_col=0, parse_dates=True)
det_logemp= pd.read_csv(DET_LOGEMP_FILE,index_col=0, parse_dates=True)

# Align all date indices to employment dates
all_dates = emp.index
date_strs = [d.strftime("%Y-%m-%d") for d in all_dates]

# Most recent and previous month labels
last_dt  = emp.index[-1]
prev_dt  = emp.index[-2]
last_lbl = month_label(last_dt)
prev_lbl = month_label(prev_dt)


# ── Build table_data.json ──────────────────────────────────────────────────────
print("Building table_data.json…")

rows = []
for _, row in mapping.iterrows():
    sid = row["series_id"]
    name = row["industry_name"]
    lvl  = int(row["display_level"])
    parent_sid = row["parent_series_id"]

    # Look up parent (denominator) name
    parent_row = mapping.loc[mapping["series_id"] == parent_sid]
    denom_name = parent_row["industry_name"].iloc[0] if len(parent_row) > 0 else name

    # Employment
    emp_col = emp[sid] if sid in emp.columns else None
    emp_recent = to_float(emp_col.iloc[-1]) if emp_col is not None else None
    emp_prev   = to_float(emp_col.iloc[-2]) if emp_col is not None else None

    # Share
    share_val = None
    share_pct = None
    if sid in shares.columns:
        s_series = shares[sid]
        share_val = to_float(s_series.dropna().iloc[-1]) if s_series.dropna().shape[0] > 0 else None
        share_pct = pct_of_score(s_series)

    # Detrended MA60
    det_ma60_val = None; det_ma60_pct = None
    if sid in det_ma60.columns:
        s = det_ma60[sid]
        det_ma60_val = to_float(s.dropna().iloc[-1]) if s.dropna().shape[0] > 0 else None
        det_ma60_pct = pct_of_score(s)

    # Detrended Hamilton
    det_ham_val = None; det_ham_pct = None
    if sid in det_ham.columns:
        s = det_ham[sid]
        det_ham_val = to_float(s.dropna().iloc[-1]) if s.dropna().shape[0] > 0 else None
        det_ham_pct = pct_of_score(s)

    # Detrended Poly3
    det_poly3_val = None; det_poly3_pct = None
    if sid in det_poly3.columns:
        s = det_poly3[sid]
        det_poly3_val = to_float(s.dropna().iloc[-1]) if s.dropna().shape[0] > 0 else None
        det_poly3_pct = pct_of_score(s)

    rows.append({
        "series_id":            sid,
        "industry_name":        name,
        "display_level":        lvl,
        "emp_recent":           emp_recent,
        "emp_recent_label":     last_lbl,
        "emp_prev":             emp_prev,
        "emp_prev_label":       prev_lbl,
        "share":                share_val,
        "share_pct":            share_pct,
        "denom_name":           denom_name,
        "detrended_ma60":       det_ma60_val,
        "detrended_ma60_pct":   det_ma60_pct,
        "detrended_hamilton":   det_ham_val,
        "detrended_hamilton_pct": det_ham_pct,
        "detrended_poly3":      det_poly3_val,
        "detrended_poly3_pct":  det_poly3_pct,
    })

with open(DATA_OUT / "table_data.json", "w") as f:
    json.dump({"rows": rows, "last_label": last_lbl, "prev_label": prev_lbl}, f)

print(f"  → {len(rows)} rows written to table_data.json")


# ── Build per-industry JSON + CSV ──────────────────────────────────────────────
print("Building per-industry files…")

n = len(mapping)
for i, mrow in mapping.iterrows():
    sid  = mrow["series_id"]
    name = mrow["industry_name"]
    parent_sid = mrow["parent_series_id"]

    parent_row = mapping.loc[mapping["series_id"] == parent_sid]
    denom_name = parent_row["industry_name"].iloc[0] if len(parent_row) > 0 else name

    # Helper: series → list with None for NaN, aligned to all_dates
    def series_to_list(df, col):
        if col not in df.columns:
            return [None] * len(all_dates)
        s = df[col].reindex(all_dates)
        return [to_float(v) for v in s]

    share_list   = series_to_list(shares,    sid)
    ma60_list    = series_to_list(det_ma60,  sid)
    ham_list     = series_to_list(det_ham,   sid)
    poly3_list   = series_to_list(det_poly3, sid)

    # Log-emp components: numerator = sid, denominator = parent_sid
    logemp_num   = series_to_list(det_logemp, sid)
    logemp_denom = series_to_list(det_logemp, parent_sid)

    industry_data = {
        "series_id":              sid,
        "industry_name":          name,
        "denom_name":             denom_name,
        "parent_series_id":       parent_sid,
        "dates":                  date_strs,
        "share":                  share_list,
        "detrended_ma60":         ma60_list,
        "detrended_hamilton":     ham_list,
        "detrended_poly3":        poly3_list,
        "log_emp_num_poly3":      logemp_num,
        "log_emp_denom_poly3":    logemp_denom,
        "csv_filename":           f"{sid}_export.csv",
    }

    with open(DATA_OUT / f"{sid}.json", "w") as f:
        json.dump(industry_data, f)

    # ── Per-industry export CSV ──────────────────────────────────────────────
    # Build a DataFrame with all series aligned to all_dates
    emp_num   = emp[sid].reindex(all_dates) if sid in emp.columns else pd.Series([np.nan]*len(all_dates), index=all_dates)
    emp_denom = emp[parent_sid].reindex(all_dates) if parent_sid in emp.columns else pd.Series([np.nan]*len(all_dates), index=all_dates)

    share_s   = shares[sid].reindex(all_dates)    if sid in shares.columns    else pd.Series([np.nan]*len(all_dates), index=all_dates)
    ma60_s    = det_ma60[sid].reindex(all_dates)  if sid in det_ma60.columns  else pd.Series([np.nan]*len(all_dates), index=all_dates)
    ham_s     = det_ham[sid].reindex(all_dates)   if sid in det_ham.columns   else pd.Series([np.nan]*len(all_dates), index=all_dates)
    poly3_s   = det_poly3[sid].reindex(all_dates) if sid in det_poly3.columns else pd.Series([np.nan]*len(all_dates), index=all_dates)
    logemp_num_s   = det_logemp[sid].reindex(all_dates)        if sid in det_logemp.columns        else pd.Series([np.nan]*len(all_dates), index=all_dates)
    logemp_denom_s = det_logemp[parent_sid].reindex(all_dates) if parent_sid in det_logemp.columns else pd.Series([np.nan]*len(all_dates), index=all_dates)

    export_df = pd.DataFrame({
        "date":                                    all_dates.strftime("%Y-%m-%d"),
        "numerator_series_id":                     sid,
        "denominator_series_id":                   parent_sid,
        "numerator_name":                          name,
        "denominator_name":                        denom_name,
        "emp_numerator":                           emp_num.values,
        "emp_denominator":                         emp_denom.values,
        "share":                                   share_s.values,
        "detrended_share_ma60":                    ma60_s.values,
        "detrended_share_hamilton_logshare":       ham_s.values,
        "detrended_share_poly3_logempdiff":        poly3_s.values,
        "detrended_log_emp_numerator_poly3":       logemp_num_s.values,
        "detrended_log_emp_denominator_poly3":     logemp_denom_s.values,
    })
    export_df.to_csv(DATA_OUT / f"{sid}_export.csv", index=False)

    if (i + 1) % 100 == 0 or (i + 1) == n:
        print(f"  {i+1}/{n} industries processed…")

print("Done! All files written to website/data/")

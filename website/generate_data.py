"""
generate_data.py — Pre-process BLS B-1a data into JSON files for the website.

Run from the website/ directory:
    python generate_data.py

Outputs:
    data/table_data.json          — All 842 rows for main table
    data/{series_id}.json         — Per-industry chart data (8 charts)
    data/{series_id}_export.csv   — Per-industry downloadable CSV
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent          # BLS/
DATA_OUT = Path(__file__).parent / "data"
DATA_OUT.mkdir(exist_ok=True)

MAPPING_FILE = BASE / "b1a_mapping_with_parent.csv"
EMP_FILE     = BASE / "b1a_wide_seriesid.csv"
SHARES_FILE  = BASE / "b1a_employment_shares.csv"


# ── Helpers ────────────────────────────────────────────────────────────────────
def pct_of_score(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0:
        return None
    return round(float(percentileofscore(s, s.iloc[-1], kind="rank")), 1)


def to_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 6)
    except (TypeError, ValueError):
        return None


def month_label(dt) -> str:
    return dt.strftime("%b-%y")


def arr_to_list(arr, rd=6):
    """Convert numpy array to list with None for NaN/inf."""
    out = []
    for v in arr:
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            out.append(None)
        else:
            out.append(round(v, rd))
    return out


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data…")

mapping = pd.read_csv(MAPPING_FILE)
mapping = mapping.sort_values("row_order").reset_index(drop=True)

emp    = pd.read_csv(EMP_FILE,    index_col=0, parse_dates=True)
shares = pd.read_csv(SHARES_FILE, index_col=0, parse_dates=True)

all_dates = emp.index
date_strs = [d.strftime("%Y-%m-%d") for d in all_dates]
n_dates   = len(all_dates)

# Time index: integer months since Jan 2000 (= 0)
time_index = np.array([(d.year - 2000) * 12 + (d.month - 1) for d in all_dates])

# Pre-COVID mask: t ≤ Feb 2020
feb2020      = pd.Timestamp("2020-02-01")
pre_covid    = np.array(all_dates <= feb2020)          # bool array
feb2020_idx  = int(np.where(pre_covid)[0][-1])        # last pre-COVID position

# Trend fit window: Jan 2010 – Feb 2020
jan2010       = pd.Timestamp("2010-01-01")
fit_window    = np.array((all_dates >= jan2010) & (all_dates <= feb2020))  # bool array
fit_start_idx = int(np.where(fit_window)[0][0])       # first Jan-2010 position

MARCH2020_STR = "2020-03-01"

last_dt  = all_dates[-1]
prev_dt  = all_dates[-2]
last_lbl = month_label(last_dt)
prev_lbl = month_label(prev_dt)
today_str = date.today().strftime("%Y-%m-%d")


# ── Detrending functions ───────────────────────────────────────────────────────

def compute_trend(log_vals: np.ndarray):
    """
    Fit quadratic (degree-2) trend on Jan 2010 – Feb 2020.
    Extrapolates from Jan 2010 onwards; NaN before Jan 2010.
    Returns (trend_log, resid_log) numpy arrays.
    """
    t    = time_index
    mask = fit_window & ~np.isnan(log_vals)
    if mask.sum() < 10:
        nans = np.full(n_dates, np.nan)
        return nans, nans

    t_fit, y_fit = t[mask], log_vals[mask]
    X_fit = np.column_stack([np.ones(len(t_fit)), t_fit, t_fit**2])
    X_all = np.column_stack([np.ones(n_dates),   t,    t**2])

    coeffs    = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
    trend_all = X_all @ coeffs

    # NaN before Jan 2010
    trend = np.where(np.arange(n_dates) >= fit_start_idx, trend_all, np.nan)
    resid = log_vals - trend
    return trend, resid


def compute_hamilton(log_vals: np.ndarray):
    """
    Estimate Hamilton (2018) OLS on Jan 2010 – Feb 2020.
    Apply coefficients to actual lagged values at every date (no recursion).
    Returns (implied_log, resid_log) or (None, None) if insufficient data.
    """
    y = log_vals.copy()

    # Need ≥60 usable observations in fit window
    fit_count = int(np.sum(fit_window & ~np.isnan(y)))
    if fit_count < 60:
        return None, None

    # Build OLS on fit_window with all four lags available
    X_rows, y_rows = [], []
    for t_idx in range(27, n_dates):
        if not fit_window[t_idx]:
            continue
        if np.isnan(y[t_idx]):
            continue
        lags = y[t_idx-24], y[t_idx-25], y[t_idx-26], y[t_idx-27]
        if any(np.isnan(lags)):
            continue
        X_rows.append([1.0, *lags])
        y_rows.append(y[t_idx])

    if len(X_rows) < 10:
        return None, None

    coeffs = np.linalg.lstsq(np.array(X_rows), np.array(y_rows), rcond=None)[0]
    alpha, b1, b2, b3, b4 = coeffs

    # Apply directly using actual lagged values at every date (no recursion)
    # Only report from Jan 2010 onwards (matching estimation window)
    cf = np.full(n_dates, np.nan)
    for t_idx in range(27, n_dates):
        if t_idx < fit_start_idx:
            continue
        lags = y[t_idx-24], y[t_idx-25], y[t_idx-26], y[t_idx-27]
        if any(np.isnan(lags)):
            continue
        cf[t_idx] = alpha + b1*lags[0] + b2*lags[1] + b3*lags[2] + b4*lags[3]

    resid = y - cf
    return cf, resid


# ── Pre-compute detrended series for all industries ────────────────────────────
print("Computing detrended series for all industries…")

# DataFrames to collect results
trend_lvl_cf  = pd.DataFrame(index=all_dates)   # quadratic trend log level (Jan2010 fit)
trend_shr_cf  = pd.DataFrame(index=all_dates)   # quadratic trend log share (Jan2010 fit)
resid_tlvl    = pd.DataFrame(index=all_dates)
resid_tshr    = pd.DataFrame(index=all_dates)
ham_cf_lvl    = pd.DataFrame(index=all_dates)   # Hamilton cf log level
ham_cf_shr    = pd.DataFrame(index=all_dates)   # Hamilton cf log share
resid_hlvl    = pd.DataFrame(index=all_dates)
resid_hshr    = pd.DataFrame(index=all_dates)

n_series = len(mapping)
for loop_i, (_, mrow) in enumerate(mapping.iterrows()):
    sid = mrow["series_id"]

    # Log employment level
    if sid in emp.columns:
        ev = emp[sid].reindex(all_dates).values.astype(float)
    else:
        ev = np.full(n_dates, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_lvl = np.where(ev > 0, np.log(ev), np.nan)

    # Log employment share
    if sid in shares.columns:
        sv = shares[sid].reindex(all_dates).values.astype(float)
    else:
        sv = np.full(n_dates, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_shr = np.where(sv > 0, np.log(sv), np.nan)

    # Method 2 — Trend extrapolation (quadratic, Jan 2010–Feb 2020 fit window)
    t_lvl, r_tlvl = compute_trend(log_lvl)
    t_shr, r_tshr = compute_trend(log_shr)
    trend_lvl_cf[sid] = t_lvl
    trend_shr_cf[sid] = t_shr
    resid_tlvl[sid]   = r_tlvl
    resid_tshr[sid]   = r_tshr

    # Method 1 — Hamilton filter
    cf_lvl, r_hlvl = compute_hamilton(log_lvl)
    cf_shr, r_hshr = compute_hamilton(log_shr)
    ham_cf_lvl[sid] = cf_lvl if cf_lvl is not None else np.nan
    ham_cf_shr[sid] = cf_shr if cf_shr is not None else np.nan
    resid_hlvl[sid] = r_hlvl if r_hlvl is not None else np.nan
    resid_hshr[sid] = r_hshr if r_hshr is not None else np.nan

    if (loop_i + 1) % 100 == 0 or (loop_i + 1) == n_series:
        print(f"  {loop_i + 1}/{n_series} series computed…")


# ── Build table_data.json ──────────────────────────────────────────────────────
print("Building table_data.json…")

def last_nonnan(df, col):
    """Most recent non-NaN value from df[col], rounded to 3 dp."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    if len(s) == 0:
        return None
    v = float(s.iloc[-1])
    return None if (np.isnan(v) or np.isinf(v)) else round(v, 3)


rows = []
for _, mrow in mapping.iterrows():
    sid        = mrow["series_id"]
    name       = mrow["industry_name"]
    lvl        = int(mrow["display_level"])
    parent_sid = mrow["parent_series_id"]

    parent_row = mapping.loc[mapping["series_id"] == parent_sid]
    denom_name = parent_row["industry_name"].iloc[0] if len(parent_row) > 0 else name

    emp_col    = emp[sid] if sid in emp.columns else None
    emp_recent = to_float(emp_col.iloc[-1]) if emp_col is not None else None
    emp_prev   = to_float(emp_col.iloc[-2]) if emp_col is not None else None

    share_val = share_pct = None
    if sid in shares.columns:
        s_series = shares[sid]
        s_dropna = s_series.dropna()
        if len(s_dropna) > 0:
            share_val = to_float(s_dropna.iloc[-1])
            share_pct = pct_of_score(s_series)

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
        "dev_trend_level":      last_nonnan(resid_tlvl, sid),
        "dev_trend_share":      last_nonnan(resid_tshr, sid),
        "dev_hamilton_level":   last_nonnan(resid_hlvl, sid),
        "dev_hamilton_share":   last_nonnan(resid_hshr, sid),
    })

with open(DATA_OUT / "table_data.json", "w") as f:
    json.dump({"rows": rows, "last_label": last_lbl, "prev_label": prev_lbl}, f)

print(f"  → {len(rows)} rows written to table_data.json")


# ── Build per-industry JSON + CSV ──────────────────────────────────────────────
print("Building per-industry files…")

n = len(mapping)
for loop_i, (_, mrow) in enumerate(mapping.iterrows()):
    sid        = mrow["series_id"]
    name       = mrow["industry_name"]
    parent_sid = mrow["parent_series_id"]

    parent_row = mapping.loc[mapping["series_id"] == parent_sid]
    denom_name = parent_row["industry_name"].iloc[0] if len(parent_row) > 0 else name

    # Raw values
    ev = emp[sid].reindex(all_dates).values.astype(float)   if sid in emp.columns    else np.full(n_dates, np.nan)
    sv = shares[sid].reindex(all_dates).values.astype(float) if sid in shares.columns else np.full(n_dates, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_lvl = np.where(ev > 0, np.log(ev), np.nan)
        log_shr = np.where(sv > 0, np.log(sv), np.nan)

    # Trend CFs (log space)
    t_lvl_v  = trend_lvl_cf[sid].values
    t_shr_v  = trend_shr_cf[sid].values
    h_lvl_v  = ham_cf_lvl[sid].values
    h_shr_v  = ham_cf_shr[sid].values
    r_tlvl_v = resid_tlvl[sid].values
    r_tshr_v = resid_tshr[sid].values
    r_hlvl_v = resid_hlvl[sid].values
    r_hshr_v = resid_hshr[sid].values

    with np.errstate(invalid="ignore", divide="ignore"):
        trend_lvl_impl  = np.where(~np.isnan(t_lvl_v), np.exp(t_lvl_v), np.nan)
        trend_shr_impl  = np.where(~np.isnan(t_shr_v), np.exp(t_shr_v), np.nan)  # decimal
        ham_lvl_impl    = np.where(~np.isnan(h_lvl_v), np.exp(h_lvl_v), np.nan)
        ham_shr_impl    = np.where(~np.isnan(h_shr_v), np.exp(h_shr_v), np.nan)  # decimal

    industry_data = {
        "series_id":            sid,
        "industry_name":        name,
        "denom_name":           denom_name,
        "parent_series_id":     parent_sid,
        "dates":                date_strs,
        "march2020":            MARCH2020_STR,
        # Actual series
        "emp_level":            arr_to_list(ev,         2),   # thousands
        "emp_share_pct":        arr_to_list(sv * 100,   4),   # percentage points
        # Method 2 — Trend extrapolation
        "trend_level_cf":       arr_to_list(trend_lvl_impl,   2),  # trend-implied emp (thousands)
        "trend_share_cf_pct":   arr_to_list(trend_shr_impl * 100, 4),  # trend-implied share (ppt)
        "resid_trend_level":    arr_to_list(r_tlvl_v,  6),
        "resid_trend_share":    arr_to_list(r_tshr_v,  6),
        # Method 1 — Hamilton filter
        "hamilton_cf_level":    arr_to_list(ham_lvl_impl,     2),  # Hamilton-implied emp (thousands)
        "hamilton_cf_share_pct":arr_to_list(ham_shr_impl * 100, 4),
        "resid_hamilton_level": arr_to_list(r_hlvl_v,  6),
        "resid_hamilton_share": arr_to_list(r_hshr_v,  6),
        "csv_filename":         f"{sid}_export.csv",
    }

    with open(DATA_OUT / f"{sid}.json", "w") as f:
        json.dump(industry_data, f)

    # ── Per-industry export CSV ──────────────────────────────────────────────
    export_df = pd.DataFrame({
        "date":                         all_dates.strftime("%Y-%m"),
        "employment_level":             ev,
        "employment_share":             sv,
        "log_level":                    log_lvl,
        "log_share":                    log_shr,
        "trend_level":                  t_lvl_v,
        "trend_share":                  t_shr_v,
        "predicted_level_trend":        trend_lvl_impl,
        "predicted_share_trend":        trend_shr_impl,
        "resid_log_level_linear_trend": r_tlvl_v,
        "resid_log_share_quad_trend":   r_tshr_v,
        "hamilton_cf_log_level":        h_lvl_v,
        "hamilton_cf_log_share":        h_shr_v,
        "predicted_level_hamilton":     ham_lvl_impl,
        "predicted_share_hamilton":     ham_shr_impl,
        "resid_log_level_hamilton":     r_hlvl_v,
        "resid_log_share_hamilton":     r_hshr_v,
    })

    csv_path = DATA_OUT / f"{sid}_export.csv"
    with open(csv_path, "w") as f:
        f.write(f"# Industry: {name}\n")
        f.write(f"# Series ID: {sid}\n")
        f.write(f"# Denominator: {denom_name} ({parent_sid})\n")
        f.write(f"# Generated: {today_str}\n")
        export_df.to_csv(f, index=False)

    if (loop_i + 1) % 100 == 0 or (loop_i + 1) == n:
        print(f"  {loop_i + 1}/{n} industries processed…")

print("Done! All files written to website/data/")

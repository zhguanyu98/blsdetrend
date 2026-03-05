"""
generate_data.py — Pre-process BLS B-1a data into JSON files for the website.

Run from the website/ directory:
    python generate_data.py

Outputs:
    data/table_data.json          — All 842 rows for main table
    data/{series_id}.json         — Per-industry chart data (6 charts)
    data/{series_id}_export.csv   — Per-industry downloadable CSV

Detrending methods (all fit on Jan 2010 – Feb 2020, reported from Jan 2010):
  1. Log-linear level : log(emp) ~ linear trend → residual = log deviation
  2. Log-linear share : log(share) ~ linear trend → residual = log deviation
  3. Raw-share trend  : share ~ quadratic (fallback to linear if t² p>0.05)
                        → residual = (actual − predicted) / predicted
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, t as t_dist

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
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
    out = []
    for v in arr:
        v = float(v)
        out.append(None if (np.isnan(v) or np.isinf(v)) else round(v, rd))
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

# Fit window: Jan 2010 – Feb 2020
jan2010       = pd.Timestamp("2010-01-01")
feb2020       = pd.Timestamp("2020-02-01")
fit_window    = np.array((all_dates >= jan2010) & (all_dates <= feb2020))
fit_start_idx = int(np.where(fit_window)[0][0])   # first Jan-2010 position

MARCH2020_STR = "2020-03-01"

last_dt   = all_dates[-1]
prev_dt   = all_dates[-2]
last_lbl  = month_label(last_dt)
prev_lbl  = month_label(prev_dt)
today_str = date.today().strftime("%Y-%m-%d")


# ── Detrending functions ───────────────────────────────────────────────────────

def fit_log_linear(log_vals: np.ndarray):
    """
    Fit LINEAR trend on log values over Jan 2010 – Feb 2020.
    Returns (trend_log, resid_log); both NaN before Jan 2010.
    """
    t    = time_index
    mask = fit_window & ~np.isnan(log_vals)
    if mask.sum() < 10:
        nans = np.full(n_dates, np.nan)
        return nans, nans

    t_fit, y_fit = t[mask], log_vals[mask]
    X_fit = np.column_stack([np.ones(len(t_fit)), t_fit])
    X_all = np.column_stack([np.ones(n_dates),   t])

    coeffs    = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
    trend_all = X_all @ coeffs

    trend = np.where(np.arange(n_dates) >= fit_start_idx, trend_all, np.nan)
    resid = log_vals - trend
    return trend, resid


def fit_raw_quadratic(raw_vals: np.ndarray):
    """
    Fit QUADRATIC trend on raw values over Jan 2010 – Feb 2020.
    If the t² coefficient p-value > 0.05, fall back to linear.
    Returns (trend_raw, resid_pct) where resid_pct = (actual−predicted)/predicted;
    both NaN before Jan 2010.
    """
    t    = time_index
    mask = fit_window & ~np.isnan(raw_vals)
    if mask.sum() < 10:
        nans = np.full(n_dates, np.nan)
        return nans, nans

    t_fit, y_fit = t[mask], raw_vals[mask]
    n_obs = len(y_fit)

    # Try quadratic
    X2    = np.column_stack([np.ones(n_obs), t_fit, t_fit**2])
    c2    = np.linalg.lstsq(X2, y_fit, rcond=None)[0]
    res2  = y_fit - X2 @ c2
    s2    = np.dot(res2, res2) / (n_obs - 3)
    try:
        se_q  = np.sqrt(max(s2 * np.linalg.inv(X2.T @ X2)[2, 2], 0.0))
        p_q   = 2 * t_dist.sf(abs(c2[2] / se_q), df=n_obs - 3) if se_q > 0 else 1.0
    except np.linalg.LinAlgError:
        p_q = 1.0

    if p_q <= 0.05:
        X_all     = np.column_stack([np.ones(n_dates), t, t**2])
        trend_all = X_all @ c2
    else:
        X1        = np.column_stack([np.ones(n_obs), t_fit])
        c1        = np.linalg.lstsq(X1, y_fit, rcond=None)[0]
        X_all     = np.column_stack([np.ones(n_dates), t])
        trend_all = X_all @ c1

    trend = np.where(np.arange(n_dates) >= fit_start_idx, trend_all, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        resid_pct = np.where(trend != 0, (raw_vals - trend) / trend, np.nan)
        resid_pct = np.where(np.arange(n_dates) >= fit_start_idx, resid_pct, np.nan)
    return trend, resid_pct


# ── Pre-compute detrended series for all industries ────────────────────────────
print("Computing detrended series for all industries…")

# Store results column-by-column; convert to arrays dict at end
results = {}   # sid → dict of arrays

n_series = len(mapping)
for loop_i, (_, mrow) in enumerate(mapping.iterrows()):
    sid = mrow["series_id"]

    ev = emp[sid].reindex(all_dates).values.astype(float)    if sid in emp.columns    else np.full(n_dates, np.nan)
    sv = shares[sid].reindex(all_dates).values.astype(float) if sid in shares.columns else np.full(n_dates, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_lvl = np.where(ev > 0, np.log(ev), np.nan)
        log_shr = np.where(sv > 0, np.log(sv), np.nan)

    t_ll,  r_ll  = fit_log_linear(log_lvl)   # method 1: log-linear level
    t_ls,  r_ls  = fit_log_linear(log_shr)   # method 2: log-linear share
    t_rs,  r_rs  = fit_raw_quadratic(sv)     # method 3: raw share quad/linear

    results[sid] = dict(
        ev=ev, sv=sv, log_lvl=log_lvl, log_shr=log_shr,
        trend_ll=t_ll, resid_ll=r_ll,
        trend_ls=t_ls, resid_ls=r_ls,
        trend_rs=t_rs, resid_rs=r_rs,
    )

    if (loop_i + 1) % 100 == 0 or (loop_i + 1) == n_series:
        print(f"  {loop_i + 1}/{n_series} series computed…")


# ── Build table_data.json ──────────────────────────────────────────────────────
print("Building table_data.json…")


def last_nonnan3(arr):
    """Most recent non-NaN/inf value, rounded to 3 dp."""
    for v in reversed(arr):
        if v is not None and not np.isnan(v) and not np.isinf(v):
            return round(float(v), 3)
    return None


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

    r = results.get(sid, {})
    rows.append({
        "series_id":        sid,
        "industry_name":    name,
        "display_level":    lvl,
        "emp_recent":       emp_recent,
        "emp_recent_label": last_lbl,
        "emp_prev":         emp_prev,
        "emp_prev_label":   prev_lbl,
        "share":            share_val,
        "share_pct":        share_pct,
        "denom_name":       denom_name,
        "dev_log_level":    last_nonnan3(r.get("resid_ll", [])),
        "dev_log_share":    last_nonnan3(r.get("resid_ls", [])),
        "dev_raw_share_pct":last_nonnan3(r.get("resid_rs", [])),
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

    r = results[sid]
    ev, sv         = r["ev"], r["sv"]
    t_ll, r_ll     = r["trend_ll"], r["resid_ll"]
    t_ls, r_ls     = r["trend_ls"], r["resid_ls"]
    t_rs, r_rs     = r["trend_rs"], r["resid_rs"]

    with np.errstate(invalid="ignore", divide="ignore"):
        trend_ll_impl = np.where(~np.isnan(t_ll), np.exp(t_ll), np.nan)   # thousands
        trend_ls_impl = np.where(~np.isnan(t_ls), np.exp(t_ls) * 100, np.nan)  # ppt

    industry_data = {
        "series_id":           sid,
        "industry_name":       name,
        "denom_name":          denom_name,
        "parent_series_id":    parent_sid,
        "dates":               date_strs,
        "march2020":           MARCH2020_STR,
        # Chart 1 & 2 — Log-linear level
        "emp_level":           arr_to_list(ev,              2),   # actual (thousands)
        "trend_ll_level":      arr_to_list(trend_ll_impl,   2),   # trend-implied (thousands)
        "resid_ll_level":      arr_to_list(r_ll,            6),   # log deviation
        # Chart 3 & 4 — Log-linear share
        "emp_share_pct":       arr_to_list(sv * 100,        4),   # actual share (%)
        "trend_ls_share_pct":  arr_to_list(trend_ls_impl,   4),   # trend-implied share (%)
        "resid_ls_share":      arr_to_list(r_ls,            6),   # log deviation
        # Chart 5 & 6 — Raw-share quadratic/linear trend
        "trend_rs_share_pct":  arr_to_list(t_rs * 100,      4),   # trend-implied share (%)
        "resid_rs_pct":        arr_to_list(r_rs,            6),   # (actual−pred)/pred
        "csv_filename":        f"{sid}_export.csv",
    }

    with open(DATA_OUT / f"{sid}.json", "w") as f:
        json.dump(industry_data, f)

    # ── Per-industry export CSV ──────────────────────────────────────────────
    export_df = pd.DataFrame({
        "date":                          all_dates.strftime("%Y-%m"),
        "employment_level":              ev,
        "employment_share":              sv,
        "log_level":                     r["log_lvl"],
        "log_share":                     r["log_shr"],
        "trend_log_level":               t_ll,
        "predicted_level":               trend_ll_impl,
        "resid_log_level":               r_ll,
        "trend_log_share":               t_ls,
        "predicted_share_log_trend":     np.where(~np.isnan(t_ls), np.exp(t_ls), np.nan),
        "resid_log_share":               r_ls,
        "trend_raw_share":               t_rs,
        "resid_raw_share_pct_dev":       r_rs,
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

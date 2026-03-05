"""
generate_data.py — Pre-process BLS B-1a data into JSON files for the website.

Run from the website/ directory:
    python generate_data.py

Outputs:
    data/table_data.json          — All 842 rows + 5 denominator options for main table
    data/{series_id}.json         — Per-industry chart data (6 charts, 5 denom options)
    data/{series_id}_export.csv   — Per-industry downloadable CSV with all 5 options

Detrending methods (all fit on Jan 2010 – Feb 2020, reported from Jan 2010):
  Level:   log(emp) ~ linear trend         → residual = log deviation   (all denom options)
  Share 1: log(share) ~ linear trend       → residual = log deviation   (per denom option)
  Share 2: share ~ quadratic/linear trend  → residual = (actual−pred)/pred (per denom option)
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

MAPPING_FILE = BASE / "b1a_mapping_with_denominators.csv"
EMP_FILE     = BASE / "b1a_wide_seriesid.csv"

OPT_LABELS = {
    1: "Level 4 parent (default)",
    2: "Level 3 parent",
    3: "Level 2 parent",
    4: "Goods/Service-providing total",
    5: "Total private / Total government",
}


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

emp       = pd.read_csv(EMP_FILE, index_col=0, parse_dates=True)
all_dates = emp.index
date_strs = [d.strftime("%Y-%m-%d") for d in all_dates]
n_dates   = len(all_dates)

# Time index: integer months since Jan 2000 (= 0)
time_index = np.array([(d.year - 2000) * 12 + (d.month - 1) for d in all_dates])

# Fit window: Jan 2010 – Feb 2020
jan2010       = pd.Timestamp("2010-01-01")
feb2020       = pd.Timestamp("2020-02-01")
fit_window    = np.array((all_dates >= jan2010) & (all_dates <= feb2020))
fit_start_idx = int(np.where(fit_window)[0][0])

MARCH2020_STR = "2020-03-01"
last_lbl  = month_label(all_dates[-1])
prev_lbl  = month_label(all_dates[-2])
today_str = date.today().strftime("%Y-%m-%d")

# Look-up dicts from mapping
id_to_name = dict(zip(mapping["series_id"], mapping["industry_name"]))


# ── Detrending functions ───────────────────────────────────────────────────────

def fit_log_linear(log_vals: np.ndarray):
    """Linear trend on log values, Jan 2010–Feb 2020. NaN before Jan 2010."""
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
    Quadratic trend on raw values, Jan 2010–Feb 2020.
    Falls back to linear if t² coefficient p-value > 0.05.
    Returns (trend_raw, resid_pct) where resid_pct = (actual−pred)/pred.
    NaN before Jan 2010.
    """
    t    = time_index
    mask = fit_window & ~np.isnan(raw_vals)
    if mask.sum() < 10:
        nans = np.full(n_dates, np.nan)
        return nans, nans
    t_fit, y_fit = t[mask], raw_vals[mask]
    n_obs = len(y_fit)

    X2   = np.column_stack([np.ones(n_obs), t_fit, t_fit**2])
    c2   = np.linalg.lstsq(X2, y_fit, rcond=None)[0]
    res2 = y_fit - X2 @ c2
    s2   = np.dot(res2, res2) / (n_obs - 3)
    try:
        se_q = np.sqrt(max(s2 * np.linalg.inv(X2.T @ X2)[2, 2], 0.0))
        p_q  = 2 * t_dist.sf(abs(c2[2] / se_q), df=n_obs - 3) if se_q > 0 else 1.0
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
        resid_pct = np.where((trend != 0) & (np.arange(n_dates) >= fit_start_idx),
                             (raw_vals - trend) / trend, np.nan)
    return trend, resid_pct


# ── Pre-compute all series ─────────────────────────────────────────────────────
print("Computing detrended series for all industries (5 denominator options)…")

# results[sid] = {
#   ev, log_lvl, trend_ll, resid_ll,          ← level (option-independent)
#   opts: {1: {sv, trend_ls, resid_ls, trend_rs, resid_rs, denom_id}, ...}
# }
results = {}

n_series = len(mapping)
for loop_i, (_, mrow) in enumerate(mapping.iterrows()):
    sid = mrow["series_id"]

    ev = emp[sid].reindex(all_dates).values.astype(float) if sid in emp.columns else np.full(n_dates, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_lvl = np.where(ev > 0, np.log(ev), np.nan)

    trend_ll, resid_ll = fit_log_linear(log_lvl)

    opts = {}
    for opt in range(1, 6):
        denom_id = mrow[f"denominator_opt{opt}"]
        dv = emp[denom_id].reindex(all_dates).values.astype(float) if denom_id in emp.columns else np.full(n_dates, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            sv      = np.where((ev > 0) & (dv > 0), ev / dv, np.nan)
            log_shr = np.where(sv > 0, np.log(sv), np.nan)

        trend_ls, resid_ls = fit_log_linear(log_shr)
        trend_rs, resid_rs = fit_raw_quadratic(sv)

        opts[opt] = dict(sv=sv, trend_ls=trend_ls, resid_ls=resid_ls,
                         trend_rs=trend_rs, resid_rs=resid_rs,
                         denom_id=denom_id)

    results[sid] = dict(ev=ev, log_lvl=log_lvl, trend_ll=trend_ll, resid_ll=resid_ll, opts=opts)

    if (loop_i + 1) % 100 == 0 or (loop_i + 1) == n_series:
        print(f"  {loop_i + 1}/{n_series} series computed…")


# ── Build table_data.json ──────────────────────────────────────────────────────
print("Building table_data.json…")


def last_nonnan3(arr):
    for v in reversed(arr):
        if v is not None and not np.isnan(float(v)) and not np.isinf(float(v)):
            return round(float(v), 3)
    return None


rows = []
for _, mrow in mapping.iterrows():
    sid        = mrow["series_id"]
    name       = mrow["industry_name"]
    lvl        = int(mrow["display_level"])
    parent_sid = mrow["parent_series_id"]

    parent_name = id_to_name.get(parent_sid, name)

    emp_col    = emp[sid] if sid in emp.columns else None
    emp_recent = to_float(emp_col.iloc[-1]) if emp_col is not None else None
    emp_prev   = to_float(emp_col.iloc[-2]) if emp_col is not None else None

    r = results[sid]

    # Build per-option summary for the table
    opts_summary = {}
    for opt in range(1, 6):
        o       = r["opts"][opt]
        sv_ser  = pd.Series(o["sv"], index=all_dates)
        s_drop  = sv_ser.dropna()
        share_val = to_float(s_drop.iloc[-1]) if len(s_drop) > 0 else None
        share_pct = pct_of_score(sv_ser)

        opts_summary[str(opt)] = {
            "share":            share_val,
            "share_pct":        share_pct,
            "dev_log_share":    last_nonnan3(o["resid_ls"]),
            "dev_raw_share_pct":last_nonnan3(o["resid_rs"]),
            "denom_name":       id_to_name.get(o["denom_id"], ""),
        }

    rows.append({
        "series_id":        sid,
        "industry_name":    name,
        "display_level":    lvl,
        "emp_recent":       emp_recent,
        "emp_recent_label": last_lbl,
        "emp_prev":         emp_prev,
        "emp_prev_label":   prev_lbl,
        "dev_log_level":    last_nonnan3(r["resid_ll"]),
        "opts":             opts_summary,
    })

with open(DATA_OUT / "table_data.json", "w") as f:
    json.dump({"rows": rows, "last_label": last_lbl, "prev_label": prev_lbl,
               "opt_labels": {str(k): v for k, v in OPT_LABELS.items()}}, f)

print(f"  → {len(rows)} rows written to table_data.json")


# ── Build per-industry JSON + CSV ──────────────────────────────────────────────
print("Building per-industry files…")

n = len(mapping)
for loop_i, (_, mrow) in enumerate(mapping.iterrows()):
    sid        = mrow["series_id"]
    name       = mrow["industry_name"]
    parent_sid = mrow["parent_series_id"]
    parent_name = id_to_name.get(parent_sid, name)

    r = results[sid]
    ev       = r["ev"]
    trend_ll = r["trend_ll"]
    resid_ll = r["resid_ll"]

    with np.errstate(invalid="ignore", divide="ignore"):
        trend_ll_impl = np.where(~np.isnan(trend_ll), np.exp(trend_ll), np.nan)

    # Per-option share data for charts 3–6
    options_json = {}
    for opt in range(1, 6):
        o       = r["opts"][opt]
        sv      = o["sv"]
        t_ls    = o["trend_ls"]
        r_ls    = o["resid_ls"]
        t_rs    = o["trend_rs"]
        r_rs    = o["resid_rs"]
        denom_id = o["denom_id"]

        with np.errstate(invalid="ignore", divide="ignore"):
            t_ls_impl = np.where(~np.isnan(t_ls), np.exp(t_ls) * 100, np.nan)

        options_json[str(opt)] = {
            "denom_name":         id_to_name.get(denom_id, ""),
            "denom_id":           denom_id,
            "emp_share_pct":      arr_to_list(sv * 100,      4),
            "trend_ls_share_pct": arr_to_list(t_ls_impl,     4),
            "resid_ls_share":     arr_to_list(r_ls,          6),
            "trend_rs_share_pct": arr_to_list(t_rs * 100,    4),
            "resid_rs_pct":       arr_to_list(r_rs,          6),
        }

    industry_data = {
        "series_id":        sid,
        "industry_name":    name,
        "parent_series_id": parent_sid,
        "dates":            date_strs,
        "march2020":        MARCH2020_STR,
        # Charts 1 & 2: log-linear level (same for all denom options)
        "emp_level":        arr_to_list(ev,            2),
        "trend_ll_level":   arr_to_list(trend_ll_impl, 2),
        "resid_ll_level":   arr_to_list(resid_ll,      6),
        # Charts 3–6: per-denominator-option share data
        "options":          options_json,
        "opt_labels":       {str(k): v for k, v in OPT_LABELS.items()},
        "csv_filename":     f"{sid}_export.csv",
    }

    with open(DATA_OUT / f"{sid}.json", "w") as f:
        json.dump(industry_data, f)

    # ── Per-industry export CSV ──────────────────────────────────────────────
    rows_csv = {"date": all_dates.strftime("%Y-%m"), "employment_level": ev,
                "log_level": r["log_lvl"], "trend_log_level": trend_ll,
                "predicted_level": trend_ll_impl, "resid_log_level": resid_ll}

    for opt in range(1, 6):
        o = r["opts"][opt]
        denom_id = o["denom_id"]
        dv = emp[denom_id].reindex(all_dates).values.astype(float) if denom_id in emp.columns else np.full(n_dates, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_shr = np.where(o["sv"] > 0, np.log(o["sv"]), np.nan)
            pred_ls = np.where(~np.isnan(o["trend_ls"]), np.exp(o["trend_ls"]), np.nan)

        rows_csv[f"share_opt{opt}"]              = o["sv"]
        rows_csv[f"denom_employment_opt{opt}"]   = dv
        rows_csv[f"log_share_opt{opt}"]          = log_shr
        rows_csv[f"trend_log_share_opt{opt}"]    = o["trend_ls"]
        rows_csv[f"predicted_share_opt{opt}"]    = pred_ls
        rows_csv[f"resid_log_share_opt{opt}"]    = o["resid_ls"]
        rows_csv[f"trend_raw_share_opt{opt}"]    = o["trend_rs"]
        rows_csv[f"resid_raw_pct_dev_opt{opt}"]  = o["resid_rs"]

    export_df = pd.DataFrame(rows_csv)
    csv_path  = DATA_OUT / f"{sid}_export.csv"
    with open(csv_path, "w") as f:
        f.write(f"# Industry: {name}\n# Series ID: {sid}\n")
        f.write(f"# Parent series: {parent_name} ({parent_sid})\n")
        f.write(f"# Generated: {today_str}\n")
        f.write("# Denominator options:\n")
        for opt in range(1, 6):
            f.write(f"#   opt{opt}: {OPT_LABELS[opt]} = {id_to_name.get(results[sid]['opts'][opt]['denom_id'], '?')}\n")
        export_df.to_csv(f, index=False)

    if (loop_i + 1) % 100 == 0 or (loop_i + 1) == n:
        print(f"  {loop_i + 1}/{n} industries processed…")

print("Done! All files written to website/data/")

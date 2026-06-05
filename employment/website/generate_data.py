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
from scipy.stats import percentileofscore
from statsmodels.tsa.filters.hp_filter import hpfilter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
DATA_OUT = Path(__file__).parent / "data"
DATA_OUT.mkdir(exist_ok=True)

MAPPING_FILE    = BASE / "b1a_mapping_with_denominators.csv"
EMP_FILE        = BASE / "b1a_wide_seriesid.csv"
EMP_PREREVIS    = BASE / "b1a_wide_seriesid_pre_revision_2026_04.csv"
APR2026_DATE    = pd.Timestamp("2026-04-01")

OPT_LABELS = {
    1: "Level 4 parent (default)",
    2: "Level 3 parent",
    3: "Level 2 parent",
    4: "Goods/Service-providing total",
    5: "Total private / Total government",
    6: "Total nonfarm",
}

DENOM_OPT6_ID = "CES0000000001"   # Total nonfarm — same for every industry
SNAP_LAGS     = [1, 3, 6, 9, 12, 15]  # month lags stored for direction / 2nd-deriv


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


def last_valid_idx_np(arr: np.ndarray) -> int:
    """Return index of last non-NaN, non-inf value, or -1 if none."""
    mask = np.isfinite(arr)
    idxs = np.where(mask)[0]
    return int(idxs[-1]) if len(idxs) > 0 else -1


def scalar_at(arr: np.ndarray, idx: int):
    """Return arr[idx] rounded to 6dp, or None if out of range / not finite."""
    if idx < 0 or idx >= len(arr):
        return None
    v = float(arr[idx])
    return None if not np.isfinite(v) else round(v, 6)


def peak_covid_scalar(arr: np.ndarray, idxs: list):
    """Return the value at the month with the highest absolute deviation
    among the given indices (Mar/Apr/May 2020). Ties broken by later month."""
    best_val, best_abs = None, -1.0
    for idx in idxs:
        v = scalar_at(arr, idx)
        if v is not None and abs(v) > best_abs:
            best_abs = abs(v)
            best_val = v
    return best_val


def covid_share_scalar(sv: np.ndarray, covid_idxs: list):
    """Return the share level at the month of max |month-over-month share change|
    in Mar/Apr/May 2020 (dynamically identified shock month)."""
    best_val, best_abs = None, -1.0
    for idx in covid_idxs:
        if idx < 1 or idx >= len(sv):
            continue
        v_curr = sv[idx]
        v_prev = sv[idx - 1]
        if not (np.isfinite(v_curr) and np.isfinite(v_prev)):
            continue
        chg = abs(float(v_curr) - float(v_prev))
        if chg > best_abs:
            best_abs = chg
            best_val = round(float(v_curr), 6)
    return best_val


def lag_snapshots(arr: np.ndarray, lv_idx: int) -> dict:
    """Return {_1m, _3m, _6m, _9m, _12m, _15m} scalars relative to lv_idx."""
    return {f"_{n}m": scalar_at(arr, lv_idx - n) if lv_idx >= n else None
            for n in SNAP_LAGS}


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

# Load pre-revision April 2026 snapshot for revision comparison
if EMP_PREREVIS.exists():
    emp_pre = pd.read_csv(EMP_PREREVIS, index_col=0, parse_dates=True)
    apr2026_pre = emp_pre.loc[APR2026_DATE] if APR2026_DATE in emp_pre.index else None
else:
    apr2026_pre = None

apr2026_revised = emp.loc[APR2026_DATE] if APR2026_DATE in emp.index else None
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

COVID_MONTHS = ["2020-03-01", "2020-04-01", "2020-05-01"]
covid_idxs = [date_strs.index(m) for m in COVID_MONTHS if m in date_strs]

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


FEB2020_TIDX = (2020 - 2000) * 12 + (2 - 1)  # time_index value for Feb 2020 = 241


def fit_raw_linear(raw_vals: np.ndarray):
    """
    Linear trend on raw values, Jan 2010–Feb 2020.
    Returns (trend_raw, resid_pct) where resid_pct = (actual−pred)/pred.
    NaN before Jan 2010.
    """
    t    = time_index
    mask = fit_window & ~np.isnan(raw_vals)
    if mask.sum() < 10:
        nans = np.full(n_dates, np.nan)
        return nans, nans
    t_fit, y_fit = t[mask], raw_vals[mask]
    X_fit     = np.column_stack([np.ones(len(t_fit)), t_fit])
    X_all     = np.column_stack([np.ones(n_dates), t])
    coeffs    = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
    trend_all = X_all @ coeffs
    trend = np.where(np.arange(n_dates) >= fit_start_idx, trend_all, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        resid_pct = np.where((trend != 0) & (np.arange(n_dates) >= fit_start_idx),
                             (raw_vals - trend) / trend, np.nan)
    return trend, resid_pct


def fit_hp_log_share(log_shr: np.ndarray):
    """
    HP filter (λ=129,600) on log share, Jan 2010–Feb 2020 window.
    Extrapolates post-Feb 2020 as a straight line using the last-24-month slope.
    Returns (hp_cf, resid_hp):
      - hp_cf: HP trend pre-COVID, linear extrapolation post-COVID (NaN before Jan 2010)
      - resid_hp: log_shr - hp_cf
    Returns (NaN, NaN) arrays if fewer than 60 pre-COVID months available.
    """
    mask = fit_window & ~np.isnan(log_shr)
    n_fit = int(mask.sum())
    if n_fit < 60:
        nans = np.full(n_dates, np.nan)
        return nans, nans

    fit_idxs = np.where(mask)[0]
    y_fit = log_shr[fit_idxs]
    _, hp_trend_vals = hpfilter(y_fit, lamb=129600)
    hp_trend_vals = np.asarray(hp_trend_vals, dtype=float)

    hp_cf = np.full(n_dates, np.nan)
    hp_cf[fit_idxs] = hp_trend_vals

    # Slope from last 24 months (robust to HP endpoint bias)
    n_slope = min(24, len(hp_trend_vals) - 1)
    slope = (hp_trend_vals[-1] - hp_trend_vals[-1 - n_slope]) / n_slope

    feb2020_val = float(hp_trend_vals[-1])
    post_mask = time_index > FEB2020_TIDX
    hp_cf[post_mask] = feb2020_val + slope * (time_index[post_mask] - FEB2020_TIDX)

    resid_hp = np.where(~np.isnan(hp_cf), log_shr - hp_cf, np.nan)
    return hp_cf, resid_hp


# ── Pre-compute all series ─────────────────────────────────────────────────────
print("Computing detrended series for all industries (6 denominator options)…")

# results[sid] = {
#   ev, log_lvl, trend_ll, resid_ll,          ← level (option-independent)
#   opts: {1..6: {sv, trend_ls, resid_ls, trend_rs, resid_rs,
#                  trend_hp, resid_hp, denom_id}, ...}
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
    for opt in range(1, 7):
        denom_id = DENOM_OPT6_ID if opt == 6 else mrow[f"denominator_opt{opt}"]
        dv = emp[denom_id].reindex(all_dates).values.astype(float) if denom_id in emp.columns else np.full(n_dates, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            sv      = np.where((ev > 0) & (dv > 0), ev / dv, np.nan)
            log_shr = np.where(sv > 0, np.log(sv), np.nan)

        trend_ls, resid_ls = fit_log_linear(log_shr)
        trend_rs, resid_rs = fit_raw_linear(sv)
        trend_hp, resid_hp = fit_hp_log_share(log_shr)

        opts[opt] = dict(sv=sv, trend_ls=trend_ls, resid_ls=resid_ls,
                         trend_rs=trend_rs, resid_rs=resid_rs,
                         trend_hp=trend_hp, resid_hp=resid_hp,
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

    # Level snapshots (option-independent)
    resid_ll_np  = r["resid_ll"]
    lv_ll        = last_valid_idx_np(resid_ll_np)
    ll_snaps     = lag_snapshots(resid_ll_np, lv_ll)
    ll_covid     = peak_covid_scalar(resid_ll_np, covid_idxs)

    # Build per-option summary for the table
    opts_summary = {}
    for opt in range(1, 7):
        o       = r["opts"][opt]
        sv_ser  = pd.Series(o["sv"], index=all_dates)
        s_drop  = sv_ser.dropna()
        share_val = to_float(s_drop.iloc[-1]) if len(s_drop) > 0 else None
        share_pct = pct_of_score(sv_ser)

        resid_ls_np = o["resid_ls"]
        resid_hp_np = o["resid_hp"]
        resid_rs_np = o["resid_rs"]
        sv_np       = o["sv"]

        ls_snaps = lag_snapshots(resid_ls_np, last_valid_idx_np(resid_ls_np))
        hp_snaps = lag_snapshots(resid_hp_np, last_valid_idx_np(resid_hp_np))
        rs_snaps = lag_snapshots(resid_rs_np, last_valid_idx_np(resid_rs_np))
        sv_snaps = lag_snapshots(sv_np,       last_valid_idx_np(sv_np))
        sv_covid = covid_share_scalar(sv_np, covid_idxs)

        opts_summary[str(opt)] = {
            "share":       share_val,
            "share_pct":   share_pct,
            "denom_name":  id_to_name.get(o["denom_id"], ""),
            # share level lags (for direction / 2nd-deriv / COVID)
            **{f"share{k}": sv_snaps[k] for k in sv_snaps},
            "share_covid": sv_covid,
            # log-linear share — current + lags + covid
            "dev_log_share":        last_nonnan3(resid_ls_np),
            **{f"dev_log_share{k}":        ls_snaps[k] for k in ls_snaps},
            "dev_log_share_covid":  peak_covid_scalar(resid_ls_np, covid_idxs),
            # HP filter share
            "dev_log_share_hp":        last_nonnan3(resid_hp_np),
            **{f"dev_log_share_hp{k}":     hp_snaps[k] for k in hp_snaps},
            "dev_log_share_hp_covid":  peak_covid_scalar(resid_hp_np, covid_idxs),
            # raw-linear share
            "dev_raw_share_pct":        last_nonnan3(resid_rs_np),
            **{f"dev_raw_share_pct{k}":    rs_snaps[k] for k in rs_snaps},
            "dev_raw_share_pct_covid":  peak_covid_scalar(resid_rs_np, covid_idxs),
        }

    # April 2026 revision fields
    pre_val  = to_float(apr2026_pre[sid])  if (apr2026_pre  is not None and sid in apr2026_pre.index)  else None
    rev_val  = to_float(apr2026_revised[sid]) if (apr2026_revised is not None and sid in apr2026_revised.index) else None
    revision = (round(rev_val - pre_val, 6) if (pre_val is not None and rev_val is not None) else None)
    if pre_val is not None and rev_val is not None and pre_val != 0:
        revision_pct = round((rev_val - pre_val) / pre_val, 6)
    else:
        revision_pct = None

    rows.append({
        "series_id":           sid,
        "industry_name":       name,
        "display_level":       lvl,
        "emp_recent":          emp_recent,
        "emp_recent_label":    last_lbl,
        "emp_prev":            emp_prev,
        "emp_prev_label":      prev_lbl,
        "dev_log_level":       last_nonnan3(r["resid_ll"]),
        **{f"dev_log_level{k}": ll_snaps[k] for k in ll_snaps},
        "dev_log_level_covid": ll_covid,
        "apr2026_preliminary": pre_val,
        "apr2026_revised":     rev_val,
        "revision":            revision,
        "revision_pct":        revision_pct,
        "opts":                opts_summary,
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

    # Per-option share data for charts 3–8 (6 denom options)
    options_json = {}
    for opt in range(1, 7):
        o        = r["opts"][opt]
        sv       = o["sv"]
        t_ls     = o["trend_ls"]
        r_ls     = o["resid_ls"]
        t_rs     = o["trend_rs"]
        r_rs     = o["resid_rs"]
        t_hp     = o["trend_hp"]
        r_hp     = o["resid_hp"]
        denom_id = o["denom_id"]

        with np.errstate(invalid="ignore", divide="ignore"):
            t_ls_impl = np.where(~np.isnan(t_ls), np.exp(t_ls) * 100, np.nan)
            t_hp_impl = np.where(~np.isnan(t_hp), np.exp(t_hp) * 100, np.nan)

        options_json[str(opt)] = {
            "denom_name":         id_to_name.get(denom_id, ""),
            "denom_id":           denom_id,
            "emp_share_pct":      arr_to_list(sv * 100,      4),
            "trend_ls_share_pct": arr_to_list(t_ls_impl,     4),
            "resid_ls_share":     arr_to_list(r_ls,          6),
            "trend_rs_share_pct": arr_to_list(t_rs * 100,    4),
            "resid_rs_pct":       arr_to_list(r_rs,          6),
            "hp_cf_share_pct":    arr_to_list(t_hp_impl,     4),
            "resid_hp":           arr_to_list(r_hp,          6),
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

    for opt in range(1, 7):
        o = r["opts"][opt]
        denom_id = o["denom_id"]
        dv = emp[denom_id].reindex(all_dates).values.astype(float) if denom_id in emp.columns else np.full(n_dates, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            log_shr = np.where(o["sv"] > 0, np.log(o["sv"]), np.nan)
            pred_ls = np.where(~np.isnan(o["trend_ls"]), np.exp(o["trend_ls"]), np.nan)
            pred_hp = np.where(~np.isnan(o["trend_hp"]), np.exp(o["trend_hp"]) * 100, np.nan)

        rows_csv[f"share_opt{opt}"]               = o["sv"]
        rows_csv[f"denom_employment_opt{opt}"]    = dv
        rows_csv[f"log_share_opt{opt}"]           = log_shr
        rows_csv[f"trend_log_share_opt{opt}"]     = o["trend_ls"]
        rows_csv[f"predicted_share_opt{opt}"]     = pred_ls
        rows_csv[f"resid_log_share_opt{opt}"]     = o["resid_ls"]
        rows_csv[f"trend_raw_share_opt{opt}"]     = o["trend_rs"]
        rows_csv[f"resid_raw_pct_dev_opt{opt}"]   = o["resid_rs"]
        rows_csv[f"hp_cf_log_share_opt{opt}"]     = o["trend_hp"]
        rows_csv[f"predicted_share_hp_opt{opt}"]  = pred_hp
        rows_csv[f"resid_log_share_hp_opt{opt}"]  = o["resid_hp"]

    export_df = pd.DataFrame(rows_csv)
    csv_path  = DATA_OUT / f"{sid}_export.csv"
    with open(csv_path, "w") as f:
        f.write(f"# Industry: {name}\n# Series ID: {sid}\n")
        f.write(f"# Parent series: {parent_name} ({parent_sid})\n")
        f.write(f"# Generated: {today_str}\n")
        f.write("# Denominator options:\n")
        for opt in range(1, 7):
            f.write(f"#   opt{opt}: {OPT_LABELS[opt]} = {id_to_name.get(results[sid]['opts'][opt]['denom_id'], '?')}\n")
        export_df.to_csv(f, index=False)

    if (loop_i + 1) % 100 == 0 or (loop_i + 1) == n:
        print(f"  {loop_i + 1}/{n} industries processed…")

print("Done! All files written to website/data/")

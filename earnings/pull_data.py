"""
pull_data.py — Pull BLS B-3a average hourly earnings (AHE) data and CPI-U.

Run from the earnings/ directory:
    python pull_data.py

Outputs:
    b3a_series.csv     — metadata for all AHE series (series_id, industry_code, ...)
    b3a_mapping.csv    — b3a_series + display_level, benchmark_id, benchmark_name
    b3a_wide.csv       — wide-format AHE data (date index × series_id columns)
    cpiu.csv           — two columns: date, cpiu
"""

import csv
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BLS_API_KEY = "12de604065914ed48cc3b31f0fc15d88"
BLS_API_V2_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CE_SERIES_URL = "https://download.bls.gov/pub/time.series/ce/ce.series"

START_YEAR = 2006
END_YEAR = datetime.now().year

BENCHMARK_TOTAL_PRIVATE = "CES0500000003"
BENCHMARK_GOODS = "CES0600000003"
BENCHMARK_SERVICES = "CES0800000003"

# Goods-producing supersector codes
GOODS_SUPERSECTORS = {10, 15, 20, 25, 30, 31, 32, 35}
# Private service-providing supersector codes
SERVICES_SUPERSECTORS = {40, 41, 42, 43, 44, 50, 55, 60, 65, 70, 75, 80, 85}

CPIU_SERIES = "CUSR0000SA0"

HERE = Path(__file__).parent

CE_INDUSTRY_URL = "https://download.bls.gov/pub/time.series/ce/ce.industry"

_LOWER_WORDS = {"a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
                "as", "at", "by", "in", "of", "on", "to", "up", "with"}
_SA_RE = re.compile(r",?\s*seasonally adjusted\s*$", re.IGNORECASE)


def title_case(s: str) -> str:
    """Title case with standard English rules for articles/prepositions/conjunctions."""
    words = s.title().split()
    result = []
    for i, w in enumerate(words):
        if i == 0 or w.lower() not in _LOWER_WORDS:
            result.append(w)
        else:
            result.append(w.lower())
    return " ".join(result)


# ── Helpers ────────────────────────────────────────────────────────────────────

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_bls_batch(series_ids, start_year: int, end_year: int) -> list:
    """Call BLS API v2 for up to 50 series in one request."""
    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": BLS_API_KEY,
    }
    for attempt in range(3):
        try:
            resp = requests.post(BLS_API_V2_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "REQUEST_NOT_PROCESSED":
                print(f"  BLS error: {data.get('message')}", file=sys.stderr)
                time.sleep(5)
                continue
            return data.get("Results", {}).get("series", [])
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(5)
    return []


def bls_series_to_df(series_list: list) -> dict:
    """Convert BLS API series list to dict of {series_id: pd.Series(date→value)}."""
    out = {}
    for s in series_list:
        sid = s["seriesID"]
        rows = []
        for obs in s.get("data", []):
            if obs.get("period", "").startswith("M") and obs["period"] != "M13":
                month = int(obs["period"][1:])
                date_str = f"{obs['year']}-{month:02d}"
                try:
                    rows.append((date_str, float(obs["value"])))
                except ValueError:
                    pass
        if rows:
            idx, vals = zip(*rows)
            out[sid] = pd.Series(vals, index=pd.Index(idx, name="date"), name=sid)
    return out


# ── Step 1: Fetch series metadata ──────────────────────────────────────────────

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BLS-research/1.0; guanyuzhou98@gmail.com)"}


def fetch_ce_series_metadata() -> pd.DataFrame:
    """Download ce.series metadata file from BLS and filter for AHE series."""
    print("Downloading ce.series metadata…")
    resp = requests.get(CE_SERIES_URL, timeout=120, headers=HEADERS)
    resp.raise_for_status()
    df = pd.read_csv(
        io.StringIO(resp.text),
        sep="\t",
        dtype=str,
        skipinitialspace=True,
    )
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Filter: data_type_code == 03 (AHE all employees), seasonal == S
    df = df[
        (df["data_type_code"] == "03") &
        (df["seasonal"] == "S")
    ].copy()

    # Exclude government supersectors (90–93)
    df["supersector_code"] = df["series_id"].str[3:5].astype(int)
    df = df[df["supersector_code"] < 90].copy()

    # Ensure benchmark series are present
    for bid in [BENCHMARK_TOTAL_PRIVATE, BENCHMARK_GOODS, BENCHMARK_SERVICES]:
        if bid not in df["series_id"].values:
            print(f"  WARNING: benchmark {bid} not in metadata, will add manually")

    # Parse industry_code and series_title
    df["industry_code"] = df["industry_code"].str.strip()
    df["series_title"] = df["series_title"].str.strip()

    # Parse industry_name: strip prefix + SA suffix, apply title case
    prefix = "Average hourly earnings of all employees, "
    df["industry_name"] = (
        df["series_title"]
        .str.replace(prefix, "", regex=False)
        .str.strip()
        .apply(lambda s: _SA_RE.sub("", s).strip())
        .apply(title_case)
    )

    # Preserve row_order
    df = df.reset_index(drop=True)
    df["row_order"] = df.index

    print(f"  Found {len(df)} AHE series (private, seasonally adjusted)")

    # Download ce.industry for display_level
    print("Downloading ce.industry hierarchy…")
    ind_resp = requests.get(CE_INDUSTRY_URL, timeout=120, headers=HEADERS)
    ind_resp.raise_for_status()
    ce_ind = pd.read_csv(
        io.StringIO(ind_resp.text),
        sep="\t",
        dtype=str,
        skipinitialspace=True,
    )
    ce_ind.columns = [c.strip() for c in ce_ind.columns]
    for col in ce_ind.columns:
        ce_ind[col] = ce_ind[col].str.strip()
    print(f"  ce.industry columns: {list(ce_ind.columns)}")

    if "display_level" in ce_ind.columns and "industry_code" in ce_ind.columns:
        dl_map = ce_ind.set_index("industry_code")["display_level"].to_dict()
        df["display_level"] = df["industry_code"].map(dl_map)
        missing = int(df["display_level"].isna().sum())
        if missing > 0:
            print(
                f"  WARNING: {missing} series have industry_code not found in "
                f"ce.industry; display_level set to None",
                file=sys.stderr,
            )
    else:
        print(
            "  WARNING: ce.industry missing expected columns; display_level will be None",
            file=sys.stderr,
        )
        df["display_level"] = None

    return df[["series_id", "industry_code", "industry_name", "series_title",
               "supersector_code", "begin_year", "end_year", "row_order",
               "display_level"]].copy()


# ── Step 2: Benchmark assignment ───────────────────────────────────────────────

def assign_benchmark(series_id: str, supersector_code: int, industry_code: str) -> tuple[str, str]:
    """Return (benchmark_id, benchmark_name) for a given series."""
    code = str(industry_code).strip().zfill(8)

    # Total private itself → no benchmark (use self as reference, omit rel calc)
    if series_id == BENCHMARK_TOTAL_PRIVATE:
        return BENCHMARK_TOTAL_PRIVATE, "Total Private (self)"

    # Goods-producing or Private service-providing → Total Private
    if series_id in (BENCHMARK_GOODS, BENCHMARK_SERVICES):
        return BENCHMARK_TOTAL_PRIVATE, "Total Private"

    # Goods-producing subsectors
    if supersector_code in GOODS_SUPERSECTORS:
        return BENCHMARK_GOODS, "Goods-producing"

    # Service-providing subsectors
    if supersector_code in SERVICES_SUPERSECTORS:
        return BENCHMARK_SERVICES, "Private service-providing"

    # Fallback
    return BENCHMARK_TOTAL_PRIVATE, "Total Private"


def build_mapping(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Add benchmark_id and benchmark_name columns. display_level comes from metadata."""
    df = meta_df.copy()
    results = df.apply(
        lambda r: assign_benchmark(r["series_id"], int(r["supersector_code"]), r["industry_code"]),
        axis=1,
    )
    df["benchmark_id"] = [r[0] for r in results]
    df["benchmark_name"] = [r[1] for r in results]
    return df


# ── Step 3: Pull time-series data ───────────────────────────────────────────────

def pull_all_series(all_series_ids: list) -> dict:
    """Batch-fetch all AHE series + CPI-U from BLS API v2."""
    all_ids = list(all_series_ids) + [CPIU_SERIES]
    # BLS API allows max 20 years per call; window in 10-year chunks if needed
    year_windows = []
    for y in range(START_YEAR, END_YEAR + 1, 20):
        year_windows.append((y, min(y + 19, END_YEAR)))

    combined: dict[str, pd.Series] = {}

    total_batches = len(list(chunks(all_ids, 50))) * len(year_windows)
    batch_num = 0

    for y_start, y_end in year_windows:
        for batch_ids in chunks(all_ids, 50):
            batch_num += 1
            print(f"  Batch {batch_num}/{total_batches}: {len(batch_ids)} series, {y_start}–{y_end}…")
            results = fetch_bls_batch(batch_ids, y_start, y_end)
            batch_data = bls_series_to_df(results)
            for sid, s in batch_data.items():
                if sid in combined:
                    combined[sid] = combined[sid].combine_first(s)
                else:
                    combined[sid] = s
            time.sleep(0.5)  # be polite to BLS API

    return combined


# ── Step 4: Build wide DataFrames and save ─────────────────────────────────────

def build_wide_df(series_dict: dict[str, pd.Series], series_ids: list[str]) -> pd.DataFrame:
    """Combine series into a wide DataFrame, sorted by date."""
    frames = {sid: series_dict[sid] for sid in series_ids if sid in series_dict}
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index = pd.Index(sorted(df.index), name="date")
    return df


def main():
    print("=== BLS B-3a AHE Data Pull ===")

    # 1. Fetch metadata
    meta_df = fetch_ce_series_metadata()
    meta_df.to_csv(HERE / "b3a_series.csv", index=False)
    print(f"Saved b3a_series.csv ({len(meta_df)} rows)")

    # 2. Build mapping
    mapping_df = build_mapping(meta_df)
    mapping_df.to_csv(HERE / "b3a_mapping.csv", index=False)
    print(f"Saved b3a_mapping.csv ({len(mapping_df)} rows)")

    # 3. Pull time-series data
    ahe_ids = mapping_df["series_id"].tolist()
    print(f"\nPulling {len(ahe_ids)} AHE series + CPI-U from BLS API…")
    all_data = pull_all_series(ahe_ids)
    print(f"\nReceived data for {len(all_data)} series")

    # 4. Save AHE wide CSV
    ahe_df = build_wide_df(all_data, ahe_ids)
    ahe_df.to_csv(HERE / "b3a_wide.csv")
    print(f"Saved b3a_wide.csv ({len(ahe_df)} rows × {len(ahe_df.columns)} series)")

    # 5. Save CPI-U CSV
    if CPIU_SERIES in all_data:
        cpiu_s = all_data[CPIU_SERIES].sort_index()
        cpiu_df = cpiu_s.reset_index()
        cpiu_df.columns = ["date", "cpiu"]
        cpiu_df.to_csv(HERE / "cpiu.csv", index=False)
        print(f"Saved cpiu.csv ({len(cpiu_df)} rows)")
    else:
        print("WARNING: CPI-U series not returned by API", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()

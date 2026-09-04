"""
pull_data.py — Pull BLS Table B-1a employment data from the BLS API.

Run from anywhere:
    python employment/pull_data.py

Before overwriting b1a_wide_seriesid.csv it archives the existing file to
    b1a_wide_seriesid_pre_revision_<YYYY_MM>.csv
named for that file's own last month — the month that was still preliminary at the
last pull. generate_data.py diffs the newest such snapshot against the fresh data to
produce the "Preliminary vs Revised" columns on the homepage. Existing snapshots are
never overwritten, so re-running the pull cannot destroy a preliminary vintage.

(Before 2026-08 these snapshots were created by hand; the 2026_04 / 2026_05 / 2026_06
files predate this script and follow the same convention.)

Outputs (all in employment/):
    b1a_wide_seriesid.csv                         — employment levels (thousands)
    b1a_wide_seriesid_pre_revision_<YYYY_MM>.csv  — snapshot of the previous vintage
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).parent
MAPPING_PATH = BASE / "b1a_mapping_with_denominators.csv"
WIDE_PATH    = BASE / "b1a_wide_seriesid.csv"

BLS_API_KEY = "12de604065914ed48cc3b31f0fc15d88"

START_YEAR = 2000
END_YEAR = datetime.now().year

BLS_API_V2_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fetch_bls_monthly_windowed(
    series_ids,
    start_year,
    end_year,
    api_key,
    series_batch_size=50,
    year_window=10,
    sleep_s=0.25,
):
    """
    Pull monthly data by (series batches) x (year windows) to avoid the ~240 obs cap.
    Returns tidy df: series_id, date, value
    """
    headers = {"Content-type": "application/json"}
    out = []

    for y0 in range(start_year, end_year + 1, year_window):
        y1 = min(end_year, y0 + year_window - 1)

        for batch in chunks(list(series_ids), series_batch_size):
            payload = {
                "seriesid": batch,
                "startyear": str(y0),
                "endyear": str(y1),
                "registrationkey": api_key,
            }

            resp = requests.post(BLS_API_V2_URL, data=json.dumps(payload), headers=headers, timeout=60)
            resp.raise_for_status()
            j = resp.json()

            if j.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS API error: status={j.get('status')} message={j.get('message')}")

            for s in j["Results"]["series"]:
                sid = s["seriesID"]
                for item in s.get("data", []):
                    period = item.get("period", "")
                    if period.startswith("M") and period != "M13":
                        year = int(item["year"])
                        month = int(period[1:])
                        date = pd.Timestamp(year=year, month=month, day=1)
                        out.append({"series_id": sid, "date": date, "value": float(item["value"])})

            time.sleep(sleep_s)

    df = pd.DataFrame(out)
    if df.empty:
        return df

    # De-duplicate in case windows overlap or API repeats endpoints
    df = df.drop_duplicates(subset=["series_id", "date"]).sort_values(["series_id", "date"]).reset_index(drop=True)
    return df


def archive_previous_vintage():
    """Snapshot the existing wide CSV before it is overwritten.

    Named for its own last month — b1a_wide_seriesid_pre_revision_<YYYY_MM>.csv — and
    an existing snapshot is never overwritten. That makes the pull safe to re-run: a
    second run in the same month writes a *new* filename rather than clobbering the
    vintage that holds the preliminary estimates generate_data.py needs.
    """
    if not WIDE_PATH.exists():
        print("No existing b1a_wide_seriesid.csv — skipping snapshot step.")
        return
    prev = pd.read_csv(WIDE_PATH, index_col=0, parse_dates=True)
    dest = BASE / f"b1a_wide_seriesid_pre_revision_{prev.index[-1]:%Y_%m}.csv"
    if dest.exists():
        print(f"Snapshot {dest.name} already exists — keeping it, not re-writing.")
        return
    shutil.copy2(WIDE_PATH, dest)
    print(f"Snapshotted previous vintage (last month {prev.index[-1]:%Y-%m}) → {dest.name}")


def main():
    mapping = pd.read_csv(MAPPING_PATH, dtype=str)
    if "series_id" not in mapping.columns:
        raise KeyError(f"'series_id' not found in {MAPPING_PATH}. Columns are: {list(mapping.columns)}")

    series_ids = mapping["series_id"].dropna().astype(str).str.strip().unique().tolist()

    archive_previous_vintage()

    tidy = fetch_bls_monthly_windowed(
        series_ids=series_ids,
        start_year=START_YEAR,
        end_year=END_YEAR,
        api_key=BLS_API_KEY,
        series_batch_size=50,
        year_window=10,
        sleep_s=0.25,
    )

    # enforce >= 2000-01
    tidy = tidy[tidy["date"] >= pd.Timestamp(START_YEAR, 1, 1)].copy()

    wide = (
        tidy.pivot_table(index="date", columns="series_id", values="value", aggfunc="first")
            .sort_index()
    )

    print("Series count:", len(series_ids))
    print("Date range:", wide.index.min(), "to", wide.index.max())
    print("Wide shape:", wide.shape)

    wide.to_csv(WIDE_PATH)
    print(f"Wrote {WIDE_PATH.relative_to(BASE)}")


if __name__ == "__main__":
    main()

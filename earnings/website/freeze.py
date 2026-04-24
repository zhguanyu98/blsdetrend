"""
freeze.py — Build a static site for GitHub Pages deployment.

Run from the earnings/website/ directory:
    python freeze.py

Output: earnings/website/build/
"""

import shutil
from pathlib import Path

from flask_frozen import Freezer

from app import DATA_DIR, _TABLE_DATA, app

app.config.update(
    FREEZER_DESTINATION="build",
    FREEZER_RELATIVE_URLS=True,
    FREEZER_IGNORE_MIMETYPE_WARNINGS=True,
    FREEZER_REMOVE_EXTRA_FILES=True,
)

freezer = Freezer(app)

_sids = [r["series_id"] for r in _TABLE_DATA["rows"]]


@freezer.register_generator
def industry():
    for sid in _sids:
        yield {"series_id": sid}


@freezer.register_generator
def serve_data():
    for csv_path in sorted(DATA_DIR.glob("*_export.csv")):
        yield {"filename": csv_path.name}
    for json_path in sorted(DATA_DIR.glob("*_alt.json")):
        yield {"filename": json_path.name}


@freezer.register_generator
def analysis():
    yield {}


@freezer.register_generator
def download():
    return []


if __name__ == "__main__":
    build_dir = Path(__file__).parent / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    print(f"Freezing {len(_sids) + 2} pages…")
    freezer.freeze()

    (build_dir / ".nojekyll").touch()

    pages = sum(1 for _ in build_dir.rglob("index.html"))
    csvs = len(list((build_dir / "data").glob("*.csv"))) if (build_dir / "data").exists() else 0
    print(f"Done → build/  ({pages} HTML pages, {csvs} CSV files)")

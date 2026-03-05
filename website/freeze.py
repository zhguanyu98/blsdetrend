"""
freeze.py — Build a static site for GitHub Pages deployment.

Run from the website/ directory:
    python freeze.py

Output: website/build/
    index.html
    {series_id}/index.html   (× 842, each with embedded chart data)
    data/{series_id}_export.csv  (× 842, for download links)
    static/css/styles.css
    .nojekyll                (prevents Jekyll processing on GitHub Pages)

After building, enable GitHub Pages on the gh-pages branch in repo Settings.
"""

import shutil
from pathlib import Path

from flask_frozen import Freezer

from app import DATA_DIR, _TABLE_DATA, app

app.config.update(
    FREEZER_DESTINATION="build",
    FREEZER_RELATIVE_URLS=True,           # converts url_for() → relative paths
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
    # Yield every CSV file so Frozen-Flask saves them to build/data/
    for csv_path in sorted(DATA_DIR.glob("*_export.csv")):
        yield {"filename": csv_path.name}


@freezer.register_generator
def download():
    return []  # legacy route, no longer linked from HTML


if __name__ == "__main__":
    # Remove stale build directory to avoid conflicts between old flat-file
    # structure and new directory-based structure.
    build_dir = Path(__file__).parent / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    print(f"Freezing {len(_sids) + 1} pages + {len(list(DATA_DIR.glob('*_export.csv')))} CSV files…")
    freezer.freeze()

    build_dir = Path(__file__).parent / "build"
    (build_dir / ".nojekyll").touch()

    pages = sum(1 for _ in build_dir.rglob("index.html"))
    csvs  = len(list((build_dir / "data").glob("*.csv"))) if (build_dir / "data").exists() else 0
    print(f"Done → build/  ({pages} HTML pages, {csvs} CSV files)")

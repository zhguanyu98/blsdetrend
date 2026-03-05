"""
app.py — Flask application for BLS B-1a Employment Analysis website.
"""

import json
import os
from pathlib import Path

from flask import Flask, abort, render_template, send_file

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Pre-load table data at startup so the first request (including health checks) is fast
_table_data_path = DATA_DIR / "table_data.json"
with open(_table_data_path) as _f:
    _TABLE_DATA = json.load(_f)


def load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        abort(404)
    with open(path) as f:
        return json.load(f)


@app.route("/health")
def health():
    return "ok", 200


@app.route("/")
def index():
    return render_template(
        "index.html",
        rows=_TABLE_DATA["rows"],
        last_label=_TABLE_DATA.get("last_label", ""),
        prev_label=_TABLE_DATA.get("prev_label", ""),
    )


@app.route("/<series_id>")
def industry(series_id: str):
    # Basic validation: BLS series IDs are alphanumeric
    if not series_id.replace("CES", "").isdigit() and not series_id.startswith("CES"):
        abort(404)
    series_data = load_json(f"{series_id}.json")
    return render_template(
        "industry.html",
        series_data=series_data,
        series_json=json.dumps(series_data),
    )


@app.route("/download/<series_id>")
def download(series_id: str):
    if not series_id.startswith("CES"):
        abort(404)
    csv_path = DATA_DIR / f"{series_id}_export.csv"
    if not csv_path.exists():
        abort(404)
    return send_file(
        csv_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{series_id}_export.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)

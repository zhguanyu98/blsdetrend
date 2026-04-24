"""
app.py — Flask application for BLS B-3a Average Weekly Earnings website.
"""

import json
import os
from pathlib import Path

from flask import Flask, abort, make_response, render_template, send_file

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Pre-load both table datasets at startup (passes Render health check)
_table_data_path = DATA_DIR / "table_data.json"
with open(_table_data_path) as _f:
    _TABLE_DATA = json.load(_f)

_table_data_alt_path = DATA_DIR / "table_data_alt.json"
with open(_table_data_alt_path) as _f:
    _TABLE_DATA_ALT = json.load(_f)


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
    resp = make_response(render_template(
        "index.html",
        rows=_TABLE_DATA["rows"],
        rows_alt=_TABLE_DATA_ALT["rows"],
        latest_label=_TABLE_DATA.get("latest_label", ""),
        prev_label=_TABLE_DATA.get("prev_label", ""),
    ))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/analysis/")
def analysis():
    resp = make_response(render_template(
        "analysis.html",
        rows=_TABLE_DATA["rows"],
        rows_alt=_TABLE_DATA_ALT["rows"],
        latest_label=_TABLE_DATA.get("latest_label", ""),
    ))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/<series_id>/", strict_slashes=False)
def industry(series_id: str):
    if not series_id.startswith("CES"):
        abort(404)
    series_data = load_json(f"{series_id}.json")
    for row in _TABLE_DATA.get("rows", []):
        if row.get("series_id") == series_id:
            series_data["industry_name"] = row.get("industry_name", series_id)
            break
    else:
        series_data["industry_name"] = series_id
    return render_template(
        "industry.html",
        series_data=series_data,
        series_json=json.dumps(series_data),
    )


@app.route("/data/<filename>")
def serve_data(filename: str):
    if "." not in filename or filename.rsplit(".", 1)[1] not in ("json", "csv"):
        abort(404)
    path = DATA_DIR / filename
    if not path.exists():
        abort(404)
    return send_file(path)


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
    app.run(debug=True, port=5002)

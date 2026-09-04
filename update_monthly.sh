#!/usr/bin/env bash
#
# update_monthly.sh — run after each BLS Employment Situation release.
#
#   ./update_monthly.sh              both sites
#   ./update_monthly.sh employment   employment only
#   ./update_monthly.sh earnings     earnings only
#
# Steps per site: pull → generate_data → freeze. Nothing is committed or pushed;
# review the summary, then `git add -A && git commit && git push origin main`.
#
set -euo pipefail
cd "$(dirname "$0")"

# ── Interpreter ──────────────────────────────────────────────────────────────
# `python3` on this machine can resolve to Anaconda 3.9, which lacks Frozen-Flask.
# Pick the first interpreter that has every dependency.
PY=""
for cand in /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \
            /usr/local/bin/python3 python3; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" - <<'EOF' >/dev/null 2>&1
import importlib.util as u
assert not [m for m in ("flask","flask_frozen","pandas","numpy","scipy","statsmodels","requests")
            if u.find_spec(m) is None]
EOF
  then PY="$cand"; break; fi
done
[ -n "$PY" ] || { echo "ERROR: no interpreter with flask/frozen-flask/pandas/scipy/statsmodels found." >&2; exit 1; }
echo "Interpreter: $PY ($("$PY" -V 2>&1))"

WHICH="${1:-both}"

run_employment() {
  echo
  echo "════ EMPLOYMENT ════"
  before=$("$PY" -c "import pandas as pd;print(pd.read_csv('employment/b1a_wide_seriesid.csv',index_col=0).index[-1][:7])")
  echo "current data ends: $before"

  "$PY" employment/pull_data.py

  after=$("$PY" -c "import pandas as pd;print(pd.read_csv('employment/b1a_wide_seriesid.csv',index_col=0).index[-1][:7])")
  echo "data now ends:     $after"
  if [ "$before" = "$after" ]; then
    echo "WARNING: the pull did not advance the month — BLS may not have posted yet."
    echo "         Revision columns will be skipped. Stopping so nothing is overwritten."
    return 1
  fi

  ( cd employment/website && "$PY" generate_data.py )
  ( cd employment/website && "$PY" freeze.py )
}

run_earnings() {
  echo
  echo "════ EARNINGS ════"
  ( cd earnings && "$PY" pull_data.py )
  ( cd earnings/website && "$PY" generate_data.py )
  ( cd earnings/website && "$PY" freeze.py )
}

case "$WHICH" in
  employment) run_employment ;;
  earnings)   run_earnings ;;
  both)       run_employment; run_earnings ;;
  *) echo "usage: $0 [employment|earnings|both]" >&2; exit 2 ;;
esac

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "════ SUMMARY ════"
"$PY" - <<'EOF'
import json, pathlib
p = pathlib.Path('employment/website/data/table_data.json')
if p.exists():
    d = json.load(open(p))
    r = next(x for x in d['rows'] if x['series_id'] == 'CES0000000001')
    pk, rk = d.get('prelim_key'), d.get('revised_key')
    print(f"Employment  latest {d['last_label']} (prev {d['prev_label']})")
    print(f"            revision month: {d['revision_month_label'] or 'NONE — no pre_revision snapshot older than the data'}")
    print(f"            Total nonfarm   {r['emp_recent']:,}   M/M {r['mom']*100:+.2f}%   Y/Y {r['yoy']*100:+.2f}%")
    if pk and r.get(pk) is not None:
        print(f"            {pk} {r[pk]:,} -> {r[rk]:,}  revision {r['revision']:+,}")
    n = sum(1 for x in d['rows'] if x['revision'] is not None)
    print(f"            {n} of {len(d['rows'])} series carry a revision")
q = pathlib.Path('earnings/website/data/table_data.json')
if q.exists():
    e = json.load(open(q))
    rows = e['rows'] if isinstance(e, dict) else e
    print(f"Earnings    {len(rows)} rows regenerated")
EOF
echo
echo "Review, then:  git add -A && git commit -m 'Monthly BLS update' && git push origin main"

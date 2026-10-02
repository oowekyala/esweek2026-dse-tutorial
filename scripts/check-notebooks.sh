#!/usr/bin/env bash
# Executes the three notebooks in order, on one core, the way a Binder kernel
# would run them, and reports how long each took. Executed copies land in
# work/executed/; the notebooks themselves are not modified.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/work/executed"
mkdir -p "$out"

pin=()
if command -v taskset >/dev/null 2>&1; then
  pin=(taskset -c 0)
fi

status=0
for nb in 01_space 02_evaluate 03_search; do
  start=$(date +%s)
  if MPLBACKEND=Agg "${pin[@]}" jupyter nbconvert --to notebook --execute \
       --ExecutePreprocessor.timeout=900 --output-dir="$out" \
       "$root/notebooks/$nb.ipynb" > "$out/$nb.log" 2>&1; then
    printf '%-14s ok   %4ds\n' "$nb" "$(( $(date +%s) - start ))"
  else
    printf '%-14s FAILED  (see %s)\n' "$nb" "$out/$nb.log"
    grep -E 'Error' "$out/$nb.log" | tail -3
    status=1
  fi
done
exit $status

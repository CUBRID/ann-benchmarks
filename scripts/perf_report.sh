#!/bin/bash
set -e

# script dir: <root>/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# project root: <root>
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# output base: <root>/perf_output
OUT_DIR="$ROOT_DIR/perf_output"

mkdir -p \
  "$OUT_DIR/folded" \
  "$OUT_DIR/flame" \
  "$OUT_DIR/perf_txt"

export PATH=/home/hgryoo/tools/FlameGraph:$PATH

f="$1"

# resolve relative path -> absolute path
f="$(realpath "$f")"

if [ ! -s "$f" ]; then
  echo "[skip] zero-sized or missing: $f"
  exit 0
fi

echo "[parsing perf] $f"

base="$(basename "$f" .data)"

sudo perf script -f -i "$f" \
  | stackcollapse-perf.pl \
  > "$OUT_DIR/folded/${base}.folded"

flamegraph.pl "$OUT_DIR/folded/${base}.folded" \
  > "$OUT_DIR/flame/${base}.svg"

{
  echo "==== $base ===="
  echo "--- Top functions ---"
  perf report --stdio -i "$f" --sort symbol --percent-limit 1
  echo
  echo "--- Call graph ---"
  perf report --stdio -i "$f" -g graph --percent-limit 2
  echo
  echo "--- DSOs ---"
  perf report --stdio -i "$f" --sort dso --percent-limit 1

  # echo "--- branch/cache events ---"
  #perf report --stdio -i "$f" -e branch-misses --sort symbol --percent-limit 1
  # perf report --stdio -i "$f" -e cache-misses  --sort symbol --percent-limit 1

} > "$OUT_DIR/perf_txt/${base}.txt"

#!/usr/bin/env bash
set -euo pipefail

# script dir: <root>/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# project root: <root>
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# output base: <root>/perf_results
OUT_BASE="$ROOT_DIR/perf_results"
mkdir -p "$OUT_BASE"

echo "[perf] watching PERF_HINT logs..."

docker events \
  --filter 'event=start' \
  --filter 'label=annb.role=benchmark' |
while read -r _ _ _ cid _; do
  echo "[perf] container started: $cid"

  docker logs -f "$cid" | while read -r line; do
    [[ "$line" =~ ^\[PERF_HINT\] ]] || continue

    echo "[perf] hint: $line"

    action=$(echo "$line" | awk '{print $2}')
    phase=$(echo "$line" | sed -n 's/.*phase=\([^ ]*\).*/\1/p')
    pid=$(echo "$line" | sed -n 's/.*pid=\([0-9]*\).*/\1/p')

    [[ -n "$pid" ]] || continue

    out="$OUT_BASE/perf_${cid}_${phase}.data"
    pidfile="/tmp/perf_${cid}_${phase}.pid"

    if [[ "$action" == "START" ]]; then
      echo "[perf] START perf pid=$pid phase=$phase"

      perf record \
        -e cycles,instructions,branches,branch-misses,cache-references,cache-misses \
        -F 99 \
        -g \
        --call-graph dwarf \
        -p "$pid" \
        -o "$out" &
      
      echo "$!" > "$pidfile"

    elif [[ "$action" == "STOP" ]]; then
      perf_pid=$(cat "$pidfile" 2>/dev/null || true)
      [[ -n "$perf_pid" ]] || continue

      echo "[perf] STOP perf phase=$phase"
      kill -INT "$perf_pid"
      wait "$perf_pid" || true
      rm -f "$pidfile"

      echo "[perf] saved: $out"
    fi
  done &
done


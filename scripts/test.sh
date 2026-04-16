#!/usr/bin/env bash
set -e
sudo -v

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -a, --algorithm NAME     Algorithm name (e.g. cubrid)
  -d, --dataset  NAME      Dataset name
  -c, --count    N         Vector count
  -l, --local              Run in local mode
  -r, --runs     N         Number of runs (default: 1)
  -h, --help               Show this help and exit

Example:
  $(basename "$0") -a cubrid -d random-xs-20-angular -c 10 -r 1
EOF
}

# defaults
RUNS=1
LOCAL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--algorithm)
      ALGO="$2"; shift 2;;
    -d|--dataset)
      DATASET="$2"; shift 2;;
    -c|--count)
      COUNT="$2"; shift 2;;
    -r|--runs)
      RUNS="$2"; shift 2;;
    -l|--local)
      LOCAL="--local"; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1"
      usage; exit 1;;
  esac
done

# validation
: "${ALGO:?--algorithm is required}"
: "${DATASET:?--dataset is required}"
: "${COUNT:?--count is required}"

uv run python run.py \
  --algorithm "$ALGO" \
  --dataset "$DATASET" \
  --count "$COUNT" \
  --runs "$RUNS" \
  --force \
  $LOCAL

RESULTS_DIR="$BASE_DIR/../results"

if [[ -d "$RESULTS_DIR" ]]; then
  # restore ownership
  sudo chown -R "$(id -u):$(id -g)" "$RESULTS_DIR"
fi

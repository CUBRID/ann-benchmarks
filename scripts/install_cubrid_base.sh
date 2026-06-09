#!/usr/bin/env bash
set -euo pipefail

# defaults
GIT_REMOTE_URL="http://github.com/cubrid/cubrid"
GIT_COMMIT=""
GIT_BRANCH="cubvec/m1"

usage() {
  echo "Usage:"
  echo "  $0 --commit=<commit_id> [--remote=<git_url>]"
  echo "  $0 --branch=<branch_name> [--remote=<git_url>]"
  echo "  $0 --help"
  echo "Example:"
  echo "  $0 --commit=1234567890 --remote=https://github.com/CUBRID/cubrid.git"
  echo "  $0 --branch=cubvec/cubvec --remote=https://github.com/CUBRID/cubrid.git"
  exit 1
}

# parse args
for arg in "$@"; do
  case "$arg" in
    --commit=*)
      GIT_COMMIT="${arg#*=}"
      ;;
    --branch=*)
      GIT_BRANCH="${arg#*=}"
      ;;
    --remote=*)
      GIT_REMOTE_URL="${arg#*=}"
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      ;;
  esac
done

# validation
if [[ -n "$GIT_COMMIT" && -n "$GIT_BRANCH" ]]; then
  echo "Error: --commit and --branch cannot be used together"
  usage
fi

if [[ -z "$GIT_COMMIT" && -z "$GIT_BRANCH" ]]; then
  echo "Error: one of --commit or --branch must be specified"
  usage
fi

# build args
BUILD_ARGS=()

if [[ -n "$GIT_REMOTE_URL" ]]; then
  BUILD_ARGS+=("GIT_REMOTE_URL=$GIT_REMOTE_URL")
fi
if [[ -n "$GIT_BRANCH" ]]; then
  BUILD_ARGS+=("GIT_BRANCH=$GIT_BRANCH")
fi
if [[ -n "$GIT_COMMIT" ]]; then
  BUILD_ARGS+=("GIT_COMMIT=$GIT_COMMIT")
fi

echo "[install] algorithm=cubrid-baseline"
echo "[install] build args:"
printf '  %s\n' "${BUILD_ARGS[@]}"

uv run python install.py \
  --algorithm cubrid-baseline \
  --build-arg "${BUILD_ARGS[@]}"

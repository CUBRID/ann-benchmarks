#!/usr/bin/env bash
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WEBSITE_DIR="$BASE_DIR/../.websites/temp"
CREATE_WEBSITE="$BASE_DIR/../create_website.py"

sudo rm -rf "$WEBSITE_DIR"
mkdir -p "$WEBSITE_DIR"

uv run python "$CREATE_WEBSITE" \
  --outputdir "$WEBSITE_DIR" \
  --scatter \
  --recompute

uv run python -m http.server 8080 --directory "$WEBSITE_DIR"

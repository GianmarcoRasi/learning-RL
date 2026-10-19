#!/usr/bin/env bash
# Stage the course Markdown into site/_src and build the static site into site/_build.
# Run from anywhere: bash site/build.sh  (add "serve" to preview at http://127.0.0.1:8000)
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf site/_src
mkdir -p site/_src
cp README.md site/_src/index.md
cp -r lessons lessons-hands-on docs flashcards site/_src/
cp site/support.md site/_src/support.md
cp -r site/javascripts site/stylesheets site/_src/
if [ "${1:-}" = "serve" ]; then
  mkdocs serve -f site/mkdocs.yml
else
  mkdocs build --strict -f site/mkdocs.yml
fi

#!/bin/sh
# Assemble a clean copy of exactly the files the Hugging Face Space needs.
# Usage: sh space/stage.sh <output-dir>      (run from the repository root)
set -eu
out="${1:?usage: sh space/stage.sh <output-dir>}"
rm -rf "$out"
mkdir -p "$out/space" "$out/results"
cp space/Dockerfile "$out/Dockerfile"
cp space/README.md "$out/README.md"
cp space/requirements.txt "$out/space/requirements.txt"
[ -f LICENSE ] && cp LICENSE "$out/LICENSE"
for d in src configs corpus; do
  cp -R "$d" "$out/$d"
done
find "$out" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$out" -name '*.egg-info' -type d -prune -exec rm -rf {} + 2>/dev/null || true
n=0
for f in reference reference_v11 calibration signatures rule_citations; do
  if [ -f "results/$f.json" ]; then cp "results/$f.json" "$out/results/$f.json"; n=$((n + 1)); fi
done
[ "$n" -gt 0 ] || : > "$out/results/.keep"
echo "staged $(find "$out" -type f | wc -l) files, $(du -sh "$out" | cut -f1), results statistics: $n"

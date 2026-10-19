#!/bin/bash
# Download 360p GoPro videos for the given recording ids (default: the demo set in data/videos/selection.txt).
# Links come from vendor/downloader/metadata/download_links.json (public Box static links).
set -u
cd "$(dirname "$0")/.."
OUT=data/videos; mkdir -p "$OUT"
if [ $# -gt 0 ]; then IDS="$*"; else IDS=$(awk '{print $1}' "$OUT/selection.txt"); fi
for rid in $IDS; do
  url=$(python3 -I -c "import json,sys; print(json.load(open('vendor/downloader/metadata/download_links.json'))[sys.argv[1]].get('gopro_360p',''))" "$rid")
  [ -z "$url" ] && { echo "no gopro_360p link for $rid"; continue; }
  f="$OUT/${rid}_360p.mp4"
  [ -s "$f" ] && { echo "skip $f"; continue; }
  echo "downloading $rid"
  curl -L --retry 3 -o "$f.part" "$url" && mv "$f.part" "$f" && echo "done $f ($(du -h "$f" | cut -f1)) $(file -b "$f" | cut -c1-40)"
done
echo VIDEOS_DONE

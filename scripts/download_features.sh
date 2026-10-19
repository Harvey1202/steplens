#!/bin/bash
# Download the CaptainCook4D segment features we use from the public Box archive.
# Sizes: omnivore 1.29 GB, slowfast 0.51 GB, imagebind 2s video/audio/text ~0.65 GB each.
set -u
OUT="${1:-data/features}"
SHARED="zzuglo0j0loo8ymdsxfr2zzgf9q2jajc"
mkdir -p "$OUT/gopro_1s" "$OUT/gopro_2s"
dl() {  # dl <file_id> <dest>
  local url="https://utdallas.app.box.com/index.php?rm=box_download_shared_file&shared_name=$SHARED&file_id=$1"
  if [ -s "$2" ]; then echo "skip $2 (exists)"; return; fi
  echo "downloading $2"
  curl -L --retry 5 --retry-delay 10 -C - -o "$2.part" "$url" && mv "$2.part" "$2" && echo "done $2 ($(du -h "$2" | cut -f1))"
}
dl f_1686926320731 "$OUT/gopro_1s/omnivore.zip"
dl f_1686934057785 "$OUT/gopro_1s/slowfast.zip"
dl f_1687061570323 "$OUT/gopro_2s/video.zip"
dl f_1687061089933 "$OUT/gopro_2s/audio.zip"
dl f_1687065395414 "$OUT/gopro_2s/text.zip"
echo ALL_DONE

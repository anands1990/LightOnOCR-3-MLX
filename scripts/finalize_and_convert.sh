#!/usr/bin/env bash
# Finalize the manually-downloaded HF cache and convert to MLX (offline).
#
# We bypassed huggingface_hub's (Xet) downloader by curl-inging the LFS files
# straight into the cache blob paths. This script wires them up and runs the
# converter in offline mode so it never tries to hit the (broken) downloader.
set -euo pipefail

REPO="models--lightonai--LightOnOCR-3-4B"
CACHE="$HOME/.cache/huggingface/hub/$REPO"
REV="a06e5c5459551c9d1696468aece70ffbcf01ae62"
SNAP="$CACHE/snapshots/$REV"
WEIGHTS_BLOB="6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67"
EXPECT="9078620536"

echo "==> checking model.safetensors blob size"
SZ=$(stat -f%z "$CACHE/blobs/$WEIGHTS_BLOB")
echo "    size=$SZ expect=$EXPECT"
if [ "$SZ" != "$EXPECT" ]; then
  echo "!!! model.safetensors incomplete ($SZ != $EXPECT); aborting." >&2
  exit 1
fi

echo "==> creating snapshot symlink for model.safetensors"
ln -sf "../../blobs/$WEIGHTS_BLOB" "$SNAP/model.safetensors"

echo "==> verifying every snapshot file resolves to a real (non-stub) file"
missing=0
for f in config.json generation_config.json preprocessor_config.json \
         processor_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt chat_template.jinja model.safetensors; do
  path="$SNAP/$f"
  if [ ! -s "$path" ]; then echo "    MISSING/empty: $f"; missing=1; fi
done
[ "$missing" = "0" ] || { echo "!!! some files missing"; exit 1; }
echo "    all present"

echo "==> running mlx_vlm.convert (offline, bf16)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 mlx_vlm.convert \
  --hf-path "lightonai/LightOnOCR-3-4B" \
  --mlx-path "LightOnOCR-3-4B-mlx/bf16" \
  --dtype bfloat16

echo "==> done. Output:"
ls -la LightOnOCR-3-4B-mlx/bf16

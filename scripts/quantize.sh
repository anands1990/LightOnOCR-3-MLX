#!/usr/bin/env bash
# Build a 4-bit quantized MLX model from the local BF16 conversion.
#
# Using the local BF16 dir as the source avoids re-downloading from HF (whose
# Xet downloader is flaky for this repo) and lets the model-card step read the
# local README.md so it works fully offline.
set -euo pipefail

SRC="LightOnOCR-3-4B-mlx/bf16"
DST="LightOnOCR-3-4B-mlx/4bit"

echo "==> quantizing $SRC -> $DST (4-bit, group 64)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 mlx_vlm.convert \
  --hf-path "$SRC" \
  --mlx-path "$DST" \
  --dtype bfloat16 \
  -q --q-bits 4 --q-group-size 64

echo "==> done"
ls -la "$DST"
du -sh "$DST"

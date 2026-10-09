#!/usr/bin/env python3
"""Run LightOnOCR-3-4B (MLX) on an image.

Modes:
    plain      -> transcription only (this is the model default: call with an
                  empty prompt to get the full page text as markdown)
    grounding  -> every block comes back with a label + bounding box, images get
                  a description, charts become an HTML data table

Usage:
    python run_ocr.py plain   [path/to/image.jpg]
    python run_ocr.py grounding [path/to/image.jpg]
"""
import sys

import os

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = os.environ.get("MLX_MODEL", "LightOnOCR-3-4B-mlx/bf16")
MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else "tests/images/receipt.jpeg"


def main():
    prompt = "grounding" if MODE == "grounding" else ""
    print(f"[mlx] loading {MODEL} ...")
    model, processor = load(MODEL)
    config = model.config

    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=1
    )

    result = generate(
        model,
        processor,
        formatted_prompt,
        image=[IMAGE],
        max_tokens=2048,
        temperature=0.2,
        top_p=0.9,
        verbose=False,
    )
    print("=" * 60)
    print(f"[mlx] {MODE} OUTPUT:")
    print(result.text)
    print("=" * 60)
    out_path = f"tests/out_{MODE}_mlx.txt"
    with open(out_path, "w") as f:
        f.write(result.text)
    print(f"saved -> {out_path}")
    total = getattr(result, "total_tokens", None)
    tps = getattr(result, "generation_tps", None)
    if tps:
        print(f"generation: {tps:.1f} tok/s" + (f", {total} tokens" if total else ""))


if __name__ == "__main__":
    main()

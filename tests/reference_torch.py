#!/usr/bin/env python3
"""PyTorch/transformers reference for LightOnOCR-3-4B.

Used ONLY to validate that the MLX-converted model produces the same OCR text.
Run this after `mlx_vlm.convert` has populated the HF cache so it doesn't
re-download the 9 GB weights.

Usage:
    python tests/reference_torch.py plain       # transcription mode (empty prompt)
    python tests/reference_torch.py grounding   # grounding mode
"""
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_ID = "lightonai/LightOnOCR-3-4B"
IMAGE = "tests/images/receipt.jpeg"
MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"


def main():
    print(f"[torch] loading {MODEL_ID} (bf16, mps)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    image = Image.open(IMAGE).convert("RGB")
    content = [{"type": "image", "image": image}]
    if MODE == "grounding":
        content.append({"type": "text", "text": "grounding"})
    conversation = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    print(f"[torch] mode={MODE} prompt_tokens={inputs['input_ids'].shape[-1]} generating...")
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.2, top_p=0.9)
    gen = out[0, inputs["input_ids"].shape[-1]:]
    text = processor.decode(gen, skip_special_tokens=True)
    print("=" * 60)
    print(f"[torch] {MODE} OUTPUT:")
    print(text)
    print("=" * 60)
    with open(f"tests/out_{MODE}_torch.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()

# LightOnOCR-3-4B → MLX

Make [lightonai/LightOnOCR-3-4B](https://huggingface.co/lightonai/LightOnOCR-3-4B)
runnable on Apple Silicon with [MLX](https://github.com/ml-explore/mlx) via
[`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm).

> Code, scripts & tests: https://github.com/anands1990/LightOnOCR-3-MLX

## TL;DR

`LightOnOCR-3-4B` is a Qwen3.5 vision-language OCR model (`model_type: qwen3_5`).
That architecture is **already implemented in mlx-vlm ≥ 0.6.5**, so this is a
pure weight conversion — no model code needs to change.

```bash
# 1. Convert (offline cache bootstrapping in scripts/finalize_and_convert.sh)
mlx_vlm.convert --hf-path lightonai/LightOnOCR-3-4B \
                 --mlx-path LightOnOCR-3-4B-mlx/bf16 --dtype bfloat16

# 2. (optional) 4-bit quantized build
bash scripts/quantize.sh

# 3. Run OCR
mlx_vlm.generate --model LightOnOCR-3-4B-mlx/4bit \
                  --image tests/images/receipt.jpeg --prompt "" \
                  --max-tokens 2048 --temperature 0.2 --top-p 0.9
```

## Published on the Hub

Live MLX builds (Apache-2.0 weights, served via `mlx-vlm`, no patches needed):

- **4-bit** (recommended): [`AnandSingh/LightOnOCR-3-4B-4bit`](https://huggingface.co/AnandSingh/LightOnOCR-3-4B-4bit) — ~2.9 GB, ~37 tok/s on M4, ~3.7 GB peak RAM
- **BF16**: [`AnandSingh/LightOnOCR-3-4B-bf16`](https://huggingface.co/AnandSingh/LightOnOCR-3-4B-bf16) — ~8.5 GB, full precision, ~11 tok/s

Both were verified to load and OCR directly from the Hub:
```bash
mlx_vlm.generate --model AnandSingh/LightOnOCR-3-4B-4bit \
  --image page.png --prompt "" --max-tokens 2048 --temperature 0.2 --top-p 0.9
```

> These live under the personal `AnandSingh` namespace (the `mlx-community` org
> restricts who can create repos). Rename/move to `mlx-community/…` later if you
> get write access — no re-upload needed.

## What's in this repo

| Path | Description |
|---|---|
| `run_ocr.py` | `MLX_MODEL=<dir> python run_ocr.py plain` / `grounding` [image] |
| `scripts/finalize_and_convert.sh` | Offline BF16 convert from a manually-populated HF cache |
| `scripts/quantize.sh` | Build a 4-bit model from the local BF16 dir |
| `scripts/publish.py` | Create the HF repo(s) and upload the folders |
| `tests/reference_torch.py` | PyTorch/`transformers` reference for numerical parity |
| `tests/images/receipt.jpeg` | Sample SROIE receipt |
| `tests/out_*.txt` | Saved model outputs (see Results) |

The converted weights themselves (multi-GB) live on the Hub, not in git.

## Why it works with no model changes

The model is a fine-tune of `Qwen/Qwen3.5-4B` (hybrid gated-delta + full-attention
text backbone, Qwen3-VL vision tower). mlx-vlm's `qwen3_5` implementation covers
exactly this. Verified mappings:

| Checkpoint fact | mlx-vlm handling |
|---|---|
| `model.language_model.*`, `model.visual.*` | `sanitize()` renames to `language_model` / `vision_tower` |
| `full_attention_interval=4` → `layer_types=[L,L,L,F]×8` | `is_linear = (i+1) % 4 != 0` reproduces it exactly |
| packed `q_proj [8192]` (`attn_output_gate`) | split into query + gate (`n_heads*head_dim*2`) |
| interleaved MRoPE (`mrope_section=[11,11,10]`, `partial_rotary_factor=0.25`) | `Qwen3_5RotaryEmbedding(style="interleaved")` |
| Conv1d `[C,1,k]`, 5-D patch embed, RMSNorm +1.0 offset | all applied in `sanitize()` |
| `Qwen3VLProcessor` + single `<\|image_pad\|>` template | reused as-is; `enable_thinking=False` auto-set |

## Note on downloading

On some machines `huggingface_hub`'s Xet downloader hangs on this repo (EU
mirror). If `mlx_vlm.convert` stalls at "Fetching files", download the LFS files
directly with `curl` into the HF cache blobs and convert offline — see
`scripts/finalize_and_convert.sh` (`model.safetensors` blob sha:
`6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67`).

## Results

Benchmarked on Apple M4 (25 GB unified memory). Numerically validated against
the original PyTorch/`transformers` checkpoint — MLX transcribes the sample
receipt with identical content to the reference (only sub-token sampling noise at
`temperature=0.2`), which rules out weight-mapping errors.

**Plain (transcription) — MLX** (`python run_ocr.py plain`):

```
Document No : TD01167104

Date : 25/12/2018 8:13:39 PM

Cashier : MANIS

Member :

## CASH BILL

<table>
  <thead>
    <tr>
      <th>CODE/DESC</th>
      <th>PRICE</th>
      <th>Disc</th>
      <th>AMOUNT</th>
    </tr>
    <tr>
      <th>QTY</th>
      <th>RM</th>
      <th></th>
      <th>RM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9556939040116</td>
      <td>RF MODELLING CLAY KIDDY FISH</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1 PC *</td>
      <td>9.000</td>
      <td>0.00</td>
      <td>9.00</td>
    </tr>
    <tr>
      <td colspan="3">Total :</td>
      <td>9.00</td>
    </tr>
    <tr>
      <td colspan="3">Rour ding Adjustment :</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td colspan="3">Round ed Total (RM):</td>
      <td>9.00</td>
    </tr>
  </tbody>
</table>
```

**Grounding — MLX** (`python run_ocr.py grounding`) — each block gets a
`![label](x0,y0,x1,y1)` box in 0–1000 page coordinates:

```
![image](191,0,864,105) barcode

![text](108,142,737,361) Document No : TD01167104 Date : 25/12/2018 8:13:39 PM Cashier : MANIS Member :

![title](407,395,651,434) ## CASH BILL

![table](44,487,970,970) <table>
  <thead>
    <tr>
      <th>CODE/DESC</th>
      <th>PRICE</th>
      <th>Disc</th>
      <th>AMOUNT</th>
    </tr>
    <tr>
      <th>QTY</th>
      <th>RM</th>
      <th></th>
      <th>RM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9556939040118</td>
      <td>KE MODELLING CLAY KIDDY FISH</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>1 PC *</td>
      <td>9.000</td>
      <td>0.00</td>
      <td>9.00</td>
    </tr>
    <tr>
      <td colspan="3">Total :</td>
      <td>9.00</td>
    </tr>
    <tr>
      <td colspan="3">Rour ding Adjustment :</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td colspan="3">Round ed Total (RM):</td>
      <td>9.00</td>
    </tr>
  </tbody>
</table>
```

Raw transcripts are saved under `tests/`:
`out_plain_mlx.txt`, `out_grounding_mlx.txt`, `out_plain_torch.txt` (reference).

| Build | Decode | Peak RAM |
|---|---|---|
| BF16 | ~11 tok/s | ~8.5 GB |
| 4-bit | ~37 tok/s | ~3.7 GB |

## License

Model weights: Apache-2.0 (per source). Conversion scripts: MIT.

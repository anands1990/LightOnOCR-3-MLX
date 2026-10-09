#!/usr/bin/env python3
"""Render an animated grounding demo of LightOnOCR-3-4B (MLX).

Draws the real bounding boxes the model outputs (0-1000 page coords) onto the
receipt image progressively, while the detected-block list reveals on the right.
Outputs demo/lightonocr_grounding.mp4 (+ .gif) via ffmpeg.
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "frames")
OUTDIR = os.path.join(HERE, "demo")
IMG = os.path.join(HERE, "tests/images/receipt.jpeg")

W, H = 1280, 720
FPS = 30

# ---- palette -------------------------------------------------------------
BG = (13, 17, 23)
PANEL = (22, 27, 34)
FG = (233, 236, 239)
MUTED = (140, 148, 158)
ACCENT = (94, 234, 212)

# ---- the model's actual grounding output (label, 0-1000 box, preview) -----
BLOCKS = [
    dict(label="barcode", box=(191, 0, 864, 105),
         preview="barcode image", color=(240, 193, 0)),
    dict(label="text", box=(108, 142, 737, 361),
         preview="Document No : TD01167104", color=(76, 201, 240)),
    dict(label="title", box=(407, 395, 651, 434),
         preview="## CASH BILL", color=(247, 37, 133)),
    dict(label="table", box=(44, 487, 970, 970),
         preview="table - 6x5 - Total RM 9.00", color=(74, 222, 128)),
]


def font(size, bold=False):
    bold_c = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    reg_c = "/System/Library/Fonts/Supplemental/Arial.ttf"
    mono_c = "/System/Library/Fonts/Menlo.ttc"
    for path in ([bold_c, reg_c] if bold else [reg_c]):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.truetype(mono_c, size, index=0)


F_TITLE = font(30, bold=True)
F_SUB = font(20)
F_HEAD = font(22, bold=True)
F_LABEL = font(24, bold=True)
F_CHIP = font(20, bold=True)
F_PREVIEW = font(21)
F_CAP = font(22, bold=True)

# layout
IMG_SIZE = 470
IX, IY = 72, 150          # image top-left on canvas
PX = 600                  # right panel x


def norm_to_canvas(box):
    x0, y0, x1, y1 = [v / 1000.0 for v in box]
    return (IX + x0 * IMG_SIZE, IY + y0 * IMG_SIZE,
            IX + x1 * IMG_SIZE, IY + y1 * IMG_SIZE)


def rounded(draw, box, rad, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=rad, fill=fill, outline=outline, width=width)


def draw_partial_rect(draw, box, color, width, p):
    """Draw a rectangle outline whose perimeter is filled up to fraction p in [0,1]."""
    x0, y0, x1, y1 = box
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    per = (x1 - x0) + (y1 - y0) + (x1 - x0) + (y1 - y0)
    target = max(0.0, min(1.0, p)) * per
    segs = []
    acc = 0.0
    for i in range(4):
        a = pts[i]
        b = pts[i + 1]
        L = (abs(b[0] - a[0]) + abs(b[1] - a[1])) or 1e-6
        if acc + L <= target:
            segs.append((a, b))
            acc += L
        else:
            r = target - acc
            t = r / L
            c = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            segs.append((a, c))
            acc = target
            break
    for a, b in segs:
        draw.line([a, b], fill=color, width=width)


def render_frame(t):
    base = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(base)

    # ---- title (fade 0 -> 0.6s) ----
    tp = min(1.0, t / 0.6)
    draw.text((60, 44), "LightOnOCR-3-4B", font=F_TITLE, fill=FG)
    draw.text((60 + draw.textlength("LightOnOCR-3-4B", font=F_TITLE) + 18, 52),
              "grounding mode", font=F_HEAD, fill=ACCENT)
    if tp < 1.0:
        # dim the whole header band while fading
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, int(200 * (1 - tp))))
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(base)
        draw.text((60, 44), "LightOnOCR-3-4B", font=F_TITLE, fill=FG)
        draw.text((60 + draw.textlength("LightOnOCR-3-4B", font=F_TITLE) + 18, 52),
                  "grounding mode", font=F_HEAD, fill=ACCENT)
    draw.text((60, 96), "Qwen3.5 vision-language model  -  MLX / Apple Silicon  -  37 tok/s  -  3.7 GB RAM",
              font=F_SUB, fill=MUTED)

    # ---- receipt image (fade in 0.7 -> 1.2s) ----
    fp = max(0.0, min(1.0, (t - 0.7) / 0.5))
    if fp > 0:
        img = Image.open(IMG).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        # panel behind the image
        draw.rounded_rectangle((IX - 10, IY - 10, IX + IMG_SIZE + 10, IY + IMG_SIZE + 10),
                               radius=14, fill=PANEL, outline=(45, 52, 64), width=1)
        if fp < 1.0:
            a = int(255 * fp)
            rgba = img.convert("RGBA")
            rgba.putalpha(a)
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            layer.paste(rgba, (IX, IY), rgba)
            base = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
            draw = ImageDraw.Draw(base)
        else:
            base.paste(img, (IX, IY))
            draw = ImageDraw.Draw(base)
        draw.text((IX, IY + IMG_SIZE + 22), "input: SROIE receipt", font=F_SUB, fill=MUTED)

    # ---- right panel ----
    rounded(draw, (PX - 24, 140, W - 60, H - 90), 16, fill=PANEL)
    draw.text((PX, 158), "detected blocks", font=F_HEAD, fill=ACCENT)

    # ---- blocks ----
    base_i = 48          # frames until first box starts
    stride = 46          # frames per block
    ddraw = 16           # box outline draw frames
    dtext = 22           # panel reveal frames

    for i, blk in enumerate(BLOCKS):
        start = base_i + i * stride
        # box outline progress
        p_draw = max(0.0, min(1.0, (t * FPS - start) / ddraw)) if fp >= 1.0 else 0.0
        box = norm_to_canvas(blk["box"])
        if p_draw > 0 and fp >= 1.0:
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            od.rectangle(box, fill=blk["color"] + (34,))
            base = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
            draw = ImageDraw.Draw(base)
            draw_partial_rect(draw, box, blk["color"], 4, p_draw)
            if p_draw >= 1.0:
                # label chip at box top-left
                label = blk["label"]
                cw = draw.textlength(label, font=F_CHIP)
                chip = (box[0], max(0, box[1] - 34), box[0] + cw + 22, max(0, box[1] - 34) + 30)
                rounded(draw, chip, 7, fill=blk["color"])
                draw.text((chip[0] + 11, chip[1] + 4), label, font=F_CHIP, fill=(10, 12, 16))

        # right-panel row reveal
        tp_row = max(0.0, min(1.0, (t * FPS - (start + ddraw)) / dtext))
        if tp_row > 0:
            ry = 210 + i * 92
            draw.rounded_rectangle((PX, ry, PX + 18, ry + 18), 4, fill=blk["color"])
            num = f"#{i+1}"
            draw.text((PX + 32, ry - 2), num, font=F_SUB, fill=MUTED)
            draw.text((PX + 70, ry - 4), blk["label"], font=F_LABEL, fill=FG)
            prev = blk["preview"]
            n = int(tp_row * len(prev))
            draw.text((PX + 70, ry + 30), prev[:n], font=F_PREVIEW, fill=MUTED)

    # ---- end caption ----
    end_t = (base_i + (len(BLOCKS) - 1) * stride + ddraw + dtext) / FPS
    if t >= end_t:
        draw.text((PX, H - 118), "one model  -  transcription + grounding",
                  font=F_CAP, fill=FG)
        draw.text((PX, H - 88), "github.com/anands1990/LightOnOCR-3-MLX",
                  font=F_SUB, fill=ACCENT)
    return base


def main():
    total_s = 10.0
    n = int(total_s * FPS)
    os.makedirs(FRAMES, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[demo] rendering {n} frames @ {FPS}fps -> {FRAMES}")
    for k in range(n):
        t = k / FPS
        render_frame(t).save(os.path.join(FRAMES, f"f{k:04d}.png"))
        if k % 45 == 0:
            print(f"  frame {k}/{n}")
    mp4 = os.path.join(OUTDIR, "lightonocr_grounding.mp4")
    gif = os.path.join(OUTDIR, "lightonocr_grounding.gif")
    print("[demo] encoding mp4 ...")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i",
                    os.path.join(FRAMES, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[demo] encoding gif ...")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i",
                    os.path.join(FRAMES, "f%04d.png"),
                    "-vf", "fps=18,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                    gif],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[demo] done:")
    print("  ", mp4, os.path.getsize(mp4) // 1024, "KB")
    print("  ", gif, os.path.getsize(gif) // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())

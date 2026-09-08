# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "pillow",
# ]
# ///
"""Reproducible crops for talk/public/figures/.

Each source figure below is a multi-panel image reproduced from a paper.
For slide legibility at 1600x900 projection we crop out the single panel (or
pair of panels) the talk actually discusses, leaving the untouched original
file in place and writing a new file alongside it.

Panel boundaries were found by scanning for the (near-)white gutter rows/
columns that separate matplotlib subplots, then confirmed by eye. Where a
crop's rectangle unavoidably grazes a neighboring panel's caption text (the
qwen2vl-mrope grid, whose shared bottom caption bleeds a few pixels into the
left-half crop), we paint over the stray fragment with the source image's
background white -- the caption itself is not part of the grid panel.

Usage: uv run talk/experiments/crop_figures.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

FIGURES_DIR = Path(__file__).resolve().parent.parent / "public" / "figures"


def crop_conformal_rag_removal() -> None:
    """(c)+(d): bottom row of the 2x2 coverage/removal grid (NeuCLIR, RAGTIME)."""
    src = FIGURES_DIR / "paper-conformal-rag-coverage.png"
    dst = FIGURES_DIR / "paper-conformal-rag-removal.png"
    img = Image.open(src).convert("RGB")
    w, h = img.size
    # Row gutter between the coverage row (a,b) and the removal row (c,d)
    # spans y=1678-1860; crop a few pixels above the "(c)"/"(d)" titles.
    crop = img.crop((0, 1840, w, h))
    crop.save(dst)
    print(f"[crop] wrote {dst} {crop.size}")


def crop_anyloc_radar() -> None:
    """The Recall@1 radar/polygon chart (rightmost panel of the splash figure)."""
    src = FIGURES_DIR / "paper-anyloc-splash.png"
    dst = FIGURES_DIR / "paper-anyloc-radar.png"
    img = Image.open(src).convert("RGB")
    # Content is only in the top ~442px; the bottom half of the 1600x900
    # canvas is blank. The radar chart's leftmost dark pixel (its outline /
    # "Recall@1" label) starts at x=1228; a small margin avoids clipping it
    # while still trimming almost all of the connecting-arrow graphic to its
    # left (a thin sliver of the arrow remains -- the two panels physically
    # overlap in the source image with no clean gutter between them).
    crop = img.crop((1225, 0, 1600, 445))
    crop.save(dst)
    print(f"[crop] wrote {dst} {crop.size}")


def crop_tetra_recall_memory() -> None:
    """Accuracy-vs-model-scale panel (top of 3 stacked, shared-x-axis panels).

    The figure has no single scatter panel plotting Recall@1 directly against
    memory; the closest available content is the accuracy (R@1) panel,
    per the fallback instruction. Because the three panels share one x-axis
    with tick labels drawn only under the bottom (memory) panel, the top
    panel alone would have no x-axis labels. We crop the top panel and the
    bottom panel's tick-label strip separately (both are the same pixel
    width, so their column positions already line up) and stack them.
    """
    src = FIGURES_DIR / "paper-tetra-tradeoff.jpg"
    dst = FIGURES_DIR / "paper-tetra-recall-memory.jpg"
    img = Image.open(src).convert("RGB")
    w, _h = img.size
    accuracy_panel = img.crop((0, 0, w, 741))
    tick_labels = img.crop((0, 2078, w, 2328))
    out = Image.new("RGB", (w, accuracy_panel.height + tick_labels.height), "white")
    out.paste(accuracy_panel, (0, 0))
    out.paste(tick_labels, (0, accuracy_panel.height))
    out.save(dst, quality=95)
    print(f"[crop] wrote {dst} {out.size}")


def crop_qwen2vl_mrope_grid() -> None:
    """Left-half position-decomposition grid, excluding the text-example panel."""
    src = FIGURES_DIR / "paper-qwen2vl-mrope.png"
    dst = FIGURES_DIR / "paper-qwen2vl-mrope-grid.png"
    img = Image.open(src).convert("RGB")
    crop = img.crop((0, 0, 1090, img.size[1]))
    # The figure's shared bottom caption ("Multimodal Rotary Position
    # Embedding (M-RoPE)") is centered under the whole source figure and
    # starts bleeding into this crop's bottom-right corner; it belongs to
    # neither the grid nor is legible fragmented, so paint it out with the
    # source's white background.
    draw = ImageDraw.Draw(crop)
    draw.rectangle((990, 518, crop.width - 1, 585), fill=(255, 255, 255))
    crop.save(dst)
    print(f"[crop] wrote {dst} {crop.size}")


def main() -> None:
    crop_conformal_rag_removal()
    crop_anyloc_radar()
    crop_tetra_recall_memory()
    crop_qwen2vl_mrope_grid()


if __name__ == "__main__":
    main()

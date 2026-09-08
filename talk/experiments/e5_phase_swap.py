# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "numpy",
#     "pillow",
#     "matplotlib",
# ]
# ///
"""Oppenheim & Lim (1981) magnitude/phase swap.

Two of the talk author's own images (poodle, amigurumi bird) are
centre-cropped to a common square, Fourier transformed, and their
magnitude/phase spectra are swapped and recombined. The claim under
test: image identity lives almost entirely in the phase spectrum, not
the magnitude spectrum.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "public" / "figures"
RESULTS_DIR = HERE / "results"

CROP_SIZE = 320

BG = "#12151c"
FG = "#e8ecf1"
MUTED = "#98a2b3"


def load_grey_square(path: Path, size: int) -> np.ndarray:
    """Load an image, convert to greyscale float64 in [0, 1], centre-crop to a size x size square."""
    img = Image.open(path).convert("L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def to_display(x: np.ndarray) -> np.ndarray:
    """Clip a real-valued image to [0, 1] for display."""
    return np.clip(x, 0.0, 1.0)


def stretch_display(x: np.ndarray) -> np.ndarray:
    """Percentile-based contrast stretch of a real-valued image to [0, 1].

    Phase-only and magnitude-only reconstructions do not live on the
    original 0-1 grey scale (unit-magnitude phase reconstructions are near
    zero everywhere; magnitude-only reconstructions with zero phase put a
    single extreme spike at the origin), so they need a per-panel contrast
    stretch to be visible at all. A 1st/99.5th percentile stretch (rather
    than raw min/max) keeps that single spike pixel from crushing the rest
    of the image to black.
    """
    lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.5)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def recombine(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Inverse FFT of magnitude * exp(i * phase), real part."""
    spectrum = magnitude * np.exp(1j * phase)
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))


def normalised_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two same-shape images, flattened."""
    a_flat = a.ravel() - a.mean()
    b_flat = b.ravel() - b.mean()
    denom = np.sqrt(np.sum(a_flat**2) * np.sum(b_flat**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a_flat * b_flat) / denom)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    poodle_path = FIGURES_DIR / "kcca-poodle-query.jpg"
    bird_path = FIGURES_DIR / "kcca-amigurumi-query.jpg"

    a = load_grey_square(poodle_path, CROP_SIZE)  # "A" = poodle
    b = load_grey_square(bird_path, CROP_SIZE)  # "B" = amigurumi bird

    fft_a = np.fft.fftshift(np.fft.fft2(a))
    fft_b = np.fft.fftshift(np.fft.fft2(b))

    mag_a, phase_a = np.abs(fft_a), np.angle(fft_a)
    mag_b, phase_b = np.abs(fft_b), np.angle(fft_b)

    # Hybrids: magnitude of one, phase of the other.
    hybrid_magA_phaseB = to_display(recombine(mag_a, phase_b))
    hybrid_magB_phaseA = to_display(recombine(mag_b, phase_a))

    # Phase-only: unit magnitude, each image's own phase. Contrast-stretched:
    # the natural amplitude of a unit-magnitude reconstruction is tiny (~1e-2)
    # and would render as flat black under a [0, 1] clip.
    phase_only_a = stretch_display(recombine(np.ones_like(mag_a), phase_a))
    phase_only_b = stretch_display(recombine(np.ones_like(mag_b), phase_b))

    # Magnitude-only: each image's own magnitude, zero phase. Contrast-stretched:
    # zero phase concentrates energy near the origin, producing a bright spike
    # that saturates a [0, 1] clip.
    mag_only_a = stretch_display(recombine(mag_a, np.zeros_like(phase_a)))
    mag_only_b = stretch_display(recombine(mag_b, np.zeros_like(phase_b)))

    # Normalise the a/b originals the same way the hybrids were built (real part of an
    # identity recombination) so the correlation comparison is apples-to-apples.
    a_recon = to_display(recombine(mag_a, phase_a))
    b_recon = to_display(recombine(mag_b, phase_b))

    correlations = {
        "magA_phaseB__vs__A": normalised_cross_correlation(hybrid_magA_phaseB, a_recon),
        "magA_phaseB__vs__B": normalised_cross_correlation(hybrid_magA_phaseB, b_recon),
        "magB_phaseA__vs__A": normalised_cross_correlation(hybrid_magB_phaseA, a_recon),
        "magB_phaseA__vs__B": normalised_cross_correlation(hybrid_magB_phaseA, b_recon),
    }

    results = {
        "crop_size": CROP_SIZE,
        "images": {"A": "kcca-poodle-query.jpg (poodle)", "B": "kcca-amigurumi-query.jpg (bird)"},
        "correlations": correlations,
        "phase_donor_wins": {
            # A hybrid should correlate more strongly with the image whose PHASE it carries.
            "magA_phaseB_correlates_more_with_B": (
                correlations["magA_phaseB__vs__B"] > correlations["magA_phaseB__vs__A"]
            ),
            "magB_phaseA_correlates_more_with_A": (
                correlations["magB_phaseA__vs__A"] > correlations["magB_phaseA__vs__B"]
            ),
        },
    }

    results_path = RESULTS_DIR / "e5_phase_swap.json"
    results_path.write_text(json.dumps(results, indent=2))

    # --- Figure ---
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": FG,
        }
    )

    fig = plt.figure(figsize=(16, 9.0), dpi=150)
    gs = fig.add_gridspec(
        2, 4, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.06, left=0.02, right=0.98, top=0.96, bottom=0.09
    )

    row1 = [
        (a, "A: poodle (original)"),
        (hybrid_magA_phaseB, "magnitude of poodle, phase of bird"),
        (hybrid_magB_phaseA, "magnitude of bird, phase of poodle"),
        (b, "B: bird (original)"),
    ]
    row2 = [
        (phase_only_a, "phase-only, poodle"),
        (mag_only_a, "magnitude-only, poodle"),
        (mag_only_b, "magnitude-only, bird"),
        (phase_only_b, "phase-only, bird"),
    ]

    for col, (img, label) in enumerate(row1):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, color=MUTED, fontsize=15, pad=8, y=-0.16, verticalalignment="top")

    for col, (img, label) in enumerate(row2):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, color=MUTED, fontsize=14, pad=6, y=-0.18, verticalalignment="top")

    fig_path = FIGURES_DIR / "exp-e5-phase-swap.png"
    fig.savefig(fig_path, facecolor=BG)
    plt.close(fig)

    print(f"Wrote {results_path}")
    print(f"Wrote {fig_path}")
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()

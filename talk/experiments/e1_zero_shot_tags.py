# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch",
#   "open_clip_torch",
#   "transformers",
#   "sentencepiece",
#   "pillow",
#   "numpy",
#   "matplotlib",
#   "requests",
# ]
# ///
"""E1: zero-shot tag ranking over the real MIRFlickr-25k vocabulary.

Ranks the MIRFlickr-25k tag vocabulary against three query images with OpenAI
CLIP ViT-B/32 (via open_clip) and, optionally, SigLIP. Compares against the
2018 hand-built KCCA tag-denoising system that used the same images and the
same tag vocabulary (see talk/slides.md, "My favourite result from my PhD").

Downloads mirflickr25k.zip (~2.9 GB) into talk/experiments/data/ if not
already present, and reads the tag vocabulary directly out of the zip without
extracting the 25,000 images it also contains.

Usage: uv run talk/experiments/e1_zero_shot_tags.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import torch

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE.parent / "public" / "figures"


os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))

import open_clip  # noqa: E402  (import after HF_HOME is set)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

MIRFLICKR_URL = "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip"
MIRFLICKR_MD5 = "a23d0a8564ee84cda5622a6c2f947785"
MIRFLICKR_ZIP = DATA_DIR / "mirflickr25k.zip"
MIRFLICKR_SIZE = 3_069_184_257  # bytes, for a resume sanity check

# Ground-truth Flickr tags for the three query images, taken from the 2018
# dissertation figure captions reproduced in talk/slides.md ("My favourite
# result from my PhD" and "The same system's failure row").
QUERIES = {
    "poodle": {
        "path": FIGURES_DIR / "kcca-poodle-query.jpg",
        "ground_truth": ["red", "dog", "poodle", "standard", "standardpoodle"],
        "label": "poodle",
    },
    "amigurumi": {
        "path": FIGURES_DIR / "kcca-amigurumi-query.jpg",
        "ground_truth": ["yellow", "bird", "handmade", "crochet", "amigurimi"],
        "label": "bird",
    },
    "failure": {
        "path": FIGURES_DIR / "kcca-failure-query.jpg",
        "ground_truth": ["green", "pink", "home", "studio", "owl"],
        "label": "owl",
    },
}

# Vocabulary tags that are the obvious correct concept for a query image but
# do not literally match that image's ground-truth tag string (e.g. a
# spelling variant that never crossed the >= 20-occurrence vocabulary
# threshold). Rendered as a hit in the figure, with an explanatory note.
NEAR_MISS_TAGS = {
    "amigurumi": {
        "image": "amigurumi",
        "note": "Flickr tag spelled \u2018amigurimi\u2019; absent from the vocabulary",
    },
}

TEMPLATES = {
    "a photo of {tag}": "a photo of {tag}",
    "bare tag": "{tag}",
}
FIGURE_TEMPLATE = "a photo of {tag}"

MODEL_SPECS = [
    {
        "key": "clip-vit-b-32-openai",
        "arch": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "required": True,
    },
    {
        "key": "siglip-vit-b-16-webli",
        "arch": "ViT-B-16-SigLIP",
        "pretrained": "webli",
        "required": False,
    },
    {
        "key": "clip-vit-l-14-openai",
        "arch": "ViT-L-14-quickgelu",
        "pretrained": "openai",
        "required": False,
    },
    {
        "key": "siglip-so400m-webli",
        "arch": "ViT-SO400M-14-SigLIP",
        "pretrained": "webli",
        "required": False,
        # so400m is ~877M params; fall back to a smaller SigLIP if it does
        # not fit in 16 GB of unified memory.
        "fallback": {
            "key": "siglip-vit-l-16-256-webli",
            "arch": "ViT-L-16-SigLIP-256",
            "pretrained": "webli",
        },
    },
]

# Short display names for the scale-comparison figure. models_out preserves
# MODEL_SPECS iteration order (small -> large), so no separate ordering list
# is needed; a fallback key (e.g. if so400m does not fit) still resolves here.
MODEL_DISPLAY_NAMES = {
    "clip-vit-b-32-openai": "CLIP ViT-B/32",
    "siglip-vit-b-16-webli": "SigLIP ViT-B/16",
    "clip-vit-l-14-openai": "CLIP ViT-L/14",
    "siglip-so400m-webli": "SigLIP SoViT-400M/14",
    "siglip-vit-l-16-256-webli": "SigLIP ViT-L/16-256",
}


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_mirflickr() -> None:
    """Resume-capable streaming download of mirflickr25k.zip, MD5-checked."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MIRFLICKR_ZIP.exists():
        head = requests.head(MIRFLICKR_URL, timeout=30, allow_redirects=True)
        remote_size = int(head.headers.get("content-length", MIRFLICKR_SIZE))
        local_size = MIRFLICKR_ZIP.stat().st_size
        if local_size >= remote_size:
            print(f"[e1] {MIRFLICKR_ZIP.name} already present ({local_size} bytes); verifying md5")
            digest = md5sum(MIRFLICKR_ZIP)
            if digest.lower() == MIRFLICKR_MD5.lower():
                print("[e1] md5 OK, skipping download")
                return
            print(f"[e1] md5 mismatch (got {digest}), re-downloading from scratch")
            MIRFLICKR_ZIP.unlink()

    resume_from = MIRFLICKR_ZIP.stat().st_size if MIRFLICKR_ZIP.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    mode = "ab" if resume_from else "wb"
    print(f"[e1] downloading {MIRFLICKR_URL} from byte {resume_from}")
    with requests.get(MIRFLICKR_URL, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with MIRFLICKR_ZIP.open(mode) as f:
            downloaded = resume_from
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)

    digest = md5sum(MIRFLICKR_ZIP)
    if digest.lower() != MIRFLICKR_MD5.lower():
        raise RuntimeError(f"mirflickr25k.zip md5 mismatch: expected {MIRFLICKR_MD5}, got {digest}")
    print("[e1] download complete, md5 verified")


TAG_FILE_RE = re.compile(r"meta/tags/(?:tags|im)(\d+)\.txt$", re.IGNORECASE)
COMMON_TAGS_RE = re.compile(r"doc/common_tags\.txt$", re.IGNORECASE)


def load_vocabulary(zf: zipfile.ZipFile) -> tuple[list[str], str]:
    """Return (vocabulary, provenance note)."""
    names = zf.namelist()
    doc_candidates = [n for n in names if COMMON_TAGS_RE.search(n)]
    tag_files = sorted(
        (n for n in names if TAG_FILE_RE.search(n)),
        key=lambda n: int(TAG_FILE_RE.search(n).group(1)),
    )

    if doc_candidates:
        doc_name = doc_candidates[0]
        raw = zf.read(doc_name).decode("utf-8", errors="replace")
        vocab: list[str] = []
        seen = set()
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            tag = line.split()[0].strip().lower()
            if tag and tag not in seen:
                seen.add(tag)
                vocab.append(tag)
        provenance = (
            f"verbatim from {doc_name} inside mirflickr25k.zip "
            f"({len(vocab)} tags, one per line in the file's own frequency order)"
        )
        return vocab, provenance

    if not tag_files:
        raise RuntimeError(
            "found neither doc/common_tags.txt nor meta/tags/*.txt inside mirflickr25k.zip"
        )

    print(f"[e1] no common_tags.txt found; building vocabulary from {len(tag_files)} per-image tag files")
    counts: Counter[str] = Counter()
    for i, name in enumerate(tag_files):
        raw = zf.read(name).decode("utf-8", errors="replace")
        tags = {t.strip().lower() for t in raw.splitlines() if t.strip()}
        counts.update(tags)
        if (i + 1) % 5000 == 0:
            print(f"[e1]   scanned {i + 1}/{len(tag_files)} tag files")

    vocab = sorted((t for t, c in counts.items() if c >= 20), key=lambda t: (-counts[t], t))
    provenance = (
        f"built by this script: every tag occurring in >= 20 of the {len(tag_files)} "
        f"per-image files under mirflickr/meta/tags/, sorted by descending frequency "
        f"({len(vocab)} tags). mirflickr/doc/common_tags.txt was not present in the archive."
    )
    return vocab, provenance


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def encode_text_batched(model, tokenizer, texts: list[str], device: str, batch_size: int = 256) -> torch.Tensor:
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = tokenizer(batch).to(device)
            f = model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def encode_image(model, preprocess, image_path: Path, device: str) -> torch.Tensor:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu()


def rank_vocab(image_feat: torch.Tensor, text_feats: torch.Tensor, vocab: list[str]) -> list[tuple[str, float]]:
    scores = (image_feat @ text_feats.T).squeeze(0).numpy()
    order = np.argsort(-scores)
    return [(vocab[i], float(scores[i])) for i in order]


def ground_truth_ranks(ranking: list[tuple[str, float]], ground_truth: list[str]) -> dict[str, int | None]:
    rank_of = {tag: r + 1 for r, (tag, _score) in enumerate(ranking)}
    return {gt: rank_of.get(gt) for gt in ground_truth}


def _load_and_run_one(key: str, arch: str, pretrained: str, vocab: list[str], device: str) -> dict:
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, cache_dir=str(CACHE_DIR)
    )
    tokenizer = open_clip.get_tokenizer(arch)

    model.eval()
    try:
        model = model.to(device)
        dev = device
    except Exception:
        model = model.to("cpu")
        dev = "cpu"
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"[e1] {key} loaded on {dev} ({num_parameters / 1e6:.0f}M params)")

    image_feats = {}
    for name, q in QUERIES.items():
        image_feats[name] = encode_image(model, preprocess, q["path"], dev)

    template_results = {}
    for template_name, template in TEMPLATES.items():
        texts = [template.format(tag=tag) for tag in vocab]
        text_feats = encode_text_batched(model, tokenizer, texts, dev)

        per_image = {}
        for name, q in QUERIES.items():
            ranking = rank_vocab(image_feats[name], text_feats, vocab)
            top10 = ranking[:10]
            top5_tags = [t for t, _ in ranking[:5]]
            gt_set = set(q["ground_truth"])
            per_image[name] = {
                "label": q["label"],
                "ground_truth": q["ground_truth"],
                "top10": [[t, round(s, 4)] for t, s in top10],
                "top5_hits": [t for t in top5_tags if t in gt_set],
                "top5_hit_count": sum(1 for t in top5_tags if t in gt_set),
                "ground_truth_ranks": ground_truth_ranks(ranking, q["ground_truth"]),
            }
        template_results[template_name] = per_image

    del model
    return {
        "arch": arch,
        "pretrained": pretrained,
        "device": dev,
        "num_parameters": int(num_parameters),
        "templates": template_results,
    }


def run_model(spec: dict, vocab: list[str], device: str) -> tuple[str, dict | None]:
    """Run one model spec, falling back to spec["fallback"] on failure (e.g. OOM)."""
    try:
        return spec["key"], _load_and_run_one(spec["key"], spec["arch"], spec["pretrained"], vocab, device)
    except Exception as e:  # pragma: no cover - only hit if a model fails to load/fit
        fallback = spec.get("fallback")
        if fallback is not None:
            print(f"[e1] {spec['key']} failed ({e}); falling back to {fallback['key']}")
            try:
                result = _load_and_run_one(
                    fallback["key"], fallback["arch"], fallback["pretrained"], vocab, device
                )
                result["fallback_from"] = {
                    "key": spec["key"],
                    "arch": spec["arch"],
                    "pretrained": spec["pretrained"],
                    "reason": str(e),
                }
                return fallback["key"], result
            except Exception as e2:
                if spec["required"]:
                    raise
                print(f"[e1] fallback {fallback['key']} also failed: {e2}")
                return spec["key"], None
        if spec["required"]:
            raise
        print(f"[e1] skipping optional model {spec['key']}: {e}")
        return spec["key"], None


def make_figure(clip_result: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    BG = "#12151c"
    TEXT = "#e8ecf1"
    MUTED = "#98a2b3"
    EMERALD = "#34d399"
    BLUE = "#7cc4ff"
    SHARED_XMAX = 0.35  # shared cosine-similarity x-axis across all three bar panels

    template_data = clip_result["templates"][FIGURE_TEMPLATE]
    order = ["poodle", "amigurumi", "failure"]

    fig, axes = plt.subplots(
        3, 2, figsize=(16, 9), dpi=150, gridspec_kw={"width_ratios": [1, 2.6]}
    )
    fig.patch.set_facecolor(BG)

    for row, name in enumerate(order):
        q = QUERIES[name]
        d = template_data[name]

        ax_img = axes[row][0]
        ax_img.set_facecolor(BG)
        img = Image.open(q["path"]).convert("RGB")
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(False)
        ax_img.set_ylabel(
            q["label"], color=TEXT, fontsize=16, fontweight="bold", labelpad=10
        )

        ax_bar = axes[row][1]
        ax_bar.set_facecolor(BG)
        top5 = d["top10"][:5][::-1]  # reverse so rank 1 is on top
        tags = [t for t, _ in top5]
        scores = [s for _, s in top5]
        near_miss = NEAR_MISS_TAGS.get(name)
        near_miss_tag = near_miss["image"] if near_miss else None
        gt_set = set(q["ground_truth"])
        colors = [EMERALD if (t in gt_set or t == near_miss_tag) else BLUE for t in tags]

        y = np.arange(len(tags))
        ax_bar.barh(y, scores, color=colors, height=0.6)
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(tags, color=TEXT, fontsize=14)
        ax_bar.tick_params(axis="x", colors=MUTED, labelsize=13)
        for spine_name, spine in ax_bar.spines.items():
            if spine_name in ("top", "right"):
                spine.set_visible(False)
            else:
                spine.set_color(MUTED)
        ax_bar.set_xlim(0, SHARED_XMAX)
        for yi, s in zip(y, scores):
            ax_bar.text(
                s + SHARED_XMAX * 0.02, yi, f"{s:.3f}", va="center", ha="left",
                color=MUTED, fontsize=12,
            )
        xlabel_parts = []
        if row == len(order) - 1:
            xlabel_parts.append("cosine similarity")
        if near_miss:
            xlabel_parts.append(near_miss["note"])
        if xlabel_parts:
            ax_bar.set_xlabel("   |   ".join(xlabel_parts), color=MUTED, fontsize=13)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=EMERALD, label="in Flickr ground truth"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="not in ground truth"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.68, 1.02),
        ncol=2,
        frameon=False,
        labelcolor=TEXT,
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def make_owl_rank_figure(models_out: dict, out_path: Path) -> None:
    """Bar chart: rank of the tag 'owl' (for the failure image) vs model scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG = "#12151c"
    TEXT = "#e8ecf1"
    MUTED = "#98a2b3"
    ORANGE = "#e8894a"

    keys = list(models_out.keys())
    ranks = [
        models_out[k]["templates"][FIGURE_TEMPLATE]["failure"]["ground_truth_ranks"]["owl"]
        for k in keys
    ]
    labels = [
        f"{MODEL_DISPLAY_NAMES.get(k, k)}\n{models_out[k]['num_parameters'] / 1e6:.0f}M params"
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = np.arange(len(keys))
    ax.bar(x, ranks, color=ORANGE, width=0.55)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=MUTED, fontsize=14)
    ax.tick_params(axis="y", colors=MUTED, labelsize=13)
    ax.set_ylabel("rank of 'owl' among 1,386 tags (log scale)", color=TEXT, fontsize=15)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(MUTED)
    ax.grid(axis="y", which="major", color=MUTED, alpha=0.15, linewidth=0.8)
    for xi, r in zip(x, ranks):
        ax.text(xi, r * 1.1, str(r), ha="center", va="bottom", color=TEXT, fontsize=14)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the MIRFlickr download and model inference; re-render "
        "exp-e1-zero-shot-tags.png from the existing results JSON.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "e1_zero_shot_tags.json"

    if args.plot_only:
        output = json.loads(results_path.read_text())
        clip_key = output["meta"]["figure_model"]
        figure_path = FIGURES_DIR / "exp-e1-zero-shot-tags.png"
        make_figure(output["models"][clip_key], figure_path)
        print(f"[e1] wrote {figure_path}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_mirflickr()
    with zipfile.ZipFile(MIRFLICKR_ZIP) as zf:
        vocab, provenance = load_vocabulary(zf)

    print(f"[e1] vocabulary: {len(vocab)} tags ({provenance})")
    vocab_path = RESULTS_DIR / "mirflickr_vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")

    device = get_device()
    print(f"[e1] device: {device}")

    models_out = {}
    for spec in MODEL_SPECS:
        print(f"[e1] running {spec['key']} ({spec['arch']}, pretrained={spec['pretrained']})")
        key, result = run_model(spec, vocab, device)
        if result is not None:
            models_out[key] = result

    clip_key = "clip-vit-b-32-openai"
    figure_path = FIGURES_DIR / "exp-e1-zero-shot-tags.png"
    make_figure(models_out[clip_key], figure_path)

    owl_rank_figure_path = FIGURES_DIR / "exp-e1-owl-rank-vs-scale.png"
    make_owl_rank_figure(models_out, owl_rank_figure_path)

    output = {
        "meta": {
            "vocab_file": str(vocab_path.relative_to(HERE)),
            "vocab_size": len(vocab),
            "vocab_provenance": provenance,
            "seed": SEED,
            "templates": list(TEMPLATES.keys()),
            "figure_template": FIGURE_TEMPLATE,
            "figure_model": clip_key,
            "figure_path": str(figure_path.relative_to(HERE.parent)),
            "owl_rank_figure_path": str(owl_rank_figure_path.relative_to(HERE.parent)),
        },
        "models": models_out,
    }

    results_path.write_text(json.dumps(output, indent=2))
    print(f"[e1] wrote {results_path}")
    print(f"[e1] wrote {figure_path}")
    print(f"[e1] wrote {owl_rank_figure_path}")


if __name__ == "__main__":
    main()

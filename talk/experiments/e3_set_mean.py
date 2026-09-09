# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "gensim>=4.3.3",
#     "sentence-transformers>=3.0",
#     "numpy<2",
#     "scipy",
#     "matplotlib",
# ]
# ///
"""Reproduces the 2018 "colour word overwhelms the plain average" pooling
failure (dissertation ch. 2, KCCA tag-denoising baseline) with real
word/text vectors, ranking the actual MIRFlickr tag vocabulary by cosine
to the plain mean -- the same operation the 2018 system performed.

Two photos from the 2018 figure:
  - poodle: a red standard poodle. Flickr tags: red, dog, poodle, standard,
    standardpoodle. Plain-average pooling in 2018 returned dog, explore,
    red, green, blue -- the colour word swamped the object. Weighted
    pooling returned dog, dogs, poodle, standard, standardpoodle.
  - bird: a yellow crochet bird (amigurumi). Flickr tags: yellow, bird,
    handmade, crochet, amigurimi [sic]. Plain-average pooling returned
    macro, explored. Weighted pooling returned handmade, etsy.

For each photo's tag set and each of two vector sources -- GloVe 6B 300d
(the 2018-era vector type) and BAAI/bge-small-en-v1.5 (a modern sentence
embedding model) -- this script:
  1. Builds the plain mean of the tag-set's member vectors.
  2. Ranks the MIRFlickr candidate vocabulary (1,386 real Flickr tags, the
     same universe of tags the 2018 system pooled over) by cosine to that
     mean, and reports the top 10.
  3. Repeats the ranking for a mean that excludes the query's own colour
     word, to see what the colour word alone drags into the neighbourhood.
  4. Reports the gap between (a) the average cosine of the mean to its own
     members and (b) the median cosine of the mean to the rest of the
     vocabulary -- the geometric mechanism behind the failure.

Also emits a compact browser fixture (talk/public/data/set-mean-vocab.json +
set-mean-vocab.f32) for the SetMean.vue slide component: every MIRFlickr
vocabulary word's raw bge-small-en-v1.5 embedding (384-d, for exact cosine
search in the browser) plus a single fixed 2-component PCA projection of the
same embeddings (for display only -- never used for the nearest-neighbour
search).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("GENSIM_DATA_DIR", str(EXPERIMENTS_DIR / "data" / "gensim-data"))
os.environ.setdefault("HF_HOME", str(EXPERIMENTS_DIR / ".cache"))
os.environ.setdefault("TORCH_HOME", str(EXPERIMENTS_DIR / ".cache"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(EXPERIMENTS_DIR / ".cache"))

import gensim.downloader as gensim_api  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SEED = 0
np.random.seed(SEED)

RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = EXPERIMENTS_DIR.parent / "public" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = EXPERIMENTS_DIR.parent / "public" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GLOVE_MODEL_NAME = "glove-wiki-gigaword-300"
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VOCAB_PATH = RESULTS_DIR / "mirflickr_vocab.txt"

# -- 2018 tag sets, from the dissertation's KCCA pooling baseline figure --
TAG_SETS = {
    "poodle": {
        "query_note": (
            "red standard poodle; Flickr tags: red, dog, poodle, standard, "
            "standardpoodle"
        ),
        "members": ["red", "dog", "poodle", "standard", "standardpoodle"],
        "colour_word": "red",
    },
    "bird": {
        "query_note": (
            "yellow crochet bird; Flickr tags: yellow, bird, handmade, crochet, "
            "amigurimi (Flickr's spelling; 'amigurumi' is the correct spelling "
            "and is what we embed for both vector sources)"
        ),
        "members": ["yellow", "bird", "handmade", "crochet", "amigurumi"],
        "colour_word": "yellow",
    },
}

# The bird tag set is the slide's real interactive example.
QUERY_IMAGE_PATH = "/figures/kcca-amigurumi-query.jpg"
INITIAL_TAGS = TAG_SETS["bird"]["members"]

# Per Main's explicit list: named colours plus colour/color spelling variants
# present in the MIRFlickr vocabulary.
COLOUR_WORDS = {
    "red", "green", "blue", "yellow", "pink", "orange", "purple", "black", "white",
    "color", "colors", "colour", "colours", "colorful", "colourful", "coloured", "colored",
}

BG = "#12151c"
FG = "#e8ecf1"
MUTED = "#98a2b3"
ORANGE = "#e8894a"
BLUE = "#7cc4ff"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_compound_split(word: str, key_to_index: dict) -> tuple[str, str] | None:
    """Try every two-way split of `word`; return the split (by combined
    frequency rank, lower = more frequent) where both halves are in-vocab,
    or None if no such split exists."""
    candidates = []
    for i in range(2, len(word) - 1):
        a, b = word[:i], word[i:]
        if a in key_to_index and b in key_to_index:
            candidates.append((a, b, key_to_index[a] + key_to_index[b]))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[2])
    return candidates[0][0], candidates[0][1]


def glove_member_vectors(words: list[str], model) -> tuple[dict, dict]:
    """Vectors for the 5 tag-set member words specifically (not the
    candidate vocabulary): real GloVe vector if in-vocab, else the mean of
    an in-vocab two-part compound split, else omitted with a note."""
    vectors: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}
    for w in words:
        if w in model.key_to_index:
            vectors[w] = np.asarray(model[w], dtype=np.float64)
            continue
        split = find_compound_split(w, model.key_to_index)
        if split is not None:
            a, b = split
            va = np.asarray(model[a], dtype=np.float64)
            vb = np.asarray(model[b], dtype=np.float64)
            vectors[w] = (va + vb) / 2.0
            notes[w] = f"not in GloVe vocab; used mean of subword split {a!r} + {b!r}"
        else:
            notes[w] = (
                "not in GloVe vocab; no two-part in-vocab subword split found "
                "(checked every split point); vector unavailable, excluded from mean"
            )
    return vectors, notes


def load_mirflickr_vocab() -> list[str]:
    return [line.strip() for line in VOCAB_PATH.read_text().splitlines() if line.strip()]


def glove_candidate_vectors(vocab: list[str], model) -> tuple[dict, list[str]]:
    """Candidate-vocabulary vectors for GloVe: skip any MIRFlickr tag not
    literally present in GloVe's own 400k-word vocabulary."""
    vectors: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for w in vocab:
        if w in model.key_to_index:
            vectors[w] = np.asarray(model[w], dtype=np.float64)
        else:
            skipped.append(w)
    return vectors, skipped


def bge_candidate_vectors(vocab: list[str], model: SentenceTransformer) -> tuple[dict, list[str]]:
    embeddings = model.encode(
        vocab, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False
    )
    vectors = {w: np.asarray(v, dtype=np.float64) for w, v in zip(vocab, embeddings)}
    return vectors, []


def pca_fit_transform(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-center and SVD. Axis signs are fixed deterministically -- each
    retained component is flipped, if needed, so its largest-magnitude
    loading is positive -- so the projection does not depend on SVD's
    arbitrary sign convention. Returns (scores[N, n_components],
    components[n_components, D], explained_variance_ratio[n_components])."""
    centered = x - x.mean(axis=0)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].copy()
    for i in range(n_components):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] *= -1
    scores = centered @ components.T
    explained_variance = (s**2) / (x.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    return scores, components, explained_variance_ratio[:n_components]


def write_browser_fixture(vocab: list[str], bge_candidates: dict[str, np.ndarray], bge_dim: int) -> Path:
    """Emits the compact browser fixture consumed by SetMean.vue:
      - set-mean-vocab.f32: every MIRFlickr vocabulary word's raw
        bge-small-en-v1.5 embedding, float32, row-major, one row per word in
        `vocab` order -- the only thing the browser's cosine search touches.
      - set-mean-vocab.json: that same vocabulary order, a fixed 2-component
        PCA projection of the same embeddings for display only, and the
        initial example (image + tags) and provenance.
    """
    vectors = np.stack([bge_candidates[w] for w in vocab])
    scores, _components, evr = pca_fit_transform(vectors, n_components=2)

    bin_path = DATA_DIR / "set-mean-vocab.f32"
    vectors.astype("<f4").tofile(bin_path)

    metadata = {
        "provenance": {
            "generated_by": "talk/experiments/e3_set_mean.py",
            "seed": SEED,
            "model": BGE_MODEL_NAME,
            "dim": bge_dim,
            "vocab_source_file": "talk/experiments/results/mirflickr_vocab.txt",
        },
        "vocab_size": len(vocab),
        "vocabulary": vocab,
        "vectors": {
            "file": "/data/set-mean-vocab.f32",
            "dtype": "float32",
            "byte_order": "little-endian",
            "shape": [len(vocab), bge_dim],
            "row_order_matches": "vocabulary",
            "note": (
                "raw (unnormalised) bge-small-en-v1.5 sentence embeddings, one row "
                "per vocabulary word in the same order as `vocabulary`. This is the "
                "original 384-d space: the nearest-neighbour search runs here, by "
                "cosine, never on the 2-d projection below."
            ),
        },
        "projection": {
            "method": (
                "PCA, 2 components, fit once (via full SVD) on all vocabulary "
                "embeddings; axis signs fixed by making each component's "
                "largest-magnitude loading positive"
            ),
            "explained_variance_ratio": [float(v) for v in evr],
            "coordinates": [[float(row[0]), float(row[1])] for row in scores],
            "coordinate_order_matches": "vocabulary",
            "note": "for display only -- not used for the nearest-neighbour search.",
        },
        "initial_example": {
            "image": QUERY_IMAGE_PATH,
            "tags": INITIAL_TAGS,
            "note": TAG_SETS["bird"]["query_note"],
        },
    }
    meta_path = DATA_DIR / "set-mean-vocab.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return meta_path


def rank_candidates(mean_vec: np.ndarray, candidate_vectors: dict, topn: int = 10):
    cosines = {w: cosine(mean_vec, v) for w, v in candidate_vectors.items()}
    ranked = sorted(cosines.keys(), key=lambda w: cosines[w], reverse=True)
    top = [{"word": w, "cosine": cosines[w]} for w in ranked[:topn]]
    return top, cosines


def categorize(word: str, members: list[str]) -> str:
    if word in members:
        return "member"
    if word in COLOUR_WORDS:
        return "colour"
    return "other"


def analyze_source(tag_set: dict, member_vectors: dict, candidate_vectors: dict, skipped: list[str]) -> dict:
    members = tag_set["members"]
    colour_word = tag_set["colour_word"]

    present_members = [w for w in members if w in member_vectors]
    missing_members = [w for w in members if w not in member_vectors]
    member_vecs = np.stack([member_vectors[w] for w in present_members])
    mean_vec = member_vecs.mean(axis=0)

    no_colour_members = [w for w in present_members if w != colour_word]
    no_colour_vecs = np.stack([member_vectors[w] for w in no_colour_members])
    mean_no_colour = no_colour_vecs.mean(axis=0)

    top10_full, cosines_full = rank_candidates(mean_vec, candidate_vectors, topn=10)
    top10_no_colour, _ = rank_candidates(mean_no_colour, candidate_vectors, topn=10)

    member_cosines = [cosines_full[w] for w in present_members if w in cosines_full]
    avg_member_cosine = float(np.mean(member_cosines)) if member_cosines else None

    other_cosines = [c for w, c in cosines_full.items() if w not in members]
    median_other_cosine = float(np.median(other_cosines)) if other_cosines else None

    gap = (
        avg_member_cosine - median_other_cosine
        if avg_member_cosine is not None and median_other_cosine is not None
        else None
    )

    return {
        "members_present": present_members,
        "members_missing": missing_members,
        "candidate_vocab_size": len(candidate_vectors),
        "candidate_vocab_skipped": len(skipped),
        "candidate_vocab_skipped_words": skipped,
        "top10_nearest_vocab_tags": top10_full,
        "top10_nearest_vocab_tags_excluding_colour_word": top10_no_colour,
        "colour_word_excluded": colour_word,
        "avg_cosine_mean_to_own_members": avg_member_cosine,
        "median_cosine_mean_to_other_vocab_tags": median_other_cosine,
        "gap_member_minus_other_median": gap,
        "_mean_vector": mean_vec,
    }


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def make_figure(figure_data: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_names = list(TAG_SETS.keys())  # poodle, bird
    sources = [("glove", "GloVe 300d (2018-era)"), ("bge", "bge-small-en-v1.5 (modern)")]

    fig, axes = plt.subplots(len(set_names), len(sources), figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor(BG)

    for row, set_name in enumerate(set_names):
        members = TAG_SETS[set_name]["members"]
        for col, (source_key, source_label) in enumerate(sources):
            ax = axes[row, col]
            ax.set_facecolor(BG)
            stats = figure_data[set_name][source_key]
            top10 = stats["top10_nearest_vocab_tags"]

            words = [entry["word"] for entry in top10]
            values = [entry["cosine"] for entry in top10]
            categories = [categorize(w, members) for w in words]
            colors = [
                BLUE if c == "member" else ORANGE if c == "colour" else MUTED
                for c in categories
            ]

            # sort ascending so the strongest match is drawn at the top
            order = sorted(range(len(words)), key=lambda i: values[i])
            words = [words[i] for i in order]
            values = [values[i] for i in order]
            colors = [colors[i] for i in order]

            y_pos = np.arange(len(words))
            ax.barh(y_pos, values, color=colors, height=0.62)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, color=FG, fontsize=14)
            ax.tick_params(axis="x", colors=MUTED, labelsize=13)
            ax.set_xlabel("cosine to plain mean", color=MUTED, fontsize=15)
            ax.axvline(0, color=MUTED, linewidth=0.8)
            for spine_name, spine in ax.spines.items():
                if spine_name in ("top", "right"):
                    spine.set_visible(False)
                else:
                    spine.set_color(MUTED)
            ax.text(
                0.02,
                1.06,
                f"{set_name} \u00b7 {source_label} \u00b7 top-10 nearest MIRFlickr tags",
                transform=ax.transAxes,
                color=FG,
                fontsize=14,
                va="bottom",
                ha="left",
                fontweight="bold",
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="member (tag on the photo)"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="colour word"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, label="other vocabulary tag"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, -0.02),
    )
    for text in legend.get_texts():
        text.set_color(FG)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path = FIGURES_DIR / "exp-e3-set-mean.png"
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    import torch

    torch.manual_seed(SEED)

    print(f"Loading GloVe ({GLOVE_MODEL_NAME}) ...")
    glove = gensim_api.load(GLOVE_MODEL_NAME)
    print(f"Loading {BGE_MODEL_NAME} ...")
    bge = SentenceTransformer(BGE_MODEL_NAME, device="cpu")

    vocab = load_mirflickr_vocab()
    print(f"Loaded {len(vocab)} MIRFlickr candidate tags")

    glove_candidates, glove_skipped = glove_candidate_vectors(vocab, glove)
    bge_candidates, bge_skipped = bge_candidate_vectors(vocab, bge)
    print(f"GloVe: {len(glove_skipped)} / {len(vocab)} candidate tags not in GloVe vocab, skipped")
    print(f"bge: {len(bge_skipped)} / {len(vocab)} candidate tags skipped")

    fixture_path = write_browser_fixture(vocab, bge_candidates, int(bge.get_embedding_dimension()))
    print(f"Wrote {fixture_path}")

    figure_data: dict[str, dict[str, dict]] = {}
    results: dict = {
        "seed": SEED,
        "sources": {
            "glove": {"model": GLOVE_MODEL_NAME, "dim": int(glove.vector_size), "vocab_size": len(glove)},
            "bge": {"model": BGE_MODEL_NAME, "dim": int(bge.get_embedding_dimension())},
        },
        "candidate_vocabulary": {
            "source_file": "talk/experiments/results/mirflickr_vocab.txt",
            "size": len(vocab),
        },
        "tag_sets": {},
    }

    for set_name, tag_set in TAG_SETS.items():
        glove_member_vecs, glove_notes = glove_member_vectors(tag_set["members"], glove)
        bge_member_vecs = {w: bge_candidates[w] for w in tag_set["members"]}

        glove_stats = analyze_source(tag_set, glove_member_vecs, glove_candidates, glove_skipped)
        glove_stats["vector_notes"] = glove_notes
        bge_stats = analyze_source(tag_set, bge_member_vecs, bge_candidates, bge_skipped)

        figure_data[set_name] = {"glove": glove_stats, "bge": bge_stats}

        results["tag_sets"][set_name] = {
            "definition": {
                "query_note": tag_set["query_note"],
                "members": tag_set["members"],
                "colour_word": tag_set["colour_word"],
            },
            "glove": {k: v for k, v in glove_stats.items() if not k.startswith("_")},
            "bge": {k: v for k, v in bge_stats.items() if not k.startswith("_")},
        }

    results_path = RESULTS_DIR / "e3_set_mean.json"
    results_path.write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"Wrote {results_path}")

    fig_path = make_figure(figure_data)
    print(f"Wrote {fig_path}")

    print()
    for set_name in TAG_SETS:
        for source_key, source_label in (("glove", "GloVe-300d"), ("bge", "bge-small-en-v1.5")):
            stats = figure_data[set_name][source_key]
            top_words = ", ".join(e["word"] for e in stats["top10_nearest_vocab_tags"])
            print(f"[{source_label}/{set_name}] top10: {top_words}")
            print(
                f"[{source_label}/{set_name}] avg-member-cosine="
                f"{stats['avg_cosine_mean_to_own_members']:.4f} "
                f"median-other-cosine={stats['median_cosine_mean_to_other_vocab_tags']:.4f} "
                f"gap={stats['gap_member_minus_other_median']:.4f}"
            )


if __name__ == "__main__":
    main()

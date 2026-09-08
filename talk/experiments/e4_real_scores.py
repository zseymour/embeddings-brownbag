# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "sentence-transformers>=3.0",
#     "numpy",
#     "matplotlib",
#     "requests",
# ]
# ///
"""E4: real retrieval scores, replacing the deck's invented top-5 (0.91/0.87/0.83/0.81/0.78).

Corpus/queries: BEIR SciFact (5,183 docs, 300 test queries, qrels/test.tsv), downloaded
straight from the BEIR mirror - no `beir` package, since all we need is the jsonl/tsv already
in BEIR's own format. Falls back to NFCorpus from the same mirror if the SciFact host is down.

Model: BAAI/bge-small-en-v1.5 via sentence-transformers, cosine similarity (embeddings are
L2-normalised so cosine = dot product). Per the model card, queries get the instruction prefix
"Represent this sentence for searching relevant passages: "; passages/corpus do not.
Runs on CPU only for determinism (bge-small is small enough that this is fast).

Exhibit A (slide 20 replacement): the real query whose top-5 has the most alternating
relevant/non-relevant pattern, plus the distribution of "best relevant score per query" vs
"best non-relevant score per query" across all 300 test queries.

Exhibit B (slide 24 companion): split-conformal lower cutoff. 300 queries split 150/150
(seed 0). Calibration nonconformity score per query = the lowest cosine score among that
query's relevant documents. Cutoff at target coverage T = the ceil((1-T)*(n+1))-th smallest
calibration score (standard finite-sample conformal correction). Evaluated on the held-out
150: empirical coverage (all relevant docs score above cutoff) and context reduction against
fixed top-10 and top-20 baselines, for T in {0.80, 0.90, 0.95}.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE.parent / "public" / "figures"

os.environ.setdefault("HF_HOME", str(HERE / ".cache"))
os.environ.setdefault("TORCH_HOME", str(HERE / ".cache"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HERE / ".cache"))

from sentence_transformers import SentenceTransformer  # noqa: E402  (import after HF_HOME is set)

SEED = 0
np.random.seed(SEED)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
PRIMARY_DATASET = "scifact"
FALLBACK_DATASET = "nfcorpus"

BG = "#12151c"
FG = "#e8ecf1"
MUTED = "#98a2b3"
ORANGE = "#e8894a"
BLUE = "#7cc4ff"
EMERALD = "#34d399"

TARGET_COVERAGES = [0.80, 0.90, 0.95]
TOPK_BASELINES = [10, 20]


def download_beir_dataset(name: str) -> Path:
    """Download and extract a BEIR dataset zip (corpus.jsonl, queries.jsonl, qrels/*.tsv)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dir = DATA_DIR / name
    if (dataset_dir / "corpus.jsonl").exists() and (dataset_dir / "qrels" / "test.tsv").exists():
        print(f"[e4] {name} already present at {dataset_dir}, skipping download")
        return dataset_dir

    zip_path = DATA_DIR / f"{name}.zip"
    url = f"{BEIR_BASE}/{name}.zip"
    print(f"[e4] downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()
    print(f"[e4] extracted to {dataset_dir}")
    return dataset_dir


def load_beir_dataset() -> tuple[str, bool, Path]:
    """Try SciFact first; fall back to NFCorpus if the primary host is unreachable."""
    try:
        return PRIMARY_DATASET, False, download_beir_dataset(PRIMARY_DATASET)
    except Exception as exc:  # noqa: BLE001 - any network/zip failure triggers the fallback
        print(f"[e4] {PRIMARY_DATASET} download failed ({exc!r}); falling back to {FALLBACK_DATASET}")
        return FALLBACK_DATASET, True, download_beir_dataset(FALLBACK_DATASET)


def load_corpus_queries_qrels(dataset_dir: Path) -> tuple[dict, dict, dict]:
    corpus: dict[str, dict] = {}
    with (dataset_dir / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row.get("title", ""), "text": row.get("text", "")}

    queries: dict[str, str] = {}
    with (dataset_dir / "queries.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    qrels_path = dataset_dir / "qrels" / "test.tsv"
    with qrels_path.open() as f:
        next(f)  # header: query-id  corpus-id  score
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)

    return corpus, queries, qrels


def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_corpus(model: SentenceTransformer, corpus_ids: list[str], corpus: dict) -> np.ndarray:
    texts = [f"{corpus[cid]['title']} {corpus[cid]['text']}".strip() for cid in corpus_ids]
    return model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def embed_queries(model: SentenceTransformer, query_ids: list[str], queries: dict) -> np.ndarray:
    texts = [QUERY_INSTRUCTION + queries[qid] for qid in query_ids]
    return model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def per_query_relevant_nonrelevant_scores(
    scores: np.ndarray, corpus_ids: list[str], test_query_ids: list[str], qrels: dict
) -> tuple[np.ndarray, np.ndarray]:
    """For each test query: (min score among its relevant docs, max score among its non-relevant docs)."""
    corpus_index = {cid: i for i, cid in enumerate(corpus_ids)}
    min_rel = np.empty(len(test_query_ids), dtype=np.float64)
    max_nonrel = np.empty(len(test_query_ids), dtype=np.float64)
    for qi, qid in enumerate(test_query_ids):
        rel_ids = {cid for cid, s in qrels[qid].items() if s >= 1 and cid in corpus_index}
        rel_idx = np.array([corpus_index[cid] for cid in rel_ids], dtype=np.int64)
        row = scores[qi]
        mask = np.ones(row.shape[0], dtype=bool)
        mask[rel_idx] = False
        min_rel[qi] = row[rel_idx].min()
        max_nonrel[qi] = row[mask].max()
    return min_rel, max_nonrel


def select_exhibit_a(
    scores: np.ndarray,
    corpus_ids: list[str],
    corpus: dict,
    test_query_ids: list[str],
    queries: dict,
    qrels: dict,
) -> dict:
    candidates = []
    for qi, qid in enumerate(test_query_ids):
        row = scores[qi]
        top5_idx = np.argsort(-row)[:5]
        rel_ids = {cid for cid, s in qrels[qid].items() if s >= 1}
        top5_doc_ids = [corpus_ids[i] for i in top5_idx]
        rel_flags = [doc_id in rel_ids for doc_id in top5_doc_ids]
        top5_scores = [float(row[i]) for i in top5_idx]
        spread = max(top5_scores) - min(top5_scores)
        alternations = sum(rel_flags[i] != rel_flags[i + 1] for i in range(4))
        candidates.append(
            {
                "query_id": qid,
                "top5_idx": top5_idx,
                "top5_doc_ids": top5_doc_ids,
                "rel_flags": rel_flags,
                "top5_scores": top5_scores,
                "spread": spread,
                "alternations": alternations,
            }
        )

    spread_eligible = [c for c in candidates if c["spread"] >= 0.05]
    pool = spread_eligible if spread_eligible else candidates
    max_alt = max(c["alternations"] for c in pool)
    best_pool = [c for c in pool if c["alternations"] == max_alt]
    best = max(best_pool, key=lambda c: c["spread"])
    spread_ok = best["spread"] >= 0.05
    global_max_alt = max(c["alternations"] for c in candidates)

    note_parts = []
    if global_max_alt > max_alt:
        tightest = max(
            c["spread"] for c in candidates if c["alternations"] == global_max_alt
        )
        note_parts.append(
            f"the single most-interleaved top-5 in the whole test set alternates on every rank "
            f"({global_max_alt}/4) but only among near-tied scores (spread {tightest:.3f}); requiring "
            f"a >=0.05 spread selects this {max_alt}/4-alternation query instead, which is still "
            f"clearly interleaved and not just score noise."
        )
    if not spread_ok:
        note_parts.append(
            f"even so, no query hit both the spread and full-interleaving bar; this is the best "
            f"trade-off SciFact/bge-small actually produced (spread {best['spread']:.3f})."
        )
    note = " ".join(note_parts) if note_parts else "a fully interleaved (4-alternation), >=0.05-spread top-5 was found."

    top5 = []
    for rank, (idx, doc_id, rel, score) in enumerate(
        zip(best["top5_idx"], best["top5_doc_ids"], best["rel_flags"], best["top5_scores"]), start=1
    ):
        title = corpus[doc_id]["title"].strip()
        top5.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "title": title,
                "title_60": title[:60],
                "score": round(score, 3),
                "relevant": rel,
            }
        )

    return {
        "query_id": best["query_id"],
        "query_text": queries[best["query_id"]],
        "top5": top5,
        "relevance_pattern": ["relevant" if r else "non-relevant" for r in best["rel_flags"]],
        "selection": {
            "alternations": best["alternations"],
            "max_possible_alternations": 4,
            "score_spread": round(best["spread"], 4),
            "spread_threshold_met": spread_ok,
            "candidates_considered": len(candidates),
            "note": note,
        },
    }


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "min": round(float(np.min(values)), 4),
        "median": round(float(np.median(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def lower_conformal_cutoff(calib_scores: np.ndarray, target_coverage: float) -> tuple[float, int]:
    """Split-conformal lower threshold: P(new min-relevant-score >= cutoff) >= target_coverage.

    Standard split-conformal correction, in its usual "upper" form: m = ceil((n+1)(1-alpha)) is
    the calibration rank used to build an upper conformal quantile at level (1-alpha). Mirrored
    onto our one-sided LOWER bound (nonconformity score = -score), the equivalent order-statistic
    index is j = (n+1) - m; cutoff = the j-th smallest calibration score. (Algebraically this is
    j = floor(alpha*(n+1)) for non-integer alpha*(n+1) - both forms are implemented here so the
    textbook m and the applied j are both visible.)
    """
    n = len(calib_scores)
    alpha = 1.0 - target_coverage
    m = math.ceil((n + 1) * (1.0 - alpha))
    j = (n + 1) - m
    if j < 1:
        return -math.inf, 0  # not enough calibration data to guarantee this coverage at all
    j = min(j, n)
    cutoff = float(np.sort(calib_scores)[j - 1])
    return cutoff, j


def theoretical_marginal_coverage(n_calib: int, j: int) -> float:
    """Exact marginal coverage of the lower_conformal_cutoff order statistic under exchangeability."""
    if j < 1:
        return 1.0
    return (n_calib + 1 - j) / (n_calib + 1)


def build_exhibit_b(
    scores: np.ndarray, min_rel: np.ndarray, max_nonrel: np.ndarray, test_query_ids: list[str]
) -> dict:
    n_queries = len(test_query_ids)
    split_rng = np.random.RandomState(SEED)
    perm = split_rng.permutation(n_queries)
    half = n_queries // 2
    calib_idx = perm[:half]
    test_idx = perm[half:]

    calib_min_rel = min_rel[calib_idx]
    test_min_rel = min_rel[test_idx]
    test_scores_matrix = scores[test_idx]  # (n_test_queries, n_corpus)

    targets = []
    for target in TARGET_COVERAGES:
        cutoff, j = lower_conformal_cutoff(calib_min_rel, target)
        empirical_coverage = float(np.mean(test_min_rel >= cutoff))
        theo_coverage = theoretical_marginal_coverage(len(calib_idx), j)
        docs_retained = (test_scores_matrix >= cutoff).sum(axis=1)
        mean_retained = float(np.mean(docs_retained))
        baselines = {
            f"top{k_base}": {
                "baseline_mean_docs": k_base,
                "context_reduction": round(1.0 - mean_retained / k_base, 4),
            }
            for k_base in TOPK_BASELINES
        }
        targets.append(
            {
                "target_coverage": target,
                "alpha": round(1.0 - target, 4),
                "calib_n": int(len(calib_idx)),
                "test_n": int(len(test_idx)),
                "conformal_rank_j": j,
                "cutoff": round(cutoff, 4),
                "theoretical_marginal_coverage": round(theo_coverage, 4),
                "test_empirical_coverage": round(empirical_coverage, 4),
                "mean_docs_retained_test": round(mean_retained, 2),
                "baselines": baselines,
            }
        )

    return {
        "split": {
            "seed": SEED,
            "calib_n": int(len(calib_idx)),
            "test_n": int(len(test_idx)),
            "calib_min_relevant_score_summary": summarize(calib_min_rel),
            "test_min_relevant_score_summary": summarize(test_min_rel),
        },
        "nonconformity_score": "min cosine score among a query's relevant documents (per-query worst relevant hit)",
        "targets": targets,
    }


def build_multi_split_stats(
    scores: np.ndarray, min_rel: np.ndarray, test_query_ids: list[str], n_splits: int = 500
) -> dict:
    """Repeat the 150/150 split-conformal procedure over `n_splits` random seeds (0..n_splits-1)
    to see whether a single split's coverage gap from target is real sampling noise or bias.
    Reuses the already-embedded score matrix; no re-embedding."""
    n_queries = len(test_query_ids)
    half = n_queries // 2
    per_target = {t: {"coverage": [], "cutoff": [], "mean_retained": []} for t in TARGET_COVERAGES}

    for split_seed in range(n_splits):
        perm = np.random.RandomState(split_seed).permutation(n_queries)
        calib_idx = perm[:half]
        test_idx = perm[half:]
        calib_min_rel = min_rel[calib_idx]
        test_min_rel = min_rel[test_idx]
        test_scores_matrix = scores[test_idx]

        for target in TARGET_COVERAGES:
            cutoff, _j = lower_conformal_cutoff(calib_min_rel, target)
            per_target[target]["coverage"].append(float(np.mean(test_min_rel >= cutoff)))
            per_target[target]["cutoff"].append(cutoff)
            per_target[target]["mean_retained"].append(
                float(np.mean((test_scores_matrix >= cutoff).sum(axis=1)))
            )

    targets = []
    for target in TARGET_COVERAGES:
        cov = np.array(per_target[target]["coverage"])
        cut = np.array(per_target[target]["cutoff"])
        ret = np.array(per_target[target]["mean_retained"])
        mean_ret = float(np.mean(ret))
        targets.append(
            {
                "target_coverage": target,
                "n_splits": n_splits,
                "coverage_mean": round(float(np.mean(cov)), 4),
                "coverage_p10": round(float(np.percentile(cov, 10)), 4),
                "coverage_p90": round(float(np.percentile(cov, 90)), 4),
                "cutoff_mean": round(float(np.mean(cut)), 4),
                "mean_docs_retained_per_query": round(mean_ret, 2),
                "baseline_top10_docs": 10,
                "baseline_top20_docs": 20,
                "context_reduction_vs_top10": round(1.0 - mean_ret / 10.0, 4),
                "context_reduction_vs_top20": round(1.0 - mean_ret / 20.0, 4),
            }
        )

    return {
        "seeds": f"0..{n_splits - 1}",
        "split_size": {"calib_n": half, "test_n": n_queries - half},
        "correction": (
            "m = ceil((n+1)(1-alpha)); cutoff = the ((n+1)-m)-th smallest calibration score "
            "(standard split-conformal correction, mirrored for a one-sided lower bound)"
        ),
        "targets": targets,
    }


def make_top5_figure(exhibit_a: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top5 = exhibit_a["top5"][::-1]  # reverse so rank 1 draws on top
    labels = [f"{t['title_60']}{'…' if len(t['title']) > 60 else ''}" for t in top5]
    values = [t["score"] for t in top5]
    colors = [EMERALD if t["relevant"] else ORANGE for t in top5]

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y = np.arange(len(top5))
    ax.barh(y, values, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=FG, fontsize=14)
    ax.tick_params(axis="x", colors=MUTED, labelsize=13)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(MUTED)

    min_v, max_v = min(values), max(values)
    pad = max(0.02, (max_v - min_v) * 0.15)
    ax.set_xlim(max(0, min_v - pad * 2), max_v + pad)
    for yi, s in zip(y, values):
        ax.text(s + pad * 0.15, yi, f"{s:.3f}", va="center", ha="left", color=MUTED, fontsize=13)
    ax.set_xlabel("cosine similarity (bge-small-en-v1.5)", color=MUTED, fontsize=14)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=EMERALD, label="relevant (SciFact qrels)"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="not relevant"),
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=12)
    for text in legend.get_texts():
        text.set_color(FG)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def make_conformal_figure(
    min_rel: np.ndarray, max_nonrel: np.ndarray, exhibit_b: dict, multi_split: dict, out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ninety = next(t for t in exhibit_b["targets"] if t["target_coverage"] == 0.90)

    fig, (ax_hist, ax_cov) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    fig.patch.set_facecolor(BG)

    # -- left: overlapping histograms, seed-0 cutoff --
    ax_hist.set_facecolor(BG)
    bins = np.linspace(
        min(min_rel.min(), max_nonrel.min()), max(min_rel.max(), max_nonrel.max()), 30
    )
    ax_hist.hist(min_rel, bins=bins, color=EMERALD, alpha=0.7, label="lowest relevant score / query")
    ax_hist.hist(max_nonrel, bins=bins, color=BLUE, alpha=0.55, label="highest non-relevant score / query")
    ax_hist.axvline(ninety["cutoff"], color=ORANGE, linewidth=2, linestyle="--")
    ax_hist.text(
        ninety["cutoff"],
        ax_hist.get_ylim()[1] if ax_hist.get_ylim()[1] else 1,
        f"  seed-0 cutoff (90% target) = {ninety['cutoff']:.3f}",
        color=ORANGE,
        fontsize=11,
        va="top",
        ha="left",
        rotation=0,
    )
    ax_hist.tick_params(colors=MUTED, labelsize=12)
    ax_hist.set_xlabel("cosine similarity", color=MUTED, fontsize=13)
    ax_hist.set_ylabel("test queries", color=MUTED, fontsize=13)
    for spine_name, spine in ax_hist.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(MUTED)
    ax_hist.set_title("calibration-half score distributions", color=FG, fontsize=14, pad=10)
    legend = ax_hist.legend(loc="upper left", frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color(FG)

    # -- right: coverage vs target (mean + 10-90th pct band over 500 splits), twin axis for
    #    mean docs retained per query, with fixed top-10/top-20 baselines as reference lines --
    ax_cov.set_facecolor(BG)
    ms_targets = [t["target_coverage"] for t in multi_split["targets"]]
    cov_mean = [t["coverage_mean"] for t in multi_split["targets"]]
    cov_p10 = [t["coverage_p10"] for t in multi_split["targets"]]
    cov_p90 = [t["coverage_p90"] for t in multi_split["targets"]]
    yerr_lo = [m - p for m, p in zip(cov_mean, cov_p10)]
    yerr_hi = [p - m for m, p in zip(cov_mean, cov_p90)]

    y_lo = min(0.70, min(cov_p10) - 0.03)
    diag = np.linspace(y_lo, 1.0, 10)
    ax_cov.plot(diag, diag, color=MUTED, linewidth=1, linestyle="--", label="ideal (mean = target)")
    ax_cov.errorbar(
        ms_targets,
        cov_mean,
        yerr=[yerr_lo, yerr_hi],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        elinewidth=2.5,
        capsize=7,
        markersize=9,
        zorder=3,
        label="mean over 500 splits (10th-90th pct band)",
    )
    for t in multi_split["targets"]:
        ax_cov.annotate(
            f"{t['coverage_mean']*100:.0f}% cov\n~{t['mean_docs_retained_per_query']:.0f} docs/query",
            (t["target_coverage"], t["coverage_mean"]),
            textcoords="offset points",
            xytext=(10, -16),
            color=MUTED,
            fontsize=10,
        )
    ax_cov.set_xlim(0.75, 1.0)
    ax_cov.set_ylim(y_lo, 1.02)
    ax_cov.tick_params(colors=MUTED, labelsize=12)
    ax_cov.set_xlabel("target coverage", color=MUTED, fontsize=13)
    ax_cov.set_ylabel("empirical coverage (500 held-out halves)", color=BLUE, fontsize=13)
    ax_cov.tick_params(axis="y", colors=BLUE)
    for spine_name, spine in ax_cov.spines.items():
        if spine_name == "top":
            spine.set_visible(False)
        elif spine_name == "left":
            spine.set_color(BLUE)
        else:
            spine.set_color(MUTED)

    ax_docs = ax_cov.twinx()
    docs_mean = [t["mean_docs_retained_per_query"] for t in multi_split["targets"]]
    ax_docs.set_yscale("log")
    ax_docs.plot(
        ms_targets, docs_mean, color=ORANGE, marker="s", markersize=8, linestyle=":",
        linewidth=1.5, zorder=2, label="mean docs retained / query",
    )
    ax_docs.axhline(10, color=MUTED, linestyle="--", linewidth=1)
    ax_docs.axhline(20, color=MUTED, linestyle="--", linewidth=1)
    ax_docs.text(0.751, 10, "top-10", color=MUTED, fontsize=10, va="bottom", ha="left")
    ax_docs.text(0.751, 20, "top-20", color=MUTED, fontsize=10, va="bottom", ha="left")
    ax_docs.set_ylim(5, max(docs_mean) * 2)
    ax_docs.set_ylabel("mean docs retained / query (log scale)", color=ORANGE, fontsize=13)
    ax_docs.tick_params(axis="y", colors=ORANGE, labelsize=12)
    for spine_name, spine in ax_docs.spines.items():
        if spine_name == "right":
            spine.set_color(ORANGE)
        else:
            spine.set_visible(False)

    ax_cov.set_title("coverage vs target, with mean context retained", color=FG, fontsize=14, pad=10)
    handles1, labels1 = ax_cov.get_legend_handles_labels()
    handles2, labels2 = ax_docs.get_legend_handles_labels()
    legend = ax_cov.legend(
        handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, fontsize=10
    )
    for text in legend.get_texts():
        text.set_color(FG)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_name, used_fallback, dataset_dir = load_beir_dataset()
    corpus, queries, qrels = load_corpus_queries_qrels(dataset_dir)
    test_query_ids = sorted(qrels.keys(), key=int)
    corpus_ids = sorted(corpus.keys(), key=int)
    print(f"[e4] dataset={dataset_name} fallback={used_fallback} corpus={len(corpus_ids)} test_queries={len(test_query_ids)}")

    all_qrel_scores = {s for d in qrels.values() for s in d.values()}
    print(f"[e4] qrel score values observed: {sorted(all_qrel_scores)}")

    model = get_model()
    print("[e4] embedding corpus")
    corpus_emb = embed_corpus(model, corpus_ids, corpus)
    print("[e4] embedding queries")
    query_emb = embed_queries(model, test_query_ids, queries)

    scores = query_emb @ corpus_emb.T  # (n_queries, n_corpus), cosine since both L2-normalised
    print(f"[e4] score matrix shape={scores.shape}")

    min_rel, max_nonrel = per_query_relevant_nonrelevant_scores(scores, corpus_ids, test_query_ids, qrels)
    top1_rel = _top1_relevant(scores, corpus_ids, test_query_ids, qrels)

    exhibit_a = select_exhibit_a(scores, corpus_ids, corpus, test_query_ids, queries, qrels)
    exhibit_a["top1_relevant_vs_nonrelevant"] = {
        "description": (
            "Per test query: the score of its best-scoring relevant document ('top-1 relevant') vs "
            "the score of its best-scoring non-relevant document ('top-1 non-relevant'), across all "
            f"{len(test_query_ids)} test queries."
        ),
        "top1_relevant_scores": [round(float(v), 4) for v in top1_rel],
        "top1_nonrelevant_scores": [round(float(v), 4) for v in max_nonrel],
        "summary": {
            "top1_relevant": summarize(top1_rel),
            "top1_nonrelevant": summarize(max_nonrel),
        },
    }

    exhibit_b = build_exhibit_b(scores, min_rel, max_nonrel, test_query_ids)

    print("[e4] running 500-split conformal coverage sweep")
    multi_split = build_multi_split_stats(scores, min_rel, test_query_ids, n_splits=500)

    top5_fig_path = FIGURES_DIR / "exp-e4-top5.png"
    make_top5_figure(exhibit_a, top5_fig_path)
    print(f"[e4] wrote {top5_fig_path}")

    conformal_fig_path = FIGURES_DIR / "exp-e4-conformal.png"
    make_conformal_figure(min_rel, max_nonrel, exhibit_b, multi_split, conformal_fig_path)
    print(f"[e4] wrote {conformal_fig_path}")

    results = {
        "meta": {
            "dataset": dataset_name,
            "dataset_fallback_used": used_fallback,
            "corpus_size": len(corpus_ids),
            "num_test_queries": len(test_query_ids),
            "qrel_score_values_observed": sorted(all_qrel_scores),
            "model": MODEL_NAME,
            "query_instruction": QUERY_INSTRUCTION,
            "similarity": "cosine (dot product of L2-normalised embeddings)",
            "device": "cpu",
            "seed": SEED,
        },
        "exhibit_a": exhibit_a,
        "exhibit_b": exhibit_b,
        "multi_split": multi_split,
        "figures": {
            "top5": str(top5_fig_path.relative_to(FIGURES_DIR.parent.parent)),
            "conformal": str(conformal_fig_path.relative_to(FIGURES_DIR.parent.parent)),
        },
    }

    results_path = RESULTS_DIR / "e4_real_scores.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[e4] wrote {results_path}")

    print("[e4] exhibit A query:", exhibit_a["query_text"])
    print("[e4] exhibit A pattern:", exhibit_a["relevance_pattern"], "scores:", [t["score"] for t in exhibit_a["top5"]])
    for t in exhibit_b["targets"]:
        print(
            f"[e4] target={t['target_coverage']} cutoff={t['cutoff']:.3f} "
            f"empirical_coverage={t['test_empirical_coverage']:.3f} "
            f"mean_retained={t['mean_docs_retained_test']:.1f} "
            f"reduction_vs_top10={t['baselines']['top10']['context_reduction']:.3f} "
            f"reduction_vs_top20={t['baselines']['top20']['context_reduction']:.3f}"
        )
    print("[e4] multi-split (500 seeds) coverage sweep:")
    for t in multi_split["targets"]:
        print(
            f"[e4]   target={t['target_coverage']} cutoff_mean={t['cutoff_mean']:.3f} "
            f"coverage_mean={t['coverage_mean']:.3f} "
            f"coverage_p10={t['coverage_p10']:.3f} coverage_p90={t['coverage_p90']:.3f} "
            f"mean_retained={t['mean_docs_retained_per_query']:.1f} "
            f"reduction_vs_top10={t['context_reduction_vs_top10']:.3f} "
            f"reduction_vs_top20={t['context_reduction_vs_top20']:.3f}"
        )


def _top1_relevant(
    scores: np.ndarray, corpus_ids: list[str], test_query_ids: list[str], qrels: dict
) -> np.ndarray:
    corpus_index = {cid: i for i, cid in enumerate(corpus_ids)}
    out = np.empty(len(test_query_ids), dtype=np.float64)
    for qi, qid in enumerate(test_query_ids):
        rel_idx = np.array(
            [corpus_index[cid] for cid, s in qrels[qid].items() if s >= 1 and cid in corpus_index],
            dtype=np.int64,
        )
        out[qi] = scores[qi, rel_idx].max()
    return out


if __name__ == "__main__":
    main()

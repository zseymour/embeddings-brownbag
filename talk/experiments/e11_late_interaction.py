# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "sentence-transformers>=3.0",
#     "pylate>=1.3",
#     "pytrec_eval",
#     "numpy",
#     "torch",
#     "matplotlib",
# ]
# ///
"""E11: single-vector vs late-interaction retrieval on BEIR SciFact.

"More than one vector" - replaces the deck's "Algebraic structure" section. Two systems,
same corpus/queries, same eval harness:

  - Single-vector: BAAI/bge-small-en-v1.5 (sentence-transformers). One 384-dim embedding per
    document and per query, cosine similarity = dot product of L2-normalised vectors. Query
    gets the model-card instruction prefix "Represent this sentence for searching relevant
    passages: "; passages/corpus do not.

  - Late interaction: answerdotai/answerai-colbert-small-v1 (33M params) via pylate
    (pylate.models.ColBERT). Each document keeps one 96-dim vector per surviving token
    (punctuation stripped by the model's skiplist, truncated to document_length=300 tokens);
    each query is padded/augmented to a fixed query_length=32 tokens (ColBERT's [MASK] query
    augmentation - no masking on the query side). Score = MaxSim = sum over query tokens of the
    max, over document tokens, of the dot product between L2-normalised token embeddings
    (pylate.scores.colbert_scores). Exhaustive over the whole 5,183-doc corpus per query - no
    approximate index.

Corpus/queries/qrels: BEIR SciFact, already on disk at data/scifact/{corpus.jsonl,queries.jsonl,
qrels/test.tsv} (5,183 docs, 300 test queries). Both systems run on CPU only, for determinism
and because the storage/latency numbers are the point of the section.

Metrics: nDCG@10, Recall@10, Recall@100 via pytrec_eval (standard trec_eval C implementation).
MRR@10 follows BEIR's own convention: truncate each system's ranking to its top 10 documents per
query, then take pytrec_eval's `recip_rank` on that truncated run (0 if no relevant doc is in the
top 10).

Storage: total bytes of the served representation for the whole corpus, float32 and float16, plus
the float16:float16 ratio. For late interaction this is "sum of actual token vectors after
document-length truncation and skiplist filtering", not a fixed per-doc count.

Latency: per query (batch size 1), median/p95 wall time to (a) encode the query and (b) score it
exhaustively against the whole corpus, CPU, after a warm-up prefix that is excluded from the
statistics.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE.parent / "public" / "figures"

os.environ.setdefault("HF_HOME", str(HERE / ".cache"))
os.environ.setdefault("TORCH_HOME", str(HERE / ".cache"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HERE / ".cache"))

import pytrec_eval  # noqa: E402  (import after HF_HOME is set)
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_DIR = DATA_DIR / "scifact"

SINGLE_VECTOR_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

LATE_INTERACTION_MODEL = "answerdotai/answerai-colbert-small-v1"

BG = "#12151c"
FG = "#e8ecf1"
MUTED = "#98a2b3"
ORANGE = "#e8894a"
BLUE = "#7cc4ff"
EMERALD = "#34d399"

N_WARMUP = 5  # per-query latency samples before this index are excluded from the statistics
RUN_DEPTH = 1000  # candidates kept per query in the trec_eval "run" (>> any cutoff we score)


# ------------------------------------------------------------------ data loading


def load_corpus_queries_qrels() -> tuple[dict, dict, dict]:
    corpus: dict[str, dict] = {}
    with (DATASET_DIR / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row.get("title", ""), "text": row.get("text", "")}

    queries: dict[str, str] = {}
    with (DATASET_DIR / "queries.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with (DATASET_DIR / "qrels" / "test.tsv").open() as f:
        next(f)  # header: query-id  corpus-id  score
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)

    return corpus, queries, qrels


def doc_texts_for(corpus_ids: list[str], corpus: dict) -> list[str]:
    return [f"{corpus[cid]['title']} {corpus[cid]['text']}".strip() for cid in corpus_ids]


# ------------------------------------------------------------------ latency helpers


def summarize_latency_s(samples_s: list[float]) -> dict:
    arr = np.asarray(samples_s, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median_ms": float(np.median(arr) * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000),
        "mean_ms": float(np.mean(arr) * 1000),
    }


# ------------------------------------------------------------------ single-vector system


def run_single_vector(
    corpus_ids: list[str], corpus: dict, test_query_ids: list[str], queries: dict
) -> dict:
    model = SentenceTransformer(SINGLE_VECTOR_MODEL, device="cpu")
    dim = model.get_sentence_embedding_dimension()

    texts = doc_texts_for(corpus_ids, corpus)
    t0 = time.perf_counter()
    corpus_emb = model.encode(
        texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32)
    corpus_encode_s = time.perf_counter() - t0
    print(f"[e11] single-vector: corpus encoded in {corpus_encode_s:.1f}s -> {corpus_emb.shape}")

    n_docs = len(corpus_ids)
    scores = np.empty((len(test_query_ids), n_docs), dtype=np.float32)
    encode_latencies: list[float] = []
    score_latencies: list[float] = []

    for i, qid in enumerate(test_query_ids):
        text = QUERY_INSTRUCTION + queries[qid]

        t0 = time.perf_counter()
        qv = model.encode(
            [text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0].astype(np.float32)
        dt_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        row = corpus_emb @ qv  # cosine, both L2-normalised
        dt_score = time.perf_counter() - t0

        scores[i] = row
        if i >= N_WARMUP:
            encode_latencies.append(dt_encode)
            score_latencies.append(dt_score)

    bytes_f32 = n_docs * dim * 4
    bytes_f16 = n_docs * dim * 2

    return {
        "scores": scores,
        "meta": {
            "model": SINGLE_VECTOR_MODEL,
            "dim": dim,
            "n_docs": n_docs,
            "query_instruction": QUERY_INSTRUCTION,
            "scoring_formula": (
                "cosine similarity = dot product of L2-normalised sentence embeddings "
                f"(dim={dim})"
            ),
            "corpus_encode_seconds_total": corpus_encode_s,
        },
        "storage": {
            "representation": "one dense vector per document",
            "dim": dim,
            "n_docs": n_docs,
            "bytes_float32": bytes_f32,
            "bytes_float16": bytes_f16,
        },
        "latency": {
            "query_encode": summarize_latency_s(encode_latencies),
            "corpus_scoring": summarize_latency_s(score_latencies),
            "warmup_queries_excluded": N_WARMUP,
        },
    }


# ------------------------------------------------------------------ late-interaction system


def run_late_interaction(
    corpus_ids: list[str], corpus: dict, test_query_ids: list[str], queries: dict
) -> dict:
    from pylate import models as pylate_models
    from pylate import scores as pylate_scores

    model = pylate_models.ColBERT(model_name_or_path=LATE_INTERACTION_MODEL, device="cpu")
    document_length = int(getattr(model, "document_length", 0))
    query_length = int(getattr(model, "query_length", 0))
    do_query_expansion = bool(getattr(model, "do_query_expansion", False))

    texts = doc_texts_for(corpus_ids, corpus)
    t0 = time.perf_counter()
    doc_token_embs = model.encode(
        texts, is_query=False, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    corpus_encode_s = time.perf_counter() - t0
    n_docs = len(doc_token_embs)
    dim = int(doc_token_embs[0].shape[1])
    token_counts = np.array([e.shape[0] for e in doc_token_embs], dtype=np.int64)
    print(
        f"[e11] late-interaction: corpus encoded in {corpus_encode_s:.1f}s -> {n_docs} docs, "
        f"dim={dim}, mean tokens/doc={token_counts.mean():.1f}, max={token_counts.max()}"
    )

    l_max = int(token_counts.max())
    doc_padded = np.zeros((n_docs, l_max, dim), dtype=np.float32)
    doc_mask = np.zeros((n_docs, l_max), dtype=np.float32)
    for i, e in enumerate(doc_token_embs):
        length = e.shape[0]
        doc_padded[i, :length] = e
        doc_mask[i, :length] = 1.0
    doc_padded_t = torch.from_numpy(doc_padded)
    doc_mask_t = torch.from_numpy(doc_mask)

    scores = np.empty((len(test_query_ids), n_docs), dtype=np.float32)
    encode_latencies: list[float] = []
    score_latencies: list[float] = []

    for i, qid in enumerate(test_query_ids):
        text = queries[qid]

        t0 = time.perf_counter()
        qv = model.encode(
            [text], is_query=True, batch_size=1, show_progress_bar=False, convert_to_numpy=True
        )[0].astype(np.float32)  # (query_length, dim); query augmentation -> no mask needed
        dt_encode = time.perf_counter() - t0

        q_t = torch.from_numpy(qv).unsqueeze(0)  # (1, query_length, dim)

        t0 = time.perf_counter()
        row_t = pylate_scores.colbert_scores(
            queries_embeddings=q_t,
            documents_embeddings=doc_padded_t,
            documents_mask=doc_mask_t,
        )[0]  # (n_docs,)
        row = row_t.numpy()
        dt_score = time.perf_counter() - t0

        scores[i] = row
        if i >= N_WARMUP:
            encode_latencies.append(dt_encode)
            score_latencies.append(dt_score)

    total_tokens = int(token_counts.sum())
    bytes_f32 = total_tokens * dim * 4
    bytes_f16 = total_tokens * dim * 2

    return {
        "scores": scores,
        "meta": {
            "model": LATE_INTERACTION_MODEL,
            "via": "pylate.models.ColBERT + pylate.scores.colbert_scores",
            "dim": dim,
            "n_docs": n_docs,
            "document_length_tokens": document_length,
            "query_length_tokens": query_length,
            "query_augmentation": do_query_expansion,
            "scoring_formula": (
                "MaxSim = sum over query tokens of (max over document tokens of dot product), "
                f"on L2-normalised token embeddings (dim={dim}); documents keep every "
                "non-punctuation token up to document_length after truncation; queries are "
                "padded/augmented with [MASK] to a fixed query_length and every position "
                "(real token or mask) participates in scoring"
            ),
            "corpus_encode_seconds_total": corpus_encode_s,
        },
        "storage": {
            "representation": "one dense vector per surviving document token (post skiplist + truncation)",
            "dim": dim,
            "n_docs": n_docs,
            "total_token_vectors": total_tokens,
            "mean_tokens_per_doc": float(token_counts.mean()),
            "max_tokens_per_doc": int(token_counts.max()),
            "min_tokens_per_doc": int(token_counts.min()),
            "bytes_float32": bytes_f32,
            "bytes_float16": bytes_f16,
        },
        "latency": {
            "query_encode": summarize_latency_s(encode_latencies),
            "corpus_scoring": summarize_latency_s(score_latencies),
            "warmup_queries_excluded": N_WARMUP,
        },
    }


# ------------------------------------------------------------------ metrics


def compute_metrics(
    scores: np.ndarray, corpus_ids: list[str], test_query_ids: list[str], qrels: dict
) -> dict:
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        row = scores[i]
        order = np.argsort(-row)[:RUN_DEPTH]
        run[qid] = {corpus_ids[j]: float(row[j]) for j in order}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10", "recall_100"})
    per_query = evaluator.evaluate(run)

    # MRR@10, BEIR-style: truncate the run to the top 10 per query, then take recip_rank
    # (0 if no relevant document is among the top 10).
    run_top10 = {
        qid: dict(sorted(run[qid].items(), key=lambda kv: -kv[1])[:10]) for qid in test_query_ids
    }
    evaluator_mrr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    per_query_mrr = evaluator_mrr.evaluate(run_top10)

    ndcg10 = float(np.mean([per_query[qid]["ndcg_cut_10"] for qid in test_query_ids]))
    recall10 = float(np.mean([per_query[qid]["recall_10"] for qid in test_query_ids]))
    recall100 = float(np.mean([per_query[qid]["recall_100"] for qid in test_query_ids]))
    mrr10 = float(np.mean([per_query_mrr[qid]["recip_rank"] for qid in test_query_ids]))

    return {
        "ndcg_cut_10": ndcg10,
        "recall_10": recall10,
        "recall_100": recall100,
        "mrr_cut_10": mrr10,
        "n_queries": len(test_query_ids),
        "trec_eval_measures": ["ndcg_cut_10", "recall_10", "recall_100"],
        "mrr_cut_10_method": (
            "recip_rank (pytrec_eval) on the run truncated to each query's top 10 documents; "
            "0 if no relevant document is in the top 10"
        ),
    }


# ------------------------------------------------------------------ figure


def make_figure(single_vector: dict, late_interaction: dict, metrics_sv: dict, metrics_li: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_q, ax_s) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    fig.patch.set_facecolor(BG)

    # -- left: grouped bars, nDCG@10 and Recall@100 --
    ax_q.set_facecolor(BG)
    groups = ["nDCG@10", "Recall@100"]
    sv_vals = [metrics_sv["ndcg_cut_10"], metrics_sv["recall_100"]]
    li_vals = [metrics_li["ndcg_cut_10"], metrics_li["recall_100"]]
    x = np.arange(len(groups))
    width = 0.32
    ax_q.bar(x - width / 2, sv_vals, width, color=BLUE, label="single-vector (bge-small)")
    ax_q.bar(x + width / 2, li_vals, width, color=ORANGE, label="late interaction (ColBERT)")
    for xi, v in zip(x - width / 2, sv_vals):
        ax_q.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", color=MUTED, fontsize=12)
    for xi, v in zip(x + width / 2, li_vals):
        ax_q.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", color=MUTED, fontsize=12)
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(groups, color=FG, fontsize=14)
    ax_q.set_ylim(0, max(sv_vals + li_vals) * 1.22)
    ax_q.tick_params(axis="y", colors=MUTED, labelsize=12)
    for spine_name, spine in ax_q.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(MUTED)
    ax_q.set_ylabel("score (SciFact test, 300 queries)", color=MUTED, fontsize=13)
    legend = ax_q.legend(loc="upper right", frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color(FG)

    # -- right: bytes/doc, float16, log scale, ratio annotated --
    ax_s.set_facecolor(BG)
    sv_bytes_per_doc = single_vector["storage"]["bytes_float16"] / single_vector["storage"]["n_docs"]
    li_bytes_per_doc = late_interaction["storage"]["bytes_float16"] / late_interaction["storage"]["n_docs"]
    labels = ["single-vector", "late interaction"]
    vals = [sv_bytes_per_doc, li_bytes_per_doc]
    colors = [BLUE, ORANGE]
    xb = np.arange(2)
    ax_s.bar(xb, vals, width=0.5, color=colors)
    ax_s.set_yscale("log")
    ax_s.set_xticks(xb)
    ax_s.set_xticklabels(labels, color=FG, fontsize=14)
    ax_s.tick_params(axis="y", colors=MUTED, labelsize=12)
    for xi, v in zip(xb, vals):
        ax_s.text(xi, v * 1.15, f"{v:,.0f} B", ha="center", va="bottom", color=MUTED, fontsize=12)
    for spine_name, spine in ax_s.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(MUTED)
    ax_s.set_ylabel("bytes / document, float16 (log scale)", color=MUTED, fontsize=13)
    ratio = li_bytes_per_doc / sv_bytes_per_doc
    ax_s.annotate(
        f"{ratio:.1f}\u00d7 more storage",
        xy=(1, li_bytes_per_doc),
        xytext=(0.5, li_bytes_per_doc * 2.2),
        color=EMERALD,
        fontsize=13,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels = load_corpus_queries_qrels()
    corpus_ids = sorted(corpus.keys(), key=int)
    test_query_ids = sorted(qrels.keys(), key=int)
    print(f"[e11] corpus={len(corpus_ids)} test_queries={len(test_query_ids)}")

    print("[e11] running single-vector system (bge-small-en-v1.5)")
    sv = run_single_vector(corpus_ids, corpus, test_query_ids, queries)
    print("[e11] running late-interaction system (answerai-colbert-small-v1 via pylate)")
    li = run_late_interaction(corpus_ids, corpus, test_query_ids, queries)

    print("[e11] scoring metrics")
    metrics_sv = compute_metrics(sv["scores"], corpus_ids, test_query_ids, qrels)
    metrics_li = compute_metrics(li["scores"], corpus_ids, test_query_ids, qrels)
    print(f"[e11] single-vector   nDCG@10={metrics_sv['ndcg_cut_10']:.4f} recall@100={metrics_sv['recall_100']:.4f}")
    print(f"[e11] late-interaction nDCG@10={metrics_li['ndcg_cut_10']:.4f} recall@100={metrics_li['recall_100']:.4f}")

    storage_ratio_f16 = li["storage"]["bytes_float16"] / sv["storage"]["bytes_float16"]
    storage_ratio_f32 = li["storage"]["bytes_float32"] / sv["storage"]["bytes_float32"]
    latency_ratio_scoring_median = (
        li["latency"]["corpus_scoring"]["median_ms"] / sv["latency"]["corpus_scoring"]["median_ms"]
    )

    figure_path = FIGURES_DIR / "exp-e11-late-interaction.png"
    make_figure(sv, li, metrics_sv, metrics_li, figure_path)
    print(f"[e11] wrote {figure_path}")

    out = {
        "meta": {
            "dataset": "scifact",
            "corpus_size": len(corpus_ids),
            "num_test_queries": len(test_query_ids),
            "seed": SEED,
            "device": "cpu",
        },
        "single_vector": {k: v for k, v in sv.items() if k != "scores"} | {"metrics": metrics_sv},
        "late_interaction": {k: v for k, v in li.items() if k != "scores"} | {"metrics": metrics_li},
        "comparison": {
            "storage_ratio_late_interaction_over_single_vector_float16": storage_ratio_f16,
            "storage_ratio_late_interaction_over_single_vector_float32": storage_ratio_f32,
            "scoring_latency_ratio_late_interaction_over_single_vector_median": latency_ratio_scoring_median,
            "ndcg_cut_10_delta_late_interaction_minus_single_vector": (
                metrics_li["ndcg_cut_10"] - metrics_sv["ndcg_cut_10"]
            ),
            "late_interaction_beats_single_vector_ndcg10": (
                metrics_li["ndcg_cut_10"] > metrics_sv["ndcg_cut_10"]
            ),
        },
    }

    out_path = RESULTS_DIR / "e11_late_interaction.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e11] wrote {out_path}")

    print(
        f"[e11] headline: single-vector nDCG@10={metrics_sv['ndcg_cut_10']:.4f}, "
        f"late-interaction nDCG@10={metrics_li['ndcg_cut_10']:.4f}, "
        f"storage ratio (f16) = {storage_ratio_f16:.1f}x, "
        f"scoring-latency ratio (median) = {latency_ratio_scoring_median:.1f}x"
    )


if __name__ == "__main__":
    main()

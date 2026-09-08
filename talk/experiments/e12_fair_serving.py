# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "sentence-transformers>=3.0",
#     "pylate>=1.3",
#     "pytrec_eval",
#     "numpy",
#     "torch",
#     "matplotlib",
#     "hnswlib",
#     "scikit-learn",
# ]
# ///
"""E12: fair serving-stack comparison, single-vector vs late-interaction, BEIR SciFact.

E11 showed "59x storage, 480x scoring time" for late interaction vs single-vector. That
number compares a brute-force float16 MaxSim (no index at all) against an exhaustive
float16 dot product (also no index at all). Neither system in that comparison is what a
production deployment would actually serve. This experiment replaces it with configurations
that each reflect a real serving stack:

Single-vector, BAAI/bge-small-en-v1.5 (384-d), cosine via dot product of L2-normalised
vectors:
  1. float16, exhaustive dot product (E11's baseline - reproduced here for a same-script
     comparison; drawn hollow in the figure because it is the unfair reference, not a
     serving config).
  2. float16 vectors in an hnswlib HNSW index (M=16, ef_construction=200), ef in {64, 128},
     top-100.
  3. int8 scalar-quantised vectors (per-dimension min/max scale) in the same HNSW index,
     ef=128, top-100.

Late interaction, answerdotai/answerai-colbert-small-v1 (96-d/token), MaxSim:
  4. float16, exhaustive MaxSim over the whole corpus (E11's baseline, reproduced; hollow).
  5. ColBERTv2-style residual compression, implemented directly: k-means centroids (k=4096
     = 2^12, fit on a subsample of all surviving token vectors with MiniBatchKMeans; every
     token assigned to its nearest centroid), residual quantised to 2 bits/dim and,
     separately, 1 bit/dim using a shared scalar codebook (bucket cutoffs from quantiles of
     a residual sample, reconstruction value = mean residual in that bucket - this mirrors
     ColBERTv2's own bucketing, which is not per-dimension). Scored by reconstructing every
     document token vector and running exact MaxSim over the *whole* corpus - this isolates
     the storage/quality cost of compression from any retrieval-latency change, per the
     assignment.
  6. PLAID-style retrieval via pylate.indexes.PLAID (fast-plaid backend, all defaults:
     nbits=4 product-quantised residuals, IVF-style centroid probing for candidate
     generation, then PLAID's own internal decompressed-residual rerank of the top
     candidates). Queried at k=100; PLAID's returned score already is "top-100 then rerank
     with exact MaxSim over reconstructed vectors" - that reranking is what n_full_scores
     controls internally, left at its default.
  7. MUVERA fixed-dimensional encoding, implemented directly from arXiv:2405.19504: SimHash
     partitioning (k_sim random hyperplanes -> 2^k_sim buckets) plus an inner Gaussian
     projection to d_proj dims, repeated R_reps times and concatenated. Every token vector is
     first centred on the corpus's global mean token vector before hashing/projecting - the
     raw answerai-colbert-small-v1 token embeddings are strongly anisotropic (mean vector norm
     ~0.90 out of unit-norm tokens), so origin-centred SimHash collapses almost every token
     into one or two of the 2^k_sim buckets and the FDE carries essentially no signal; this
     centring is not spelled out in the paper (whose evaluated encoders are apparently closer
     to zero-mean) but is necessary here and is a standard LSH preprocessing step. Documents
     sum projected token vectors per bucket (empty buckets filled from the nearest non-empty
     bucket by Hamming distance in the SimHash code, per the paper's fill-empty-partitions
     default); queries average per bucket, no fill. Two sizes: "small" (k_sim=4, d_proj=16,
     R=20 -> 5,120 dims) and "large" (k_sim=5, d_proj=32, R=20 -> 20,480 dims). Retrieval is
     exhaustive inner product over the FDE (cheap at 5,183 docs); the small FDE is also put
     in an hnswlib index with the same M/ef_construction as config 2 to report that
     retrieval's latency. Reported with and without an exact-MaxSim rerank of the top-100
     over the original float16 token vectors.

Every configuration reports: nDCG@10 and Recall@100 (pytrec_eval), bytes actually served per
document (vectors plus any codebooks/centroids/graph overhead, amortised over the corpus),
and batch-1 CPU latency (p50/p95) to retrieve+score one query, excluding query encoding
(which is measured once per model and reported separately, since it is identical across
every configuration sharing that model).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE.parent / "public" / "figures"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(CACHE_DIR))

import hnswlib  # noqa: E402
import pytrec_eval  # noqa: E402  (import after HF_HOME is set)
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402

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
RUN_DEPTH_EXHAUSTIVE = 1000  # candidates kept in the "run" for configs that score the whole corpus
RUN_DEPTH_ANN = 100  # candidates kept for configs that only ever retrieve top-100 (ANN/PLAID/MUVERA)

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200

KMEANS_K = 4096  # 2**12 centroids for ColBERTv2-style residual compression
KMEANS_SAMPLE = 200_000

MUVERA_SETTINGS = {
    "small": {"k_sim": 4, "d_proj": 16, "r_reps": 20},
    "large": {"k_sim": 5, "d_proj": 32, "r_reps": 20},
}


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


# ------------------------------------------------------------------ small helpers


def summarize_latency_s(samples_s: list[float]) -> dict:
    arr = np.asarray(samples_s, dtype=np.float64)
    return {
        "n": int(arr.size),
        "p50_ms": float(np.median(arr) * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000),
        "mean_ms": float(np.mean(arr) * 1000),
    }


def compute_metrics_from_run(run: dict[str, dict[str, float]], qrels: dict, test_query_ids: list[str]) -> dict:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_100"})
    per_query = evaluator.evaluate(run)
    ndcg10 = float(np.mean([per_query[qid]["ndcg_cut_10"] for qid in test_query_ids]))
    recall100 = float(np.mean([per_query[qid]["recall_100"] for qid in test_query_ids]))
    return {"ndcg_cut_10": ndcg10, "recall_100": recall100, "n_queries": len(test_query_ids)}


def run_from_dense_row(row: np.ndarray, corpus_ids: list[str], depth: int) -> dict[str, float]:
    order = np.argsort(-row)[:depth]
    return {corpus_ids[j]: float(row[j]) for j in order}


def run_from_candidates(ids: list[str], scores: np.ndarray) -> dict[str, float]:
    return {cid: float(s) for cid, s in zip(ids, scores)}


def pad_token_arrays(token_arrays: list[np.ndarray], dim: int) -> tuple[np.ndarray, np.ndarray]:
    l_max = max(a.shape[0] for a in token_arrays)
    n = len(token_arrays)
    padded = np.zeros((n, l_max, dim), dtype=np.float32)
    mask = np.zeros((n, l_max), dtype=np.float32)
    for i, a in enumerate(token_arrays):
        length = a.shape[0]
        padded[i, :length] = a
        mask[i, :length] = 1.0
    return padded, mask


def split_by_counts(flat_array: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    out = []
    offset = 0
    for c in counts:
        out.append(flat_array[offset : offset + int(c)])
        offset += int(c)
    return out


def bytes_of_dir(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def hnsw_graph_overhead_bytes_per_doc(index: "hnswlib.Index", n_docs: int, dim: int, tmp_path: Path) -> float:
    """Bytes/doc consumed by the HNSW graph structure alone (index file minus raw float32
    vectors), measured empirically by serialising the index. hnswlib always stores vectors
    internally as float32 regardless of the "served" precision we report, so we subtract
    that float32 vector cost to get the graph-only overhead, then add back whatever
    precision (float16, int8, ...) we are actually claiming to serve.
    """
    index.save_index(str(tmp_path))
    total = tmp_path.stat().st_size
    tmp_path.unlink()
    vector_bytes_f32_total = n_docs * dim * 4
    return max(0.0, (total - vector_bytes_f32_total) / n_docs)


# ------------------------------------------------------------------ encoding (shared across configs)


def encode_single_vector(
    corpus_ids: list[str], corpus: dict, test_query_ids: list[str], queries: dict
) -> dict:
    cache_path = CACHE_DIR / "e12_sv_encodings.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if int(z["n_docs"]) == len(corpus_ids) and int(z["n_queries"]) == len(test_query_ids):
            print("[e12] single-vector: loaded cached encodings")
            return {
                "corpus_emb": z["corpus_emb"],
                "query_vecs": z["query_vecs"],
                "dim": int(z["dim"]),
                "encode_latency": summarize_latency_s(list(z["encode_latencies_s"])),
                "corpus_encode_seconds_total": float(z["corpus_encode_seconds_total"]),
            }

    model = SentenceTransformer(SINGLE_VECTOR_MODEL, device="cpu")
    dim = model.get_sentence_embedding_dimension()

    texts = doc_texts_for(corpus_ids, corpus)
    t0 = time.perf_counter()
    corpus_emb = model.encode(
        texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32)
    corpus_encode_s = time.perf_counter() - t0
    print(f"[e12] single-vector: corpus encoded in {corpus_encode_s:.1f}s -> {corpus_emb.shape}")

    query_vecs = np.empty((len(test_query_ids), dim), dtype=np.float32)
    encode_latencies: list[float] = []
    for i, qid in enumerate(test_query_ids):
        text = QUERY_INSTRUCTION + queries[qid]
        t0 = time.perf_counter()
        qv = model.encode(
            [text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0].astype(np.float32)
        dt = time.perf_counter() - t0
        query_vecs[i] = qv
        if i >= N_WARMUP:
            encode_latencies.append(dt)

    np.savez(
        cache_path,
        corpus_emb=corpus_emb,
        query_vecs=query_vecs,
        dim=dim,
        n_docs=len(corpus_ids),
        n_queries=len(test_query_ids),
        encode_latencies_s=np.asarray(encode_latencies),
        corpus_encode_seconds_total=corpus_encode_s,
    )

    return {
        "corpus_emb": corpus_emb,
        "query_vecs": query_vecs,
        "dim": dim,
        "encode_latency": summarize_latency_s(encode_latencies),
        "corpus_encode_seconds_total": corpus_encode_s,
    }


def encode_late_interaction(
    corpus_ids: list[str], corpus: dict, test_query_ids: list[str], queries: dict
) -> dict:
    cache_path = CACHE_DIR / "e12_li_encodings.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if int(z["n_docs"]) == len(corpus_ids) and int(z["n_queries"]) == len(test_query_ids):
            print("[e12] late-interaction: loaded cached encodings")
            doc_token_embs = split_by_counts(z["all_doc_tokens"], z["token_counts"])
            return {
                "doc_token_embs": doc_token_embs,
                "token_counts": z["token_counts"],
                "query_token_embs": z["query_token_embs"],
                "dim": int(z["dim"]),
                "query_length": int(z["query_length"]),
                "document_length": int(z["document_length"]),
                "encode_latency": summarize_latency_s(list(z["encode_latencies_s"])),
                "corpus_encode_seconds_total": float(z["corpus_encode_seconds_total"]),
            }

    from pylate import models as pylate_models

    model = pylate_models.ColBERT(model_name_or_path=LATE_INTERACTION_MODEL, device="cpu")
    document_length = int(getattr(model, "document_length", 0))
    query_length = int(getattr(model, "query_length", 0))

    texts = doc_texts_for(corpus_ids, corpus)
    t0 = time.perf_counter()
    doc_token_embs = model.encode(
        texts, is_query=False, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    doc_token_embs = [np.asarray(e, dtype=np.float32) for e in doc_token_embs]
    corpus_encode_s = time.perf_counter() - t0
    dim = int(doc_token_embs[0].shape[1])
    token_counts = np.array([e.shape[0] for e in doc_token_embs], dtype=np.int64)
    print(
        f"[e12] late-interaction: corpus encoded in {corpus_encode_s:.1f}s -> {len(doc_token_embs)} docs, "
        f"dim={dim}, mean tokens/doc={token_counts.mean():.1f}"
    )

    query_token_embs = np.empty((len(test_query_ids), query_length, dim), dtype=np.float32)
    encode_latencies: list[float] = []
    for i, qid in enumerate(test_query_ids):
        text = queries[qid]
        t0 = time.perf_counter()
        qv = model.encode(
            [text], is_query=True, batch_size=1, show_progress_bar=False, convert_to_numpy=True
        )[0].astype(np.float32)
        dt = time.perf_counter() - t0
        query_token_embs[i] = qv
        if i >= N_WARMUP:
            encode_latencies.append(dt)

    np.savez(
        cache_path,
        all_doc_tokens=np.concatenate(doc_token_embs, axis=0),
        token_counts=token_counts,
        query_token_embs=query_token_embs,
        dim=dim,
        query_length=query_length,
        document_length=document_length,
        n_docs=len(corpus_ids),
        n_queries=len(test_query_ids),
        encode_latencies_s=np.asarray(encode_latencies),
        corpus_encode_seconds_total=corpus_encode_s,
    )

    return {
        "doc_token_embs": doc_token_embs,
        "token_counts": token_counts,
        "query_token_embs": query_token_embs,
        "dim": dim,
        "query_length": query_length,
        "document_length": document_length,
        "encode_latency": summarize_latency_s(encode_latencies),
        "corpus_encode_seconds_total": corpus_encode_s,
    }


# ------------------------------------------------------------------ single-vector configs


def sv_f16_exhaustive(corpus_ids, test_query_ids, qrels, corpus_emb, query_vecs, dim) -> dict:
    n_docs = len(corpus_ids)
    latencies = []
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        t0 = time.perf_counter()
        row = corpus_emb @ query_vecs[i]
        dt = time.perf_counter() - t0
        run[qid] = run_from_dense_row(row, corpus_ids, RUN_DEPTH_EXHAUSTIVE)
        if i >= N_WARMUP:
            latencies.append(dt)
    metrics = compute_metrics_from_run(run, qrels, test_query_ids)
    return {
        "system": "single-vector",
        "config": "bge-small f16 exhaustive dot product",
        "hollow": True,
        "ndcg10": metrics["ndcg_cut_10"],
        "recall100": metrics["recall_100"],
        "bytes_per_doc": dim * 2,
        "latency": summarize_latency_s(latencies),
        "notes": "E11 baseline reproduced: no index at all, brute-force dot product over the full corpus.",
    }


def sv_hnsw(corpus_ids, test_query_ids, qrels, corpus_emb, query_vecs, dim, ef: int) -> dict:
    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=len(corpus_ids), M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, random_seed=SEED)
    index.add_items(corpus_emb, np.arange(len(corpus_ids)))
    index.set_ef(ef)

    graph_overhead = hnsw_graph_overhead_bytes_per_doc(
        index, len(corpus_ids), dim, CACHE_DIR / f"_tmp_sv_hnsw_ef{ef}.bin"
    )

    latencies = []
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        t0 = time.perf_counter()
        labels, distances = index.knn_query(query_vecs[i], k=RUN_DEPTH_ANN)
        dt = time.perf_counter() - t0
        ids = [corpus_ids[j] for j in labels[0]]
        run[qid] = run_from_candidates(ids, -distances[0])
        if i >= N_WARMUP:
            latencies.append(dt)
    metrics = compute_metrics_from_run(run, qrels, test_query_ids)
    return {
        "system": "single-vector",
        "config": f"bge-small f16 HNSW ef={ef}",
        "hollow": False,
        "ndcg10": metrics["ndcg_cut_10"],
        "recall100": metrics["recall_100"],
        "bytes_per_doc": dim * 2 + graph_overhead,
        "latency": summarize_latency_s(latencies),
        "notes": (
            f"hnswlib, M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, ef={ef}, top-100. "
            f"bytes/doc = {dim * 2} (float16 vectors) + {graph_overhead:.1f} (HNSW graph, measured "
            "from a serialised index)."
        ),
    }


def sv_int8_hnsw(corpus_ids, test_query_ids, qrels, corpus_emb, query_vecs, dim, ef: int) -> dict:
    scale = np.maximum(np.abs(corpus_emb).max(axis=0), 1e-8) / 127.0  # per-dim symmetric scale
    quantized = np.round(corpus_emb / scale[None, :]).clip(-127, 127).astype(np.int8)
    dequantized = (quantized.astype(np.float32)) * scale[None, :]

    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=len(corpus_ids), M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, random_seed=SEED)
    index.add_items(dequantized, np.arange(len(corpus_ids)))
    index.set_ef(ef)

    graph_overhead = hnsw_graph_overhead_bytes_per_doc(
        index, len(corpus_ids), dim, CACHE_DIR / f"_tmp_sv_int8_hnsw_ef{ef}.bin"
    )
    scale_overhead_per_doc = (dim * 4) / len(corpus_ids)  # float32 per-dim scale, amortised over the corpus

    latencies = []
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        t0 = time.perf_counter()
        labels, distances = index.knn_query(query_vecs[i], k=RUN_DEPTH_ANN)
        dt = time.perf_counter() - t0
        ids = [corpus_ids[j] for j in labels[0]]
        run[qid] = run_from_candidates(ids, -distances[0])
        if i >= N_WARMUP:
            latencies.append(dt)
    metrics = compute_metrics_from_run(run, qrels, test_query_ids)
    return {
        "system": "single-vector",
        "config": f"bge-small int8 HNSW ef={ef}",
        "hollow": False,
        "ndcg10": metrics["ndcg_cut_10"],
        "recall100": metrics["recall_100"],
        "bytes_per_doc": dim * 1 + graph_overhead + scale_overhead_per_doc,
        "latency": summarize_latency_s(latencies),
        "notes": (
            "Per-dimension symmetric int8 scalar quantisation (scale = max|value| / 127), dequantised "
            f"on the fly for scoring; query stays float32. bytes/doc = {dim} (int8 vectors) + "
            f"{graph_overhead:.1f} (graph) + {scale_overhead_per_doc:.2f} (per-dim scales, amortised)."
        ),
    }


# ------------------------------------------------------------------ late-interaction configs


def li_f16_exhaustive(corpus_ids, test_query_ids, qrels, doc_padded, doc_mask, token_counts, query_token_embs, dim) -> dict:
    from pylate import scores as pylate_scores

    doc_padded_t = torch.from_numpy(doc_padded)
    doc_mask_t = torch.from_numpy(doc_mask)

    latencies = []
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        q_t = torch.from_numpy(query_token_embs[i]).unsqueeze(0)
        t0 = time.perf_counter()
        row = pylate_scores.colbert_scores(
            queries_embeddings=q_t, documents_embeddings=doc_padded_t, documents_mask=doc_mask_t
        )[0].numpy()
        dt = time.perf_counter() - t0
        run[qid] = run_from_dense_row(row, corpus_ids, RUN_DEPTH_EXHAUSTIVE)
        if i >= N_WARMUP:
            latencies.append(dt)
    metrics = compute_metrics_from_run(run, qrels, test_query_ids)
    total_tokens = int(token_counts.sum())
    bytes_per_doc = total_tokens * dim * 2 / len(corpus_ids)
    return {
        "system": "late-interaction",
        "config": "ColBERT f16 exhaustive MaxSim",
        "hollow": True,
        "ndcg10": metrics["ndcg_cut_10"],
        "recall100": metrics["recall_100"],
        "bytes_per_doc": bytes_per_doc,
        "latency": summarize_latency_s(latencies),
        "notes": "E11 baseline reproduced: no index at all, exact MaxSim over every token of every document.",
    }


def fit_residual_compression(doc_token_embs: list[np.ndarray], dim: int) -> dict:
    all_tokens = np.concatenate(doc_token_embs, axis=0)
    total_tokens = all_tokens.shape[0]
    rng = np.random.default_rng(SEED)

    sample_n = min(KMEANS_SAMPLE, total_tokens)
    sample_idx = rng.choice(total_tokens, size=sample_n, replace=False)
    t0 = time.perf_counter()
    km = MiniBatchKMeans(n_clusters=KMEANS_K, batch_size=10_000, n_init=1, max_iter=100, random_state=SEED)
    km.fit(all_tokens[sample_idx])
    centroids = km.cluster_centers_.astype(np.float32)
    kmeans_fit_s = time.perf_counter() - t0
    print(f"[e12] residual: k-means (k={KMEANS_K}) fit on {sample_n} tokens in {kmeans_fit_s:.1f}s")

    # assign every token to its nearest centroid, chunked to bound memory
    t0 = time.perf_counter()
    centroid_sq = (centroids**2).sum(axis=1)
    chunk = 20_000
    assignment = np.empty(total_tokens, dtype=np.int32)
    for i in range(0, total_tokens, chunk):
        xb = all_tokens[i : i + chunk]
        d2 = centroid_sq[None, :] - 2 * (xb @ centroids.T)
        assignment[i : i + chunk] = np.argmin(d2, axis=1)
    assign_s = time.perf_counter() - t0
    print(f"[e12] residual: assigned {total_tokens} tokens to centroids in {assign_s:.1f}s")

    residual = all_tokens - centroids[assignment]
    return {"centroids": centroids, "assignment": assignment, "residual": residual, "all_tokens": all_tokens}


def quantize_residual(residual: np.ndarray, nbits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shared (not per-dimension) scalar codebook, quantile-bucketed, ColBERTv2-style."""
    rng = np.random.default_rng(seed)
    nlevels = 2**nbits
    flat = residual.reshape(-1)
    sample_size = min(2_000_000, flat.size)
    sample = rng.choice(flat, size=sample_size, replace=False)
    quantiles = np.linspace(0, 100, nlevels + 1)[1:-1]
    cutoffs = np.percentile(sample, quantiles).astype(np.float32)
    bucket_ids = np.searchsorted(cutoffs, flat).astype(np.int32)
    codebook = np.zeros(nlevels, dtype=np.float32)
    sample_buckets = np.searchsorted(cutoffs, sample)
    for b in range(nlevels):
        m = sample_buckets == b
        if m.any():
            codebook[b] = sample[m].mean()
    reconstructed = codebook[bucket_ids].reshape(residual.shape)
    return reconstructed.astype(np.float32), codebook


def li_residual(
    corpus_ids, test_query_ids, qrels, query_token_embs, dim, token_counts, fitted: dict, nbits: int
) -> dict:
    from pylate import scores as pylate_scores

    n_docs = len(corpus_ids)
    total_tokens = fitted["all_tokens"].shape[0]

    recon_residual, codebook = quantize_residual(fitted["residual"], nbits, seed=SEED + nbits)
    reconstructed_flat = fitted["centroids"][fitted["assignment"]] + recon_residual
    norms = np.linalg.norm(reconstructed_flat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    reconstructed_flat = (reconstructed_flat / norms).astype(np.float32)

    reconstructed_per_doc = split_by_counts(reconstructed_flat, token_counts)
    doc_padded, doc_mask = pad_token_arrays(reconstructed_per_doc, dim)
    doc_padded_t = torch.from_numpy(doc_padded)
    doc_mask_t = torch.from_numpy(doc_mask)

    latencies = []
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        q_t = torch.from_numpy(query_token_embs[i]).unsqueeze(0)
        t0 = time.perf_counter()
        row = pylate_scores.colbert_scores(
            queries_embeddings=q_t, documents_embeddings=doc_padded_t, documents_mask=doc_mask_t
        )[0].numpy()
        dt = time.perf_counter() - t0
        run[qid] = run_from_dense_row(row, corpus_ids, RUN_DEPTH_EXHAUSTIVE)
        if i >= N_WARMUP:
            latencies.append(dt)
    metrics = compute_metrics_from_run(run, qrels, test_query_ids)

    per_token_bytes = 2 + nbits * dim / 8  # 2-byte centroid id + quantised residual
    centroid_table_bytes = KMEANS_K * dim * 2  # float16 centroid table, amortised
    codebook_bytes = len(codebook) * 4
    bytes_per_doc = (total_tokens * per_token_bytes + centroid_table_bytes + codebook_bytes) / n_docs

    return {
        "system": "late-interaction",
        "config": f"ColBERTv2-style residual ({nbits}-bit)",
        "hollow": False,
        "ndcg10": metrics["ndcg_cut_10"],
        "recall100": metrics["recall_100"],
        "bytes_per_doc": bytes_per_doc,
        "latency": summarize_latency_s(latencies),
        "notes": (
            f"k-means k={KMEANS_K} (2^12) fit on a {min(KMEANS_SAMPLE, total_tokens):,}-token subsample "
            f"(MiniBatchKMeans); every token: 2-byte centroid id + {nbits}-bit/dim residual against a "
            f"{2**nbits}-value shared scalar codebook (quantile buckets, ColBERTv2-style, not per-dim). "
            "Reconstructed and exact-MaxSim'd over the whole corpus per query - isolates storage cost, "
            "not retrieval latency (latency is close to the exhaustive f16 baseline by design)."
        ),
    }


def li_plaid(corpus_ids, test_query_ids, qrels, doc_token_embs, query_token_embs, dim, doc_padded, doc_mask) -> dict:
    from pylate.indexes import PLAID

    index_folder = CACHE_DIR / "e12_plaid_index"
    shutil.rmtree(index_folder, ignore_errors=True)
    index = PLAID(
        index_folder=str(index_folder),
        index_name="e12",
        override=True,
        show_progress=False,
        device="cpu",
    )
    t0 = time.perf_counter()
    index.add_documents(documents_ids=corpus_ids, documents_embeddings=doc_token_embs)
    build_s = time.perf_counter() - t0
    print(f"[e12] PLAID: index built in {build_s:.1f}s")

    index_bytes = bytes_of_dir(index_folder)
    bytes_per_doc = index_bytes / len(corpus_ids)

    latencies = []
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        q = query_token_embs[i : i + 1]
        t0 = time.perf_counter()
        results = index(q, k=RUN_DEPTH_ANN)[0]
        dt = time.perf_counter() - t0
        run[qid] = {r["id"]: float(r["score"]) for r in results}
        if i >= N_WARMUP:
            latencies.append(dt)
    metrics = compute_metrics_from_run(run, qrels, test_query_ids)

    return {
        "system": "late-interaction",
        "config": "PLAID (pylate, default settings)",
        "hollow": False,
        "ndcg10": metrics["ndcg_cut_10"],
        "recall100": metrics["recall_100"],
        "bytes_per_doc": bytes_per_doc,
        "latency": summarize_latency_s(latencies),
        "notes": (
            "pylate.indexes.PLAID, fast-plaid backend, unmodified defaults (nbits=4 product-quantised "
            "residuals, n_ivf_probe=8, n_full_scores=8192); top-100 per query. Its returned score is "
            "PLAID's own centroid-pruned candidate generation reranked internally with reconstructed "
            f"residual vectors. bytes/doc measured directly from the serialised index on disk "
            f"({index_bytes:,} bytes total)."
        ),
    }


def build_muvera_projections(dim: int, k_sim: int, d_proj: int, r_reps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hyperplanes = rng.standard_normal((r_reps, k_sim, dim)).astype(np.float32)
    proj_matrices = (rng.standard_normal((r_reps, dim, d_proj)) / np.sqrt(d_proj)).astype(np.float32)
    return hyperplanes, proj_matrices


def bucket_ids_for_tokens(tokens: np.ndarray, hyperplane: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    centered = tokens - mean_vec
    bits = (centered @ hyperplane.T) > 0
    weights = (1 << np.arange(bits.shape[1])).astype(np.int64)
    return bits.astype(np.int64) @ weights


def scatter_sum(keys: np.ndarray, values: np.ndarray, n_bins: int) -> np.ndarray:
    d_proj = values.shape[1]
    out = np.empty((n_bins, d_proj), dtype=np.float64)
    for c in range(d_proj):
        out[:, c] = np.bincount(keys, weights=values[:, c], minlength=n_bins)
    return out.astype(np.float32)


def hamming_neighbor_order(k_sim: int) -> np.ndarray:
    B = 1 << k_sim
    dist = np.zeros((B, B), dtype=np.int32)
    for i in range(B):
        dist[i] = [bin(i ^ j).count("1") for j in range(B)]
    return np.argsort(dist, axis=1, kind="stable")


def build_doc_fdes(
    doc_token_embs: list[np.ndarray],
    hyperplanes: np.ndarray,
    proj_matrices: np.ndarray,
    k_sim: int,
    d_proj: int,
    r_reps: int,
    mean_vec: np.ndarray,
) -> np.ndarray:
    n_docs = len(doc_token_embs)
    B = 1 << k_sim
    fde = np.zeros((n_docs, r_reps * B * d_proj), dtype=np.float32)
    doc_ids_per_token = np.concatenate(
        [np.full(t.shape[0], i, dtype=np.int64) for i, t in enumerate(doc_token_embs)]
    )
    all_tokens = np.concatenate(doc_token_embs, axis=0)
    centered_tokens = all_tokens - mean_vec
    neighbor_order = hamming_neighbor_order(k_sim)
    for r in range(r_reps):
        bucket_ids = bucket_ids_for_tokens(all_tokens, hyperplanes[r], mean_vec)
        projected = centered_tokens @ proj_matrices[r]
        combined = doc_ids_per_token * B + bucket_ids
        sums = scatter_sum(combined, projected, n_docs * B).reshape(n_docs, B, d_proj)
        counts = np.bincount(combined, minlength=n_docs * B).reshape(n_docs, B)
        empty = counts == 0
        docs_with_empty = np.where(empty.any(axis=1))[0]
        for di in docs_with_empty:
            doc_counts = counts[di]
            for b in np.where(empty[di])[0]:
                for cand in neighbor_order[b]:
                    if doc_counts[cand] > 0:
                        sums[di, b] = sums[di, cand]
                        break
        fde[:, r * B * d_proj : (r + 1) * B * d_proj] = sums.reshape(n_docs, -1)
    return fde


def build_query_fde(
    query_tokens: np.ndarray,
    hyperplanes: np.ndarray,
    proj_matrices: np.ndarray,
    k_sim: int,
    d_proj: int,
    r_reps: int,
    mean_vec: np.ndarray,
) -> np.ndarray:
    B = 1 << k_sim
    fde = np.zeros(r_reps * B * d_proj, dtype=np.float32)
    centered_tokens = query_tokens - mean_vec
    for r in range(r_reps):
        bucket_ids = bucket_ids_for_tokens(query_tokens, hyperplanes[r], mean_vec)
        projected = centered_tokens @ proj_matrices[r]
        sums = scatter_sum(bucket_ids, projected, B)
        counts = np.bincount(bucket_ids, minlength=B)
        nz = counts > 0
        sums[nz] /= counts[nz, None]
        fde[r * B * d_proj : (r + 1) * B * d_proj] = sums.reshape(-1)
    return fde


def rerank_candidates(
    candidate_idx: np.ndarray, doc_padded: np.ndarray, doc_mask: np.ndarray, query_tokens: np.ndarray
) -> np.ndarray:
    from pylate import scores as pylate_scores

    q_t = torch.from_numpy(query_tokens).unsqueeze(0)
    d_t = torch.from_numpy(doc_padded[candidate_idx])
    m_t = torch.from_numpy(doc_mask[candidate_idx])
    return pylate_scores.colbert_scores(queries_embeddings=q_t, documents_embeddings=d_t, documents_mask=m_t)[
        0
    ].numpy()


def li_muvera(
    corpus_ids,
    test_query_ids,
    qrels,
    doc_token_embs,
    query_token_embs,
    dim,
    token_counts,
    doc_padded,
    doc_mask,
    size_name: str,
    use_hnsw: bool,
) -> list[dict]:
    settings = MUVERA_SETTINGS[size_name]
    k_sim, d_proj, r_reps = settings["k_sim"], settings["d_proj"], settings["r_reps"]
    fde_dim = r_reps * (1 << k_sim) * d_proj
    n_docs = len(corpus_ids)

    hyperplanes, proj_matrices = build_muvera_projections(dim, k_sim, d_proj, r_reps, seed=SEED)
    mean_vec = np.concatenate(doc_token_embs, axis=0).mean(axis=0).astype(np.float32)

    t0 = time.perf_counter()
    doc_fdes = build_doc_fdes(doc_token_embs, hyperplanes, proj_matrices, k_sim, d_proj, r_reps, mean_vec)
    fde_build_s = time.perf_counter() - t0
    print(f"[e12] MUVERA {size_name}: doc FDEs (dim={fde_dim}) built in {fde_build_s:.1f}s")

    projection_bytes_total = hyperplanes.nbytes + proj_matrices.nbytes
    fde_bytes_per_doc = fde_dim * 2 + projection_bytes_total / n_docs  # float16 FDE + amortised projections
    raw_token_bytes_per_doc = int(token_counts.sum()) * dim * 2 / n_docs  # needed only if reranking

    results = []

    # -- exhaustive retrieval, no rerank / with rerank --
    latencies_norerank, latencies_rerank = [], []
    run_norerank: dict[str, dict[str, float]] = {}
    run_rerank: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        t0 = time.perf_counter()
        qfde = build_query_fde(query_token_embs[i], hyperplanes, proj_matrices, k_sim, d_proj, r_reps, mean_vec)
        scores = doc_fdes @ qfde
        top_idx = np.argpartition(-scores, RUN_DEPTH_ANN - 1)[:RUN_DEPTH_ANN]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        dt_norerank = time.perf_counter() - t0
        run_norerank[qid] = {corpus_ids[j]: float(scores[j]) for j in top_idx}

        rerank_scores = rerank_candidates(top_idx, doc_padded, doc_mask, query_token_embs[i])
        dt_rerank = time.perf_counter() - t0
        run_rerank[qid] = {corpus_ids[top_idx[j]]: float(rerank_scores[j]) for j in range(len(top_idx))}

        if i >= N_WARMUP:
            latencies_norerank.append(dt_norerank)
            latencies_rerank.append(dt_rerank)

    metrics_norerank = compute_metrics_from_run(run_norerank, qrels, test_query_ids)
    metrics_rerank = compute_metrics_from_run(run_rerank, qrels, test_query_ids)

    results.append(
        {
            "system": "late-interaction",
            "config": f"MUVERA FDE {fde_dim}-d, no rerank",
            "hollow": False,
            "ndcg10": metrics_norerank["ndcg_cut_10"],
            "recall100": metrics_norerank["recall_100"],
            "bytes_per_doc": fde_bytes_per_doc,
            "latency": summarize_latency_s(latencies_norerank),
            "notes": (
                f"k_sim={k_sim} (B={1 << k_sim} buckets), d_proj={d_proj}, R_reps={r_reps} -> "
                f"dim={fde_dim}. Exhaustive inner product over all {n_docs} docs' FDEs, top-100 taken "
                "as the final answer (no MaxSim rerank). bytes/doc is the FDE alone."
            ),
        }
    )
    results.append(
        {
            "system": "late-interaction",
            "config": f"MUVERA FDE {fde_dim}-d + MaxSim rerank",
            "hollow": False,
            "ndcg10": metrics_rerank["ndcg_cut_10"],
            "recall100": metrics_rerank["recall_100"],
            "bytes_per_doc": fde_bytes_per_doc + raw_token_bytes_per_doc,
            "latency": summarize_latency_s(latencies_rerank),
            "notes": (
                f"Same FDE (dim={fde_dim}) retrieval, top-100 reranked with exact MaxSim over the "
                "original float16 token vectors. bytes/doc therefore adds the full per-token float16 "
                f"store ({raw_token_bytes_per_doc:.0f} B/doc) on top of the FDE - reranking needs the "
                "original vectors available for every document, not just the FDE."
            ),
        }
    )

    if not use_hnsw:
        return results

    # -- same HNSW index as config 2, built over the FDE, + rerank --
    index = hnswlib.Index(space="ip", dim=fde_dim)
    index.init_index(max_elements=n_docs, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, random_seed=SEED)
    index.add_items(doc_fdes, np.arange(n_docs))
    index.set_ef(128)
    graph_overhead = hnsw_graph_overhead_bytes_per_doc(
        index, n_docs, fde_dim, CACHE_DIR / f"_tmp_muvera_hnsw_{size_name}.bin"
    )
    fde_hnsw_bytes_per_doc = fde_dim * 2 + graph_overhead + projection_bytes_total / n_docs

    latencies_hnsw = []
    run_hnsw: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(test_query_ids):
        t0 = time.perf_counter()
        qfde = build_query_fde(query_token_embs[i], hyperplanes, proj_matrices, k_sim, d_proj, r_reps, mean_vec)
        labels, _distances = index.knn_query(qfde, k=RUN_DEPTH_ANN)
        cand_idx = labels[0]
        rerank_scores = rerank_candidates(cand_idx, doc_padded, doc_mask, query_token_embs[i])
        dt = time.perf_counter() - t0
        run_hnsw[qid] = {corpus_ids[cand_idx[j]]: float(rerank_scores[j]) for j in range(len(cand_idx))}
        if i >= N_WARMUP:
            latencies_hnsw.append(dt)
    metrics_hnsw = compute_metrics_from_run(run_hnsw, qrels, test_query_ids)

    results.append(
        {
            "system": "late-interaction",
            "config": f"MUVERA FDE {fde_dim}-d HNSW + rerank",
            "hollow": False,
            "ndcg10": metrics_hnsw["ndcg_cut_10"],
            "recall100": metrics_hnsw["recall_100"],
            "bytes_per_doc": fde_hnsw_bytes_per_doc + raw_token_bytes_per_doc,
            "latency": summarize_latency_s(latencies_hnsw),
            "notes": (
                f"FDE (dim={fde_dim}) placed in the same hnswlib index as the single-vector HNSW config "
                f"(M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, ef=128), top-100, then exact-MaxSim "
                "rerank over original float16 token vectors. bytes/doc adds the FDE (with its own "
                "measured graph overhead) and the full per-token float16 store needed for reranking."
            ),
        }
    )
    return results


# ------------------------------------------------------------------ figure


def _place_labels(ax, fig, entries: list[dict], fontsize: float) -> None:
    """Greedily place each (point, text) label at the first non-overlapping candidate offset,
    trying offsets at increasing radius and 12 angles around the point, connected back to the
    point with a thin leader line. Avoids the label soup that a fixed offset produces once
    several points share similar x/y (as happens here - several configs land within a few
    percent of each other in quality and an order of magnitude of each other in bytes/doc).
    """
    pt_to_px = fig.dpi / 72.0
    placed_boxes: list[tuple[float, float, float, float]] = []
    step = fontsize + 3
    candidates = [
        (radius * step * np.cos(np.radians(angle)), radius * step * np.sin(np.radians(angle)))
        for radius in range(1, 9)
        for angle in range(15, 360, 30)
    ]
    ax_box = ax.get_window_extent()
    for e in entries:
        px, py = ax.transData.transform((e["x"], e["y"]))
        text = e["text"]
        w_px = 0.58 * fontsize * len(text) * pt_to_px
        h_px = 1.25 * fontsize * pt_to_px
        best_in_bounds, best_any = None, None
        for dx, dy in candidates:
            dx_px, dy_px = dx * pt_to_px, dy * pt_to_px
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"
            bx0 = px + dx_px if ha == "left" else px + dx_px - w_px
            by0 = py + dy_px if va == "bottom" else py + dy_px - h_px
            box = (bx0 - 2, by0 - 2, bx0 + w_px + 2, by0 + h_px + 2)
            collides = any(box[0] < pb[2] and box[2] > pb[0] and box[1] < pb[3] and box[3] > pb[1] for pb in placed_boxes)
            if collides:
                continue
            candidate = (dx, dy, ha, va, box)
            if best_any is None:
                best_any = candidate
            in_bounds = box[0] >= ax_box.x0 and box[2] <= ax_box.x1 and box[1] >= ax_box.y0 and box[3] <= ax_box.y1
            if in_bounds:
                best_in_bounds = candidate
                break
        chosen = best_in_bounds or best_any
        if chosen is None:
            dx, dy = candidates[-1]
            ha, va = ("left" if dx >= 0 else "right"), ("bottom" if dy >= 0 else "top")
            chosen = (dx, dy, ha, va, (px, py, px + w_px, py + h_px))
        dx, dy, ha, va, box = chosen
        placed_boxes.append(box)
        ax.annotate(
            text,
            xy=(e["x"], e["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            color=MUTED,
            fontsize=fontsize,
            ha=ha,
            va=va,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkA": 0, "shrinkB": 5},
        )


FIGURE_CONFIGS = [
    ("bge-small f16 exhaustive dot product", "exhaustive"),
    ("bge-small f16 HNSW ef=128", "HNSW f16"),
    ("bge-small int8 HNSW ef=128", "HNSW int8"),
    ("ColBERT f16 exhaustive MaxSim", "exhaustive MaxSim"),
    ("ColBERTv2-style residual (2-bit)", "residual 2-bit"),
    ("ColBERTv2-style residual (1-bit)", "residual 1-bit"),
    ("PLAID (pylate, default settings)", "PLAID"),
    ("MUVERA FDE 5120-d + MaxSim rerank", "MUVERA + rerank"),
]


def make_figure(table: list[dict], quality_bar_ndcg: float, best_config: str | None, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_config = {row["config"]: row for row in table}
    rows = [(by_config[cfg], label) for cfg, label in FIGURE_CONFIGS if cfg in by_config]
    off_chart = [
        by_config[cfg]["ndcg10"]
        for cfg in ("MUVERA FDE 5120-d, no rerank", "MUVERA FDE 20480-d, no rerank")
        if cfg in by_config
    ]

    fig, (ax_b, ax_l) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=150)
    fig.patch.set_facecolor(BG)

    for ax, get_x, xlabel in (
        (ax_b, lambda row: row["bytes_per_doc"], "bytes / document (log scale)"),
        (ax_l, lambda row: row["latency"]["p50_ms"], "p50 latency, ms (log scale)"),
    ):
        ax.set_facecolor(BG)
        for row, _label in rows:
            color = BLUE if row["system"] == "single-vector" else ORANGE
            hollow = row.get("hollow", False)
            x, y = get_x(row), row["ndcg10"]
            if hollow:
                ax.scatter([x], [y], s=230, facecolors="none", edgecolors=color, linewidths=2.5, zorder=3)
            elif row["config"] == best_config:
                ax.scatter([x], [y], s=230, facecolors=color, edgecolors=EMERALD, linewidths=2.5, zorder=4)
            else:
                ax.scatter([x], [y], s=180, color=color, zorder=3)
        ax.axhline(quality_bar_ndcg, color=EMERALD, linewidth=1, linestyle="--", alpha=0.6, zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, color=MUTED, fontsize=13)
        ax.set_ylabel("nDCG@10 (SciFact test, 300 queries)", color=MUTED, fontsize=13)
        ax.tick_params(axis="both", colors=MUTED, labelsize=11)
        for spine_name, spine in ax.spines.items():
            if spine_name in ("top", "right"):
                spine.set_visible(False)
            else:
                spine.set_color(MUTED)
        ax.set_ylim(0.66, 0.76)
        x_vals = [get_x(row) for row, _label in rows]
        ax.set_xlim(min(x_vals) / 1.8, max(x_vals) * 2.5)

    # labels are placed after axis limits/log-scale are final and a draw has fixed the pixel
    # transform, so the greedy placer works in real display coordinates
    fig.canvas.draw()
    for ax, get_x in ((ax_b, lambda row: row["bytes_per_doc"]), (ax_l, lambda row: row["latency"]["p50_ms"])):
        entries = [{"x": get_x(row), "y": row["ndcg10"], "text": label} for row, label in rows]
        _place_labels(ax, fig, entries, fontsize=14)

    if off_chart:
        note = "MUVERA without rerank: " + " / ".join(f"{v:.2f}" for v in off_chart) + ", off chart"
        ax_l.text(
            0.5, -0.14, note, transform=ax_l.transAxes, color=MUTED, fontsize=11, ha="center", va="top",
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=12, label="single-vector"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=12, label="late interaction"),
        plt.Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=FG, markersize=12,
            label="E11 exhaustive baseline (unfair, hollow)",
        ),
        plt.Line2D([0], [0], color=EMERALD, linewidth=1.5, linestyle="--", label="within 0.01 nDCG@10 of exhaustive MaxSim"),
    ]
    legend = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    for text in legend.get_texts():
        text.set_color(FG)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels = load_corpus_queries_qrels()
    corpus_ids = sorted(corpus.keys(), key=int)
    test_query_ids = sorted(qrels.keys(), key=int)
    print(f"[e12] corpus={len(corpus_ids)} test_queries={len(test_query_ids)}")

    print("[e12] encoding single-vector (bge-small-en-v1.5)")
    sv = encode_single_vector(corpus_ids, corpus, test_query_ids, queries)
    print("[e12] encoding late-interaction (answerai-colbert-small-v1)")
    li = encode_late_interaction(corpus_ids, corpus, test_query_ids, queries)

    doc_padded, doc_mask = pad_token_arrays(li["doc_token_embs"], li["dim"])

    table: list[dict] = []

    print("[e12] single-vector: exhaustive f16")
    table.append(sv_f16_exhaustive(corpus_ids, test_query_ids, qrels, sv["corpus_emb"], sv["query_vecs"], sv["dim"]))
    for ef in (64, 128):
        print(f"[e12] single-vector: HNSW f16 ef={ef}")
        table.append(sv_hnsw(corpus_ids, test_query_ids, qrels, sv["corpus_emb"], sv["query_vecs"], sv["dim"], ef))
    print("[e12] single-vector: HNSW int8 ef=128")
    sv_int8_row = sv_int8_hnsw(corpus_ids, test_query_ids, qrels, sv["corpus_emb"], sv["query_vecs"], sv["dim"], 128)
    f16_row_128 = table[-1]  # bge-small f16 HNSW ef=128, just appended
    sv_int8_row["notes"] += (
        f" vs the float16 HNSW at the same ef=128: nDCG@10 {f16_row_128['ndcg10']:.4f} -> {sv_int8_row['ndcg10']:.4f} "
        f"({sv_int8_row['ndcg10'] - f16_row_128['ndcg10']:+.4f}), recall@100 {f16_row_128['recall100']:.4f} -> "
        f"{sv_int8_row['recall100']:.4f} ({sv_int8_row['recall100'] - f16_row_128['recall100']:+.4f})."
    )
    table.append(sv_int8_row)

    print("[e12] late-interaction: exhaustive f16")
    table.append(
        li_f16_exhaustive(
            corpus_ids, test_query_ids, qrels, doc_padded, doc_mask, li["token_counts"], li["query_token_embs"], li["dim"]
        )
    )

    print("[e12] late-interaction: fitting residual compression (k-means)")
    fitted = fit_residual_compression(li["doc_token_embs"], li["dim"])
    for nbits in (2, 1):
        print(f"[e12] late-interaction: residual compression {nbits}-bit")
        table.append(
            li_residual(corpus_ids, test_query_ids, qrels, li["query_token_embs"], li["dim"], li["token_counts"], fitted, nbits)
        )

    print("[e12] late-interaction: PLAID")
    table.append(
        li_plaid(corpus_ids, test_query_ids, qrels, li["doc_token_embs"], li["query_token_embs"], li["dim"], doc_padded, doc_mask)
    )

    print("[e12] late-interaction: MUVERA small (5,120-d), with HNSW variant")
    table.extend(
        li_muvera(
            corpus_ids, test_query_ids, qrels, li["doc_token_embs"], li["query_token_embs"], li["dim"], li["token_counts"],
            doc_padded, doc_mask, "small", use_hnsw=True,
        )
    )
    print("[e12] late-interaction: MUVERA large (20,480-d)")
    table.extend(
        li_muvera(
            corpus_ids, test_query_ids, qrels, li["doc_token_embs"], li["query_token_embs"], li["dim"], li["token_counts"],
            doc_padded, doc_mask, "large", use_hnsw=False,
        )
    )

    for row in table:
        print(
            f"[e12] {row['system']:16s} {row['config']:42s} nDCG@10={row['ndcg10']:.4f} "
            f"recall@100={row['recall100']:.4f} bytes/doc={row['bytes_per_doc']:.0f} "
            f"p50={row['latency']['p50_ms']:.3f}ms p95={row['latency']['p95_ms']:.3f}ms"
        )

    # headline: best late-interaction *serving* config (i.e. not the exhaustive baseline) within
    # 0.01 nDCG@10 of exhaustive MaxSim, chosen by lowest bytes/doc, compared against single-vector
    # HNSW float16 (ef=128, the higher-quality of the two single-vector HNSW operating points).
    li_exhaustive_ndcg = next(r for r in table if r["config"] == "ColBERT f16 exhaustive MaxSim")["ndcg10"]
    sv_hnsw128 = next(r for r in table if r["config"] == "bge-small f16 HNSW ef=128")
    li_serving_candidates = [
        r
        for r in table
        if r["system"] == "late-interaction"
        and not r.get("hollow")
        and r["ndcg10"] >= li_exhaustive_ndcg - 0.01
    ]
    best_li = min(li_serving_candidates, key=lambda r: r["bytes_per_doc"]) if li_serving_candidates else None

    figure_path = FIGURES_DIR / "exp-e12-fair-serving.png"
    make_figure(table, li_exhaustive_ndcg - 0.01, best_li["config"] if best_li else None, figure_path)
    print(f"[e12] wrote {figure_path}")

    out = {
        "meta": {
            "dataset": "scifact",
            "corpus_size": len(corpus_ids),
            "num_test_queries": len(test_query_ids),
            "seed": SEED,
            "device": "cpu",
            "single_vector_model": SINGLE_VECTOR_MODEL,
            "late_interaction_model": LATE_INTERACTION_MODEL,
            "single_vector_query_encode_latency": sv["encode_latency"],
            "late_interaction_query_encode_latency": li["encode_latency"],
            "single_vector_corpus_encode_seconds_total": sv["corpus_encode_seconds_total"],
            "late_interaction_corpus_encode_seconds_total": li["corpus_encode_seconds_total"],
            "late_interaction_mean_tokens_per_doc": float(li["token_counts"].mean()),
            "hnsw_params": {"M": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
            "kmeans_k": KMEANS_K,
            "muvera_settings": MUVERA_SETTINGS,
        },
        "table": table,
        "headline": {
            "late_interaction_exhaustive_ndcg10": li_exhaustive_ndcg,
            "quality_bar": "within 0.01 nDCG@10 of exhaustive MaxSim",
            "best_late_interaction_serving_config": best_li["config"] if best_li else None,
            "best_late_interaction_bytes_per_doc": best_li["bytes_per_doc"] if best_li else None,
            "best_late_interaction_p50_ms": best_li["latency"]["p50_ms"] if best_li else None,
            "single_vector_hnsw_f16_ef128_bytes_per_doc": sv_hnsw128["bytes_per_doc"],
            "single_vector_hnsw_f16_ef128_p50_ms": sv_hnsw128["latency"]["p50_ms"],
            "bytes_ratio_vs_single_vector_hnsw": (
                (best_li["bytes_per_doc"] / sv_hnsw128["bytes_per_doc"]) if best_li else None
            ),
            "latency_ratio_vs_single_vector_hnsw": (
                (best_li["latency"]["p50_ms"] / sv_hnsw128["latency"]["p50_ms"]) if best_li else None
            ),
        },
    }

    out_path = RESULTS_DIR / "e12_fair_serving.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e12] wrote {out_path}")

    if best_li:
        print(
            f"[e12] headline: best late-interaction serving config within 0.01 nDCG@10 of exhaustive "
            f"({li_exhaustive_ndcg:.4f}) is '{best_li['config']}' "
            f"(nDCG@10={best_li['ndcg10']:.4f}, {best_li['bytes_per_doc']:.0f} B/doc, "
            f"p50={best_li['latency']['p50_ms']:.3f}ms) vs single-vector HNSW f16 ef=128 "
            f"({sv_hnsw128['bytes_per_doc']:.0f} B/doc, p50={sv_hnsw128['latency']['p50_ms']:.3f}ms): "
            f"{out['headline']['bytes_ratio_vs_single_vector_hnsw']:.1f}x storage, "
            f"{out['headline']['latency_ratio_vs_single_vector_hnsw']:.1f}x latency."
        )
    else:
        print("[e12] headline: no late-interaction serving config met the 0.01 nDCG@10 quality bar")


if __name__ == "__main__":
    main()

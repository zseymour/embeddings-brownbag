# Late-interaction serving research

## Decision

### Recommended on-slide replacement

Use the **token-pooling index comparison**, not the raw float16 59× diagnostic:

> **2-bit PLAID + 2× token pooling: 760 MB → 388 MB per TREC-COVID index (1.96× smaller), with only 1.34% average effectiveness loss at pooling factor 2 after excluding the Touché and FiQA outliers.** A 16-bit dense HNSW index on the same TREC-COVID setup is 345 MB.

**Denominator and configuration:** complete on-disk index size, not vector payload size; TREC-COVID, documents truncated to 256 tokens; hierarchical token pooling applied to ColBERTv2 vectors; 2-bit PLAID for the late-interaction rows; 16-bit dense vectors in HNSW for the single-vector reference. The authors state that the 16-bit dense and 2-bit PLAID representations are selected because both have virtually no quantization-related degradation. The 1.34% figure is the paper's average degradation after outlier treatment; Table 2 also reports the per-dataset relative score (100 = no pooling), including 102.95 for TREC-COVID at factor 2.

**Primary source:** Clavié, Chaffin, and Adams, *Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling*, https://arxiv.org/pdf/2409.14683, §4.2, Table 2 (quality) and §4.3, Table 3 (full index sizes). The same paper says the method needs no model retraining or query-time processing and is a drop-in indexing step.

**Transition into MUVERA:**

> Pooling makes the retained token set smaller; it does not remove late interaction. **MUVERA changes the shortlist representation:** one fixed-dimensional encoding per document for ordinary MIPS, followed by exact MaxSim reranking of candidates.

This keeps the slide's point honest: current late-interaction serving can compress and pool its token set close to a dense-index footprint, while MUVERA changes the serving primitive rather than claiming that raw token storage is the production bill.

### Why this replaces 59×

The local 59× result is a useful pressure test but not a serving comparison. It compares exhaustive, uncompressed float16 token MaxSim with exhaustive float16 single-vector scoring on a 5,183-document SciFact corpus. It has no ANN candidate generation, no compressed index, and no production-style shortlist. The raw number should remain explicitly labelled **diagnostic** if it remains anywhere in the talk.

The recommended claim has a complete indexed representation on both sides, an explicit corpus and document-length cap, a published quality measurement, and an explicit denominator. It is therefore more defensible than an arbitrary ratio of raw token bytes to one pooled vector.

---

## Evidence matrix

### 1. Raw/uncompressed and compressed ColBERTv2 storage

| Claim | Numerator / denominator | Workload and configuration | Verdict and source location |
|---|---|---|---|
| ColBERTv2 residual coding uses 20 or 36 bytes per 128-dimensional token vector, versus 256 bytes for ColBERT's 16-bit vector | 256 / 20 = 12.8× for 1-bit residuals; 256 / 36 = 7.1× for 2-bit residuals | Per-token representation; 4-byte centroid ID plus 16 or 32 bytes of residual; not a complete index | **Usable as a raw representation fact, not as a deployment bill.** ColBERTv2, https://arxiv.org/pdf/2112.01488, §3.3 Representation. |
| MS MARCO ColBERT index 154 GiB → ColBERTv2 16 GiB (1-bit) or 25 GiB (2-bit) | 154 / 16 = 9.6×; 154 / 25 = 6.2×; paper summarizes this as 6–10× | MS MARCO, about 9M passages; the compressed numbers include 4.5 GiB for the inverted list | **Usable, but only as late-interaction-vs-late-interaction compression.** ColBERTv2, https://arxiv.org/pdf/2112.01488, §5.3 Efficiency. |
| The paper says 25 GiB is similar to one 768-dimensional 4-byte vector per passage | 25 GiB is approximately 9M × 768 × 4 bytes; the comparison excludes the single-vector ANN graph and compares it with a ColBERTv2 index that includes a 4.5 GiB inverted list | MS MARCO; single-vector arithmetic is a storage estimate, not a measured HNSW index | **Reject as a headline ratio.** The paper itself notes that HNSW can make the single-vector index larger and that single-vector quantization can make it smaller. ColBERTv2, https://arxiv.org/pdf/2112.01488, §5.3. |
| 2-bit compression preserves vanilla ColBERT quality: 36.2 MRR@10 and 82.3 Recall@50 versus 36.2 and 82.1 for uncompressed vanilla ColBERT | Same model and MS MARCO quality metric; compare compressed and uncompressed late interaction | MS MARCO; ColBERTv2 Appendix B | **Usable for compression quality, not a single-vector comparison.** ColBERTv2, https://arxiv.org/pdf/2112.01488, Appendix B, “Impact of Compression.” |

### 2. PLAID: indexed serving, quality, and latency

| Claim | Numerator / denominator | Workload and configuration | Verdict and source location |
|---|---|---|---|
| PLAID index is 21.6 GiB versus 24.6 GiB for vanilla ColBERTv2 | 24.6 / 21.6 = 1.14× smaller (12.2%) | MS MARCO v1: 8.8M passages, 597.9M tokens, 6,980 queries; both use the ColBERTv2 compressed representation | **Usable as an index-layout saving, not as a single-vector comparison.** PLAID, https://arxiv.org/pdf/2205.09707, Table 1. |
| PLAID at the conservative `k=1000` setting preserves quality while reducing latency versus vanilla ColBERTv2 | Paper reports 6.8× GPU and 45× CPU speedups while matching MRR@10 and Recall@100 | MS MARCO v1; same ColBERTv2 model and corpus; latency is average retrieval latency and the paper excludes query encoding for neural systems | **Usable when the baseline and exclusion are stated.** PLAID, https://arxiv.org/pdf/2205.09707, Table 3 and §5.1–§5.2. |
| PLAID uses compressed indexes at large scale: 24.6→21.6 GiB on MS MARCO v1, 105.2→92.0 GiB on Wikipedia, 14.0→12.3 GiB on LoTTE pooled, and 246.0→202.2 GiB on MS MARCO v2 | Each ratio is vanilla ColBERTv2 index / PLAID index; all are complete index sizes reported in the same table | 2-bit residuals except MS MARCO v2, which uses 1-bit residuals; the table's corpus sizes and token counts are explicit | **Usable for scale context; do not mix rows into one ratio.** PLAID, https://arxiv.org/pdf/2205.09707, Table 1 and §5.1. |
| PLAID's centroid-only candidate generation contains at least 99% of vanilla's top-k results when probing 10k candidates for k=1k | Candidate recall relative to vanilla ColBERTv2's top-k, not relevance recall | MS MARCO v1 and LoTTE pooled; Figure 3 | **Usable as an algorithmic explanation, not as storage or end-to-end quality.** PLAID, https://arxiv.org/pdf/2205.09707, §3.3 and Figure 3. |

PLAID is a fair same-family serving baseline, but it does not establish that late interaction costs 6–10× a dense index. Its main result is candidate pruning and optimized kernels over a residual-compressed token index.

### 3. MUVERA: fixed-dimensional encoding and what its ratios mean

| Claim | Numerator / denominator | Workload and configuration | Verdict and source location |
|---|---|---|---|
| PQ-256-8 compresses a 10,240-dimensional FDE to 1,280 bytes, a 32× payload reduction versus one float per dimension | 10,240 × 4 bytes = 40,960 bytes / 1,280 bytes = 32× | 10,240-dimensional document FDE; PQ-256-8; FDE retrieval via DiskANN and Chamfer/MaxSim rerank | **Usable as FDE payload compression only.** It excludes DiskANN graph overhead and the original token store required for exact reranking. MUVERA, https://arxiv.org/pdf/2405.19504, Abstract, §3.2, and Appendix C.4. |
| MUVERA averages 10% higher Recall@100/Recall@1000 and 90% lower latency than PLAID | Average over the six evaluated BEIR datasets and k ∈ {100, 1000}; latency is PLAID latency / MUVERA latency | MS MARCO, HotpotQA, NQ, Quora, SciDocs, and ArguAna; ColBERTv2 embeddings; same 10,240-dimensional FDEs with PQ-256-8; single-thread latency on an Intel Sapphire Rapids machine | **Usable as a published serving comparison.** It is a better next-slide result than a storage ratio. MUVERA, https://arxiv.org/pdf/2405.19504, §3.2 and Figure 8. |
| FDEs need 2–5× fewer candidates than the single-vector heuristic at comparable recall | Candidate count required to recover the Chamfer nearest neighbor; not bytes, end-to-end latency, or relevance recall | MS MARCO and other BEIR collections; Figure 5 compares FDEs with and without document-ID deduplication | **Usable only as candidate-generation evidence.** MUVERA, https://arxiv.org/pdf/2405.19504, §3.1, Figure 5, and Table 1 in Appendix C. |
| MS MARCO average document has 78.8 token embeddings; original MV representation is about 10,087 floats per document, while the FDE is 10,240 dimensions | FDE dimension / original average token-vector payload is close to 1.0 before graph and rerank storage | ColBERTv2: 128 dimensions, 32 query vectors, variable document vectors; MS MARCO | **Reject as a claim that MUVERA makes the whole index one-vector-sized.** MUVERA still retains the token store for exact reranking, and FDE graph overhead is separate. MUVERA, https://arxiv.org/pdf/2405.19504, §3 “MV Model, MV Embedding Sizes, and FDE Dimensionality” and §3.2. |

### 4. Recommended direct storage evidence: token pooling

| Claim | Numerator / denominator | Workload and configuration | Verdict and source location |
|---|---|---|---|
| 2-bit PLAID full index is 760 MB; hierarchical pooling factor 2 reduces it to 388 MB | 760 / 388 = 1.96× smaller; 372 MB saved | TREC-COVID, 256-token document cap; same ColBERTv2 indexing pipeline; complete on-disk index | **Recommended.** This is an index-level denominator, not token-payload arithmetic. Token Pooling, https://arxiv.org/pdf/2409.14683, §4.3 and Table 3. |
| A 16-bit dense HNSW index on the same TREC-COVID setup is 345 MB | Compare 388 MB pooled 2-bit PLAID with 345 MB 16-bit dense HNSW; 388 / 345 = 1.12× | Same TREC-COVID corpus and 256-token cap; the paper reports both complete index sizes | **Use as context, not as a quality claim.** The paper chooses 16-bit dense and 2-bit PLAID because both have virtually no quantization-related degradation, but it does not report a pooled-vs-dense quality comparison in Table 3. Token Pooling, https://arxiv.org/pdf/2409.14683, §4.3 and Table 3. |
| Pooling factor 2 incurs only 1.34% average effectiveness degradation after outlier treatment; factor 3 incurs 3.52% | Relative score is 100 for the same model with no pooling; these are average relative retrieval scores, not absolute nDCG/MRR | Six BEIR and three LoTTE collections in the 2-bit PLAID experiment; authors identify Touché as an outlier and FiQA as unusually sensitive | **Use with the outlier caveat.** Token Pooling, https://arxiv.org/pdf/2409.14683, §4.2 and Table 2. |
| TREC-COVID relative score at pooling factor 2 is 102.95 | 102.95 / 100 baseline; pooling happened to improve this dataset's score | 2-bit PLAID, TREC-COVID row of Table 2 | **Do not put 102.95 alone on the slide.** It can look like an absolute effectiveness metric and is not a cross-dataset guarantee. Token Pooling, https://arxiv.org/pdf/2409.14683, Table 2. |

The paper says pooling is an indexing-only step: no model retraining, model architecture change, or query-time processing. It is therefore a strong replacement for a raw storage ratio while preserving the distinction between token-set serving and MUVERA's fixed-dimensional proxy.

### 5. Later same-baseline serving evidence

#### EMVB (strong ColBERTv2/PLAID engine comparison)

*Source:* https://arxiv.org/pdf/2404.02805, §5, Tables 1–2.

EMVB and PLAID use the same ColBERTv2 model and are measured on the same single-threaded Intel Xeon Gold 5318Y CPU. On MS MARCO Table 1 at `k=1000`:

- PLAID: **260 ms/query**, **36 bytes per token embedding**, MRR@10 **39.8**, Recall@100 **91.3**, Recall@1000 **97.5**.
- EMVB (`m=16`): **93 ms/query** (2.8× faster), **20 bytes per token embedding** (1.8× smaller), MRR@10 **39.5**, Recall@100 **91.4**, Recall@1000 **97.5**.
- EMVB (`m=32`): **104 ms/query** (2.5× faster), **36 bytes per token embedding**, MRR@10 **39.9**, Recall@100 **91.4**, Recall@1000 **97.5**.

This is an excellent fair serving comparison, but the storage denominator is **bytes per embedding**, not complete index size. Use it if the slide wants a latency-quality tradeoff; do not present 20/36 bytes as whole-index storage.

On LoTTE Table 2 at `k=1000`, EMVB (`m=32`) is 142 ms versus PLAID 411 ms (2.9× faster), with Success@5 69.0 versus 69.6 and Success@100 90.1 versus 90.5. The paper reports that OPQ is used for this out-of-domain comparison because JMPQ requires training queries.

#### WARP (later multi-vector engine)

*Source:* Scheerer et al., *WARP: An Efficient Engine for Multi-Vector Retrieval*, https://arxiv.org/pdf/2501.17788, Table 2 §5.1 and Table 4 §5.3.

WARP and XTR/ScaNN use the same XTR-base model, same single-threaded CPU setup, and the same LoTTE/BEIR workloads. On LoTTE Pooled, WARP is **171.3 ms/query versus 2,156.3 ms/query** for XTR/ScaNN (12.6× lower), while Success@5 is **69.3 versus 68.4**. Table 4 reports the complete LoTTE Pooled index as **23.88 GiB for WARP b=2 versus 87.30 GiB for ScaNN** (3.7× smaller). Across the six LoTTE datasets, average Success@5 is 70.3 versus 69.8; across six BEIR datasets, average nDCG@10 is 44.8 versus 45.0, with WARP latency 70.3 ms average versus 345.8 ms for XTR/ScaNN (the per-dataset values are in Table 3).

This is a strong current serving point, but WARP is built for the XTR objective rather than ColBERTv2. Its comparison is fair **within XTR**, not a direct ColBERTv2-vs-single-vector claim. The paper also reports a 4.3× WARP-vs-PLAID latency comparison, but explicitly says the encoder and hardware differ; do not use that number as the main claim.

#### ConstBERT (learned fixed-size token-set compression)

*Source:* MacAvaney, Mallia, and Tonellotto, *Efficient Constant-Space Multi-Vector Retrieval*, https://arxiv.org/pdf/2504.01818, §3.2–§3.3, Tables 1–3.

On MS MARCO, Table 1 compares the same corpus and single-threaded dual Xeon setup, but ConstBERT is a retrained model rather than a drop-in ColBERT index. ColBERT is 22G / MRR@10 39.99; ConstBERT32 is 11G / MRR@10 39.04. On BEIR, Table 2 reports examples such as NFCorpus 23G ColBERT versus 5G ConstBERT32, and TREC-COVID 1.5G versus 0.5G. Table 3's two-stage ESPLADE + ConstBERT32 reranker reaches MRR 39.52 at 4.95 ms versus standalone ColBERT/PLAID MRR 39.99 at 51.25 ms.

Use this as later storage research, not as a headline ColBERTv2 serving baseline: the representation is retrained, and Table 3 changes the candidate-generation pipeline.

#### ColBERT-serve (RAM and concurrency, not index compression)

*Source:* Huang et al., *ColBERT-serve: Efficient Multi-Stage Memory-Mapped Scoring*, https://arxiv.org/pdf/2504.14903, Table 1, §4, and Table 2.

Memory-mapping ColBERTv2 reduces measured RSS from 23.4 GB to 2.3 GB on MS MARCO (90%) and from 98.3 GB to 8.2 GB on Wikipedia (92%). Table 1 reports MS MARCO control 128 GB RAM / $438 per month versus MMAP 16 GB / $95 (88% less RAM, 78% lower listed cost), and Wikipedia control 128 GB versus MMAP 32 GB (75% less RAM). Table 2 reports MS MARCO Dev ColBERTv2 MRR@10 39.51 versus Hybrid α=.3 MRR@10 40.22; Wikipedia NQ-dev Success@5 is 67.51 versus Hybrid α=.3 65.78 (optimal α 66.34); LoTTE Lifestyle is 74.6 versus 74.8 (optimal α 75.3).

This is useful evidence that compressed late-interaction serving can move the index out of RAM and support concurrency. It is not a storage-size comparison: the disk index remains, and the paper warns that full-ColBERT latency uses a different, higher-end machine and is not directly comparable with the MMAP machines.

#### DESSERT (later same-task latency comparison)

*Source:* Engels et al., *DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries*, https://arxiv.org/pdf/2210.15748, §6.2, Table 2.

On MS MARCO with four CPU cores/eight threads, DESSERT versus PLAID at `k=10` is 9.5/15.5 ms versus 45.1 ms across two DESSERT operating points, with MRR@10 35.7/37.2 versus 39.2. At `k=1000`, it is 22.7/32.3 ms versus 100 ms, with Recall@1000 95.1/96.0 versus 97.5. This is a fair same-workload latency-quality frontier, but it is not a storage comparison and it trades recall for speed.
#### LITE (reranking only)

*Source:* Ji et al., *Efficient Document Ranking with Learnable Late Interactions*, https://arxiv.org/pdf/2406.17968, §2 (scope), §4.4, and Table 3.

LITE is explicitly a reranker over 100 candidates, not a first-stage retrieval engine. On MS MARCO, full ColBERT stores 200 document-token embeddings and scores 1 query × 100 documents in 62 ms with MRR@10 0.383; separable LITE scores in 111 ms with MRR@10 0.393 at the same 200-token, 200×-DE storage. Small separable LITE stores 50 tokens (50×-DE storage), scores in 56 ms, and reaches MRR@10 0.391. This gives a clear storage/quality/latency denominator for reranking, but it is not evidence about corpus-wide PLAID, MUVERA, or ANN serving.

#### PLAID reproducibility study

*Source:* MacAvaney and Tonellotto, *A Reproducibility Study of PLAID*, https://arxiv.org/pdf/2404.14989, Tables 1–2 and §5.

The study reproduces PLAID's MS MARCO results and shows that the three PLAID parameters must be tuned together. It reports a single-threaded 80.5 ms/query at one operating point versus 185.5 ms/query at another, and a 7 ms/query lexical reranking point in the additional baseline study. It also reports a ColBERTv2 MS MARCO index around 22 GB versus less than 1 GB for PISA BM25, but this is a different lexical-vs-neural system comparison, not a fair single-vector late-interaction storage baseline. Use the parameter-dependence warning, not the 22 GB ratio.

#### PLAID SHIRTTT and ESPN

* PLAID SHIRTTT:* https://arxiv.org/pdf/2405.00975, §5. For 504M ClueWeb09 documents (31B passages), the streaming system reports 8.4 TB disk, 1.3 s/query, 108 CPUs with 200 GB RAM each; BM25 is 0.4 TB and 0.07 s/query on one CPU. This is useful scale and operational evidence, but the hardware, multilingual/streaming design, and BM25 baseline do not establish a fair quality-matched storage ratio for the slide.

* ESPN:* https://arxiv.org/pdf/2312.05417, Tables 1–5. ESPN offloads a ColBERTer reranking table to SSD; its MS MARCO v2 Table 5 is 271.5 ms mmap versus 85.8 ms ESPN with prefetching at 32 GB, and §5.3 reports 5–16× lower memory requirements. The paper uses ColBERTer, copied comparison rows, and an SSD/memory configuration rather than a same-index storage baseline. Keep it as systems context, not the slide metric.

---

## Claims to reject or label carefully

1. **“Late interaction costs 59× storage.”** Reject as a production claim. It is raw float16, exhaustive, local SciFact scoring with no ANN index or residual compression.
2. **“ColBERTv2 is 6–10× larger than a single-vector index.”** Reject. The 6–10× ratio in ColBERTv2 is compression from vanilla ColBERT to ColBERTv2; the paper's single-vector comparison is an arithmetic estimate that excludes ANN overhead while the late-interaction number includes an inverted list.
3. **“MUVERA is 32× smaller.”** Label as **FDE payload compression only**: 10,240 float dimensions to 1,280 PQ bytes. Full retrieval still needs a MIPS graph and the original token store for exact MaxSim reranking.
4. **“PLAID is 45× faster than single-vector retrieval.”** Reject. PLAID's 45× CPU figure is versus vanilla ColBERTv2, not a single-vector model, and its latency measurement excludes query encoding.
5. **“WARP is 3× faster than PLAID.”** Do not use as the primary claim. The WARP paper says this comparison uses different encoders and hardware. Use the same-XTR WARP/ScaNN comparison if WARP is shown.
6. **Cross-paper index-size ratios.** Do not divide ColBERTv2's 25 GiB, PLAID's 21.6 GiB, ConstBERT's 11G, or WARP's 23.88 GiB. Their token caps, model dimensions, quantizers, index formats, and corpus configurations differ.

## Minimal speaker-note caveat

> “The old 59× was raw, exhaustive float16 storage. A complete 2-bit PLAID index with 2× token pooling is 388 MB versus 760 MB without pooling on the same TREC-COVID setup, with the paper reporting only 1.34% average loss after its stated outlier treatment. That still preserves a token set. MUVERA goes one step further: it builds a fixed-dimensional proxy for the shortlist, then restores exact MaxSim only for candidates.”

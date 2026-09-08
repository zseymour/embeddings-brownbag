# References

Every arXiv entry below was resolved against arXiv (API or abstract page) and the title confirmed to match the claim it supports. Dates are first-submission dates. Grouped by the act that uses them.

A **do-not-use** list is at the bottom: claims that failed verification and must not appear on a slide.

---

## Framing — the bitter lesson and its qualifications

| Source | What it establishes |
|---|---|
| Sutton, "The Bitter Lesson" (13 Mar 2019) — http://www.incompleteideas.net/IncIdeas/BitterLesson.html | The actual claim: general methods leveraging computation win, via **search** and **learning**. Quoted verbatim on slides: the headline sentence; the four-part lesson ("1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term ... 3) in the long run it plateaus ... 4) breakthrough progress eventually arrives by an opposing approach"); "we should build in only the meta-methods that can find and capture this arbitrary complexity" (thesis slide); "We want AI agents that can discover like we can, not which contain what we have discovered" (coda). On convolution he only says modern networks "use only the notions of convolution and certain kinds of invariances": a description, not an endorsement; do not say he "keeps" or "credits" it. Timescale line for the coda note: "over a slightly longer time than a typical research project, massively more computation inevitably becomes available." |
| arXiv:2207.10551 (2022-07-21) *Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?* | Scaling *coefficients* vary by architecture and inductive bias — compute scaling is not architecture-agnostic. |
| arXiv:2302.10692 (2023-02-21) *On Inductive Biases for Machine Learning in Data Constrained Settings* | Where data is finite, reintroducing known mathematical structure improves sample efficiency. |
| arXiv:2305.16264 (2023-05-25) *Scaling Data-Constrained Language Models* | The data-constrained regime the bitter lesson does not address. |
| arXiv:2604.00025 (2026-03-11) *Brevity Constraints Reverse Performance Hierarchies in Language Models* | Inverse-scaling results flip once prompting protocol is controlled — scaling claims are protocol-dependent. |
| arXiv:2606.01090 (2026-05-31) *Measuring the Symmetry–Data Exchange Rate* | **A wrong-group constraint is measurably worse than no constraint**, CI [+0.79, +3.26]. Use this finding; the headline exchange-rate number is self-labelled exploratory. |

## Absorbed by scale — the general embedding stack

| Source | What it establishes |
|---|---|
| arXiv:2506.05176 (2025-06-05) *Qwen3 Embedding* | General large-scale contrastive pretraining beats per-task metric-learning pipelines; #1 MTEB multilingual at publication. |
| arXiv:2508.10104 (2025-08-13) *DINOv3* | 7B self-supervised vision backbone; needed both scale **and** a targeted fix (Gram anchoring) for dense-feature degradation. |
| arXiv:2304.07193 (2023-04-14) *DINOv2: Learning Robust Visual Features without Supervision* | **49.0 mIoU on ADE20K with frozen features and a linear head** (Table 10). Frozen features plus a linear probe — not zero-shot. |
| arXiv:2505.19274 (2025-05-25) *Conventional Contrastive Learning Often Falls Short* | Cross-encoder listwise distillation as the newer lever past hand-tuned triplet mining. |
| arXiv:2004.12832 (2020-04-27) *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* | MaxSim: "every query embedding interacts with all document embeddings via a MaxSim operator ... summed across query terms." Uses FAISS only for single-vector first-stage filtering; MaxSim scored outside the index. |
| arXiv:2112.01488 (2021-12-02) *ColBERTv2* | Residual compression: centroid id + 1–2 bit residual, **20 or 36 bytes per token vector** vs 256 uncompressed; MS MARCO index **154 GiB → 16–25 GiB**. Table 4, MS MARCO dev MRR@10: **ColBERTv2 39.7** vs RocketQAv2 38.8 (best single-vector in the table), ColBERT 36.0. |
| arXiv:2407.01449 (2024-06-27) *ColPali* | ViDoRe Table 2, average nDCG@5: **ColPali 81.3** vs best text pipeline (Unstructured + captioning + BGE-M3) **67.0**. **1,024 patches per page × 128 dims; 257.5 KB per page.** Not "1030 patches". Token pooling ×3 keeps 97.8% of performance at 33% of the vectors. CC0. |
| arXiv:2405.19504 (2024-05-29) *MUVERA* | Fixed Dimensional Encodings: "vectors whose inner product approximates multi-vector similarity"; retrieval with off-the-shelf MIPS (DiskANN), then exact Chamfer/MaxSim rerank. vs PLAID across BEIR: **average 10% higher recall, 90% lower latency**. |
| arXiv:2603.25248 (2026-03-26) *ColBERT-Att* | Late interaction fused with learned attention weighting. |
| arXiv:2511.00444 (2025-11-01) *LIR: First Workshop on Late Interaction and Multi-Vector Retrieval @ ECIR 2026* | Late interaction is an active paradigm, not a legacy one. |
| Own experiment, `talk/experiments/e11_late_interaction.py` | BEIR SciFact, 300 queries: bge-small-en-v1.5 nDCG@10 **0.713**, answerai-colbert-small-v1 **0.746**; Recall@100 0.942 vs 0.956; MRR@10 0.682 vs 0.719. Raw float16 storage 768 B vs 45,245 B per document (58.9×); exhaustive CPU scoring 0.12 ms vs 58.9 ms. **Brute force on both sides; do not quote these ratios as the cost of late interaction.** Both nDCG values match the models' published SciFact numbers. |
| Own experiment, `talk/experiments/e12_fair_serving.py` | Production-style serving, same corpus. Single vector, HNSW f16 (M=16, ef=128): nDCG@10 0.713, **917 B/doc, 0.23 ms p50**; int8 HNSW 533 B/doc at −0.001 nDCG. Late interaction within 0.01 nDCG of exact MaxSim: ColBERTv2-style 2-bit residual compression, k=4096: nDCG@10 0.739, **6,279 B/doc (6.8×)**, still exhaustive scoring 57 ms. PLAID (pylate defaults) 0.737, 28 KB/doc, 246 ms at this corpus size (candidate generation dominates; flips at scale). MUVERA 5,120-d + MaxSim rerank 0.708, 4.7 ms; needs the token store for rerank. MUVERA without rerank 0.46/0.54. SimHash required mean-centering on this encoder's anisotropic tokens; documented in the script. |

## Act I — Hierarchy and non-Euclidean geometry

| Source | What it establishes |
|---|---|
| arXiv:1705.08039 (2017-05-22) *Poincaré Embeddings for Learning Hierarchical Representations* | The origin of the hierarchy-in-hyperbolic-space line. |
| arXiv:2601.23064 (2026-01-30) *HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation* | 240k hyperbolic entity embeddings replace 5M+ image embeddings. **+8.8% country, +20.1% region, +43.2% subregion, +16.8% city; −19.5% mean geodesic error.** Largest gain at subregion, not monotonic in depth (city +16.8%); ranking holds across three backbones. Inference: Lorentz embeddings searched with FAISS IndexFlatIP via the sign-flip trick (Appendix A.8). CC BY 4.0 — figures reusable with attribution. |
| arXiv:2307.05845 (2023-07-11) *PIGEON: Predicting Image Geolocations* | Component-isolated ablations: haversine-smoothed loss **990.0 → 877.4 km mean error (−11.4%)**; administrative-hierarchy geocells **60.6 → 55.5 km median error (−8.4%)**. |
| arXiv:2304.09172 (2023-04-18) *Hyperbolic Image-Text Representations* (MERU) | The hyperbolic vision-language line the audit examines. |
| arXiv:2607.05268 (2026-07-06) *Is the Geometry Doing the Work? An Operating-Point Audit of Hierarchy in Hyperbolic Vision-Language Models* | Audits seven released checkpoints; **all converged ones remain near-Euclidean** on the dimensionless-radius measure. The talk's central honesty citation. |
| arXiv:2412.01023 (2024-12-02) *Learning Structured Representations with Hyperbolic Embeddings* (HypStructure) | A hyperbolic **regularizer** on Euclidean embeddings — geometry moved into the loss. |
| arXiv:2604.09550 (2026-01-26) *HyEm: Query-Adaptive Hyperbolic Retrieval for Biomedical Ontologies via Euclidean Vector Indexing* | Euclidean log-mapped vectors in a standard vector DB, hyperbolic distance only for reranking. Thesis in one system. |
| arXiv:2211.00181 (2022-10-31) *The Numerical Stability of Hyperbolic Representation Learning* | Poincaré struggles with small values, Lorentz with large; float32 cannot add values separated by >16 orders of magnitude. |
| arXiv:2405.13979 (2024-05-22) *Robust Hyperbolic Learning with Curvature-Aware Optimization* | Naive Riemannian curvature optimization is unstable; hyperbolic learning is expensive and fragile. |
| arXiv:2407.16641 (2024-07-23) *A Geometry-Aware Algorithm to Learn Hierarchical Embeddings in Hyperbolic Space* | Standard hyperbolic training stalls at local optima on very bushy real catalogs. |
| arXiv:2505.18973 (2025-05-25) *Hierarchical Mamba Meets Hyperbolic Geometry* | Hyperbolic wins on deep hierarchies; the advantage shrinks as data departs from clean trees. |
| arXiv:2509.05757 (2025-09-06) *Hyperbolic Large Language Models* | Survey. Contains the precision-depth result: embedding a chain of length ℓ within distortion ε needs Θ(ℓ/ε) mantissa bits. |
| arXiv:2505.12369 (2025-05-18) *Fully Geometric Multi-Hop Reasoning on Knowledge Graphs with Transitive Relations* | Geometric region reasoning remains active for KGs. |

## Act I — Geolocalization, and the scale rebuttal

| Source | What it establishes |
|---|---|
| arXiv:1602.05314 (2016-02-17) *PlaNet — Photo Geolocation with Convolutional Neural Networks* | S2 cells chosen over lat/lon because lat/lon cells distort near the poles. Design rationale, not an ablation. |
| arXiv:2309.16020 (2023-09-27) *GeoCLIP* | CLIP-aligned hierarchical location encoder. IM2GPS3k: 14.11/34.47/50.65/69.67/83.82% at 1/25/200/750/2500 km. |
| arXiv:2302.00275 (2023-02-01) *Learning Generalized Zero-Shot Learners for Open-Domain Image Geolocalization* (StreetCLIP) | Zero-shot foundation model beats supervised models trained on 4M+ images. |
| arXiv:2303.04249 (2023-03-07) *Where We Are and What We're Looking At* (GeoDecoder) | Learned hierarchy queries, one set per geographic level. |
| arXiv:2404.18873 (2024-04-29) *OpenStreetView-5M* | 5.1M streetview images, 225 countries; the benchmark HierLoc reports against. |
| arXiv:2306.17624 (2023-06-30) *Sphere2Vec* | Spherical loss beats Euclidean location encoders — but on geo-aware classification, **not** photo-geolocation. State the caveat. |
| arXiv:2212.12794 (2022-12-24) *GraphCast* | Icosahedral mesh over a lat-lon grid; beats operational NWP on 90% of 1380 targets. No geolocalization system has adopted an icosahedral mesh — the transfer is inference, not evidence. |
| arXiv:2601.21278 (2026-01-29) *GeoRC: A Benchmark for Geolocation Reasoning Chains* | 800 chains built with the reigning GeoGuessr world champion. Table 2, country-level accuracy: best prompted VLM **91.2%** (Gemini-2.5-Pro; GPT-5 88.4%, Gemini-3-Pro 90.4%) vs three individual human experts **90.0 / 96.7 / 97.3%**. There is no "expert team = 96%" aggregate; do not use that framing. Same models are poor at producing the reasoning chain. |
| arXiv:2502.11163 (2025-02-16) *AI Sees Your Location, But With A Bias Toward The Wealthy World* | Up to 53.8% city-level accuracy with no fine-tuning; **−12.5%** on less-developed and **−17.0%** on sparsely-populated regions. Scale takes the median case and leaves the tails. |

## Act II — Uncertainty and calibration

| Source | What it establishes |
|---|---|
| arXiv:1810.00319 (2018-09-30) *Modeling Uncertainty with Hedged Instance Embedding* | The principled form of an ad-hoc uncertainty rule. |
| arXiv:2505.05163 (2025-05-08) *Probabilistic Embeddings for Frozen Vision-Language Models* (GroVE) | Post-hoc uncertainty-calibrated embeddings from frozen VLMs. |
| arXiv:2507.20718 (2025-07-28) *Uncertainty-driven Embedding Convolution* | Current distributional-embedding work. |
| arXiv:2512.22318 (2025-12-26) *Decomposing Uncertainty in Probabilistic Knowledge Graph Embeddings: Why Entity Variance Is Not Enough* | **0.99 AUROC on random corruptions, 0.52-0.64 under temporal distribution shift.** The act's grade rests on this. |
| arXiv:2112.05872 (2021-12-11) *SLOSH: Set LOcality Sensitive Hashing via Sliced-Wasserstein Embeddings* | Distributions re-embedded into Euclidean space so ANN works at all; Wasserstein is cubic in distribution size. |
| arXiv:2404.04287 (2024-04-04) *CONFLARE: CONFormal LArge language model REtrieval* | Calibration set plus similarity cutoff gives distribution-free coverage on retrieval. |
| arXiv:2511.17908 (2025-11-22) *Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction* | Conformal filtering **cuts retained context two to three times** at target coverage. |
| arXiv:2410.19349 (2024-10-25) *pEBR: A Probabilistic Approach to Embedding Based Retrieval* | Retrieval score as a probability, so it can be thresholded. |
| arXiv:2502.10875 (2025-02-15) *A Geometric Approach to Personalized Recommendation with Set-Theoretic Constraints Using Box Embeddings* | Region embeddings as the practical middle path. |
| Radovanović, Nanopoulos, Ivanović (JMLR 2010) *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data* | The original hubness result: skew of the k-occurrence distribution grows with intrinsic dimension; points near the data mean become hubs. |
| arXiv:2112.12777 (2021-12-23) *Cross Modal Retrieval with Querybank Normalisation* (QB-Norm) | Hubness in contrastive cross-modal spaces; inference-time rescoring against a query bank, index untouched. |
| arXiv:2310.11612 (2023-10-17) *Balance Act: Mitigating Hubness in Cross-Modal Retrieval with Query and Gallery Banks* (DBNorm) | Same fix, both banks. |
| arXiv:2410.24114 (2024-10-31) *Nearest Neighbor Normalization Improves Multimodal Retrieval* (NNN) | Calls hubness "a failure mode of contrastive text-to-image retrieval"; additive per-candidate bias correction. |
| arXiv:2503.10526 (2025-03-13) *NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval* | The training-time version: penalise hub centrality in the loss. |
| arXiv:2508.02538 (2025-08-04) *Hubness Reduction with Dual Bank Sinkhorn Normalization for Cross-Modal Retrieval* | Hubness still "persistent" in 2025 contrastive models; Sinkhorn balancing at query time. |

## Pivot — the SAANE trial and modern visual place recognition

| Source | What it establishes |
|---|---|
| arXiv:1812.03402 (2018-12-08) *Semantically-Aware Attentive Neural Embeddings for Image-based Visual Localization* (SAANE, BMVC 2019) | The defendant. Reports AUC, not Recall@1 — not interconvertible with modern numbers. |
| arXiv:2308.00688 (2023-08-01) *AnyLoc: Towards Universal Visual Place Recognition* | Off-the-shelf DINOv2 plus VLAD, **no VPR training**; +6% credited to semantic properties that emerged **without supervision**. The prosecution's key witness. |
| arXiv:2311.15937 (2023-11-27) *Optimal Transport Aggregation for Visual Place Recognition* (SALAD) | DINOv2 backbone, Sinkhorn assignment. Nordland R@1 76.0%, MSLS-val 91.9%. |
| arXiv:2502.16601 (2025-02-23) *SelaVPR++* | Nordland R@1 97.2%; binary hashes for retrieval, float features for rerank. |
| arXiv:2407.02422 (2024-07-02) *Close, But Not There: Boosting Geographic Distance Sensitivity in Visual Place Recognition* (CliqueMining) | **Nordland R@1 76% → 90% with zero architectural change** — pure hard-negative graph sampling. The talk's strongest single finding. |
| arXiv:2402.19231 (2024-02-29) *CricaVPR* | Cross-image correlation attention — a living descendant of the 2018 attention idea. Pitts30k R@1 94.9%, Tokyo24/7 93.0%. |
| arXiv:2405.07364 (2024-05-12) *BoQ: A Place is Worth a Bag of Learnable Queries* | Learnable-query cross-attention; claims one-stage beats two-stage methods "orders of magnitude faster." |
| arXiv:2204.02287 (2022-04-05) *Rethinking Visual Geo-localization for Large-Scale Applications* (CosPlace) | 512-d descriptors, SF-XL scale. Pitts30k R@1 88.4%, Tokyo24/7 81.9%. |
| arXiv:2308.10832 (2023-08-21) *EigenPlaces* | Pitts30k R@1 92.5%, Tokyo24/7 92.4%, half the descriptor size of CosPlace. |
| arXiv:2303.02190 (2023-03-03) *MixVPR* | Pitts250k R@1 94.6%, MSLS 88.0%, **Nordland only 58.4%** — seasonal change stayed hard. |
| arXiv:2402.14505 (2024-02-22) *Towards Seamless Adaptation of Pre-trained Models for Visual Place Recognition* (SelaVPR) | Rerank at ~3% of two-stage RANSAC cost. |
| arXiv:2502.17237 (2025-02-24) *MegaLoc: One Retrieval to Place Them All* | Single-stage, drop-in; Tokyo24/7 R@1 96.5%. |

## Act III — Rotation, phase, and complex arithmetic

| Source | What it establishes |
|---|---|
| arXiv:2104.09864 (2021-04-20) *RoFormer: Enhanced Transformer with Rotary Position Embedding* | Relative position as rotation in the complex plane. |
| arXiv:2409.12191 (2024-09-18) *Qwen2-VL* | M-RoPE: the same prior transferred to temporal/height/width axes without re-derivation. |
| arXiv:2502.05173 (2025-02-07) *VideoRoPE* | 3D (t,h,w) rotary structure with low-frequency temporal allocation. |
| arXiv:2309.00071 (2023-08-31) *YaRN: Efficient Context Window Extension of Large Language Models* | Frequency reinterpretation, not retraining, extends context; Fig. 3 shows perplexity flat to 128k from a 4k-trained LLaMA (≈32×). Do not claim 1000×. |
| arXiv:2502.20082 (2025-02-27) *LongRoPE2* | Near-lossless context scaling to 128k on Phi3-mini and LLaMA3-8B. |
| arXiv:2502.11276 (2025-02-16) *The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval* | Even a hard constraint leaves capacity on the table. |
| arXiv:2512.07525 (2025-12-08) *Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs* | RoPE's real rotation-matrix form discards phase relative to full complex multiplication. |
| arXiv:2511.17388 (2025-11-21) *Selective Rotary Position Embedding* | Learned, input-dependent rotation angles. |
| arXiv:2410.09749 (2024-10-13) *EMWaveNet* | States the mechanism: assuming a complex-valued CNN is equivalent to a two-channel real one is erroneous — stacking discards the correlation the complex product enforces. |
| arXiv:1906.03605 (2019-06-09) *Semi-supervised Complex-valued GAN for Polarimetric SAR Image Classification* | Complex-valued treatment in SAR. |
| arXiv:2012.08931 (2020-12-15) *Deep learning for fast MR imaging: a review for learning reconstruction from incomplete k-space data* | k-space is inherently complex; phase is a physical quantity used downstream. |
| arXiv:2309.01535 (2023-09-04) *Single-Channel Speech Enhancement with Deep Complex U-Networks and Probabilistic Latent Space Models* | Complex U-Net gains up to **4.5 dB SI-SDR** over the real-valued magnitude DVU-Net: Table 2, VoiceBank+Demand, 20.21 vs 15.62 dB. On MS-DNS (Table 1) the gain is ≈3 dB anechoic and **negative (−0.7 to −1.9 dB) under reverberation**. Dataset-specific; say so. |
| arXiv:2207.08412 (2022-07-18) *Multi-branch Cascaded Swin Transformers with Attention to k-space Sampling Pattern* (McSTRA) | The counter-ablation: single-channel **magnitude** mapping beat dual-channel complex when phase was not the target. |
| arXiv:2405.09689 (2024-05-15) *Generalized Holographic Reduced Representations* | The vector-symbolic line, still primarily theoretical. |
| arXiv:2509.24425 (2025-09-29) *BiHDTrans* | The one credible recent VSA result — beats only other **binary** transformers, on multivariate time series. |

## Act IV — Cost, index geometry, and quantization

| Source | What it establishes |
|---|---|
| FAISS wiki, *Notes on MetricType and distances* — https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances | Verbatim: "There are two primary methods supported by Faiss indices, L2 and inner product." Cosine is normalize-then-inner-product. **Mahalanobis recipe: whiten by the inverse Cholesky factor of the covariance, then index in `METRIC_L2`.** Additional metrics (L1, L∞, Lp, Canberra, Bray-Curtis, Jensen-Shannon, Hamming) only on `IndexFlat`, `IndexHNSW`, `GpuIndexFlat`. No hyperbolic, no Wasserstein. |
| DiskANN docs — https://microsoft.github.io/DiskANN/ | Exactly three metrics: `l2`, `mips`, `cosine`. |
| Milvus docs — https://milvus.io/docs/v2.5.x/metric.md, https://milvus.io/docs/scann.md | L2, IP, COSINE, plus Jaccard/Hamming for binary and BM25 for sparse. |
| Qdrant 1.10 release notes — https://qdrant.tech/blog/qdrant-1.10.x/ ; search docs — https://qdrant.tech/documentation/search/search/ | Metrics: Dot, Cosine, Euclid, Manhattan (no Hamming). Native `multivector_config` with `max_sim` comparator since 1.10 (July 2024). |
| Elasticsearch 8.18 reference — https://www.elastic.co/guide/en/elasticsearch/reference/8.18/rank-vectors.html ; dense-vector.html | `rank_vectors` field with `maxSimDotProduct`, technical preview, **for second-order reranking, not primary ANN retrieval**. `dense_vector` supports `element_type: bit` with hamming. |
| Pinecone docs — https://docs.pinecone.io/guides/index-data/create-an-index ; sparse-dense guide | Metrics: cosine, dotproduct, euclidean. Sparse+dense hybrid only; no multi-vector / late-interaction primitive. |
| Milvus 2.6.4 — https://milvus.io/docs/search-with-embedding-lists.md ; blog "Array of Structs and MAX_SIM" | True ColBERT-style multi-vector (`MAX_SIM` over embedding lists) since **2.6.4**. The older "multi-vector hybrid search" (2.4) is multiple separate vector fields fused by reranking, a different feature. |
| Vespa — https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/ ; multi-vector indexing blog (Mar 2023) | Native: `tensor<float>(t{}, x[128])` fields, ColBERT embedder shipped Feb 2024. |
| Weaviate 1.31 — https://weaviate.io/blog/muvera | "Weaviate 1.31 implements the MUVERA encoding algorithm for multi-vector embeddings." Multi-vector support pre-dates it. |
| arXiv:2103.01486 (2021-03-02) *Patch-NetVLAD* | The two-stage pattern: top-100 retrieval then RANSAC verification. |
| arXiv:2211.14864 (2022-11-27) *A Faster, Lighter and Stronger Deep Learning-Based Approach for Place Recognition* | Source of the **~14,557 ms/query** RANSAC rerank measurement attributed to Patch-NetVLAD. |
| arXiv:2304.03410 (2023-04-06) *R²Former: Unified Retrieval and Reranking Transformer for Place Recognition* | Learned reranker replaces RANSAC. |
| arXiv:2503.02511 (2025-03-04) *TeTRA-VPR: A Ternary Transformer Approach for Compact Visual Place Recognition* | 2-bit ternary backbone, binarized embedding layer: **up to 69% less memory, 35% lower inference latency, no loss or slight improvement in Recall@1.** |
| arXiv:2502.06786 (2025-02-10) *Matryoshka Quantization* | Nested quantization beats standard int2 post-training quantization. |
| arXiv:2605.16608 (2026-05-15) *To MRL or not to MRL* | Plain truncation ≈ Matryoshka training **until roughly 70% compression**. Do not adopt the objective reflexively. |

## Coda — the harness around an LLM (vendor primary sources, read September 2026)

Used only on the two coda slides. Each row: the hand-built technique, what absorbed it, and the vendor's own sentence.

| Source | What it establishes |
|---|---|
| OpenAI, 2024-09 — https://community.openai.com/t/new-reasoning-models-openai-o1-preview-and-o1-mini/938081 | Prompted chain-of-thought ("let's think step by step", few-shot CoT) → OpenAI o1 — reasoning trained into the model, spends time "thinking" before responding. "We've developed these models to spend more time thinking before they respond. They can reason through complex tasks and solve harder problems than previous models in science, codin…" |
| Anthropic, 2025-02 — https://www.anthropic.com/news/visible-extended-thinking | Prompted chain-of-thought / manual 'think step by step' prompting → Claude 3.7 Sonnet extended thinking — toggleable built-in reasoning with a developer-controlled 'thinking budg. "With the new Claude 3.7 Sonnet, users can toggle 'extended thinking mode' on or off, directing the model to think more deeply about trickier questions. And developers can even set …" |
| OpenAI, 2023-06 — https://openai.com/index/function-calling-and-other-api-updates/ | Hand-written output parsers / regex over model text / 'respond in JSON → OpenAI native function calling — model trained to emit structured JSON arguments for developer-described funct. "Developers can now describe functions to gpt-4-0613 and gpt-3.5-turbo-0613, and have the model intelligently choose to output a JSON object containing arguments to call those funct…" |
| OpenAI, 2024-08 — https://openai.com/index/introducing-structured-outputs-in-the-api/ | 'Respond in JSON' prompting / retrying malformed model output against  → OpenAI Structured Outputs — schema-constrained decoding guarantees valid JSON matching a developer schema. "Developers have long been working around the limitations of LLMs in this area via open source tooling, prompting, and retrying requests repeatedly to ensure that model outputs matc…" |
| Anthropic, 2024-05 — https://www.anthropic.com/news/tool-use-ga | Hand-coded tool-invocation parsing / prompt-engineered tool routing → Claude tool use, general availability across the Claude 3 model family. "Tool use, which enables Claude to interact with external tools and APIs, is now generally available across the entire Claude 3 model family on the Anthropic Messages API, Amazon Be…" |
| Google, 2024-02 — https://cloud.google.com/blog/products/ai-machine-learning/gemini-on-vertex-ai-expands | Hand-coded tool-invocation parsing for Gemini → Gemini function calling, general availability on Vertex AI. "With function calling, now generally available, developers can connect the Gemini model to external APIs for transactions and other actions." |
| Anthropic, 2024-10 — https://www.anthropic.com/news/3-5-models-and-computer-use | Hand-coded ReAct / plan-and-execute agent loops driving screen/mouse a → Anthropic 'computer use' — Claude trained to natively issue screen, cursor, and keyboard actions. "We're also introducing a groundbreaking new capability in public beta: computer use. Available today on the API, developers can direct Claude to use computers the way people do—by …" |
| OpenAI, 2025-01 — https://openai.com/index/introducing-operator/ | Hand-coded browser-automation agent loops (Selenium/Playwright scripti → OpenAI Operator — a model (CUA) trained via reinforcement learning to interact with GUIs directly. "Operator is powered by a new model called Computer-Using Agent (CUA). Combining GPT‑4o's vision capabilities with advanced reasoning through reinforcement learning, CUA is trained …" |
| OpenAI, 2025-03 — https://openai.com/index/new-tools-for-building-agents/ | Hand-coded plan-and-execute / multi-step tool-orchestration scaffoldin → OpenAI Responses API — built-in web search, file search, and computer-use tools with native multi-step tool or. "Over the past year, we've introduced new model capabilities—such as advanced reasoning, multimodal interactions, and new safety techniques—that have laid the foundation for our mod…" |
| OpenAI, 2024-12 — https://developers.openai.com/api/docs/guides/reasoning | Self-consistency / majority voting / best-of-n sampling scripted by th → OpenAI reasoning-effort parameter — a single trained-in knob that trades latency for more internal reasoning. "The reasoning.effort parameter guides the model on how much to think when performing a task. ... Lower effort favors speed and lower token usage, while at higher effort the model t…" |
| Google, 2025-05 — https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking | Parallel sampling / majority-vote scaffolding to squeeze extra accurac → Gemini 'thinking budget' parameter — developer-set token budget for the model's own internal reasoning before . "By default, if thinking_budget is not set, the model automatically controls how much it thinks up to a maximum of 8,192 tokens. ... To use dynamic budget through the API, set think…" |
| Google, 2024-02 — https://developers.googleblog.com/gemini-15-our-next-generation-model-now-available-for-private-preview-in-google-ai-studio/ | RAG chunking + retrieval as the only way to fit large documents into c → Gemini 1.5 with up to 1-million-token context window. "We've been able to significantly increase this — running up to 1 million tokens consistently, achieving the longest context window of any large-scale foundation model. ... Before t…" |
| OpenAI, 2024-04 — https://community.openai.com/t/new-features-in-the-assistants-api/720539 | Hand-built retrieval pipeline (chunk, embed, index, rerank) glued in f → OpenAI file_search built-in tool (Assistants API v2, later Responses API) — managed vector-store retrieval. "We are announcing a variety of new features and improvements to the Assistants API and moving our Beta to a new API version... We're launching an improved retrieval tool called fil…" |
| Google, 2025-11 — https://blog.google/innovation-and-ai/technology/developers-tools/file-search-gemini-api/ | Hand-built retrieval pipeline glued in front of Gemini → Gemini File Search tool — fully managed RAG (chunking, embedding, vector storage) built into the Gemini API. "To make File Search simple and affordable for all developers, we're making storage and embedding generation at query time free of charge. You only pay [when creating embeddings whe…" |
| Anthropic, 2025-02 — https://www.anthropic.com/research/constitutional-classifiers | Hand-built guardrail classifiers layered around the model → Anthropic Constitutional Classifiers — input/output classifiers trained on a constitution and shipped as a mod. "In our new paper, we describe a system based on Constitutional Classifiers that guards models against jailbreaks. These Constitutional Classifiers are input and output classifiers …" |
| OpenAI, 2024-02 — https://openai.com/index/memory-and-new-controls-for-chatgpt/ | Hand-built long-term memory scaffold over a vector store → ChatGPT memory — vendor-managed persistent memory across chats. "We're testing memory with ChatGPT. Remembering things you discuss across all chats saves you from having to repeat information and makes future conversations more helpful. You're i…" |

**Not verified from a primary source; do not put a date on a slide:** Gemini thinking-model and Deep Think announcement dates; any OpenAI 2025–26 built-in safety classifier beyond the Moderation API; Anthropic Files API / citations GA date (sources conflict, Jan vs Jun 2025); any developer-API memory product; Claude Code as a vendor statement that multi-step tool use is trained in.

## Structure versus scale — the decision rule's evidence

| Source | What it establishes |
|---|---|
| arXiv:2510.09768 (2025-10-10) *Scaling Laws and Symmetry, Evidence from Neural Force Fields* | Equivariant force fields hold a **better scaling exponent at every tested compute budget** — the gap widens. Strongest pro-structure result. |
| arXiv:2509.21811 (2025-09-26) *Scaling Laws for Neural Material Models* | Equivariant models keep lower loss across all parameter and data ranges tested. |
| arXiv:2410.23179 (2024-10-30) *Does equivariance matter at scale?* | The balanced answer: equivariance improves data efficiency, but augmentation closes the gap given enough epochs. |
| arXiv:2510.02259 (2025-10-02) *Transformers Discover Molecular Structure Without Graph Priors* | Plain transformers on raw Cartesian coordinates match a SOTA equivariant GNN at matched compute and learn near-equivariance emergently (>0.99 cosine across rotated frames). Strongest pro-scale result. |
| arXiv:2504.08441 (2025-04-11) *SARFormer* | Acquisition-geometry encoding gives up to 17% RMSE improvement, concentrated in the limited-label regime. |
| arXiv:2509.21722 (2025-09-26) *On the Status of Foundation Models for SAR Imagery* | Generic vision foundation models fail off-the-shelf on SAR; **domain-matched** self-supervised finetuning then beats the hand-engineered incumbent. Raw scale fails, domain-matched scale wins. |
| arXiv:2405.09365 (2024-05-15) *SARATR-X* | The hand-engineered-prior incumbent in SAR target recognition. |
| arXiv:2406.18295 (2024-06-26) *Evaluating and Benchmarking Foundation Models for Earth Observation and Geospatial AI* | Foundation models beat problem-specific models by 16-86% at 100 labeled samples per region — scale winning the low-label regime. |
| arXiv:2606.19784 (2026-06-18) *EquiVLA* | SE(3)-equivariant action heads still win sample efficiency against large VLA baselines. |

## Geometry of representation — background, not load-bearing

arXiv:2311.03658 (2023-11-07) *The Linear Representation Hypothesis and the Geometry of Large Language Models* · arXiv:2405.07987 (2024-05-13) *The Platonic Representation Hypothesis* · arXiv:2507.01098 (2025-07-01) *Proof of a perfect platonic representation hypothesis* · arXiv:2505.12540 (2025-05-18) *Harnessing the Universal Geometry of Embeddings* (vec2vec) · arXiv:2510.02348 (2025-09-27) *mini-vec2vec* · arXiv:2608.05980 (2026-08-06) *How Far Do Simple Transformations Translate Across Text Embedding Models?* — translatability depends on architecture, objective, pooling, and data; not a universal law.

## Zach's own work

| Work | Link |
|---|---|
| *Leveraging Weakly-Aligned, User-Generated Data for Deep Learning Features* (Binghamton, 2019) | `.work/` in this repo |
| *Semantically-Aware Attentive Neural Embeddings for Image-based Visual Localization* (BMVC 2019) | arXiv:1812.03402 |
| *Multi-label Triplet Embeddings for Image Annotation from User-Generated Tags* (ACM ICMR 2018) | doi:10.1145/3206025.3206061 |
| *Image Annotation Retrieval with Text-Domain Label Denoising* (ACM ICMR 2018) | doi:10.1145/3206025.3206063 |
| *Multimodal Skip-gram Using Convolutional Pseudowords* (2015) | arXiv:1511.04024 |
| *MaAST: Map Attention with Semantic Transformers* (ICRA 2021) | arXiv:2103.11374 |
| *SASRA* (ICPR 2022) | arXiv:2108.11945 |
| *GraphMapper* (ICPR 2022) | arXiv:2205.08325 |
| *Recall Loss / Striking the Right Balance* (ICRA 2022) | search "Recall loss for imbalanced image classification and semantic segmentation" |

Dissertation source pointers used in the talk: `.work/chapters/chapter1/new_model.tex:5` (the commented-out RGB analogy), `:11` (5,000-character truncation "due solely to memory constraints"), `.work/chapters/chapter1/textcnn.tex:274` ("there is no free lunch"), `.work/chapters/chapter2/triplet/triplet.tex:190-265` (subset positives, hand-set margin, auxiliary loss), `.work/chapters/chapter2/kcca/kcca.tex:283-306` (5%-of-vocabulary clustering, hand-derived distance-to-centroid weight), `.work/chapters/chapter3/saane/approach.tex:30-58` (the CBAM-derived attention module).

---

## Do not use — failed verification

| Claim | Problem |
|---|---|
| Any Matryoshka p50 latency figure (e.g. "32 ms → 18 ms") | Attributed to arXiv:2604.19771, which is *Cognis: Context-Aware Memory for Conversational AI Agents* — an unrelated paper. Number has no source. |
| arXiv:2604.22390 "Region Matters" | Title and date could not be confirmed. |
| StructVPR latency figures (NetVLAD 40 ms, SFRS 207 ms) | Secondary-source table excerpt only. |
| ANTHEM ">10% over SOTA product search" | Claim-of-a-claim through a survey; trace to ANTHEM's own venue first. |
| "Comparing Euclidean and Hyperbolic Loss Functions for Hierarchical Multi-Label Text Classification" (COLING 2025) | Real ACL Anthology ID, not verified against the Anthology directly. |
| NeurIPS 2025 submission "From Bitter to Better Lessons in AI" (openreview LAXgS0xzPf) | Not verified. |
| Pinecone and Weaviate exact documentation wording | Pinecone metrics (cosine, dotproduct, euclidean; no multi-vector) and Qdrant metrics (Dot, Cosine, Euclid, Manhattan; no Hamming) were read directly in September 2026 and are safe. Weaviate's MUVERA claim is from its 1.31 blog post, read directly. |

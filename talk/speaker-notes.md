# On Uses of Neural Embeddings and the Bitter Lesson

### What the embedding systems I built taught me about scaffolds and hard boundaries

Brown bag, roughly 60 minutes total: a 42-45 minute rehearsed main path, with the rest held open for real interruptions and discussion. Internal engineering audience, mostly junior ML engineers: fluent in current tooling, thin on pre-2020 history. The previous brown bag covered text embeddings as inputs to LLMs. That knowledge is assumed and never recapped.

**Thesis.** Hand-built structure sorts into three piles. Structure that shapes learning — a loss term, a sampler, a curriculum — has no dependency on how the model gets served, so as a strong default it outlives the model architecture around it, though nothing guarantees that forever. A capability scaffold bridges a gap the model doesn't have yet; it is useful now, and it should be built so it can come out later. A hard boundary enforces a fact about your system — a permission, a budget, an index's supported metric; that kind stays explicit no matter how good the model gets, because the model getting better never changes the fact.

**Takeaway.** Structure that shapes learning is usually worth building once the three-question test in the checklist says yes. Build capability scaffolds when you need them, but put a seam around every one and define the capability threshold that removes it. Never let a hard boundary drift into "the model probably handles that now." The harness around an LLM is where the same three jobs become today's design rule.

## How this talk is run

This is a conversation with pictures, not a literature review. Rules for me:

- **No arXiv IDs on screen. Ever.** Papers get described in words, for example "a 2026 audit of seven released models," never cited aloud unless someone asks for the source. Everything is in `references.md` if they want it after.
- **At most one number per beat, said out loud, rounded.** A slide with four numbers on it is a slide nobody reads.
- **Personal spine.** Each embedding question opens on a concrete system I built, named on screen, not "my old work": the KCCA weighted tag-pooling and subset-sampling rules (both ACM ICMR 2018) open the first two acts, the agglomerative tag-cluster hierarchy (ACM ICMR 2018) opens the third, and SAANE (BMVC 2019) opens the fourth. The fixed-window dissertation chunker (Binghamton, 2019) shows up later in the first act, not as its opener. Say what replaced each one, and just as often, what part of it is still doing a job today.
- Invite interruption at the top and mean it. If a beat turns into a good argument, let it run; the serving bridge and the curated backup are what shrinks. Place recognition never shrinks.

**Budget.** See the timing table below. **If a discussion runs long, cut the serving bridge to one slide and move straight to synthesis; never cut "What shapes the embedding?"** Curated backup comes after Questions, time permitting.

## Timing

| Section | Slides (approx.) | Minutes |
|---|---|---|
| Opening (title through thesis) | 6 | 6 |
| What can one vector preserve? (divider + content) | 9 | 10 |
| What does distance mean? (divider + content) | 8 | 11 |
| What shapes the embedding? (divider + content, protected) | 6 | 7 |
| Serving bridge (not a section) | 2 | 3 |
| Synthesis: scaffold vs. hard constraint | 1 | 2 |
| Harness destination | 3 | 4 |
| **Main path total** | **~35-36** | **~43** |
| Questions, then curated backup | — | remainder of the slot |

Slide counts are approximate pending the final deck; rehearse against the minute figures.

## The personal-spine rule

Every section opens on something I actually built, not on a decade. State the system, the paper, and the year; then show what replaced it and what survived.

| Section | Opens on | Source | What replaced it | What survived |
|---|---|---|---|---|
| Opening | KCCA weighted tag-pooling | *Image Annotation Retrieval with Text-Domain Label Denoising*, ACM ICMR 2018 | CLIP zero-shot tagging, one line, no clustering | the diagnosis that naive pooling lets one loud signal (a colour word) swamp the rest — it resurfaces as a hubness and calibration problem later |
| What can one vector preserve? | the weighted tag-pooling rule and the subset-positive training rule | *Image Annotation Retrieval with Text-Domain Label Denoising* and *Multi-label Triplet Embeddings for Image Annotation from User-Generated Tags*, both ACM ICMR 2018 | learned pooling and probabilistic embeddings, then token- and patch-level late interaction | the diagnosis, not the fix — restated later in the act by the fixed-window dissertation chunker (Binghamton, 2019), which loses exactly what a query needs when the query is a subspan |
| What does distance mean? | the agglomerative tag-cluster hierarchy, weighted by distance to centroid | *Image Annotation Retrieval with Text-Domain Label Denoising*, ACM ICMR 2018 (kcca.tex) | Poincaré-ball hierarchy losses trained end-to-end | hierarchy as a loss term, not a stored geometry — the served vector stays flat |
| What shapes the embedding? | SAANE's hand-wired semantic-attention branches | *Semantically-Aware Attentive Neural Embeddings for Image-based Visual Localization*, BMVC 2019 | frozen self-supervised backbones (DINOv2) plus hard-negative mining | the negative-mining intuition, now automated instead of hand-picked |

## Curated backup (seven topics, shown only after Questions)

Nothing outside this list gets a backup slide. If a question needs a source not on this list, answer from the backstage citations below and offer to follow up.

1. WordNet mammals in the Poincaré disk (Nickel & Kiela, 2017) — the figure that started the hyperbolic-embedding line.
2. Prompted general-model geolocation vs. human experts (GeoRC, 2026) — no geolocation architecture at all, matching the field's specialist models.
3. CLIP hubness evidence (QB-Norm, DBNorm, NNN, NeighborRetr) — the failure mode, and its training-time and inference-time fixes.
4. The fair production-serving comparison (own e12 script) — single vector, ColBERTv2-style compression, PLAID, and MUVERA on the same corpus, at production-realistic storage and latency. (e11's brute-force quality comparison stays in the main path.)
5. Paper-scale storage/latency figures (ColBERTv2 compression, MUVERA vs. PLAID) — the numbers that don't fit a one-number-per-slide rule.
6. Exact-equivariance structure-vs-scale evidence (neural force fields vs. transformers learning near-equivariance) — the two-sided version of the decision rule's hardest test.
7. The phase-corruption example (RoPE / EMWaveNet) — a magnitude-only encoding throws away a relative-position signal that a complex-valued one keeps exactly.

## Backstage (only if challenged, none of this on a slide)

Do not volunteer these. `references.md` has the full grouped list with verified titles and numbers.

- Hierarchy gains, geolocation: HierLoc (arXiv:2601.23064), +43.2% subregion, largest gain at subregion; city is +16.8%, so not monotonic in depth; served via FAISS IndexFlatIP on Lorentz embeddings (sign-flip trick), holds across three backbones. Loss-side ablations: PIGEON (arXiv:2307.05845), haversine loss 990 → 877 km.
- The audit: arXiv:2607.05268, seven checkpoints, all near-Euclidean. Precision: arXiv:2211.00181. The Euclidean-index workaround: HyEm (arXiv:2604.09550), HypStructure (arXiv:2412.01023).
- Scale rebuttal: GeoRC (arXiv:2601.21278) 90% vs 96% for a single expert, not an expert-team aggregate — see references.md for the exact framing. Tails: arXiv:2502.11163, −12.5% / −17.0%.
- Uncertainty collapse under drift: arXiv:2512.22318, 0.99 → 0.52-0.64 AUROC. Index incompatibility: SLOSH (arXiv:2112.05872). Conformal: arXiv:2511.17908 (2-3x context reduction), CONFLARE (arXiv:2404.04287).
- Hubness: Radovanović et al. JMLR 2010 for the geometry. In contrastive models: QB-Norm (arXiv:2112.12777), DBNorm (arXiv:2310.11612), NNN (arXiv:2410.24114), Dual Bank Sinkhorn (arXiv:2508.02538); training-time fix NeighborRetr (arXiv:2503.10526). The colour-word-pollution sentence is kcca.tex line 437, from the KCCA weighted-pooling work behind the ICMR 2018 paper.
- Off-the-shelf VPR: AnyLoc (arXiv:2308.00688). DINOv2 frozen + linear probe, 49.0 mIoU on ADE20K, Table 10 of arXiv:2304.07193 (**frozen features plus a linear head, not zero-shot**). Sampling: CliqueMining (arXiv:2407.02422), Nordland 76.0 → 90.7 (Table 1). Attention descendants: CricaVPR (arXiv:2402.19231), BoQ (arXiv:2405.07364). Own DINOv2 clustering on SAANE's own query frames (BMVC 2019): talk/experiments/e9_dino_pca.py, purity 54/54/68% vs. the hand-wired semantic-segmentation map.
- Late interaction: ColBERT (arXiv:2004.12832), ColBERTv2 (arXiv:2112.01488), ColPali (arXiv:2407.01449), MUVERA (arXiv:2405.19504), LIR workshop at ECIR 2026 (arXiv:2511.00444). Own SciFact comparison: e11 (quality, raw cost — main path) and e12 (production stacks: 6.8x storage at 2-bit residual compression; latency depends on the index — curated backup item 4). If asked why the bill isn't 60x: e11 was brute force on both sides. Vector-DB support verified per vendor docs; see references.md.
- Phase (curated backup item 7): mechanism in arXiv:2410.09749; speech gain in arXiv:2309.01535 (VoiceBank+Demand; negative under reverberation on MS-DNS); magnitude-beats-complex counter-ablation in arXiv:2207.08412.
- Index metrics: FAISS wiki (read directly), DiskANN docs, Milvus docs. RANSAC cost measurement: arXiv:2211.14864. Quantisation: TeTRA-VPR (arXiv:2503.02511).
- Wrong-structure-is-worse: arXiv:2606.01090, CI [+0.79, +3.26].
- My own sources, with line numbers, are listed at the bottom of `references.md`.

**Do not use on a slide or aloud:** any Matryoshka latency figure (misattributed to an unrelated paper), the ANTHEM e-commerce percentage, the StructVPR latency numbers, "Region Matters," or Pinecone/Weaviate doc wording quoted verbatim.

# On Uses of Neural Embeddings and the Bitter Lesson

### What hand-built structure survived scale, and where it went

Brown bag, 60 minutes. Internal engineering audience, mostly junior ML engineers: fluent in current tooling, thin on pre-2020 history. The previous brown bag covered text embeddings as inputs to LLMs. That knowledge is assumed and never recapped.

**Thesis.** Structure survives in training and at the boundary. Inside the model's job, scale eats it. Every grade gets one of three places: training, boundary, inside.

**Takeaway.** Spend your structure budget on losses, sampling, and curricula. Keep inference boring, flat, and quantized, because that is the only shape the serving stack accepts. Coda for the room: the same scoreboard for the harness around an LLM; keep it thin, own the boundary, label the guesses.

## How this talk is run

This is a conversation with pictures, not a literature review. Rules for me:

- **No arXiv IDs on screen. Ever.** Papers get described in words, for example "a 2026 audit of seven released models," never cited aloud unless someone asks for the source. Everything is in `references.md` if they want it after.
- **At most one number per beat, said out loud, rounded.** A slide with four numbers on it is a slide nobody reads.
- **Present-first in every act.** Open on something they used this week, then reveal the 2018 ancestor.
- **The defendant is the 2019 era.** I am the cooperating witness who kept the receipts.
- Invite interruption at the top and mean it. If a beat turns into a good argument, let it run and cut Cost.

**Budget.** 5 open / 10 Hierarchy / 13 Uncertainty / 5 Place recognition / 9 More than one vector / 4 Cost / 3 coda / 3 close = 52 minutes of material. Cost is the first thing to go; the coda is the second-to-last. **If a discussion runs long, Cost goes; Place recognition never goes.** Backup after Questions: the phase-swap slide, only if asked when structure is exact.

**Grades** appear at the end of each act: held / absorbed / absorbed, not free / unresolved. My current enthusiasms get graded as hard as my old work.

## Backstage (only if challenged)

Do not volunteer these. `references.md` has the full grouped list with verified titles and numbers.

- Hierarchy gains, geolocation: HierLoc (arXiv:2601.23064), +43.2% subregion, largest gain at subregion; city is +16.8%, so not monotonic in depth; served via FAISS IndexFlatIP on Lorentz embeddings (sign-flip trick), holds across three backbones. Loss-side ablations: PIGEON (arXiv:2307.05845), haversine loss 990 → 877 km.
- The audit: arXiv:2607.05268, seven checkpoints, all near-Euclidean. Precision: arXiv:2211.00181. The Euclidean-index workaround: HyEm (arXiv:2604.09550), HypStructure (arXiv:2412.01023).
- Scale rebuttal: GeoRC (arXiv:2601.21278) 90% vs 96%. Tails: arXiv:2502.11163, −12.5% / −17.0%.
- Uncertainty collapse under drift: arXiv:2512.22318, 0.99 → 0.52-0.64 AUROC. Index incompatibility: SLOSH (arXiv:2112.05872). Conformal: arXiv:2511.17908 (2-3x context reduction), CONFLARE (arXiv:2404.04287).
- Hubness: Radovanović et al. JMLR 2010 for the geometry. In contrastive models: QB-Norm (arXiv:2112.12777), DBNorm (arXiv:2310.11612), NNN (arXiv:2410.24114), Dual Bank Sinkhorn (arXiv:2508.02538); training-time fix NeighborRetr (arXiv:2503.10526). Averaging: the 2018 sentence is kcca.tex line 437.
- Off-the-shelf VPR: AnyLoc (arXiv:2308.00688). DINOv2 frozen + linear probe, 49.0 mIoU on ADE20K, Table 10 of arXiv:2304.07193 (**frozen features plus a linear head, not zero-shot**). Sampling: CliqueMining (arXiv:2407.02422), Nordland 76.0 → 90.7 (Table 1). Attention descendants: CricaVPR (arXiv:2402.19231), BoQ (arXiv:2405.07364). Own DINOv2 clustering on the 2018 frames: talk/experiments/e9_dino_pca.py, purity 54/54/68% vs the hand-wired map.
- Late interaction: ColBERT (arXiv:2004.12832), ColBERTv2 (arXiv:2112.01488), ColPali (arXiv:2407.01449), MUVERA (arXiv:2405.19504), LIR workshop at ECIR 2026 (arXiv:2511.00444). Own SciFact comparison: e11 (quality, raw cost) and e12 (production stacks: 6.8× storage at 2-bit residual compression; latency depends on the index). If asked why the bill isn't 60×: e11 was brute force on both sides. Vector-DB support verified per vendor docs; see references.md.
- Phase (backup only): mechanism in arXiv:2410.09749; speech gain in arXiv:2309.01535 (VoiceBank+Demand; negative under reverberation on MS-DNS); magnitude-beats-complex counter-ablation in arXiv:2207.08412.
- Index metrics: FAISS wiki (read directly), DiskANN docs, Milvus docs. RANSAC cost measurement: arXiv:2211.14864. Quantisation: TeTRA-VPR (arXiv:2503.02511).
- Wrong-structure-is-worse: arXiv:2606.01090, CI [+0.79, +3.26].
- My own sources, with line numbers, are listed at the bottom of `references.md`.

**Do not use on a slide or aloud:** any Matryoshka latency figure (misattributed to an unrelated paper), the ANTHEM e-commerce percentage, the StructVPR latency numbers, "Region Matters," or Pinecone/Weaviate doc wording quoted verbatim.

# The Decision Checklist

**Thesis:** Structure survives in training and at the boundary. Inside the model's job, scale eats it.

## Before you reach for structure instead of scale, check three conditions

1. Is the constraint exact, or is it your best guess?
2. Is your budget fixed, or can you throw more data and compute at it?
3. Could augmented data teach the model the same thing?

If it's exact, your budget is fixed, and augmentation can't fake it, build it. Otherwise wait; the next model probably eats it.

## Where to spend the structure budget

- **Hierarchy-aware loss terms.** PIGEON's haversine-smoothed loss cut mean error from 990.0 km to 877.4 km, an 11.4% reduction. (arXiv:2307.05845)
- **Hard-negative sampling and mining.** CliqueMining raised Nordland Recall@1 from 76% to 90% with zero architectural change. (arXiv:2407.02422)
- **A conformal threshold on your retriever.** Conformal filtering cut retained context by two to three times at a target coverage rate. (arXiv:2511.17908)

## Where not to spend it

- **A curved metric in your index.** No production index supports one: FAISS gives you L2 and inner product; DiskANN gives you l2, mips, and cosine.
- **Distributional embeddings as your served representation.** Probabilistic knowledge-graph methods scored 0.99 AUROC on random corruptions, then collapsed to 0.52–0.64 under temporal distribution shift (arXiv:2512.22318). Wasserstein distance is cubic in distribution size, so SLOSH (arXiv:2112.05872) re-embeds distributions back into Euclidean space just to make ANN work at all.
- **A Matryoshka objective when plain truncation would do**, unless you're compressing past roughly 70%. (arXiv:2605.16608)

## The same rule for a harness around an LLM

Before you write scaffolding around a model somebody else trains, ask which kind of piece it is.

1. Is it a **fact about your system** (a permission, a budget, a data boundary), or a **guess about what the model can't do yet**?
2. Is your budget fixed, or will the next model generation arrive before this ships?
3. Could the next model learn it, given the trend of the last three?

A fact about your system lives at the boundary: build it, keep it, and keep checking it as the model improves. A guess about the model lives inside its job: build it if you must, label it, and put a seam around it so it can be deleted the day the API does it. Between 2023 and 2025 that happened to prompted chain-of-thought, output parsers, majority voting, and hand-coded agent loops, roughly eighteen months each.

## Takeaway

Keep inference boring, flat, and quantized. Keep the harness thin: the boundaries you own, the evals you trust, everything else replaceable.

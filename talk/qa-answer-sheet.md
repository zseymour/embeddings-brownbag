# Q&A Answer Sheet

Five hardest questions. Rehearse the substance, don't read verbatim.

---

## 1. "Isn't this just 'inductive bias is good' with extra steps?"

No. "Inductive bias is good" is a value claim. Mine is a placement claim, and it's falsifiable: structure that shapes learning is usually worth building because it carries no serving-time dependency, a capability scaffold is worth building only until the model absorbs it, and a hard boundary is worth building forever because it isn't a claim about the model at all — it's a fact about your system. If placement didn't matter, the same structure should help wherever you bolt it on, and it doesn't. arXiv:2606.01090 measures a wrong-group constraint as significantly worse than no constraint at all, with a confidence interval of [+0.79, +3.26] that excludes zero. Structure isn't free money; the wrong structure, in the wrong pile, actively costs you more than having none.

**Cite:** arXiv:2606.01090

---

## 2. "An audit says hyperbolic vision-language models don't even use their geometry. Why are we here?"

Because that audit is in the talk on purpose. It's evidence for the thesis, not against it. arXiv:2607.05268 audits seven released checkpoints and finds every converged one sits near-Euclidean on the dimensionless-radius measure. That's exactly what the placement claim predicts: the geometry these models actually use lives in the loss — structure shaping learning — not in the served representation, which stays flat because the index is a hard boundary that never adopts a curved metric. HierLoc (arXiv:2601.23064) is the loss-side version done right: +43.2% subregion accuracy, the largest gain of any level it reports (not monotonic with depth — city is +16.8%, lower than subregion).

**Cite:** arXiv:2607.05268 (with arXiv:2601.23064 as the loss-side counterpoint)

---

## 3. "If the geometry dies in the index, why should I care as an application engineer?"

Because it tells you where your engineering time is actually worth spending. FAISS supports two primary metrics (L2 and inner product), and its own documented recipe for Mahalanobis distance is to whiten your vectors by the inverse Cholesky factor of the covariance and then drop them into a plain L2 index. DiskANN offers exactly l2, mips, and cosine. Nobody is shipping a curved index — that's a hard boundary, not a place to spend engineering time. So stop shopping for one and put your structure budget into the loss and the sampling instead, where it still shapes learning.

**Cite:** FAISS metric documentation; DiskANN metric documentation

---

## 4. "Why not just wait for the next model to absorb this too?"

Sometimes you should — but only for a capability scaffold, never for a hard boundary. If the constraint is exact and augmented data can't cheaply substitute for it, it isn't a scaffold at all; waiting doesn't close the gap. arXiv:2510.09768 finds equivariant force fields hold a better scaling exponent than unconstrained models at every tested compute budget, so the gap widens with scale rather than shrinking. But if the constraint is only approximate, it probably is a scaffold, and the right move is to build it with a seam and expect to delete it: arXiv:2510.02259 shows plain transformers trained on raw Cartesian coordinates match a SOTA equivariant network at matched compute, learning near-equivariance emergently to above 0.99 cosine similarity across rotated frames — the capability got absorbed.

**Cite:** arXiv:2510.09768 (exact case) and arXiv:2510.02259 (approximate case)

---

## 5. "Your own career went from embeddings to plumbing. Aren't you arguing against yourself?"

Yes, and that's the honest place to start. MaAST, SASRA, and GraphMapper all treat the embedding as a component buried inside a larger learned system — that's plumbing, not an object of study. The reason to study it directly anyway is that plumbing still has a geometry, and it still has a bill, and the bill is the part scale never pays on its own; somebody still has to decide whether a given piece of that plumbing is structure worth keeping in the loss, a scaffold that should come out, or a boundary that has to stay hard. TeTRA-VPR (arXiv:2503.02511) makes the bill visible: a 2-bit ternary backbone with a binarized embedding layer gets up to 69% less memory and 35% lower inference latency, with no loss (or a slight improvement) in Recall@1.

**Cite:** arXiv:2503.02511

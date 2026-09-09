# The Decision Checklist

**Thesis:** Hand-built structure sorts into three piles, and the pile decides what you do with it.

1. **Structure that shapes learning** — a loss term, a sampler, a curriculum. It never runs at serving time, so as a strong default it outlives the model architecture around it — but that is a default, not a guarantee.
2. **A capability scaffold** — code that bridges something the model can't yet do. Useful now; needs a seam so it can come out later.
3. **A hard boundary** — a fact about your system: a permission, a budget, a data-access rule, an index's supported metric. It stays explicit no matter how good the model gets.

Getting the pile wrong is the actual failure mode. Build a hard boundary as if it were a scaffold, and it rots the day nobody remembers to check it. Build a scaffold as if it were a hard boundary, and you carry code a newer model has already made obsolete.

## 1. Structure that shapes learning

Ask:
- Does it change what the model has to discover, without ever running at serving time?
- Is the constraint exact, or is it your best guess?
- Is your budget fixed, or could you throw more data and compute at it instead?
- Could augmented data teach the model the same thing more cheaply?

If it only touches training, the constraint is exact, the budget is fixed, and augmentation can't fake it: build it and keep it. Otherwise, prefer augmentation, or wait — the next model probably absorbs the guess.

**Spend here:**
- **Hierarchy-aware loss terms.** PIGEON's haversine-smoothed loss cut mean error from 990.0 km to 877.4 km, an 11.4% reduction. (arXiv:2307.05845)
- **Hard-negative sampling and mining.** CliqueMining raised Nordland Recall@1 from 76% to 90% with zero architectural change. (arXiv:2407.02422)

## 2. A capability scaffold: useful now, must be removable

A scaffold covers something the model genuinely cannot do yet. It is not a fact about the world; it is a bet about a gap that will eventually close.

Before you build one:
- Name the specific model limitation it papers over, in one sentence.
- Put a seam around it: an interface the rest of the system calls through, not code tangled into business logic.
- Write down what "the capability got absorbed" looks like — a benchmark, a released model, a behavior you can test for — so there is a concrete trigger to delete it.

**Spoken rule:** A scaffold is allowed to be useful and temporary. Name the model limitation it covers, put a seam around it, and remove it when the capability gets absorbed. Keep real constraints hard.

**Scaffolds that already got removed:** prompted chain-of-thought, absorbed into reasoning-trained models; hand-written output parsers for JSON, absorbed into structured-output APIs; hand-coded ReAct loops for tool use, absorbed into native tool-calling; majority-vote or self-consistency sampling scripted by the caller, absorbed into a single reasoning-effort parameter.

## 3. A hard boundary: stays explicit, never delegated to the model

A hard boundary is a fact about your system, not a guess about the model. Scale never absorbs it, because absorbing it would mean letting the model decide your own permissions, budget, or data-access rules for you.

Ask:
- Is this true regardless of what the model can do — a permission, a budget, a data-access rule, an index's supported metric?
- If the model got this wrong, is the failure a security, compliance, or correctness problem, not just a quality problem?

If yes to either: keep it outside the model, enforced in code you control, and never let "the model is good enough now" become the reason to remove it.

**Concrete hard boundaries from the embedding work:**
- **The index's metric support.** FAISS gives you L2 and inner product; DiskANN gives you l2, mips, and cosine. No production index accepts a curved metric, so a curved metric can never be the thing you store and search — it has to move into the loss instead (see §1).
- **A conformal threshold on your retriever.** Conformal filtering cut retained context by two to three times at a target coverage rate — a check computed from a calibration set at serving time, not something learned inside the model. (arXiv:2511.17908)
- **Calibration thresholds derived from a held-out set**, not from the model's own confidence. Probabilistic knowledge-graph methods scored 0.99 AUROC on random corruptions, then collapsed to 0.52-0.64 under temporal distribution shift (arXiv:2512.22318): the model's own uncertainty estimate is not a boundary you can lean on.

**Concrete hard boundaries in a harness around an LLM:** permissions on what an agent can touch, a budget on tokens or spend, which data sources are in scope for a given request, and which outputs go out without a human in the loop. None of these get thinner as the model improves.

## Where structure quietly turns into wasted engineering

- **A curved metric in your index** — §3 already covers why: no production index supports one.
- **Distributional embeddings as your served representation.** Wasserstein distance is cubic in distribution size, so SLOSH (arXiv:2112.05872) re-embeds distributions back into Euclidean space just to make ANN work at all. If you need the uncertainty, get it from calibration (§3), not from a served distribution.
- **A Matryoshka objective when plain truncation would do**, unless you're compressing past roughly 70%. (arXiv:2605.16608)

## Takeaway

Use structure to shape learning: it is usually worth building once the three-question test says yes. Treat every capability workaround as a scaffold — modular, seamed, and scheduled for removal. Keep permissions, budgets, data boundaries, and trust outside the model, always. That pile does not shrink just because the model got better.

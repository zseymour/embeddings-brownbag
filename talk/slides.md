---
theme: seriph
colorSchema: dark
title: On Uses of Neural Embeddings and the Bitter Lesson
info: |
  Brown bag, about 45 minutes. What the embedding systems I built taught me about scaffolds and hard boundaries.
class: text-left
transition: fade
aspectRatio: 16/9
layout: cover
---

# On Uses of Neural Embeddings and the Bitter Lesson

What the embedding systems I built taught me about scaffolds and hard boundaries

<!--
Title, subtitle, then: "This is a talk about being wrong in public, and about the parts of the embedding systems I built that turned out not to be wrong."

Set the norm now: "Interrupt me. If something sounds like nonsense, it might be, and I'd rather find out now."
-->

---

# A hand-built pooling rule for image tags

<p class="text-lg mb-2">Hand-picked kernels, a hand-clustered vocabulary, a hand-tuned pooling rule. This system is mine, so the failure rows are available.</p>

<div class="grid grid-cols-[auto_auto_1fr_1fr] gap-x-6 gap-y-3 items-center mt-2">
  <span class="text-xs op-60 text-center">query, with its Flickr tags</span>
  <span class="text-xs op-60 text-center">1st nearest image</span>
  <span class="text-sm font-bold text-orange-400">plain average of tag vectors</span>
  <span class="text-sm font-bold text-emerald-400">weighted average (mine)</span>

  <figure class="m-0 text-center"><img src="/figures/kcca-poodle-query.jpg" alt="red standard poodle" class="max-h-24 rounded" /><figcaption class="text-xs op-60 mt-1">red, dog, poodle, standard, standardpoodle</figcaption></figure>
  <img src="/figures/kcca-poodle-nn1.jpg" alt="first nearest neighbour" class="max-h-24 rounded mx-auto" />
  <span class="font-mono text-orange-400 leading-snug"><strong>dog</strong>, explore, <strong>red</strong>, green, blue</span>
  <span class="font-mono text-emerald-400 leading-snug"><strong>dog</strong>, dogs, <strong>poodle</strong>, <strong>standard</strong>, <strong>standardpoodle</strong></span>

  <figure class="m-0 text-center"><img src="/figures/kcca-amigurumi-query.jpg" alt="crochet bird" class="max-h-24 rounded" /><figcaption class="text-xs op-60 mt-1">yellow, bird, handmade, crochet, amigurimi</figcaption></figure>
  <img src="/figures/kcca-amigurumi-nn1.jpg" alt="first nearest neighbour" class="max-h-24 rounded mx-auto" />
  <span class="font-mono text-orange-400 leading-snug"><strong>amigurimi</strong>, <strong>crochet</strong>, macro, explored, toy</span>
  <span class="font-mono text-emerald-400 leading-snug"><strong>amigurimi</strong>, <strong>crochet</strong>, toy, <strong>handmade</strong>, etsy</span>
</div>

<p class="text-xs op-60 mt-3">Bold: tag is in the ground truth. MIRFlickr-25k, image projected into a KCCA image-text space, five nearest tags returned.</p>

<p class="source">Image Annotation Retrieval with Text-Domain Label Denoising, ACM ICMR 2018</p>

<!--
"Two Flickr photos, tagged by whoever uploaded them. Both columns project the image into the same learned image-text space and pull back the five nearest tags. The difference is only how the tag word vectors are pooled."

"Plain averaging: the poodle comes back as dog, red, green, blue. The colour word swamps everything, and 'explore' is just a tag Flickr users spam. Weighted: dog, poodle, standard poodle. Same on the bird: the plain average gives macro and explored; the weighted one gives handmade, and etsy, which nobody labelled."

"Everything that produced this is gone. Nobody would build it today. The interesting question is why, because the obvious answer, 'a bigger model ate it', is only half right. Hold on to the colour-word failure; it comes back later, when we ask what one vector can preserve."
-->

---

# The same photos through a zero-shot model

<p class="text-sm op-70 mb-2">CLIP ViT-B/32, one line, ranking the same 1,386-tag MIRFlickr vocabulary. No kernels, no clustering, no pooling rule. The owl fails the same way this system failed: the photo is a shelf of crafts, and the models describe the shelf.</p>

<img src="/figures/exp-e1-zero-shot-tags.png" alt="zero-shot top-5 tags for the poodle, bird, and owl photos" class="max-h-80 mx-auto rounded" />

<p class="source">own experiment, talk/experiments/e1_zero_shot_tags.py; MIRFlickr-25k common_tags.txt</p>

<!--
"Same three photos, same vocabulary, a 2021 model with a one-line prompt. Standard poodle first. Amigurumi first, spelled correctly; the uploader's misspelling isn't in the vocabulary. And the owl: collection, crafts, handmade, decoration, display. Which is what the photo is. My hand-built pipeline said cup, vintage, retro. Nobody finds the owl, because the tag is about a small wooden owl on a shelf full of other things."

"Everything that produced my columns is gone. Nobody would build it today. The interesting question is why, because 'a bigger model ate it' is only half right."
-->

---

# Scale narrows the gap. It does not close it

<p class="text-sm op-70 mb-2">Same owl photo, same 1,386 tags, only the model changes. "Owl" moves from rank 1013 to 10, then back to 20. A large gain, but not a complete or monotonic one.</p>

<img src="/figures/exp-e1-owl-rank-vs-scale.png" alt="rank of owl vs model size" class="max-h-85 mx-auto rounded" />

<p class="source">own experiment, talk/experiments/e1_zero_shot_tags.py</p>

<!--
"Throw scale at it. Three times the parameters takes 'owl' from rank one thousand to rank ten. Six times, and it is back at twenty. Never top five, and the curve is not monotonic. Scale is a real lever, not a decoration — two orders of magnitude of rank movement is not nothing. But it doesn't explain why the owl never clears the top five. Hold that open question."
-->

---

# The bitter lesson

<blockquote class="text-3xl mt-12">
General methods that leverage computation are ultimately the most effective, and by a large margin.
</blockquote>

<p class="text-xl op-70 mt-8">Search and learning scale. Hand-built knowledge helps early, then plateaus.</p>

<p class="source">Sutton, The Bitter Lesson, 2019</p>

<!--
"The owl chart is the receipt for both halves of Sutton's claim: two more orders of magnitude of compute moved the rank a lot, a real gain, and still never found the owl, a real residual."

"That is Sutton's claim. He lays it out in four steps: researchers build knowledge into a system; it helps; it plateaus; then search and learning overtake it. The word that matters is knowledge. Encoding your guesses about a task stops paying. That is not the same as encoding a fact about the world. The essay is careful about that distinction, and this talk is built around it."
-->

---

# Three jobs for structure

<div class="grid grid-cols-3 gap-6 mt-10 text-center">
  <div class="rounded-lg border border-white/15 bg-white/5 px-5 py-6">
    <p class="text-2xl font-bold">Shape learning</p>
    <p class="text-base op-60 mt-3">loss · sampler · curriculum</p>
    <p class="text-sm op-70 mt-7">no serving dependency</p>
  </div>
  <div class="rounded-lg border border-white/15 bg-white/5 px-5 py-6">
    <p class="text-2xl font-bold">Scaffold</p>
    <p class="text-base op-60 mt-3">bridges a capability gap</p>
    <p class="text-sm op-70 mt-7">useful now, modular, removable</p>
  </div>
  <div class="rounded-lg border border-white/15 bg-white/5 px-5 py-6">
    <p class="text-2xl font-bold">Hard boundary</p>
    <p class="text-base op-60 mt-3">enforces a system fact</p>
    <p class="text-sm op-70 mt-7">explicit, durable</p>
  </div>
</div>

<div class="mt-10 text-center text-2xl font-semibold leading-snug">
  <p>A scaffold is allowed to disappear. A hard boundary should not.</p>
</div>

<p class="text-base op-50 text-center mt-4">The same rule applies to scaffolding around models we do not train ourselves.</p>

<!--
"Three jobs hand-built structure can do. Shape learning: put it in the loss, the sampler, the curriculum, and it never has to survive at inference. Scaffold: a piece that bridges something the model can't do yet. It's allowed to be useful and temporary, as long as it's modular enough to remove later. Hard boundary: a fact about your system, not a guess about the model, permissions, budgets, what data crosses a line. That one should stay explicit forever."

"Every act from here asks the same question about a piece of hand-built structure: which of these three jobs is it doing?"

"Hold onto this framework — it comes back at the very end."
-->

---
layout: section
---

# What can one vector preserve?

A photo has six tags. A document has ten thousand words. One vector has to speak for all of it.

<!--
"Every system in this act puts one item through one encoder and gets one vector back. The question is what that one vector can hold before it runs out of room, and where the field put the parts that didn't fit."
-->

---

# Two hand-built fixes for one photo, six tags

<p class="text-sm op-70 mb-4">A photo with six tags is not one thing. Two hand-built rules for representing it anyway, one at pooling time and one at training time. Both mine, both from the same annotation project.</p>

<div class="grid grid-cols-2 gap-10 text-lg">
  <div>
    <p class="font-bold text-orange-400 mb-2">Pooling: weighted average of the tag vectors</p>
    <p>Plain averaging failed on the poodle row from the opening. "When the tag vectors are just averaged, the color word quickly overwhelms the model."</p>
  </div>
  <div>
    <p class="font-bold text-orange-400 mb-2">Training: the subset rule</p>
    <ul>
      <li>A positive example is anything whose tags are a subset of mine</li>
      <li>Margin fixed by hand</li>
      <li>Training stalled, so a second loss term, weight also by hand</li>
    </ul>
  </div>
</div>

<p class="source">Image Annotation Retrieval with Text-Domain Label Denoising, ACM ICMR 2018; Multi-label Triplet Embeddings for Image Annotation from User-Generated Tags, ACM ICMR 2018</p>

<!--
"The pooling rule, you already saw. The subset rule is new: I never called 'similar' a yes/no question. Any photo sharing some of my tags counted as a positive, at a margin I set by hand, plus a second loss term when training stalled."

"Both rules are guesses about what one vector can't hold on its own. The mechanism behind the left one is next. The principled version of the right one is two slides after that."
-->

---

# Why plain averaging fails

<p class="text-sm op-70 mb-2">The mean of a set sits inside the hull of its members, by definition. Below: the real amigurumi photo, its five Flickr tags, and the actual 1,386-word MIRFlickr vocabulary, embedded with bge-small-en-v1.5. The scatter is a 2-D PCA projection for display only -- the mean and its nearest unselected neighbour are computed by cosine over the real 384 dimensions. Click any point to add its tag to the set.</p>

<SetMean />

<!--
"Five real tags on the actual photo: yellow, bird, handmade, crochet, amigurumi. Average them with bge-small-en-v1.5 and search by cosine over the real 1,386-tag MIRFlickr vocabulary. The nearest unselected tag to that mean is 'colourful'. The colour word still pulls, just less violently than the 2018 GloVe vectors did: the gap between the mean's cosine to its own members and its cosine to the rest of the vocabulary shrinks from about 0.48 with 2018-era vectors to about 0.13 here."

"Click any of the other 1,381 points to add it to the tag set, then watch the hull, mean, and nearest unselected neighbour recompute live. The scatter you're clicking on is a 2-D PCA projection for display only -- every number on screen is computed from the real 384 dimensions, not from where the dot sits."

"Attention pooling doesn't dodge this: it's still a convex combination, still a point in this same hull, it just learns which point instead of averaging blindly. The vector you serve is still one point."
-->

---

# Learned distributions: the principled fix

<p class="text-sm op-70 mb-2">The principled version of both hand-built fixes: learn a distribution instead of a point, wide for ambiguous items, tight for unambiguous ones. Standard ANN indexes cannot compare distributions, so the workaround (SLOSH) re-embeds them as ordinary points before they ever reach an index.</p>

<img src="/figures/paper-hib-corrupted.png" alt="Hedged Instance Embedding: ambiguous inputs map to wide Gaussians" class="max-h-72 mx-auto rounded bg-white mt-2" />

<p class="source">Hedged Instance Embedding, 2018 (Fig. 3); Decomposing Uncertainty in Probabilistic KG Embeddings, 2025; SLOSH, 2021</p>

<!--
"It's elegant, and the field's verdict on it is rough. Detecting randomly corrupted facts: 0.99 AUROC. Under temporal drift, the only kind of drift you get in production, it drops to 0.52 to 0.64."

"Comparing two distributions properly is expensive enough that the standard workaround squashes them back into ordinary vectors before they ever reach an index. Same shape as both hand-built fixes on the last two slides: a good idea about representing uncertainty, and no serving path for it yet."
-->

---

# Fixed windows commit before the query arrives

<div class="grid grid-cols-2 gap-10 mt-6 text-lg items-start">
  <div>
    <p class="font-bold text-orange-400 mb-2">My 2019 pipeline</p>
    <ul>
      <li>Three 300-character windows</li>
      <li>25% overlap</li>
      <li>Truncate after 5,000 characters</li>
    </ul>
  </div>
  <div>
    <p class="font-bold text-emerald-400 mb-2">The lasting problem</p>
    <p>Any fixed boundary can split the evidence a query needs. A different window only moves the split.</p>
  </div>
</div>

<p class="source">Leveraging Weakly-Aligned, User-Generated Data for Deep Learning Features (Binghamton, 2019)</p>

<!--
"My text model, from the dissertation: three windows, 300 characters each, 25% overlap, and anything longer truncated at 5,000 characters. The numbers were a reasonable response to a memory limit at the time, not a mistake."

"The durable problem isn't the numbers. It's that no fixed window knows what the later query will need. Any boundary you pick can split the evidence apart, and a different window just moves the split somewhere else. Next slide is what changes when you stop choosing the unit in advance: ColBERT changes when the unit gets chosen, from before the query to after it."
-->

---

# Late interaction: keep every token

<p class="text-sm op-70 mb-2">ColBERT, 2020. Encode the query and the document into one vector per token. Score = sum over query tokens of the maximum similarity to any document token. No pooling; the document is stored as a set.</p>

<img src="/figures/paper-colbert-paradigms.png" alt="representation-based, interaction, all-to-all, and late interaction" class="max-h-50 mx-auto rounded bg-white" />

<p class="text-sm op-70 mt-3">Left: the single-vector model every earlier slide served. Right: late interaction, which keeps the per-token vectors and defers the comparison to query time. ColBERTv2 on MS MARCO: MRR@10 39.7 against 38.8 for the best single-vector model in the same table.</p>

<p class="source">ColBERT, 2020 (Fig. 2); ColBERTv2, 2022 (Table 4)</p>

<!--
"Panel a is everything so far: one vector each side, one dot product. Panel d keeps every token vector and does the matching late. No mean, no attention pooling, no chunk boundary picked before the query arrives — nothing chosen in advance. The document stays a set."
-->

---

# ColPali: keep every patch, skip the text pipeline

<p class="text-sm op-70 mb-2">The same idea on documents as images. Standard pipeline: OCR, layout detection, chunking, captioning, text embedding. ColPali: embed the page image, keep every patch, score with MaxSim. On ViDoRe, averaged over ten tasks: +14 points of nDCG@5 over the best text pipeline.</p>

<img src="/figures/paper-colpali-pipeline.png" alt="standard OCR and chunking pipeline vs ColPali page-patch pipeline" class="max-h-70 mx-auto rounded bg-white" />

<p class="source">ColPali, 2024 (Fig. 1; Table 2, CC0)</p>

<!--
"Documents as pictures. The top pipeline is what most document RAG looks like: OCR, layout, chunk, caption, embed, five hand-designed stages. The bottom one embeds the page and keeps a vector per patch: nDCG@5 81.3 against 67.0 for the best text pipeline, fourteen points, no text pipeline at all. That's the chunking slide, absorbed."
-->

---

# Keeping every token helps. Now we have to serve it

<p class="text-sm op-70 mb-2">SciFact, 300 queries. One 384-dim vector per document (bge-small) against one 96-dim vector per token (answerai-colbert-small, 236 tokens per document on average), both 33M-parameter encoders, exhaustive scoring.</p>

<img src="/figures/exp-e11-late-interaction.png" alt="nDCG@10 and Recall@100, and bytes per document, single vector vs late interaction" class="max-h-70 mx-auto rounded" />

<p class="text-sm op-70 mt-2">nDCG@10: 0.713 → 0.746, keeping every token instead of one vector. Byte panel: raw float16, uncompressed. Not the production cost.</p>

<p class="source">own experiment, talk/experiments/e11_late_interaction.py; BEIR SciFact</p>

<!--
"Now test that answer on SciFact, same size encoders. Keep every token and quality goes up three points of nDCG; recall and MRR move the same direction, 0.942 to 0.956 and 0.682 to 0.719."

"The right-hand panel of the figure is the raw cost before anyone engineers it: sixty times the bytes in float16, no compression. That's not what it costs once someone builds an index for it — the next slide is how you pay for keeping every token inside an ordinary index, and what's in the backup is the fully engineered bill."
-->

---

# The index still wants one vector

<p class="text-sm op-70 mb-2">MUVERA, 2024: compress each document's token set into one fixed-dimensional vector whose inner product approximates MaxSim, retrieve with an ordinary MIPS index, then rerank the candidates with exact MaxSim.</p>

<div class="flex flex-wrap items-center justify-center gap-x-3 gap-y-4 mt-8">
  <div class="w-40 rounded-lg border border-white/15 bg-white/5 px-3 py-4 text-center">
    <p class="font-bold text-sm">Token set</p>
    <p class="text-xs op-60 mt-1">one vector per token</p>
  </div>
  <div class="text-2xl op-40">→</div>
  <div class="w-40 rounded-lg border border-white/15 bg-white/5 px-3 py-4 text-center">
    <p class="font-bold text-sm">One fixed-dimensional encoding</p>
  </div>
  <div class="text-2xl op-40">→</div>
  <div class="w-40 rounded-lg border border-orange-400/40 bg-orange-400/10 px-3 py-4 text-center">
    <p class="font-bold text-sm text-orange-400">Ordinary MIPS shortlist</p>
    <p class="text-xs op-60 mt-1">approximate</p>
  </div>
  <div class="text-2xl op-40">→</div>
  <div class="w-40 rounded-lg border border-emerald-400/40 bg-emerald-400/10 px-3 py-4 text-center">
    <p class="font-bold text-sm text-emerald-400">Exact MaxSim rerank</p>
    <p class="text-xs op-60 mt-1">exact</p>
  </div>
</div>

<p class="text-base text-center mt-6">One vector is enough to shortlist. The full token set comes back to rerank the candidates, not to decide how many of them to return.</p>

<p class="text-lg font-semibold text-center mt-3">Reranking still produces a ranking. Something else has to decide where it gets cut.</p>

<p class="source">MUVERA, 2024 (Fig. 1); Weaviate 1.31 ships it as the default multi-vector encoding</p>

<!--
"Two stages instead of four. The fixed-dimensional encoding is a single vector whose dot product approximates the multi-vector score, so an ordinary MIPS index, DiskANN off the shelf, can do the candidate retrieval, approximate. Then exact MaxSim reranks the shortlist, and the structure you compressed away comes back, exact. Against PLAID, ColBERT's own multi-stage engine: on average 10% higher recall at 90% lower latency across BEIR."

"Notice what reranking did and didn't do. It reordered the shortlist more accurately. It didn't decide how many of those reordered results anyone should actually get back. That question doesn't go away no matter how good the ranking underneath it is."
-->

---
layout: section
---

# Which results should we return?

A ranking orders candidates. Whoever calls the retriever still has to decide how many of them to keep.

<!--
"Every system so far ends the same way: a ranked list. This act is about the decision after the ranking: how many of those ranked results actually get returned. A real query, a real score gap, and what it takes to turn 'these are ranked' into 'these are the ones I'm returning.'"
-->

---

# A ranking is not an answer

<p class="text-sm op-70 mb-2">A real query against a real corpus: SciFact, 5,183 abstracts, bge-small-en-v1.5. The query asks about IL-2 and regulatory T cells. Top five by cosine, with the benchmark's relevance labels.</p>

<img src="/figures/exp-e4-top5.png" alt="top five retrieved documents with cosine scores; relevant and non-relevant interleave" class="max-h-75 mx-auto rounded" />

<p class="text-lg font-semibold mt-3">Relevant and irrelevant results interleave. No cosine threshold separates them here.</p>

<p class="source">own experiment, talk/experiments/e4_real_scores.py; BEIR SciFact query 1029</p>

<!--
"Real scores, real labels. Relevant, relevant, not, relevant, not, and the whole spread is seven hundredths. 0.804 is a similarity score, not an 80.4% chance of relevance. And every one of you has hardcoded a top-five or top-ten cutoff on a score like this and moved on."

"A ranking told you the order. It didn't tell you where to stop. That's the real question: given this list, how many of these do you actually return? Two things settle it. What risk of missing a relevant result are you willing to accept, and do this retriever's scores even separate relevant from irrelevant well enough to act on. Coverage first, then what it costs."
-->

---

# The cutoff cannot move the points

<p class="text-sm op-70 mb-2">Illustrative unit-normalized embeddings. Same documents and labels; different angles.</p>

<CosineCutoff />

<p class="text-sm op-70 mt-2">Widen the accepted region to keep all six relevant documents. Better separation lets us do that with fewer irrelevant results.</p>

<p class="source">toy illustration, not measured data; real SciFact calibration is next</p>

<!--
"Twenty-one illustrative documents on a unit circle: six relevant in green, fifteen irrelevant in gray. The query points right. Both panels show the same documents and labels, but the second embedding separates the relevant directions from the irrelevant ones."

"For unit vectors, cosine similarity is the dot product. The cutoff accepts the shaded wedge around the query. Lower the cutoff and the wedge widens. Both panels start at 0.94. On the left, that misses four of the six relevant documents. Press 'Keep all relevant': it chooses the highest slider setting that includes all six, 0.61 on the left and 0.95 on the right. The first embedding returns eleven documents to keep all six relevant. The second returns six."

"The button uses known labels for this one illustrative query. Real calibration uses many labelled calibration queries: record each query's lowest relevant score, take a lower percentile with a finite-sample correction, then check that cutoff on separate held-out queries without using their labels to choose it. For the experiment next, the target is that ninety percent of queries keep every labelled relevant document. That is different from the fraction of green points kept in this one drawing. The guarantee is marginal over the calibration sample and future query, and requires exchangeability."

"What that actually costs on a real retriever is next."
-->

---

# What that target costs, on this retriever

<p class="text-sm op-70 mb-3">bge-small-en-v1.5, calibrated on SciFact: 300 labelled queries, 5,183 abstracts, averaged over 500 random calibration/held-out splits.</p>

<div class="grid grid-cols-2 gap-8 text-center mt-2">
  <div class="rounded-lg border border-white/15 bg-white/5 px-6 py-6">
    <p class="text-sm op-60 uppercase tracking-wide mb-3">Target</p>
    <p class="text-lg font-semibold">90% of held-out queries keep every labelled relevant document</p>
  </div>
  <div class="rounded-lg border border-orange-400/40 bg-orange-400/10 px-6 py-6">
    <p class="text-sm op-60 uppercase tracking-wide mb-3">Observed (mean over 500 splits)</p>
    <p class="text-3xl font-bold text-orange-400">90.2% coverage</p>
    <p class="text-lg font-semibold text-orange-400 mt-2">167 documents kept per query</p>
  </div>
</div>

<p class="text-base text-center mt-4">Calibration controls miss risk. Better score separation can reduce the number of documents needed to meet the same target.</p>

<p class="source">own experiment, talk/experiments/e4_real_scores.py; BEIR SciFact</p>

<!--
"Same bge-small retriever from the ranking slide, calibrated the way I just described: 300 labelled queries, 500 random calibration/held-out splits. Averaged over those splits, coverage lands almost exactly on target, 90.2%. Any single split can miss that badly — this is the mean across splits, not a per-run guarantee."

"Hitting the target costs 167 documents per query on average, out of 5,183, where a fixed top-10 keeps 10. That's not a calibration failure. Calibration did exactly its job: it found the cutoff that controls your miss risk, honestly, given this retriever's actual scores. The cost comes from somewhere else — the relevant and irrelevant scores are too close together for a tight cutoff to also be a safe one. Calibration controls your miss risk. It can't manufacture separation the embedding doesn't have."

"Better separation between relevant and irrelevant scores can reduce how many documents we need to return at the same coverage target. For place recognition, I tried to get that separation by teaching the representation which parts of a scene should stay the same across seasons."
-->

---
layout: section
---

# What shapes the embedding?

Can we train the representation to separate places despite seasons and weather? I tried a semantic branch and attention.

<!--
"A cutoff cannot make the relevant results easier to separate. For place recognition, I tried teaching the representation what should stay the same across seasons."

Never cut this act.

"Then I moved into robot navigation. First paper: the embedding was the product. Second: the embedding fed a policy. Third: the embedding was a component inside a bigger learned system and nobody talked about it. So the trajectory of my own career is 'the embedding stopped being the deliverable.' The reason this act still matters: plumbing has a geometry and a bill, and the bill is the part scale never pays."
-->

---

# SAANE: the pipeline

<p class="text-sm op-70 mb-2">The bet: a second network segments the scene (road, building, sky, tree) and tells the appearance network where to look, so the descriptor survives seasons and weather.</p>

<img src="/figures/saane-pipeline.png" alt="SAANE pipeline: two backbones, fusion, attention, pooling" class="max-h-90 mx-auto rounded bg-white" />

<p class="source">Semantically-Aware Attentive Neural Embeddings for Image-based Visual Localization (SAANE), BMVC 2019</p>

<!--
One sentence: "A second network segments the scene, road, building, sky, tree, and tells the appearance network where to look, so the descriptor survives seasons and weather."
-->

---

# SAANE's positive case: semantics survives the season

<p class="text-base op-80 mb-3">Nordland, winter to summer: SAANE retrieves the exact location; AMOSNet misses by 13 km; App+Sem by 212 m.</p>

<img src="/figures/saane-retrieval-nordland.png" alt="Nordland positive case: SAANE retrieves the exact location across seasons" class="max-h-90 mx-auto rounded bg-white" />

<p class="source">SAANE, BMVC 2019 (supplementary figures)</p>

<!--
"First, the positive result. The query is winter and the database image is summer. SAANE retrieves the exact place. The appearance-only baseline lands thirteen kilometres away, and the fusion model without attention lands 212 metres away. Here, the semantic path and attention do exactly what I designed them to do."

"That makes the failure case meaningful rather than a cheap retrospective dismissal."
-->

---

# SAANE's failure case: attention on a shadow

<p class="text-base op-80 mb-3">Retrieval errors, same query: SAANE misses by 271 m; App+Sem by 613 m; appearance-only AMOSNet is within 1 m.</p>

<img src="/figures/saane-supp-6.png" alt="St. Lucia failure case: attention maps and retrieved-image distances" class="max-h-90 mx-auto rounded bg-white" />

<p class="source">SAANE, BMVC 2019 (supplementary figures)</p>

<!--
"Now the contrast: the same mechanism fails on St. Lucia."

"A shadow from a power pole falls across the query image, and the multimodal attention module treats it as useful structure instead of ignoring it. SAANE lands 271 meters off. My combined model without attention, App+Sem, lands 613 meters off, worse. The appearance-only baseline, no semantic branch at all, lands within a meter. The hand-built semantic path did not save either fusion model on this query. The distances are the verdict; the attention and segmentation columns in the middle are diagnostics, not results."
-->

---

# 2023: off the shelf, no place-recognition training

<p class="text-sm op-70 mb-2">AnyLoc: frozen DINOv2 features plus unsupervised VLAD, no place-recognition training. It wins Recall@1 in both structured and unstructured environments, narrowly in one, by nearly double in the other. The authors credit semantic structure the model learned without supervision.</p>

<AnyLocGains />

<p class="source">AnyLoc, 2023 (Tables III-IV)</p>

<!--
"Say the caveat out loud: my paper reported a different metric than today's papers, AUC instead of Recall@1, so it isn't a head-to-head with SAANE. This chart is AnyLoc against three methods trained end-to-end for place recognition, on AnyLoc's own benchmark split."

"Structured environments, indoor, campus, well-mapped: all four methods span 62 to 86 percent, and AnyLoc is highest, but every trained method is respectable. Unstructured environments, off-road, aerial, unmapped: the three trained methods drop to the high twenties and low thirties, and AnyLoc holds 65 percent, nearly double the best of them."

"That's DINOv2, frozen, off the shelf, zero place-recognition training, the model this chart introduces. The semantic structure the authors credit shows up exactly where hand-tuned training data was scarcest. Next: the same frozen model, run directly on the SAANE query frames."
-->

---

# The SAANE frames through a frozen model

<p class="text-sm op-70 mb-2">Frozen DINOv2 recovers the same sky and road structure across snow, night, and daylight, without labels or place-recognition training.</p>

<img src="/figures/exp-e9-dino-pca.png" alt="query frames, SAANE hand-wired segmentation, DINOv2 PCA, DINOv2 k-means" class="max-h-76 mx-auto rounded" />

<p class="source">own experiment: facebook/dinov2-base at 896 px, talk/experiments/e9_dino_pca.py</p>

<!--
"These are the three query frames from the SAANE paper (BMVC 2019), through a frozen DINOv2 with nothing trained on top. Column three is a PCA of the patch features, column four is six clusters fit across all three frames at once. Sky is one cluster in all three. Road surface is one cluster in the two frames that have asphalt, and Nordland's rail ballast got its own cluster instead of being forced into 'road'. Purity against SAANE's twelve-class hand-wired map is in the fifties and sixties, chance is eight. So: not SAANE's segmentation map, but the part of it that mattered for place recognition, sky and ground, for free."
-->

---

# Smarter negatives change training, not serving

<VprProgress />

<p class="source">MixVPR, 2023; SALAD, 2023; CliqueMining, 2024; SelaVPR++, 2025</p>

<!--
"The thing that survived hardest wasn't in the architecture at all. Get smarter about which negative examples you show the model during training, same network, not one layer changed, and accuracy on the hardest seasonal benchmark jumps from seventy-six to ninety-one percent. That's a sampling trick, and a sampling trick is what I also had in that paper, in a footnote, as a detail."

"I found that out by going back and reading my own thesis."

"But it's a training-time trick: it changes which pairs the loss sees, and it never touches what gets served. The next table is the other side of that line, what the index sees once the vector actually ships."
-->

---

# After training, the index sets the serving rules

<IndexMetrics />

<p class="source">FAISS wiki; DiskANN docs; Milvus 2.4 and 2.6.4 docs; Qdrant 1.10 notes; Elasticsearch 8.18 reference; Pinecone docs</p>

<!--
"The sampling trick from the last slide is invisible here. The index never sees which negatives trained the model; it only ever sees the vector's shape and the metric you search it with. Whatever structure survives training, or gets bent into a boundary trick, still has to clear this table."

"Source is the FAISS wiki, read directly: two metrics, L2 and inner product, cosine is normalize-then-inner-product. Same story everywhere: DiskANN three metrics, Milvus three plus Hamming."

"That's the vendor telling you: nobody ships a curved index, nobody ships a Wasserstein index. Every trick in this talk that survived, the sign flip, the re-embedding, the fixed-dimensional encoding, survived by fitting into this table."
-->

---

# The serving budget can shape training

<p class="text-sm op-70 mb-3">Deployment is a hard boundary: a fixed memory and latency budget on the device that runs this at inference. TeTRA answers it with progressive quantization-aware training: start full precision, move the backbone to ternary weights and the embedding to binary as training progresses.</p>

<div class="grid grid-cols-[1fr_1fr_1fr] gap-x-4 gap-y-2 mt-3 text-sm text-center">
  <div class="text-left op-60 pb-2 border-b border-white/20"></div>
  <div class="font-bold pb-2 border-b border-white/20">EigenPlaces (baseline)</div>
  <div class="font-bold text-orange-400 pb-2 border-b border-white/20">TeTRA-BoQ</div>

  <div class="text-left op-70 py-1 border-b border-white/10">Recall@1</div>
  <div class="py-1 border-b border-white/10">86.7%</div>
  <div class="py-1 border-b border-white/10 text-orange-400 font-bold">88.6% (+1.9 pts)</div>

  <div class="text-left op-70 py-1 border-b border-white/10">Total latency</div>
  <div class="py-1 border-b border-white/10">34 ms</div>
  <div class="py-1 border-b border-white/10 text-orange-400 font-bold">22 ms (−35%)</div>

  <div class="text-left op-70 py-1">Total memory</div>
  <div class="py-1">700 MB</div>
  <div class="py-1 text-orange-400 font-bold">215 MB (−69%)</div>
</div>

<p class="text-xs op-50 mt-2">Percent changes are vs EigenPlaces as the baseline.</p>

<p class="source">TeTRA-VPR, 2025 (Fig. 4)</p>

<!--
"Deployment is the hard boundary here: a fixed memory and latency budget on whatever device runs this at inference. EigenPlaces, full precision: 86.7% Recall@1, 34 milliseconds, 700 megabytes. TeTRA-BoQ, trained with progressive quantization-aware training down to a ternary backbone and a binarized embedding: 88.6% Recall@1, 22 milliseconds, 215 megabytes. Sixty-nine percent less memory, thirty-five percent lower latency, both against EigenPlaces as the baseline, and recall goes up, not down."

"That's the whole point of training it in: nobody is claiming post-training quantization would have lost recall here. TeTRA starts full precision and quantizes progressively during training, down to ternary weights and a binary embedding, so the model adapts to the precision it will actually ship at instead of getting quantized cold after the fact."

"The next slide is about telling a designed-in constraint like this apart from a temporary scaffold."
-->

---

# Three jobs, three design rules

<div class="grid grid-cols-3 gap-6 mt-6 text-sm">
  <div class="rounded-lg border border-white/15 bg-white/5 px-5 py-5">
    <p class="text-lg font-bold mb-3">Shape learning</p>
    <p>Hard-negative sampling (CliqueMining): changes what the loss sees, never has to survive inference.</p>
  </div>
  <div class="rounded-lg border border-white/15 bg-white/5 px-5 py-5">
    <p class="text-lg font-bold mb-3">Scaffold</p>
    <p>A hand-clustered vocabulary, absorbed by scale. Multi-vector flattened plus rerank (MUVERA), where the seam is already visible in the vendor docs.</p>
  </div>
  <div class="rounded-lg border border-white/15 bg-white/5 px-5 py-5">
    <p class="text-lg font-bold mb-3">Hard boundary</p>
    <p>A coverage target is a product requirement; the cutoff must be recalibrated whenever the retriever or the data changes. Quantisation: a fact about your memory budget.</p>
  </div>
</div>

<p class="text-xl font-semibold text-center mt-8 leading-snug">A scaffold is allowed to be useful and temporary. Name the model limitation it covers, put a seam around it, and remove it when the capability gets absorbed. Keep real constraints hard.</p>

<!--
"Same three categories from the opening, now with names attached. The hand-clustered vocabulary was a scaffold, a guess about what the model couldn't do yet, and scale absorbed it. Late interaction and MUVERA are scaffolds with the seam already showing: Milvus and Vespa added native multi-vector support in the last two years, which is what a removable scaffold looks like mid-removal."

"Quantisation isn't a scaffold: it's a fact about your hardware budget, not a guess about the model. The coverage target is a product requirement, and the cutoff that implements it isn't a fixed fact either — it has to be recalibrated whenever the retriever or the data changes, so it's not waiting to be absorbed."

"A wrong guess about structure costs more than no structure at all, so name which one you're building before you build it."
-->

---
layout: section
---

# The harness around an LLM

Every generation of models pulls more of the harness behind the API. Which parts of yours are a scaffold, and which are a hard boundary?

<p class="text-base op-70 mt-6">"We want AI agents that can discover like we can, not which contain what we have discovered." Sutton, 2019, and he meant it literally.</p>

<!--
"Now the part you're actually building. Everything so far was about a model you trained. Most of you are building around a model somebody else trains, and the same three jobs apply, shape learning, scaffold, hard boundary, just faster. Sutton's line about agents was a metaphor in 2019. It isn't now."
-->

---

# What the harness absorbed, and what's still standing

<div class="grid grid-cols-2 gap-8 mt-4 text-sm">
  <div>
    <p class="text-base font-bold text-orange-400 mb-3">Scaffolds models absorbed</p>
    <ul class="space-y-2">
      <li><strong>Prompted chain-of-thought, self-consistency, majority vote</strong> → reasoning trained in, with a thinking budget</li>
      <li><strong>Regex over model text, retry loops</strong> → native function calling, schema-constrained output</li>
      <li><strong>Hand-coded ReAct and browser-automation loops</strong> → computer use and built-in agent tools</li>
      <li><strong>Chunk, embed, index, rerank by hand</strong> → long context and managed retrieval</li>
    </ul>
  </div>
  <div>
    <p class="text-base font-bold text-emerald-400 mb-3">Boundaries still standing</p>
    <ul class="space-y-2">
      <li><strong>Tool and data permissions</strong>: nothing absorbs a constraint that is about you</li>
      <li><strong>Spend limits</strong>: a budget is a fact about you, not the model</li>
      <li><strong>Evals, and a calibrated "when do I trust it"</strong>: the calibration slide again</li>
    </ul>
  </div>
</div>

<p class="source">vendor announcements and docs, read directly, September 2026: OpenAI, Anthropic, Google</p>

<!--
"Left column, in order: prompted chain-of-thought and self-consistency or majority-vote scripts went behind the API as reasoning trained into the model, o1 in September 2024, extended thinking in February 2025, a reasoning-effort knob in December 2024, a thinking budget in May 2025. Regex over model text and retry loops became native function calling, June 2023, and schema-constrained structured outputs, August 2024. Hand-coded ReAct and browser-automation loops became computer use, October 2024, and built-in agent tools in the Responses API, March 2025. Chunk, embed, index, rerank by hand became a million-token context window, February 2024, and managed file search, April and November 2024."

"Right column: permissions and spend limits are facts about you, not guesses about the model, so nothing absorbs them. Evals and a calibrated trust threshold are the calibration slide again, just pointed at a different kind of output."
-->

---
layout: center
class: text-center
---

# The design rule

<div class="text-2xl font-semibold leading-relaxed text-left inline-block mt-6">
  <p>Use structure to shape learning.</p>
  <p class="mt-4">Treat capability workarounds as scaffolds: modular and removable.</p>
  <p class="mt-4">Keep permissions, budgets, data boundaries, and trust outside the model.</p>
</div>

<!--
"Three lines. Spend your structure budget at training time, where it compounds. Every workaround for something the model can't do yet gets a seam and a removal trigger, a specific capability threshold or benchmark you can check, not just a date on a calendar. And the things that are facts about you, not guesses about the model, permissions, budgets, data boundaries, trust, never move inside."

"My architecture didn't survive. My sampling scheme did. I wouldn't have guessed that in 2019, and I wouldn't have found out if I hadn't gone back and read my own thesis."

Hand out the decision checklist. Sources in references.md for anyone who asks.
-->

---
layout: center
class: text-center
---

# Questions

<p class="op-70">Decision checklist and full sources: ask me after.</p>

---
layout: section
---

# Backup

---

# How I hand-built a hierarchy: cluster the vocabulary

<p class="text-sm op-70 mb-2">The tagging system from the opening. Its hierarchy is one level of clusters, and every knob was set by hand.</p>

1. Cluster the tag vocabulary
2. Number of clusters picked by hand: **5% of the vocabulary**
3. A hand-derived formula: distance from the cluster centre → probability the tag is relevant
4. Two hand-picked similarity functions underneath

<p class="source">Image Annotation Retrieval with Text-Domain Label Denoising, ACM ICMR 2018</p>

<!--
Plain words, no jargon: "I clustered the tag vocabulary. I picked the number of clusters by hand, five percent of the vocabulary, because fewer made junk clusters and more made singletons. Then I invented a formula that turned distance-from-the-cluster-centre into probability-this-tag-is-relevant. Two hand-picked similarity functions underneath. Four guesses, stacked."
-->

---

# One tree, two geometries

<p class="text-sm op-70 mb-2">At depth d, a 3-way tree has 3<sup>d</sup> nodes. Give each level one unit of radius: a Euclidean plane supplies area proportional to d<sup>2</sup>; a hyperbolic plane supplies area proportional to e<sup>d</sup>.</p>

<HierarchyCrowding />

<p class="text-sm op-70 mt-2">This 2-D drawing shows same-depth separation in one radial layout. It does not prove an optimal embedding or claim that branch distances reorder.</p>

<!--
"Same tree on both sides, branching three ways. Left is an ordinary flat plane; right is a Poincaré disk, where the curved edges are that geometry's straight lines. Every leaf gets an equal-radius circle of room, measured in its own geometry's ruler."

"The slider grows the tree one level at a time. It opens at depth four, where the plane's circles already overlap completely and the disk's circles don't touch at all. Drag it down to watch the plane hold on a little longer at shallow depth, and back up to watch it collapse."

"The general math behind this: a tree branching b ways has b to the d nodes at depth d, exponential growth. A fixed-dimensional Euclidean space only offers polynomial room at radius d. A hyperbolic space offers exponential room. That mismatch, tree against plane, is the whole reason this line of work exists."

"Be careful about what this specific component does and doesn't show. It's a two-dimensional illustration of same-depth separation: can this geometry keep same-depth leaf cells from different branches apart as the tree gets bushy? It is not proof that any particular embedding is optimal, and it is not a claim that distances between branches ever reorder between the two geometries. At equal depth, they don't."
-->

---

# HierLoc: train the geographic hierarchy directly

<p class="text-sm op-70 mb-2">HierLoc, 2026: country, region, subregion, and city each get their own entity embedding in Lorentz (hyperbolic) space. 240k entity embeddings stand in for 5M image embeddings.</p>

<HierLocGains />

<p class="source">HierLoc, 2026 (OSV-5M benchmark)</p>

<!--
"HierLoc, a geolocation system from this year. Instead of one flat image embedding, it builds a hierarchy of entity embeddings, country, region, subregion, city, in Lorentz space, and swaps five million image embeddings for two hundred forty thousand of those."

"Accuracy at every level: country up 8.8%, region up 20.1%, subregion up 43.2%, city up 16.8%. Subregion is the biggest gain, not the deepest level, so it isn't monotonic with depth."

"And the index is plain FAISS inner product. Their vectors are hyperbolic, Lorentz model, but Lorentz distance is an inner product with one sign flipped, so the serving stack never finds out. That's the shape structure has to take to survive inference."
-->

---

# Do served hyperbolic embeddings use the curvature?

<p class="text-sm op-70 mb-2">A separate 2026 audit measured how much curvature seven released hyperbolic vision-language checkpoints use: all seven land near u ≈ 0.2, deep in the near-Euclidean band.</p>

<div class="grid grid-cols-[1.3fr_1fr] gap-6 items-center">
  <img src="/figures/paper-hyperbolic-audit-hu-curve.png" alt="distortion factor vs dimensionless radius, near-Euclidean band, checkpoints near u=0.2" class="max-h-75 rounded bg-white" />
  <div>
    <p class="text-3xl font-bold text-orange-400">All seven sit in the near-flat band.</p>
  </div>
</div>

<p class="source">Is the Geometry Doing the Work?, 2026; The Numerical Stability of Hyperbolic Representation Learning, 2022</p>

<!--
"This audit doesn't test HierLoc. It tests seven other released hyperbolic vision-language checkpoints: whether they actually use the curved space they were trained in."

"The trained models sit in a nearly flat region of a space they're named after. The near-Euclidean threshold is u=0.84; every checkpoint tested lands around 0.2."

"Separately, precision: float64 only represents points out to radius ≈38 in the Poincaré ball before rounding pushes them to the boundary. Deep trees need the rim, and the rim is where the float runs out."

"So the honest version: hierarchy in the loss, still standing. Hyperbolic space as the thing you store and search, unresolved. I say that as someone who finds it beautiful."

"That's where this line of work stands today: real gains in the loss function, an open question in the served index. Worth watching, not yet worth shipping unmodified."
-->

---

# A real hierarchy in the disk, 2017

<p class="text-sm op-70 mb-2">The WordNet mammals subtree, 1,180 nodes, embedded in the two-dimensional Poincaré disk. Deeper levels fan out toward the rim, where the geometry has the room. This figure is why the whole line of work exists.</p>

<img src="/figures/paper-poincare-wordnet-mammals.png" alt="WordNet mammals embedded in the Poincaré disk" class="max-h-85 mx-auto rounded bg-white" />

<p class="source">Nickel & Kiela, Poincaré Embeddings for Learning Hierarchical Representations, 2017 (Fig. 2b)</p>

<!--
"The demo on real data, 2017. Mammals at the centre, species at the rim. Everyone who saw this figure wanted to store their taxonomy in it."
-->

---

# No structure at all: a prompted general model

<p class="op-70">Geolocation accuracy, same benchmark, no hierarchy and no geolocation architecture.</p>

<div class="grid grid-cols-2 gap-8 text-center my-10">
  <div><div class="text-6xl font-bold text-emerald-400">91%</div><div class="text-sm op-70 uppercase tracking-wide mt-2">best prompted general model,<br>country level, no geolocation architecture</div></div>
  <div><div class="text-6xl font-bold text-emerald-400">90–97%</div><div class="text-sm op-70 uppercase tracking-wide mt-2">three human experts,<br>including the reigning GeoGuessr champion</div></div>
</div>

<p class="text-sm op-70">General models, no fine-tuning, in a separate 2025 audit, city level: developing 41.7% vs developed 48.8%; sparsely populated 38.5% vs populous 52.4%.</p>

<p class="source">GeoRC, 2026; AI Sees Your Location, But With A Bias Toward The Wealthy World, 2025</p>

<!--
Say it before they do. "That's from prompting. Scale falls down in the tails. Whether the tails are fatal depends on your product, not your research taste."
-->

---

# Hubness: a few points recur across queries

<p class="text-sm op-70 mb-2">Some points appear unusually often in nearest-neighbour lists. <span class="text-xs">Radovanović et al., JMLR 2010; NeighborRetr, 2025.</span></p>

<Hubness />


<!--
"Four hundred points; for each one, how many others count it among their ten nearest neighbours. Dimension is the only control, and it opens at 256, already showing the skewed distribution: the top 5% of points hold about 22% of all nearest-neighbour slots."

"Drag it down to 2 for the contrast, an ordinary bump around ten. Drag it back up to 512 and the skew sharpens further."

"Those top points are hubs: they recur as a near neighbour across many different queries, far more often than a typical point does. The available fixes are training-time losses that penalise hubs, or a rescoring wrapper applied at query time."
-->

---

# Hubness, measured on CLIP

<p class="text-sm op-70 mb-2">How many COCO captions each image is retrieved for, base CLIP against the same model after nearest-neighbour normalisation. Before the fix a few images match hundreds of captions; after it, the distribution looks like a fine-tuned model's. The fix is a rescoring wrapper; the index is untouched.</p>

<img src="/figures/paper-nnn-hubness-hist.png" alt="captions matched per image, CLIP before and after NNN" class="max-h-80 mx-auto rounded bg-white" />

<p class="source">Nearest Neighbor Normalization Improves Multimodal Retrieval, 2024 (Fig. 2)</p>

<!--
"Not a toy. This is CLIP on COCO. Left, the raw model: a few images are retrieved for hundreds of captions each. Right, after an additive correction computed from a bank of reference queries. Nothing about the index changed."
-->

---

# The bill, on production stacks

<p class="text-sm op-70 mb-2">Same corpus, both sides served the way they ship: single vector through an HNSW index; late interaction through residual compression, PLAID, and MUVERA plus rerank. Hollow points are the brute-force numbers nobody deploys.</p>

<img src="/figures/exp-e12-fair-serving.png" alt="nDCG@10 vs bytes per document and vs latency, single-vector and late-interaction serving configurations" class="max-h-64 mx-auto rounded" />

<p class="text-sm op-70 mt-2">Within 0.01 nDCG of exact MaxSim, 2-bit residual compression costs 6.8× the bytes of the HNSW single vector, matching ColBERTv2's published 6–10×. At 5,000 documents the single vector is already at 0.2 ms, so every multi-vector point looks slow; at MS MARCO scale MUVERA reports 90% lower latency than PLAID.</p>

<p class="source">own experiment, talk/experiments/e12_fair_serving.py; ColBERTv2, 2022; MUVERA, 2024</p>

<!--
"Earlier drafts of this slide said sixty times the storage and five hundred times the latency. That was brute force against brute force, which is the cost of not indexing, not the cost of late interaction. Served the way people ship it: seven times the bytes, and latency that depends entirely on whether the index was built for it. Five years of engineering went into that column, and it is why this is the one structure inside the representation that held."

Backstage: PLAID at this corpus size is slower than exhaustive (246 ms) because candidate generation dominates; that flips at scale. MUVERA needed mean-centering before SimHash on this encoder (anisotropic tokens); documented in the script.
-->

---

# The bill, at paper scale

<div class="grid grid-cols-3 gap-8 text-center my-6">
  <div><div class="text-5xl font-bold text-orange-400">6–10×</div><div class="text-sm op-70 uppercase tracking-wide mt-2">ColBERTv2 index vs uncompressed,<br>MS MARCO: 154 GiB → 16–25 GiB</div></div>
  <div><div class="text-5xl font-bold text-orange-400">257 KB</div><div class="text-sm op-70 uppercase tracking-wide mt-2">per page, ColPali,<br>1,024 patches × 128 dims</div></div>
  <div><div class="text-5xl font-bold text-emerald-400">−90%</div><div class="text-sm op-70 uppercase tracking-wide mt-2">latency, MUVERA vs PLAID,<br>at 10% higher recall</div></div>
</div>

<p class="text-sm op-70">Still one vector per token. Index support in 2026: Vespa and Qdrant native; Milvus since 2.6.4; Elasticsearch as a rerank-only field in technical preview; Weaviate via MUVERA; Pinecone, FAISS, DiskANN no.</p>

<p class="source">ColBERTv2, 2022; ColPali, 2024; MUVERA, 2024; vendor docs, verified September 2026</p>

<!--
"The field has spent five years paying this down: residual compression, token pooling, fixed-dimensional encodings, and a workshop at ECIR this year on nothing else. And half the production indexes still won't take it natively."
-->

---

# Structure vs. scale, measured

<div class="grid grid-cols-2 gap-8 mt-2">
  <div>
    <img src="/figures/paper-forcefield-scaling.svg" alt="validation loss vs compute for unconstrained and equivariant force fields" class="max-h-70 mx-auto rounded bg-white p-2" />
    <p class="text-sm op-70 mt-2">Neural force fields: the equivariant models keep a steeper scaling exponent at every compute budget, −0.40 against −0.14 for the unconstrained network. The gap widens.</p>
  </div>
  <div>
    <img src="/figures/paper-equivariance-data-scaling.svg" alt="loss vs training data for baseline, augmented, and equivariant transformers" class="max-h-70 mx-auto rounded bg-white p-2" />
    <p class="text-sm op-70 mt-2">Same question, transformers: augmentation buys the baseline's data efficiency back, but not the equivariant model's floor. A plain transformer on raw coordinates matches an equivariant GNN at matched compute and learns near-equivariance on its own.</p>
  </div>
</div>

<p class="source">Scaling Laws and Symmetry, 2025; Does equivariance matter at scale?, 2024; Transformers Discover Molecular Structure Without Graph Priors, 2025</p>

<!--
"This is the live argument in 2025, and it is not settled. Left: in force fields, the constraint changes the exponent, and the gap widens with compute. Right: with transformers, augmentation gets the baseline most of the way, and a big plain transformer matches the equivariant model at equal compute, having learned the symmetry itself. Both are true. The difference is whether the constraint is exact and whether augmented data can fake it."
-->

---

# Phase carries the shape

<p class="text-sm op-70 mb-2">Two photos, Fourier transformed, spectra swapped. Each hybrid looks like the image whose phase it carries: correlation 0.69 and 0.74 with the phase donor, 0.02 and 0.01 with the magnitude donor. Structure that is a fact about the data, not a guess about the model.</p>

<img src="/figures/exp-e5-phase-swap.png" alt="magnitude and phase swap between the poodle and bird photos" class="max-h-75 mx-auto rounded" />

<p class="source">Oppenheim & Lim, 1981; own reproduction, talk/experiments/e5_phase_swap.py</p>

<!--
Only if asked "when is a constraint exact, not a scaffold?". "The 1981 experiment. Keep one picture's magnitudes and the other's phases, invert. You see the phase donor every time. Complex-valued networks win on radar, MRI, and speech because phase is the signal; feed it as two real channels and independent scaling corrupts exactly this."
-->

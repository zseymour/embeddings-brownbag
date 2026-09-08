---
theme: seriph
colorSchema: dark
title: On Uses of Neural Embeddings and the Bitter Lesson
info: |
  Brown bag, 60 minutes. What hand-built structure survived scale, and where it went.
class: text-left
transition: fade
aspectRatio: 16/9
layout: cover
---

# On Uses of Neural Embeddings and the Bitter Lesson

What hand-built structure survived scale, and where it went

<!--
Title, subtitle, then: "This is a talk about being wrong in public, and about the one part of my old work that turned out not to be wrong."

Set the norm now: "Interrupt me. If something sounds like nonsense, it might be, and I'd rather find out now."
-->

---
layout: default
---

# Image tagging, 2018

<p class="text-lg mb-2">The way it was built then: hand-picked kernels, a hand-clustered vocabulary, a hand-tuned pooling rule. This one is mine, so the failure rows are available.</p>

<div class="grid grid-cols-[auto_auto_1fr_1fr] gap-x-6 gap-y-3 items-center mt-2">
  <span class="text-xs op-60 text-center">query, with its Flickr tags</span>
  <span class="text-xs op-60 text-center">1st nearest image</span>
  <span class="text-sm font-bold text-orange-400">plain average of tag vectors</span>
  <span class="text-sm font-bold text-emerald-400">weighted average (mine)</span>

  <figure class="m-0 text-center"><img src="/figures/kcca-poodle-query.jpg" alt="red standard poodle" class="max-h-36 rounded" /><figcaption class="text-xs op-60 mt-1">red, dog, poodle, standard, standardpoodle</figcaption></figure>
  <img src="/figures/kcca-poodle-nn1.jpg" alt="first nearest neighbour" class="max-h-36 rounded mx-auto" />
  <span class="font-mono text-orange-400 leading-snug"><strong>dog</strong>, explore, <strong>red</strong>, green, blue</span>
  <span class="font-mono text-emerald-400 leading-snug"><strong>dog</strong>, dogs, <strong>poodle</strong>, <strong>standard</strong>, <strong>standardpoodle</strong></span>

  <figure class="m-0 text-center"><img src="/figures/kcca-amigurumi-query.jpg" alt="crochet bird" class="max-h-36 rounded" /><figcaption class="text-xs op-60 mt-1">yellow, bird, handmade, crochet, amigurimi</figcaption></figure>
  <img src="/figures/kcca-amigurumi-nn1.jpg" alt="first nearest neighbour" class="max-h-36 rounded mx-auto" />
  <span class="font-mono text-orange-400 leading-snug"><strong>amigurimi</strong>, <strong>crochet</strong>, macro, explored, toy</span>
  <span class="font-mono text-emerald-400 leading-snug"><strong>amigurimi</strong>, <strong>crochet</strong>, toy, <strong>handmade</strong>, etsy</span>
</div>

<p class="text-xs op-60 mt-3">Bold: tag is in the ground truth. MIRFlickr-25k, image projected into a KCCA image-text space, five nearest tags returned.</p>

<!--
"Two Flickr photos, tagged by whoever uploaded them. Both columns project the image into the same learned image-text space and pull back the five nearest tags. The difference is only how the tag word vectors are pooled."

"Plain averaging: the poodle comes back as dog, red, green, blue. The colour word swamps everything, and 'explore' is just a tag Flickr users spam. Weighted: dog, poodle, standard poodle. Same on the bird: the plain average gives macro and explored; the weighted one gives handmade, and etsy, which nobody labelled."

"Everything that produced this is gone. Nobody would build it today. The interesting question is why, because the obvious answer, 'a bigger model ate it', is only half right. The other half is the one part of my old work that turned out not to be wrong. Hold on to the colour-word failure; it comes back in the uncertainty section."
-->

---

# The same photos through a zero-shot model

<p class="text-sm op-70 mb-2">CLIP ViT-B/32, one line, ranking the same 1,386-tag MIRFlickr vocabulary. No kernels, no clustering, no pooling rule. The owl fails the way 2018 failed: the photo is a shelf of crafts, and the models describe the shelf.</p>

<img src="/figures/exp-e1-zero-shot-tags.png" alt="zero-shot top-5 tags for the poodle, bird, and owl photos" class="max-h-80 mx-auto rounded" />

<p class="source">own experiment, talk/experiments/e1_zero_shot_tags.py; MIRFlickr-25k common_tags.txt</p>

<!--
"Same three photos, same vocabulary, a 2021 model with a one-line prompt. Standard poodle first. Amigurumi first, spelled correctly; the uploader's misspelling isn't in the vocabulary. And the owl: collection, crafts, handmade, decoration, display. Which is what the photo is. The 2018 pipeline said cup, vintage, retro. Nobody finds the owl, because the tag is about a small wooden owl on a shelf full of other things."

"Everything that produced the 2018 columns is gone. Nobody would build it today. The interesting question is why, because 'a bigger model ate it' is only half right."
-->

---

# Scale, on the one photo everything fails

<p class="text-sm op-70 mb-2">Rank of the tag "owl" among 1,386, four zero-shot models by size. Scale moves it from 1013 to 10. Nothing tested puts it in the top five, and the curve is not monotonic.</p>

<img src="/figures/exp-e1-owl-rank-vs-scale.png" alt="rank of owl vs model size" class="max-h-85 mx-auto rounded" />

<p class="source">own experiment, talk/experiments/e1_zero_shot_tags.py</p>

<!--
"Throw scale at it. Three times the parameters takes 'owl' from rank one thousand to rank ten. Six times, and it is back at twenty. Never top five. That is the shape of the bitter lesson in one chart: general methods plus compute win, by a large margin, and the residual is the data, not the architecture."
-->

---

# What the bitter lesson says

<blockquote class="text-lg">
General methods that leverage computation are ultimately the most effective, and by a large margin.
</blockquote>

<p class="text-sm op-70 mt-2">Two things scale: search and learning. The lesson itself, in his four parts:</p>

<blockquote class="text-base mt-2">
1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning.
</blockquote>

<p class="text-sm op-70 mt-3">On the surviving structure in vision he says only that modern networks "use only the notions of convolution and certain kinds of invariances". A description of how little survived, not an argument for keeping it. The four grades in this talk are his four parts.</p>

<p class="source">Sutton, The Bitter Lesson, 2019</p>

<!--
"Part two is the 2018 half of this talk: every hand-built piece helped, and it was satisfying. Part three is 'absorbed'. The word that matters is knowledge: encoding your guesses about a task stops paying. That is not the same as encoding a fact about the world, and the essay is careful about that distinction; we'll come back to it."
-->

---
layout: statement
---

# <span class="text-5xl">Structure survives in training and at the boundary.<br>Inside the model's job, scale eats it.</span>

<p class="text-base op-60 mt-6">the boundary: the index the vector sits in, the threshold you put on its score, the tools around the model</p>

<blockquote class="text-base mt-8">
"We should build in only the meta-methods that can find and capture this arbitrary complexity."
</blockquote>

<p class="text-sm op-60 mt-1">Sutton, on what does belong inside. A loss, a sampler, a calibration set, a rerank: meta-methods. A cluster tree, a segmentation branch, a chunking rule: contents.</p>

<!--
"I'm not going to tell you geometry is back. I'm going to tell you where it went: into loss functions and samplers, and into the boundary around the model: the index, the threshold, the rerank. Not into the thing the model does. Sutton says it in one sentence: build in only the meta-methods. Training and boundary are where meta-methods live. Inside is where contents live. Hold on to those three words, training, boundary, inside; every grade in this talk gets one."
-->

---

# The four grades, and three places

<div class="grid grid-cols-[auto_1fr] gap-x-8 gap-y-3 mt-6 text-xl">
  <span class="text-emerald-400 font-bold">held</span><span>still used in 2026 systems</span>
  <span class="text-orange-400 font-bold">absorbed</span><span>a larger model does it without the hand-built part</span>
  <span class="text-orange-400 font-bold">absorbed, not free</span><span>pushed out of the model to the boundary; still costs engineering</span>
  <span class="text-purple-300 font-bold">unresolved</span><span>evidence still mixed</span>
</div>

<div class="grid grid-cols-[auto_1fr] gap-x-8 gap-y-3 mt-8 text-lg">
  <span class="font-bold">training</span><span>the loss, the sampler, the curriculum; gone by inference time</span>
  <span class="font-bold">boundary</span><span>between the model and the rest of the system: the index, the threshold, the rerank, the tool</span>
  <span class="font-bold">inside</span><span>the model's own job: what it represents and how it computes</span>
</div>

<!--
"My current enthusiasms get graded as hard as my old work. Grades at the end of each section, and a scoreboard at the end."
-->

---

# Four kinds of structure

<div class="grid grid-cols-[auto_1fr_1fr] gap-x-10 gap-y-3 mt-4 text-base">
  <span class="op-60 text-sm">structure</span><span class="op-60 text-sm">how it was hand-built, 2016 to 2019</span><span class="op-60 text-sm">where it lives in 2026</span>
  <span class="font-bold">Hierarchy</span><span>cluster trees over vocabularies; curved embedding spaces <span class="op-60">(mine: a tag cluster tree)</span></span><span>hierarchy-aware losses; the served vector stays flat</span>
  <span class="font-bold">Uncertainty</span><span>hand-set similarity rules; pooling rules for sets <span class="op-60">(mine: a subset rule, weighted averaging)</span></span><span>a calibration set and a percentile</span>
  <span class="font-bold">Place recognition</span><span>hand-wired semantic branches and attention modules <span class="op-60">(mine: SAANE, 2018)</span></span><span>frozen self-supervised features; harder negatives</span>
  <span class="font-bold">More than one vector</span><span>documents chunked into fixed windows, one vector each <span class="op-60">(mine: three sliding windows)</span></span><span>one vector per token or patch, scored late; flattened again to fit the index</span>
</div>

<p class="text-sm op-70 mt-5">Then Cost: what a production index accepts. Then a coda: the same scoreboard for the harness around an LLM.</p>

<!--
"Four kinds of structure the field built by hand around 2018, one section each. Each section: the problem as you meet it today, the evidence, the 2018 version, the grade, and where the structure lived: training, boundary, or inside. Watch that column; it's the point."
-->

---
layout: section
---

# Hierarchy

Hierarchies were hand-built into vocabularies and into embedding spaces. Where does hierarchy live in an embedding system now?

---
layout: center
---

<div class="text-4xl leading-relaxed font-semibold">
United States<br>
→ California<br>
→ Bay Area<br>
→ Mountain View
</div>

<!--
"Four levels. Now the thing you did this week: you embedded a pile of stuff and dropped the vectors into a cosine index. Flat. What happens to a tree when you flatten it?"
-->

---
clicks: 3
---

# One tree, two geometries

<p class="text-sm op-70 mb-2">A tree branching 3 ways has 3<sup>d</sup> nodes at depth d. Room in a plane grows like d<sup>2</sup>; in a hyperbolic disk, like e<sup>d</sup>.</p>

<HierarchyCrowding />

<!--
Arrow keys grow the tree one level per click; the slider goes to depth 5 for questions. Start with "different branches" selected.

"Same tree on both sides, branching three ways, drawn with the same wedges. The left is an ordinary flat plane. The right is a Poincaré disk, and the curved edges are the straight lines of that geometry."

"Two nodes, highlighted. The orange path is the tree: the edges you'd walk to get from one to the other. The dashed line is what an embedding gives you, the straight-line distance in that space. Watch what happens to it as we go deeper."

At depth 4, different branches: "The tree says these two are eight edges apart. The plane says 0.3. The disk says 1.6. The plane has folded a long tree path into a short jump, and nothing you train on top of it can unfold it."

Switch to siblings: "Two edges apart. Plane: 0.3 again. It cannot tell a sibling from a stranger. The disk gives both about the same too, and that's honest: the disk isn't perfect at this curvature. To keep all eight edges you'd push the leaves into the rim, and that's where float32 runs out. Hold that thought for the audit."

"And the circles: every leaf gets the same amount of room, half an edge, measured in each geometry's own ruler. In the plane they're on top of each other by depth four. In the disk they never touch. The room grows as fast as the tree does."

Do not claim that different branches end up closer than siblings in flat space but not in curved space. At a given depth, angular neighbours are the same distance apart in both geometries whatever their tree relationship; the difference is how much of a long path either geometry keeps.
-->

---

# A real hierarchy in the disk, 2017

<p class="text-sm op-70 mb-2">The WordNet mammals subtree, 1,180 nodes, embedded in the two-dimensional Poincaré disk. Deeper levels fan out toward the rim, where the geometry has the room. This figure is why the whole line of work exists.</p>

<img src="/figures/paper-poincare-wordnet-mammals.png" alt="WordNet mammals embedded in the Poincaré disk" class="max-h-85 mx-auto rounded bg-white" />

<p class="source">Nickel & Kiela, Poincaré Embeddings for Learning Hierarchical Representations, 2017 (Fig. 2b)</p>

<!--
"That is the demo on real data, 2017. Mammals at the centre, species at the rim. Everyone who saw this figure wanted to store their taxonomy in it. Hold that thought."
-->

---
layout: center
class: text-center
---

<p class="text-3xl font-semibold text-emerald-400">Where in your own systems do you have a tree<br>that you're storing flat?</p>

<!--
Expect: product catalogs, org charts, file paths, permission scopes, taxonomies, geo. Take two or three answers. This is the beat that makes the rest of the section land, because now it's their problem.
-->

---
layout: default
---

# The country → city tree in hyperbolic space

<HierLocGains />

<p class="text-sm op-70 mt-2">240k geographic-entity embeddings in place of 5M image embeddings. Gains at every level, largest at subregion.</p>

<p class="source">HierLoc, 2026 (OSV-5M benchmark)</p>

<!--
"A geolocation system last year swapped five million image embeddings for two hundred forty thousand hierarchy embeddings: countries, regions, cities. Accuracy went up across the board: gains at every level, biggest at subregion, forty-three percent."

"And the index is plain FAISS inner product. Their vectors are hyperbolic, Lorentz model, but Lorentz distance is an inner product with one sign flipped, so the serving stack never finds out. That's the shape structure has to take to survive inference."

Backstage only if asked: HierLoc, ICLR 2026.
-->

---

# Do served hyperbolic embeddings use the curvature?

<p class="text-sm op-70 mb-2">HierLoc's gain comes from its training objective. A 2026 audit measured how much curvature seven released hyperbolic checkpoints use: the shaded band is near-Euclidean, the checkpoints sit at u ≈ 0.2, the threshold is 0.84.</p>

<div class="grid grid-cols-[1.3fr_1fr] gap-6 items-center">
  <img src="/figures/paper-hyperbolic-audit-hu-curve.png" alt="distortion factor vs dimensionless radius, near-Euclidean band, checkpoints near u=0.2" class="max-h-75 rounded bg-white" />
  <div>
    <p class="text-3xl font-bold text-orange-400">All seven sit in the near-flat band.</p>
    <p class="text-sm op-70 mt-6">Separately, precision: float64 represents points only to radius ≈38 in the Poincaré ball and ≈19 in the Lorentz model before rounding pushes them to the boundary. Deep trees need the rim.</p>
  </div>
</div>

<p class="source">Is the Geometry Doing the Work?, 2026; The Numerical Stability of Hyperbolic Representation Learning, 2022</p>

<!--
"The trained models sit in a nearly flat region of a space they're named after."

"So the honest version: hierarchy in the loss, held. Hyperbolic space as the thing you store and search, unresolved. And I say that as someone who finds it beautiful."
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

# How 2018 built a hierarchy: cluster the vocabulary

<p class="text-sm op-70 mb-2">The tagging system from the opening slide. Its hierarchy is one level of clusters, and every knob was set by hand.</p>

1. Cluster the tag vocabulary
2. Number of clusters picked by hand: **5% of the vocabulary**
3. A hand-derived formula: distance from the cluster centre → probability the tag is relevant
4. Two hand-picked similarity functions underneath

<!--
Plain words, no jargon: "I clustered the tag vocabulary. I picked the number of clusters by hand, five percent of the vocabulary, because fewer made junk clusters and more made singletons. Then I invented a formula that turned distance-from-the-cluster-centre into probability-this-tag-is-relevant. Two hand-picked similarity functions underneath. Four guesses, stacked."
-->

---

# Hierarchy, graded

<div class="grid grid-cols-[1fr_auto_auto] gap-x-10 gap-y-4 mt-8 text-xl">
  <span>Hierarchy in the loss</span><span class="text-sm op-60">training</span><span class="text-emerald-400 font-bold">held</span>
  <span>Hyperbolic space as what you store</span><span class="text-sm op-60">inside</span><span class="text-purple-300 font-bold">unresolved</span>
  <span>Hand-clustered vocabularies <span class="text-sm op-60">(mine, 2018)</span></span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
</div>

<p class="text-sm op-70 mt-4">Middle column: where the structure lived. Training time, at the boundary between the model and the rest of the system, or inside the model's own job.</p>

---
layout: section
---

# Uncertainty

Rules for what counts as similar were hand-set. What does a similarity score mean now?

---

# Which of these do you show the user?

<p class="text-sm op-70 mb-2">A real query against a real corpus: SciFact, 5,183 abstracts, bge-small-en-v1.5. The query asks about IL-2 and regulatory T cells. Top five by cosine, with the benchmark's relevance labels.</p>

<img src="/figures/exp-e4-top5.png" alt="top five retrieved documents with cosine scores; relevant and non-relevant interleave" class="max-h-75 mx-auto rounded" />

<p class="source">own experiment, talk/experiments/e4_real_scores.py; BEIR SciFact query 1029</p>

<!--
"Real scores, real labels. Relevant, relevant, not, relevant, not, and the whole spread is seven hundredths. What does 0.804 mean? Nothing. It doesn't mean 80% anything. And every one of you has hardcoded a threshold like this and moved on."
-->

---

# Hubness: a few points are everyone's neighbour

<p class="text-sm op-70 mb-2">Why a fixed cutoff fails: in high dimensions a few points score above any threshold against most queries, and most points against almost none.</p>

<Hubness />

<p class="text-xs op-50 mt-1">Radovanović et al., JMLR 2010; NeighborRetr, 2025</p>

<!--
Open at d=2 and drag the dimension slider right slowly.

"Four hundred points, and for each one I ask: how many other points count you among their ten nearest neighbours? In two dimensions that's a bump around ten, as you'd expect. Now watch it in the dimensions your embeddings actually live in."

At 256 or 512: "A handful of points are in everybody's list. Most points are in nobody's. Those are hubs, and the right-hand panel is why your threshold is broken: the hub scores above your cutoff against most queries, the typical point against almost none. Same 0.83, different meaning."

"This is not a bug in your model. It's what high-dimensional geometry does to any point that sits a little closer to the middle of the cloud, and contrastive models have it. The fixes are training-time losses that penalise hubs, and a cheap rescoring wrapper at query time. Nobody fixes it in the index."

Backstage: NNN (arXiv:2410.24114), QB-Norm, DBNorm (arXiv:2310.11612), Dual Bank Sinkhorn (arXiv:2508.02538), NeighborRetr (arXiv:2503.10526). Radovanović et al. 2010 for the original.
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
layout: center
class: text-center
---

<p class="text-3xl font-semibold text-emerald-400">How did you pick your similarity threshold?</p>

<!--
The honest answers (eyeballed it, copied it, tuned it once on a Tuesday) are the setup for the whole section. Let people be honest; be honest first if nobody volunteers.
-->

---
layout: fact
---

# 2–3×

less context in the RAG prompt, same hit rate, on two RAG benchmarks.<br>The cutoff comes from a few hundred labelled examples and a percentile, not from the model.

<img src="/figures/paper-conformal-rag-removal.png" alt="removal rate vs target coverage on two RAG benchmarks" class="max-h-60 mx-auto rounded bg-white mt-4" />

<p class="source">Principled Context Engineering for RAG, 2025 (Fig. 2); CONFLARE, 2024</p>

<!--
"What won is a wrapper. You take a few hundred labelled examples, look at where the scores fall, and set the cutoff so you get the coverage you asked for. It's a calibration set and a percentile."

"And it is principled. It's a statement about your data's statistics, not a guess about your task. It just doesn't touch your model, your embedding, or your index."

Grade: absorbed, not free. The structure moved into the calibration layer.

Monday action, say it plainly: "Go put a real threshold on your retriever this week. A calibration set and a percentile."
-->

---

# The same recipe on a retriever that barely separates

<p class="text-sm op-70 mb-2">SciFact, bge-small, 300 labelled queries split 150/150, 500 random splits. Coverage lands on target: 80.5%, 90.2%, 95.5%. The cost: guaranteeing every relevant document clears the cutoff on this corpus keeps about 22, 167, and 503 documents per query. A fixed top-10 keeps 10.</p>

<img src="/figures/exp-e4-conformal.png" alt="calibration score distributions and coverage vs target with documents retained" class="max-h-80 w-full object-contain rounded" />

<p class="source">own experiment, talk/experiments/e4_real_scores.py; BEIR SciFact</p>

<!--
"I ran the recipe. Coverage is exactly what the percentile promises, averaged over splits. Context reduction is not: the lowest relevant score and the highest non-relevant score overlap almost completely on this retriever, so a cutoff that keeps every relevant document keeps most of the corpus. The wrapper guarantees coverage. It cannot manufacture separation the embedding doesn't have. That is what 'absorbed, not free' means: the structure moved into a calibration layer, and the bill is a retriever good enough to threshold."

Grade: calibrated thresholds, held; the saving is conditional on separation, and the slide says so.
-->

---

# The other place to put uncertainty: in the vector

<p class="text-sm op-70 mb-4">A photo with six tags is not one thing. Two 2018 rules for representing it anyway, one at pooling time and one at training time. Both mine.</p>

<div class="grid grid-cols-2 gap-10 text-lg">
  <div>
    <p class="font-bold text-orange-400 mb-2">Pooling: weighted average of the tag vectors</p>
    <p>Plain averaging failed on the poodle row from the opening slide. In the dissertation: "when the tag vectors are just averaged, the color word quickly overwhelms the model."</p>
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

<!--
"So far, uncertainty lives in the threshold. The other option is to put it in the vector itself. I tried that twice in 2018 without knowing that's what I was doing."

"Left: I averaged tag word vectors to get one vector per photo, plain averaging kept the colour and lost the object, so I weighted the average. I wrote down the symptom and never named the mechanism. Right: training a triplet embedding for multi-label images, 'similar' isn't binary when one photo has six tags and another has three, so I called any subset a positive, fixed the margin by hand, and bolted on a second loss when training stalled."

"The mechanism behind the left one is next. The principled version of the right one is after that."
-->

---

# Why plain averaging fails

<p class="text-sm op-70 mb-2">The mean of a set sits inside the hull. In high dimensions the members' differences cancel and only the shared component survives, which every stranger shares too. Same mechanism as mean pooling a document today.</p>

<SetMean />

<!--
Open at n=6, d=2: "Six vectors, their convex hull, and their mean. The mean is inside the hull, by definition, and it's pulled toward the middle."

Drag dimension up: "Now in the dimensions you actually use. Two things happen. The mean gets short, about one over root n, because the parts of the members that disagree cancel. And what's left is the part they all shared, which is also the part every stranger shares. So the mean ends up more similar to strangers than any of its members was."

"That's what averaging tag vectors did to me in 2018: it kept the colour, which everything had, and cancelled the object. And it's what mean pooling does to a document today. Attention pooling is still a convex combination, it's still a point in the hull, it just learns which point. The learning happened at training time. The thing you serve is still one vector."

Grade: mean pooling as a set representation, absorbed.
-->

---

# The 2018 failure, with real vectors

<p class="text-sm op-70 mb-2">Nearest MIRFlickr tags to the plain mean of the poodle's and the bird's five tags. GloVe, the 2018 vector type: blue and yellow enter the poodle's top ten, pink and purple the bird's; drop the colour word from the set and they vanish. A 2023 text model: the colour words stop intruding, and instead everything is close to everything: the gap between members and strangers shrinks from 0.51 to 0.17.</p>

<img src="/figures/exp-e3-set-mean.png" alt="top-10 nearest vocabulary tags to the plain mean, GloVe vs bge" class="max-h-72 mx-auto rounded" />

<p class="source">own experiment: GloVe 6B-300d, BAAI/bge-small-en-v1.5, 1,386-tag MIRFlickr vocabulary; talk/experiments/e3_set_mean.py</p>

<!--
"Same setup as 2018: pool the tag vectors, rank the vocabulary. With GloVe, the mean of red-dog-poodle-standard-standardpoodle has blue and yellow in its top ten. Take 'red' out of the set and they disappear. That is the sentence from my dissertation, measured. With a modern text model the colour words stop intruding, and instead everything is close to everything: members 0.84, the median stranger 0.67. That is the hull demo's second effect, and it is why your threshold slide exists."
-->

---

# Learned distributions: results

The principled version of the subset rule and of weighted averaging: learn a distribution instead of a point. Ambiguous items get wide regions, unambiguous items get tight ones.

- Detecting randomly corrupted facts: 0.99 AUROC. <span class="text-orange-400">Under temporal drift: 0.52 to 0.64.</span>
- Standard ANN indexes cannot compare distributions; the workaround (SLOSH) re-embeds them as ordinary vectors.

So the served vector stays a point, and uncertainty stays in the threshold layer.

<img src="/figures/paper-hib-corrupted.png" alt="Hedged Instance Embedding: ambiguous inputs map to wide Gaussians" class="max-h-44 mx-auto rounded bg-white mt-2" />

<p class="source">Hedged Instance Embedding, 2018 (Fig. 3, the principled form); Decomposing Uncertainty in Probabilistic KG Embeddings, 2025; SLOSH, 2021</p>

<!--
"It's elegant. I love it. And the field's verdict on it is rough."

"Drift over time is the only kind of drift you get in production."

"Comparing two distributions properly is expensive enough that the standard workaround is to squash them back into ordinary vectors."
-->

---

# Uncertainty, graded

<div class="grid grid-cols-[1fr_auto_auto] gap-x-10 gap-y-4 mt-8 text-xl">
  <span>Mean pooling as a set representation</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Learned distributions over points</span><span class="text-sm op-60">inside → boundary</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Hubness-aware scoring</span><span class="text-sm op-60">inside → boundary</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Calibrated thresholds</span><span class="text-sm op-60">boundary</span><span class="text-emerald-400 font-bold">held</span>
</div>

---
layout: section
---

# Place recognition

Semantic branches and attention modules were hand-wired into place-recognition embeddings. One 2018 paper, mine, every part graded.

<!--
Never cut this section.

"After the PhD I worked on robot navigation. First paper: the embedding was the product. Second: the embedding fed a policy. Third: the embedding was a component inside a bigger learned system and nobody talked about it. And the last brown bag here was about embeddings as input to an LLM. Plumbing again."

"So the trajectory of my own career is 'the embedding stopped being the deliverable.' Which makes this a strange talk for me to give. The reason I'm giving it anyway: plumbing has a geometry and a bill, and the bill is the part scale never pays."
-->

---

# SAANE, 2018: the pipeline

<p class="text-sm op-70 mb-2">The bet: a second network segments the scene (road, building, sky, tree) and tells the appearance network where to look, so the descriptor survives seasons and weather.</p>

<img src="/figures/saane-pipeline.png" alt="2018 SAANE pipeline: two backbones, fusion, attention, pooling" class="max-h-90 mx-auto rounded bg-white" />

<!--
One sentence: "A second network segments the scene, road, building, sky, tree, and tells the appearance network where to look, so the descriptor survives seasons and weather."
-->

---

# Seasons: Nordland

<img src="/figures/saane-retrieval-nordland.png" alt="Nordland retrieval" class="max-h-100 mx-auto rounded bg-white" />

<!--
Seasonal change on Nordland. It worked.
-->

---

# Where it breaks

<img src="/figures/saane-supp-4.png" alt="attention failure case" class="max-h-100 mx-auto rounded bg-white" />

<!--
"Attention slides onto shadows when the semantic side is weak."
-->

---

# 2023: off the shelf, no place-recognition training

<p class="text-sm op-70 mb-2">AnyLoc: frozen DINOv2 features plus unsupervised VLAD, no place-recognition training, beats the VPR-trained methods across environments. The authors credit semantic structure the model learned without supervision; DINOv2's own figure shows it: PCA of patch features colours the same parts across images, no labels.</p>

<div class="grid grid-cols-[1fr_1.6fr] gap-6 items-center">
  <img src="/figures/paper-anyloc-radar.png" alt="AnyLoc: off-the-shelf features vs VPR-trained methods, Recall@1 across environments" class="max-h-80 mx-auto rounded bg-white" />
  <img src="/figures/paper-dinov2-pca.jpg" alt="DINOv2 Fig. 1: PCA of patch features, matching parts colour-matched" class="max-h-80 mx-auto rounded bg-white" />
</div>

<p class="source">AnyLoc, 2023 (Fig. 1); DINOv2, 2023 (Fig. 1; Table 10: 49.0 mIoU on ADE20K, frozen features + linear head)</p>

<!--
Say the caveat out loud: "My paper reported a different metric than today's papers, so I'm not claiming a head-to-head. The trend on the shared benchmark is what's comparable, and the trend is not kind to me."

Backstage: AnyLoc; DINOv2 frozen features + linear head, 49.0 mIoU on ADE20K, Table 10. Frozen + linear probe, not zero-shot.
-->

---

# The 2018 frames through a frozen model

<p class="text-sm op-70 mb-2">DINOv2, no labels, no place-recognition training. Third column: PCA of patch features per frame. Fourth: six k-means clusters fit jointly over the three frames, same colour = same cluster. Sky and road surface come out as consistent clusters across snow, night, and daylight; purity against the hand-wired map 54%, 54%, 68% (chance 8%).</p>

<img src="/figures/exp-e9-dino-pca.png" alt="query frames, 2018 hand-wired segmentation, DINOv2 PCA, DINOv2 k-means" class="max-h-88 mx-auto rounded" />

<p class="source">own experiment: facebook/dinov2-base at 896 px, talk/experiments/e9_dino_pca.py</p>

<!--
"These are the three query frames from the 2018 figures, through a frozen DINOv2 with nothing trained on top. Column three is a PCA of the patch features, column four is six clusters fit across all three frames at once. Sky is one cluster in all three. Road surface is one cluster in the two frames that have asphalt, and Nordland's rail ballast got its own cluster instead of being forced into 'road'. Purity against my twelve-class hand-wired map is in the fifties and sixties, chance is eight. The other four clusters are messier: texture and depth more than category. So: not my segmentation map, but the part of it that mattered for place recognition, sky and ground, for free."
-->

---

# Smarter negatives: what CliqueMining does

<p class="text-sm op-70 mb-2">The third thing in the 2018 paper, in a footnote: which negatives the model trains on. CliqueMining builds a graph of nearby, similar-looking places and samples training batches from cliques of them, so the hardest negatives are the ones a few hundred metres away.</p>

<img src="/figures/paper-cliquemining-method.png" alt="CliqueMining: graph of candidate places, cliques as hard negatives" class="max-h-60 mx-auto rounded bg-white mt-6" />

<p class="source">CliqueMining, 2024 (Fig. 4)</p>

<!--
"Graph on the left: places that are close together and look alike. Middle: sample cliques. Right: the batch is a set of near-identical places a few hundred metres apart, which is exactly what the network gets wrong. Nothing in the architecture changes."
-->

---

# Smarter negatives, same network

<VprProgress />

<p class="text-sm op-70 mt-2">Same network, no layer changed; only the training negatives differ.</p>

<p class="source">MixVPR, 2023; SALAD, 2023; CliqueMining, 2024; SelaVPR++, 2025</p>

<!--
"The thing that survived hardest wasn't in the architecture at all. Get smarter about which negative examples you show the model during training, same network, not one layer changed, and accuracy on the hardest seasonal benchmark jumps from seventy-six to ninety-one percent. That's a sampling trick. And a sampling trick is what I also had in that paper, in a footnote, as a detail."

"I found that out by going back and reading my own thesis."
-->

---

# SAANE, graded

<div class="grid grid-cols-[1fr_auto_auto] gap-x-10 gap-y-4 mt-8 text-xl">
  <span>Hand-wired semantic branch <span class="text-sm op-60">(SAANE)</span></span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Hand-designed attention module <span class="text-sm op-60">(SAANE; CricaVPR, BoQ today)</span></span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Hard-negative sampling <span class="text-sm op-60">(a SAANE footnote; CliqueMining today)</span></span><span class="text-sm op-60">training</span><span class="text-emerald-400 font-bold">held</span>
</div>

<!--
"When I published, I credited most of the win to the semantic branch and a few points to the attention block. Seven years later: the big contribution is the one scale ate. The small one has children. Two of the leading systems today still hand-design an attention or aggregation mechanism, and it still earns its place."

"My small idea outlived my big idea, and my footnote outlived both."
-->

---
layout: center
class: text-center
---

<p class="text-3xl font-semibold text-emerald-400">What's in a footnote of your current project<br>that's doing the work?</p>

---
layout: section
---

# More than one vector

A document is bigger than one vector. 2018 cut it into hand-sized windows. What does the field serve now?

<!--
"Every section so far served one vector per item and put the structure somewhere else. This section is the one place the field kept structure in the served representation, and what it costs."
-->

---

# 2018: a document as three fixed windows

<div class="grid grid-cols-2 gap-10 mt-4 text-lg">
  <div>
    <p class="font-bold text-orange-400 mb-2">The hand-built answer</p>
    <ul>
      <li>Documents cut into sliding windows</li>
      <li>300 characters, 25% overlap, positions fixed by me</li>
      <li>Three windows, justified in a commented-out line by analogy to RGB channels</li>
      <li>Longer documents truncated at 5,000 characters "due solely to memory constraints"</li>
    </ul>
  </div>
  <div>
    <p class="font-bold text-emerald-400 mb-2">The question it was answering</p>
    <ul>
      <li>One vector per document loses everything a single point cannot hold</li>
      <li>Mean pooling picks a point in the hull; attention pooling learns which point</li>
      <li>Chunking decides in advance what a "unit" is</li>
    </ul>
  </div>
</div>

<p class="source">dissertation, chapter 1 (new_model.tex)</p>

<!--
"My text model in 2018. Three windows, fixed size, fixed overlap, and a comment in the source justifying three of them by analogy to RGB channels. Commented out, because even I didn't believe it. The question underneath is the one the set-mean slide raised: a document is not a point. Chunking is the hand-built answer, and every RAG pipeline in this building still does it."
-->

---

# Late interaction: keep every token

<p class="text-sm op-70 mb-2">ColBERT, 2020. Encode the query and the document into one vector per token. Score = sum over query tokens of the maximum similarity to any document token. No pooling; the document is stored as a set.</p>

<img src="/figures/paper-colbert-paradigms.png" alt="representation-based, interaction, all-to-all, and late interaction" class="max-h-50 mx-auto rounded bg-white" />

<p class="text-sm op-70 mt-3">Left: the single-vector model every earlier section served. Right: late interaction, which keeps the per-token vectors and defers the comparison to query time. ColBERTv2 on MS MARCO: MRR@10 39.7 against 38.8 for the best single-vector model in the same table.</p>

<p class="source">ColBERT, 2020 (Fig. 2); ColBERTv2, 2022 (Table 4)</p>

<!--
"Panel a is everything so far: one vector each side, one dot product. Panel d keeps every token vector and does the matching late. No mean, no attention pooling, nothing chosen in advance. The document stays a set."
-->

---

# ColPali: keep every patch, skip the text pipeline

<p class="text-sm op-70 mb-2">The same idea on documents as images. Standard pipeline: OCR, layout detection, chunking, captioning, text embedding. ColPali: embed the page image, keep every patch, score with MaxSim. On ViDoRe, nDCG@5 averaged over ten tasks: best text pipeline 67.0, ColPali 81.3.</p>

<img src="/figures/paper-colpali-pipeline.png" alt="standard OCR and chunking pipeline vs ColPali page-patch pipeline" class="max-h-70 mx-auto rounded bg-white" />

<p class="source">ColPali, 2024 (Fig. 1; Table 2, CC0)</p>

<!--
"Documents as pictures. The top pipeline is what most document RAG looks like: OCR, layout, chunk, caption, embed, five hand-designed stages. The bottom one embeds the page and keeps a vector per patch. Fourteen points of nDCG on the benchmark, no text pipeline at all. That's the chunking slide, absorbed."
-->

---

# Single vector vs late interaction, on the same corpus

<p class="text-sm op-70 mb-2">SciFact, 300 queries. One 384-dim vector per document (bge-small) against one 96-dim vector per token (answerai-colbert-small, 236 tokens per document on average), both 33M-parameter encoders, exhaustive scoring.</p>

<img src="/figures/exp-e11-late-interaction.png" alt="nDCG@10 and Recall@100, and bytes per document, single vector vs late interaction" class="max-h-70 mx-auto rounded" />

<p class="text-sm op-70 mt-2">nDCG@10 0.713 → 0.746; Recall@100 0.942 → 0.956; MRR@10 0.682 → 0.719. Both numbers match each model's published SciFact score.</p>

<p class="source">own experiment, talk/experiments/e11_late_interaction.py; BEIR SciFact</p>

<!--
"Same corpus as the threshold slides, same size encoders. Keep every token and the quality goes up three points of nDCG. The right-hand panel is the raw cost before anyone engineers it: sixty times the bytes in float16 with no compression. The next slide is what it costs served properly."
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

# Getting a set back into a flat index

<p class="text-sm op-70 mb-2">MUVERA, 2024: compress each document's token set into one fixed-dimensional vector whose inner product approximates MaxSim, retrieve with an ordinary MIPS index, then rerank the candidates with exact MaxSim. Against PLAID, ColBERT's own multi-stage engine: on average 10% higher recall at 90% lower latency across BEIR.</p>

<img src="/figures/paper-muvera-fde.png" alt="MUVERA two-stage pipeline vs PLAID four-stage pipeline" class="max-h-60 mx-auto rounded bg-white" />

<p class="text-base mt-3">Third time in this talk: a curved distance became an inner product (HierLoc), a distribution became a vector (SLOSH), now a set of vectors becomes a vector. Structure inside the representation survives by moving to the boundary: the index sees one vector, the structure comes back at rerank.</p>

<p class="source">MUVERA, 2024 (Fig. 1); Weaviate 1.31 ships it as the default multi-vector encoding</p>

<!--
"Two stages instead of four. The fixed-dimensional encoding is a single vector whose dot product approximates the multi-vector score. Off-the-shelf DiskANN does the retrieval. Exact MaxSim reranks the shortlist. That is the recipe every surviving inference-time structure in this talk has followed."
-->

---

# More than one vector, graded

<div class="grid grid-cols-[1fr_auto_auto] gap-x-10 gap-y-4 mt-8 text-xl">
  <span>Late interaction as the served representation</span><span class="text-sm op-60">inside, with a stack built for it</span><span class="text-emerald-400 font-bold">held</span>
  <span>Multi-vector in the index <span class="text-sm op-60">(flattened to one vector plus a rerank)</span></span><span class="text-sm op-60">inside → boundary</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Hand-chunked documents <span class="text-sm op-60">(mine, 2018; most RAG pipelines, 2026)</span></span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
</div>

<!--
"The one place the field kept structure in what it serves. It held, it wins on quality, it costs about seven times the storage once compressed, and to get into a production index it turns back into a single vector with the structure recovered at rerank. Same shape as HierLoc's sign flip and SLOSH's squash. That's the pattern."
-->

---
layout: section
---

# Cost

Structure at inference time has to fit the index and the latency budget. What a production serving stack accepts.

<!--
Designated cut if discussion ran long.
-->

---

# What production indexes support

<IndexMetrics />

<p class="source">FAISS wiki; DiskANN docs; Milvus 2.4 and 2.6.4 docs; Qdrant 1.10 notes; Elasticsearch 8.18 reference; Pinecone docs</p>

<!--
Source is the FAISS wiki, read directly. Same story everywhere: DiskANN three metrics, Milvus three metrics plus Hamming. That's the vendor telling you.

"Nobody ships a curved index."
-->

---

# Two-stage retrieval, then one-stage

<p class="text-sm op-70 mb-2">Inference-time structure in place recognition: geometric verification, then a learned reranker, then none.</p>

- 2021, Patch-NetVLAD: top-100 retrieval, then RANSAC verification, about **15 s per query**
- 2023, R²Former: a learned reranker replaces RANSAC
- 2024, SelaVPR: reranking at about **3%** of the RANSAC cost
- 2024, BoQ: single-stage retrieval beats the two-stage methods

<p class="source">Patch-NetVLAD, 2021; A Faster, Lighter and Stronger…, 2022; R²Former, 2023; SelaVPR, 2024; BoQ, 2024</p>

<!--
"The boundary is training time versus inference time."
-->

---

# 2-bit backbone, binary vector

<div class="grid grid-cols-3 gap-8 text-center my-10">
  <div><div class="text-6xl font-bold text-emerald-400">−69%</div><div class="text-sm op-70 uppercase tracking-wide mt-2">memory</div></div>
  <div><div class="text-6xl font-bold text-emerald-400">−35%</div><div class="text-sm op-70 uppercase tracking-wide mt-2">latency</div></div>
  <div><div class="text-6xl font-bold text-emerald-400">0</div><div class="text-sm op-70 uppercase tracking-wide mt-2">recall lost</div></div>
</div>

<p class="text-sm op-70">Matryoshka training vs. plain truncation of the vector: no difference until about 70% compression.</p>

<p class="source">TeTRA-VPR, 2025; To MRL or not to MRL, 2026</p>

<!--
"Quantise the model to two bits, binarise the output vector. That's the shape of a real win in 2026, and no amount of model scale gives it to you."
-->

---

# The trade-off, measured on Tokyo 24/7

<p class="text-sm op-70 mb-2">Three TeTRA variants against float baselines, full pipeline. Accuracy stays in the same band; latency and memory drop. The one that matters is TeTRA-BoQ against EigenPlaces: better recall, a third of the memory.</p>

<img src="/figures/paper-tetra-tradeoff.jpg" alt="Accuracy, latency, and memory by model on Tokyo 24/7" class="max-h-85 mx-auto rounded bg-white" />

<p class="source">TeTRA-VPR, 2025 (Fig. 4)</p>

---

# Scoreboard

<div class="grid grid-cols-[1fr_auto_auto] gap-x-10 gap-y-1.5 text-base">
  <span class="op-60 text-xs">structure</span><span class="op-60 text-xs">where it lived</span><span class="op-60 text-xs">grade</span>
  <span>Hierarchy in the loss</span><span class="text-sm op-60">training</span><span class="text-emerald-400 font-bold">held</span>
  <span>Hyperbolic space as what you store</span><span class="text-sm op-60">inside</span><span class="text-purple-300 font-bold">unresolved</span>
  <span>Hand-clustered vocabularies</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Mean pooling as a set representation</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Learned distributions over points</span><span class="text-sm op-60">inside → boundary</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Hubness-aware scoring</span><span class="text-sm op-60">inside → boundary</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Calibrated thresholds</span><span class="text-sm op-60">boundary</span><span class="text-emerald-400 font-bold">held</span>
  <span>Hand-wired semantic branches</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Hand-designed attention modules</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Hard-negative sampling</span><span class="text-sm op-60">training</span><span class="text-emerald-400 font-bold">held</span>
  <span>Hand-chunked documents</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Late interaction as the served representation</span><span class="text-sm op-60">inside, with a stack built for it</span><span class="text-emerald-400 font-bold">held</span>
  <span>Multi-vector in the index</span><span class="text-sm op-60">inside → boundary</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Quantisation and the cost side</span><span class="text-sm op-60">boundary</span><span class="text-emerald-400 font-bold">held</span>
</div>

<p class="text-sm op-70 mt-3">Every "held" lived at training time or at the boundary. Every "absorbed" lived inside the model's job. "Absorbed, not free" is the ones that moved out to the boundary and kept the bill. One exception, and the field built a serving stack to earn it.</p>

<!--
Read down it in fifteen seconds. No citations.
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
"This is the live argument in 2025, and it is not settled. Left: in force fields, the constraint changes the exponent, and the gap widens with compute. Right: with transformers, augmentation gets the baseline most of the way, and a big plain transformer matches the equivariant model at equal compute, having learned the symmetry itself. Both are true. The three questions on the next slide are how to tell which case you are in."
-->

---

# Three questions before you build structure

1. Is the constraint **exact**, or is it your best guess about what the model can't do yet?
2. Is your budget **fixed**, or can you throw more data and compute at it?
3. Could **augmented data** teach the model the same thing?

If it's exact, your budget is fixed, and augmentation can't fake it, build it, and build it at the boundary. Otherwise wait for the next general model.

<p class="text-orange-400 mt-6">Constraining a model with the wrong structure is measurably worse than not constraining it at all.</p>

<p class="source">Measuring the Symmetry–Data Exchange Rate, 2026</p>

<!--
"Structure is a bet. Wrong structure costs more than no structure. Otherwise wait; the next model probably eats it." Backstage: symmetry-data exchange rate paper, CI excludes zero.
-->

---
layout: section
---

# The same scoreboard, for the harness around an LLM

Every generation of models pulls more of the harness behind the API. Which parts of yours are inside the model's job?

<p class="text-base op-70 mt-6">"We want AI agents that can discover like we can, not which contain what we have discovered." Sutton, 2019, and he meant it literally.</p>

<!--
"Now the part you're actually building. Everything in this talk was about a model you trained. Most of you are building around a model somebody else trains, and the same thing is happening to your scaffolding, faster. Sutton's line about agents was a metaphor in 2019. It isn't now. And his timescale, 'over a slightly longer time than a typical research project, massively more computation inevitably becomes available', is about eighteen months in this table."
-->

---

# Harness structure, 2022 to 2024, graded in 2026

<div class="grid grid-cols-[1fr_1.1fr_auto_auto] gap-x-6 gap-y-2 text-base mt-2">
  <span class="op-60 text-xs">hand-built</span><span class="op-60 text-xs">absorbed by</span><span class="op-60 text-xs">where it lived</span><span class="op-60 text-xs">grade</span>
  <span>Prompted chain-of-thought, "think step by step"</span><span class="text-sm op-80">reasoning trained in: o1 (2024-09), extended thinking with a budget (2025-02)</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Regex over model text, "respond in JSON", retry loops</span><span class="text-sm op-80">native function calling (2023-06), schema-constrained structured outputs (2024-08)</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Self-consistency, majority vote, best-of-n scripts</span><span class="text-sm op-80">a reasoning-effort knob (2024-12), a thinking budget (2025-05)</span><span class="text-sm op-60">inside</span><span class="text-orange-400 font-bold">absorbed</span>
  <span>Hand-coded ReAct and browser-automation loops</span><span class="text-sm op-80">computer use trained in (2024-10), RL-trained GUI agent (2025-01), built-in tools in the API (2025-03)</span><span class="text-sm op-60">inside → boundary</span><span class="text-orange-400 font-bold">absorbed, not free</span>
  <span>Chunk, embed, index, rerank, by hand</span><span class="text-sm op-80">1M-token context (2024-02); managed file search (2024-04, 2025-11); late interaction</span><span class="text-sm op-60">inside → boundary</span><span class="text-orange-400 font-bold">absorbed, not free</span>
</div>

<p class="text-sm op-70 mt-4">Everything written to do the model's thinking for it went behind the API, roughly eighteen months each.</p>

<p class="source">vendor announcements and docs, read directly, September 2026: OpenAI, Anthropic, Google</p>

<!--
"Same table, same three words. Everything you wrote to do the model's thinking for it, prompted reasoning, output parsing, voting, the planning loop, is inside the model's job, and it went behind the API in about eighteen months each. Retrieval and the agent loop moved out to the boundary and you still pay for them: budgets, stop conditions, the index."
-->

---

# Harness structure: what is still standing

<div class="grid grid-cols-[1fr_1.1fr_auto_auto] gap-x-6 gap-y-3 text-base mt-2">
  <span class="op-60 text-xs">hand-built</span><span class="op-60 text-xs">vendor's version</span><span class="op-60 text-xs">where it lives</span><span class="op-60 text-xs">grade</span>
  <span>Hand-built guardrail classifiers</span><span class="text-sm op-80">vendor-trained constitutional classifiers (2025-02)</span><span class="text-sm op-60">boundary</span><span class="text-purple-300 font-bold">unresolved</span>
  <span>Long-term memory over a vector store</span><span class="text-sm op-80">consumer memory (2024-02); no developer-API equivalent verified</span><span class="text-sm op-60">boundary</span><span class="text-purple-300 font-bold">unresolved</span>
  <span>Tool boundaries, permissions, spend caps</span><span class="text-sm op-80">nothing; the model cannot absorb a constraint that is about you</span><span class="text-sm op-60">boundary</span><span class="text-emerald-400 font-bold">held</span>
  <span>Evals, and a calibrated "when do I trust it"</span><span class="text-sm op-80">nothing; the calibration slide again</span><span class="text-sm op-60">boundary</span><span class="text-emerald-400 font-bold">held</span>
</div>

<p class="text-sm op-70 mt-4">What held is the boundary: the permissions, the spend cap, the eval, the threshold you calibrated.</p>

<p class="source">vendor announcements and docs, read directly, September 2026: OpenAI, Anthropic, Google</p>

<!--
"What held is the boundary: the permissions, the spend cap, the eval, the threshold you calibrated. Nobody's model can absorb a constraint that is about your product."

"Guardrails and memory: the vendors are shipping them, and I don't think it's settled whether the boundary version or the vendor version wins. Unresolved."
-->

---

# Three questions, for a harness

1. Is this piece a **fact about your system** (a permission, a budget, a data boundary), or a **guess about what the model can't do yet**?
2. Is your budget **fixed**, or will the next model generation arrive before this ships?
3. Could the **next model** learn it, given the trend of the last three?

<p class="text-lg mt-6">A fact about your system lives at the boundary; build it and keep it. A guess about the model lives inside its job; build it if you must, label it, and make it unpluggable.</p>

<p class="text-orange-400 mt-6">The failure runs both ways: 2018 treated a guess as architecture. The other mistake is letting the model absorb something that should have stayed a hard boundary.</p>

<!--
"Crutches are fine. The 2018 systems in this talk were crutches and they shipped. Know which kind each one is. A guess about the model is temporary: put a seam around it so the day the API does it, you delete a file. A fact about your system is not temporary, and the risk there is the opposite one: the model gets good enough that you stop checking, and the constraint quietly moves inside."
-->

---
layout: center
class: text-center
---

<p class="text-3xl font-semibold">Put structure in training: the loss, the sampler, the curriculum.</p>

<p class="text-3xl font-semibold text-emerald-400 mt-6">Serve flat, quantised vectors in a standard index.</p>

<p class="text-3xl font-semibold text-emerald-400 mt-6">Keep the harness thin: the boundaries you own, the evals you trust, everything else replaceable.</p>

<!--
"Spend your structure budget at training time, where it compounds. Keep inference boring, flat, and quantised; that is the only shape the serving stack accepts. And when the model is somebody else's: own the boundary, measure the trust, and keep every crutch inside the model's job labelled and unplugged-able. My architecture didn't survive. My sampling scheme did. I wouldn't have guessed that in 2019, and I wouldn't have found out if I hadn't gone back and read my own thesis."

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

# Phase carries the shape

<p class="text-sm op-70 mb-2">Two photos, Fourier transformed, spectra swapped. Each hybrid looks like the image whose phase it carries: correlation 0.69 and 0.74 with the phase donor, 0.02 and 0.01 with the magnitude donor. Structure that is a fact about the data: the exact-constraint case of the decision rule.</p>

<img src="/figures/exp-e5-phase-swap.png" alt="magnitude and phase swap between the poodle and bird photos" class="max-h-75 mx-auto rounded" />

<p class="source">Oppenheim & Lim, 1981; own reproduction, talk/experiments/e5_phase_swap.py</p>

<!--
Only if asked "when is structure exact?". "The 1981 experiment. Keep one picture's magnitudes and the other's phases, invert. You see the phase donor every time. Complex-valued networks win on radar, MRI, and speech because phase is the signal; feed it as two real channels and independent scaling corrupts exactly this."
-->

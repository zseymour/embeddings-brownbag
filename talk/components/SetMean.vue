<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

interface VocabMetadata {
  provenance: {
    generated_by: string
    seed: number
    model: string
    dim: number
    vocab_source_file: string
  }
  vocab_size: number
  vocabulary: string[]
  vectors: {
    file: string
    dtype: string
    byte_order: string
    shape: [number, number]
    row_order_matches: string
    note: string
  }
  projection: {
    method: string
    explained_variance_ratio: number[]
    coordinates: [number, number][]
    coordinate_order_matches: string
    note: string
  }
  initial_example: {
    image: string
    tags: string[]
    note: string
  }
}

const META_URL = '/data/set-mean-vocab.json'
const VECTORS_URL = '/data/set-mean-vocab.f32'
const QUERY_IMAGE = '/figures/kcca-amigurumi-query.jpg'

const meta = ref<VocabMetadata | null>(null)
const vectors = ref<Float32Array | null>(null)
const loadError = ref<string | null>(null)
const selectedTags = ref<string[]>([])
const hoveredIndex = ref<number | null>(null)

const vocab = computed(() => meta.value?.vocabulary ?? [])
const coords = computed(() => meta.value?.projection.coordinates ?? [])
const dim = computed(() => meta.value?.provenance.dim ?? 0)

const wordIndex = computed(() => {
  const m = new Map<string, number>()
  vocab.value.forEach((w, i) => m.set(w, i))
  return m
})

const selectedIndices = computed(() => {
  const idx = wordIndex.value
  const out: number[] = []
  for (const tag of selectedTags.value) {
    const i = idx.get(tag)
    if (i !== undefined) out.push(i)
  }
  return out
})

function rowAt(i: number): Float32Array {
  const v = vectors.value!
  const d = dim.value
  return v.subarray(i * d, (i + 1) * d)
}

/**
 * Plain mean of the selected tags' raw bge-small-en-v1.5 vectors, computed in
 * the original 384-d space -- never from the 2-d projection below.
 */
const meanVector = computed(() => {
  const idxs = selectedIndices.value
  if (!vectors.value || idxs.length === 0) return null
  const d = dim.value
  const m = new Float64Array(d)
  for (const i of idxs) {
    const row = rowAt(i)
    for (let j = 0; j < d; j++) m[j] += row[j]
  }
  for (let j = 0; j < d; j++) m[j] /= idxs.length
  return m
})

function cosineSim(a: Float64Array, b: Float32Array): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let j = 0; j < a.length; j++) {
    dot += a[j] * b[j]
    na += a[j] * a[j]
    nb += b[j] * b[j]
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

/**
 * Nearest vocabulary word to the mean by cosine, searched over the full
 * 384-d vectors -- the 2-d plot never enters this calculation, it only
 * displays the result. Excludes the currently selected tags themselves:
 * they sit closest to their own mean by construction (it is their average),
 * which would make "nearest" trivial. What's informative is which *other*
 * vocabulary word the plain average ends up resembling.
 */
const nearestNeighbor = computed(() => {
  const m = meanVector.value
  if (!m || !vectors.value) return null
  const excluded = new Set(selectedIndices.value)
  let bestIndex = -1
  let bestCosine = -Infinity
  for (let i = 0; i < vocab.value.length; i++) {
    if (excluded.has(i)) continue
    const c = cosineSim(m, rowAt(i))
    if (c > bestCosine) {
      bestCosine = c
      bestIndex = i
    }
  }
  if (bestIndex === -1) return null
  return { index: bestIndex, word: vocab.value[bestIndex], cosine: bestCosine }
})

/**
 * PCA is a linear (centre, then project) map, so the projection of the mean
 * of a set of vectors equals the mean of their individual projections.
 * Averaging the already-computed 2-d coordinates therefore reproduces the
 * true projected mean without needing the PCA basis at runtime. Display
 * only -- the mean used for the nearest-neighbour search above is the
 * separate, exact 384-d mean.
 */
const projectedMean = computed<[number, number] | null>(() => {
  const idxs = selectedIndices.value
  const pts = coords.value
  if (idxs.length === 0 || pts.length === 0) return null
  let sx = 0
  let sy = 0
  for (const i of idxs) {
    sx += pts[i][0]
    sy += pts[i][1]
  }
  return [sx / idxs.length, sy / idxs.length]
})

/** Andrew's monotone chain convex hull; input need not be sorted. */
function convexHull(points: [number, number][]): [number, number][] {
  if (points.length < 3) return points
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const lower: [number, number][] = []
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop()
    lower.push(p)
  }
  const upper: [number, number][] = []
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop()
    upper.push(p)
  }
  lower.pop()
  upper.pop()
  return lower.concat(upper)
}

const hullPoints = computed<[number, number][]>(() => {
  const idxs = selectedIndices.value
  const pts = coords.value
  return convexHull(idxs.map((i) => pts[i]))
})

function addTag(word: string) {
  if (!selectedTags.value.includes(word)) selectedTags.value.push(word)
}

function removeTag(word: string) {
  selectedTags.value = selectedTags.value.filter((t) => t !== word)
}

function resetSelection() {
  if (meta.value) selectedTags.value = [...meta.value.initial_example.tags]
}

const hoveredWord = computed(() => (hoveredIndex.value !== null ? vocab.value[hoveredIndex.value] : null))

// -- canvas rendering --------------------------------------------------

const COLOR_BG = '#12151c'
const COLOR_BORDER = '#2a3140'
const COLOR_TEXT = '#e8ecf1'
const COLOR_DIM = '#98a2b3'
const COLOR_ACCENT = '#6fd3ff'
const COLOR_MEAN = '#e8894a'
const COLOR_NEAREST = '#ffd76f'

const W = 860
const H = 250
const MARGIN = 14

const canvasRef = ref<HTMLCanvasElement>()
let fontFamily = 'sans-serif'
let pointScreens: [number, number][] = []

// Fixed once the fixture loads: every vocabulary point's PCA coordinate
// bounding box, padded, so the plot never rescales as the selection changes.
const domain = computed(() => {
  const pts = coords.value
  if (!pts.length) return null
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const [x, y] of pts) {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  const padX = (maxX - minX) * 0.08 || 1
  const padY = (maxY - minY) * 0.08 || 1
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY }
})

function project(x: number, y: number): [number, number] {
  const d = domain.value!
  const plotX0 = MARGIN
  const plotX1 = W - MARGIN
  const plotY0 = MARGIN
  const plotY1 = H - MARGIN
  const px = plotX0 + ((x - d.minX) / (d.maxX - d.minX)) * (plotX1 - plotX0)
  const py = plotY1 - ((y - d.minY) / (d.maxY - d.minY)) * (plotY1 - plotY0)
  return [px, py]
}

function draw() {
  const canvas = canvasRef.value
  const ctx = canvas?.getContext('2d')
  const d = domain.value
  if (!canvas || !ctx || !d) return

  const dpr = window.devicePixelRatio || 1
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, W, H)
  ctx.fillStyle = COLOR_BG
  ctx.fillRect(0, 0, W, H)
  ctx.strokeStyle = COLOR_BORDER
  ctx.lineWidth = 1
  ctx.strokeRect(MARGIN / 2, MARGIN / 2, W - MARGIN, H - MARGIN)

  const pts = coords.value
  const selected = new Set(selectedIndices.value)
  const nn = nearestNeighbor.value

  pointScreens = new Array(pts.length)
  for (let i = 0; i < pts.length; i++) pointScreens[i] = project(pts[i][0], pts[i][1])

  ctx.fillStyle = COLOR_DIM
  ctx.font = `12px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  const modelName = meta.value?.provenance.model ?? ''
  ctx.fillText(`${pts.length.toLocaleString()} MIRFlickr vocabulary tags \u00b7 ${modelName} \u00b7 click to add`, MARGIN + 6, MARGIN + 16)

  // every vocabulary point, faint, except the two categories drawn below
  ctx.fillStyle = 'rgba(152, 162, 179, 0.45)'
  for (let i = 0; i < pointScreens.length; i++) {
    if (selected.has(i) || (nn && i === nn.index)) continue
    const [px, py] = pointScreens[i]
    ctx.beginPath()
    ctx.arc(px, py, 1.6, 0, Math.PI * 2)
    ctx.fill()
  }

  // convex hull of the selected set, projected
  const hull = hullPoints.value.map(([x, y]) => project(x, y))
  if (hull.length >= 2) {
    ctx.beginPath()
    ctx.moveTo(hull[0][0], hull[0][1])
    for (let i = 1; i < hull.length; i++) ctx.lineTo(hull[i][0], hull[i][1])
    if (hull.length >= 3) {
      ctx.closePath()
      ctx.fillStyle = 'rgba(111, 211, 255, 0.08)'
      ctx.fill()
    }
    ctx.strokeStyle = COLOR_ACCENT
    ctx.lineWidth = 1.2
    ctx.stroke()
  }

  // selected tag points
  ctx.fillStyle = COLOR_ACCENT
  for (const i of selected) {
    const [px, py] = pointScreens[i]
    ctx.beginPath()
    ctx.arc(px, py, 4.2, 0, Math.PI * 2)
    ctx.fill()
  }

  // projected mean and its nearest unselected vocabulary neighbour
  const pm = projectedMean.value
  if (nn) {
    const [nx, ny] = pointScreens[nn.index]
    if (pm) {
      const [mx, my] = project(pm[0], pm[1])
      ctx.strokeStyle = 'rgba(255, 215, 111, 0.65)'
      ctx.lineWidth = 1.2
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(mx, my)
      ctx.lineTo(nx, ny)
      ctx.stroke()
      ctx.setLineDash([])
    }
    ctx.strokeStyle = COLOR_NEAREST
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.arc(nx, ny, 6, 0, Math.PI * 2)
    ctx.stroke()
    ctx.fillStyle = COLOR_NEAREST
    ctx.beginPath()
    ctx.arc(nx, ny, 2.6, 0, Math.PI * 2)
    ctx.fill()
  }

  if (pm) {
    const [mx, my] = project(pm[0], pm[1])
    ctx.fillStyle = COLOR_MEAN
    ctx.beginPath()
    ctx.moveTo(mx - 6, my)
    ctx.lineTo(mx, my - 6)
    ctx.lineTo(mx + 6, my)
    ctx.lineTo(mx, my + 6)
    ctx.closePath()
    ctx.fill()
  }

  // hover ring
  if (hoveredIndex.value !== null) {
    const [hx, hy] = pointScreens[hoveredIndex.value]
    ctx.strokeStyle = COLOR_TEXT
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(hx, hy, 7, 0, Math.PI * 2)
    ctx.stroke()
  }
}

/**
 * Maps a mouse event to the canvas's logical (pre-devicePixelRatio,
 * pre-Slidev-scale) drawing coordinates via getBoundingClientRect, whose
 * size already reflects every ancestor CSS transform (including Slidev's
 * slide-fit scaling). This keeps hit-testing correct regardless of DPR or
 * how small/large the slide is currently rendered.
 */
function eventToLogical(evt: MouseEvent): [number, number] {
  const canvas = canvasRef.value!
  const rect = canvas.getBoundingClientRect()
  const x = ((evt.clientX - rect.left) / rect.width) * W
  const y = ((evt.clientY - rect.top) / rect.height) * H
  return [x, y]
}

const HIT_RADIUS = 7

function nearestPointTo(x: number, y: number): number | null {
  let best = -1
  let bestDist = HIT_RADIUS * HIT_RADIUS
  for (let i = 0; i < pointScreens.length; i++) {
    const [px, py] = pointScreens[i]
    const dx = px - x
    const dy = py - y
    const dist = dx * dx + dy * dy
    if (dist < bestDist) {
      bestDist = dist
      best = i
    }
  }
  return best === -1 ? null : best
}

function onCanvasClick(evt: MouseEvent) {
  const [x, y] = eventToLogical(evt)
  const i = nearestPointTo(x, y)
  if (i !== null) addTag(vocab.value[i])
}

function onCanvasMove(evt: MouseEvent) {
  const [x, y] = eventToLogical(evt)
  hoveredIndex.value = nearestPointTo(x, y)
}

function onCanvasLeave() {
  hoveredIndex.value = null
}

function setupCanvasResolution() {
  const canvas = canvasRef.value
  if (!canvas) return
  const dpr = window.devicePixelRatio || 1
  canvas.width = W * dpr
  canvas.height = H * dpr
  fontFamily = getComputedStyle(canvas).fontFamily || fontFamily
}

onMounted(async () => {
  try {
    const [metaRes, vectorsRes] = await Promise.all([fetch(META_URL), fetch(VECTORS_URL)])
    if (!metaRes.ok) throw new Error(`fixture metadata fetch failed: ${metaRes.status}`)
    if (!vectorsRes.ok) throw new Error(`fixture vectors fetch failed: ${vectorsRes.status}`)
    const metaJson = (await metaRes.json()) as VocabMetadata
    const buf = await vectorsRes.arrayBuffer()
    meta.value = metaJson
    vectors.value = new Float32Array(buf)
    selectedTags.value = [...metaJson.initial_example.tags]
  } catch (err) {
    loadError.value = err instanceof Error ? err.message : String(err)
    return
  }
  setupCanvasResolution()
  draw()
})

watch([selectedTags, hoveredIndex, meta], () => draw(), { deep: true, flush: 'sync' })
</script>

<template>
  <div class="set-mean-root">
    <div class="plot-wrap">
      <canvas
        ref="canvasRef"
        width="860"
        height="250"
        @click="onCanvasClick"
        @mousemove="onCanvasMove"
        @mouseleave="onCanvasLeave"
      ></canvas>
      <img class="query-thumb" :src="QUERY_IMAGE" alt="Query photo: a yellow crocheted amigurumi bird" />
      <div v-if="loadError" class="load-error">Failed to load vocabulary fixture: {{ loadError }}</div>
    </div>
    <div class="controls">
      <div class="caption-row">
        <span class="pill pill-proj">PCA for display</span>
        <span class="pill pill-search">cosine search in 384 dimensions</span>
        <span class="hover-label">{{ hoveredWord ? `hovering: ${hoveredWord}` : 'hover a point to inspect' }}</span>
      </div>
      <div class="status-row">
        <span v-if="nearestNeighbor" class="nearest">
          nearest to mean: <strong>{{ nearestNeighbor.word }}</strong>
          <span class="muted">(cosine {{ nearestNeighbor.cosine.toFixed(3) }}, excludes selected tags)</span>
        </span>
        <span v-else class="nearest muted">select at least one tag to compute a mean</span>
      </div>
      <div class="chips-row">
        <button
          v-for="tag in selectedTags"
          :key="tag"
          class="chip"
          :title="`remove ${tag}`"
          @click="removeTag(tag)"
        >
          {{ tag }} <span class="x">&times;</span>
        </button>
        <button class="reset" @click="resetSelection">reset to amigurumi example</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.set-mean-root {
  width: 860px;
  height: 400px;
  background: #12151c;
  color: #e8ecf1;
  font-family: inherit;
  display: flex;
  flex-direction: column;
}

.plot-wrap {
  position: relative;
  width: 860px;
  height: 250px;
}

.plot-wrap canvas {
  width: 860px;
  height: 250px;
  display: block;
  cursor: crosshair;
}

.query-thumb {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 56px;
  height: 56px;
  object-fit: cover;
  border-radius: 4px;
  border: 1px solid #2a3140;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
}

.load-error {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #e8894a;
  font-size: 13px;
  padding: 0 20px;
  text-align: center;
  background: rgba(18, 21, 28, 0.9);
}

.controls {
  box-sizing: border-box;
  width: 860px;
  height: 150px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 14px;
  border-top: 1px solid #2a3140;
}

.caption-row {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: #98a2b3;
}

.pill {
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid #2a3140;
  white-space: nowrap;
}

.pill-proj {
  color: #e8894a;
}

.pill-search {
  color: #6fd3ff;
}

.hover-label {
  margin-left: auto;
  font-style: italic;
}

.status-row {
  font-size: 14px;
}

.nearest strong {
  color: #ffd76f;
}

.muted {
  color: #98a2b3;
  font-size: 12px;
}

.chips-row {
  display: flex;
  flex-wrap: wrap;
  align-content: flex-start;
  gap: 6px;
  overflow-y: auto;
  flex: 1;
}

.chip {
  background: rgba(111, 211, 255, 0.12);
  border: 1px solid #6fd3ff;
  color: #e8ecf1;
  border-radius: 12px;
  padding: 3px 8px 3px 10px;
  font-size: 13px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.chip .x {
  color: #98a2b3;
  font-size: 12px;
}

.chip:hover {
  border-color: #e8894a;
}

.chip:hover .x {
  color: #e8894a;
}

.reset {
  background: transparent;
  border: 1px solid #2a3140;
  color: #98a2b3;
  border-radius: 12px;
  padding: 3px 10px;
  font-size: 12px;
  cursor: pointer;
  height: fit-content;
}

.reset:hover {
  color: #e8ecf1;
  border-color: #6fd3ff;
}
</style>

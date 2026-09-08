<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { sampleEmbeddings, cosine } from './embeddingSample'

const DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
const N = 400
const K = 10
const SEED = 7

const COLOR_BG = '#12151c'
const COLOR_BORDER = '#2a3140'
const COLOR_TEXT = '#e8ecf1'
const COLOR_DIM = '#98a2b3'
const COLOR_CALM = '#54d6b4'
const COLOR_WARN = '#e8894a'
const COLOR_ACCENT = '#6fd3ff'

const dimIndex = ref(0)
const d = computed(() => DIMS[dimIndex.value])

const canvas = ref<HTMLCanvasElement>()

interface Stats {
  nk: Int32Array
  maxNk: number
  hubIndex: number
  typicalIndex: number
  hubCos: Float64Array
  typicalCos: Float64Array
  threshold: number
  hubPasses: number
  typicalPasses: number
  topShare: number
  skew: number
}

const stats = ref<Stats | null>(null)

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

function percentile(sorted: Float64Array, p: number): number {
  const idx = p * (sorted.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  const frac = idx - lo
  return sorted[lo] * (1 - frac) + sorted[hi] * frac
}

function computeStats(dim: number): Stats {
  const points = sampleEmbeddings(N, dim, SEED)

  // full pairwise cosine matrix, symmetric
  const sim: Float64Array[] = new Array(N)
  for (let i = 0; i < N; i++) sim[i] = new Float64Array(N)
  const pairs = new Float64Array((N * (N - 1)) / 2)
  let pairPos = 0
  for (let i = 0; i < N; i++) {
    sim[i][i] = 1
    for (let j = i + 1; j < N; j++) {
      const c = cosine(points[i], points[j])
      sim[i][j] = c
      sim[j][i] = c
      pairs[pairPos++] = c
    }
  }
  pairs.sort()
  const threshold = percentile(pairs, 0.9)

  // k nearest neighbours per point (by cosine, excluding self); Nk[j] counts
  // how many other points' k-NN lists contain j.
  const nk = new Int32Array(N)
  const idxBuf = new Array<number>(N - 1)
  for (let i = 0; i < N; i++) {
    const row = sim[i]
    let p = 0
    for (let j = 0; j < N; j++) if (j !== i) idxBuf[p++] = j
    idxBuf.sort((a, b) => row[b] - row[a])
    for (let t = 0; t < K; t++) nk[idxBuf[t]]++
  }

  const maxNk = Math.max(...nk)
  let hubIndex = 0
  for (let i = 1; i < N; i++) if (nk[i] > nk[hubIndex]) hubIndex = i

  const med = median(Array.from(nk))
  let typicalIndex = -1
  let bestDist = Infinity
  for (let i = 0; i < N; i++) {
    if (i === hubIndex) continue
    const dist = Math.abs(nk[i] - med)
    if (dist < bestDist) {
      bestDist = dist
      typicalIndex = i
    }
  }

  const hubCos = new Float64Array(N - 1)
  const typicalCos = new Float64Array(N - 1)
  let hp = 0
  for (let j = 0; j < N; j++) if (j !== hubIndex) hubCos[hp++] = sim[hubIndex][j]
  let tp = 0
  for (let j = 0; j < N; j++) if (j !== typicalIndex) typicalCos[tp++] = sim[typicalIndex][j]

  let hubPasses = 0
  for (let i = 0; i < hubCos.length; i++) if (hubCos[i] > threshold) hubPasses++
  let typicalPasses = 0
  for (let i = 0; i < typicalCos.length; i++) if (typicalCos[i] > threshold) typicalPasses++

  const sortedDesc = Array.from(nk).sort((a, b) => b - a)
  const topCount = Math.ceil(0.05 * N)
  const topSum = sortedDesc.slice(0, topCount).reduce((a, b) => a + b, 0)
  const topShare = (topSum / (N * K)) * 100

  const meanNk = nk.reduce((a, b) => a + b, 0) / N
  let m2 = 0
  let m3 = 0
  for (let i = 0; i < N; i++) {
    const diff = nk[i] - meanNk
    m2 += diff * diff
    m3 += diff * diff * diff
  }
  m2 /= N
  m3 /= N
  const skew = m3 / Math.pow(m2, 1.5)

  return { nk, maxNk, hubIndex, typicalIndex, hubCos, typicalCos, threshold, hubPasses, typicalPasses, topShare, skew }
}

const CANVAS_W = 860
const CANVAS_H = 330
const LEFT_X = 10
const LEFT_W = 480
const RIGHT_X = 510
const RIGHT_W = 340

function draw() {
  const el = canvas.value
  const s = stats.value
  if (!el || !s) return
  const ctx = el.getContext('2d')
  if (!ctx) return
  const dpr = window.devicePixelRatio || 1
  if (el.width !== CANVAS_W * dpr || el.height !== CANVAS_H * dpr) {
    el.width = CANVAS_W * dpr
    el.height = CANVAS_H * dpr
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H)
  ctx.fillStyle = COLOR_BG
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H)

  const fontFamily = getComputedStyle(el).fontFamily || 'sans-serif'
  drawHistogram(ctx, s, fontFamily)
  drawStrips(ctx, s, fontFamily)
}

function drawHistogram(ctx: CanvasRenderingContext2D, s: Stats, fontFamily: string) {
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = COLOR_TEXT
  ctx.font = `bold 14px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.fillText('How often each point is someone\u2019s neighbour', LEFT_X, 18)

  const plotX0 = LEFT_X + 6
  const plotX1 = LEFT_X + LEFT_W - 10
  const plotY0 = 34
  const plotY1 = 290
  const plotW = plotX1 - plotX0
  const plotH = plotY1 - plotY0

  const xMax = Math.max(s.maxNk, K) + 1
  const counts = new Int32Array(xMax + 1)
  for (let i = 0; i < s.nk.length; i++) counts[s.nk[i]]++
  const maxCount = Math.max(...counts)

  const barW = plotW / (xMax + 1)
  const hubNk = s.nk[s.hubIndex]

  for (let v = 0; v <= xMax; v++) {
    const c = counts[v]
    if (c === 0) continue
    const h = (c / maxCount) * plotH
    const x = plotX0 + v * barW
    ctx.fillStyle = v === hubNk ? COLOR_WARN : COLOR_ACCENT
    ctx.fillRect(x, plotY1 - h, Math.max(barW - 1, 1), h)
  }

  // axis line
  ctx.strokeStyle = COLOR_BORDER
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(plotX0, plotY1)
  ctx.lineTo(plotX1, plotY1)
  ctx.stroke()

  // dashed line at k = 10
  const kx = plotX0 + K * barW + barW / 2
  ctx.save()
  ctx.setLineDash([4, 3])
  ctx.strokeStyle = COLOR_DIM
  ctx.beginPath()
  ctx.moveTo(kx, plotY0)
  ctx.lineTo(kx, plotY1)
  ctx.stroke()
  ctx.restore()
  ctx.fillStyle = COLOR_DIM
  ctx.font = `13px ${fontFamily}`
  ctx.textAlign = 'center'
  ctx.fillText('k = 10', kx, plotY0 - 4)

  // x-axis ticks
  ctx.fillStyle = COLOR_DIM
  ctx.font = `13px ${fontFamily}`
  ctx.textAlign = 'center'
  const tickVals = [0, Math.round(xMax / 2), xMax]
  for (const v of tickVals) {
    const x = plotX0 + v * barW + barW / 2
    ctx.fillText(String(v), x, plotY1 + 16)
  }
  ctx.textAlign = 'left'
  ctx.fillText('Nk (times chosen as a neighbour)', plotX0, plotY1 + 32)
}

function drawStrips(ctx: CanvasRenderingContext2D, s: Stats, fontFamily: string) {
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = COLOR_TEXT
  ctx.font = `bold 14px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.fillText('Scores against a hub vs. a typical point', RIGHT_X, 18)

  const plotX0 = RIGHT_X + 6
  const plotX1 = RIGHT_X + RIGHT_W - 8
  const plotW = plotX1 - plotX0
  const mapX = (v: number) => plotX0 + v * plotW

  const hubLabelY = 46
  const hubStripY0 = 54
  const hubStripY1 = 78
  const typicalLabelY = 150
  const typicalStripY0 = 158
  const typicalStripY1 = 182
  const axisY = 292

  // hub strip label + count
  ctx.font = `14px ${fontFamily}`
  ctx.fillStyle = COLOR_WARN
  ctx.textAlign = 'left'
  ctx.fillText(`top hub, in ${s.nk[s.hubIndex]} lists`, plotX0, hubLabelY)
  ctx.fillStyle = COLOR_DIM
  ctx.font = `13px ${fontFamily}`
  ctx.textAlign = 'right'
  ctx.fillText(`passes for ${s.hubPasses} queries`, plotX1, hubLabelY)

  // typical strip label + count
  ctx.font = `14px ${fontFamily}`
  ctx.fillStyle = COLOR_ACCENT
  ctx.textAlign = 'left'
  ctx.fillText(`typical point, in ${s.nk[s.typicalIndex]} lists`, plotX0, typicalLabelY)
  ctx.fillStyle = COLOR_DIM
  ctx.font = `13px ${fontFamily}`
  ctx.textAlign = 'right'
  ctx.fillText(`passes for ${s.typicalPasses} queries`, plotX1, typicalLabelY)

  // strip backgrounds
  ctx.strokeStyle = COLOR_BORDER
  ctx.lineWidth = 1
  ctx.strokeRect(plotX0, hubStripY0, plotW, hubStripY1 - hubStripY0)
  ctx.strokeRect(plotX0, typicalStripY0, plotW, typicalStripY1 - typicalStripY0)

  // ticks
  ctx.strokeStyle = COLOR_WARN
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let i = 0; i < s.hubCos.length; i++) {
    const x = mapX(Math.min(Math.max(s.hubCos[i], 0), 1))
    ctx.moveTo(x, hubStripY0 + 2)
    ctx.lineTo(x, hubStripY1 - 2)
  }
  ctx.stroke()

  ctx.strokeStyle = COLOR_ACCENT
  ctx.beginPath()
  for (let i = 0; i < s.typicalCos.length; i++) {
    const x = mapX(Math.min(Math.max(s.typicalCos[i], 0), 1))
    ctx.moveTo(x, typicalStripY0 + 2)
    ctx.lineTo(x, typicalStripY1 - 2)
  }
  ctx.stroke()

  // shared x-axis line and ticks
  ctx.strokeStyle = COLOR_BORDER
  ctx.beginPath()
  ctx.moveTo(plotX0, axisY)
  ctx.lineTo(plotX1, axisY)
  ctx.stroke()
  ctx.fillStyle = COLOR_DIM
  ctx.font = `13px ${fontFamily}`
  ctx.textAlign = 'center'
  for (const v of [0, 0.5, 1]) {
    ctx.fillText(v.toFixed(1), mapX(v), axisY + 16)
  }
  ctx.fillText('cosine similarity', (plotX0 + plotX1) / 2, axisY + 32)

  // threshold dashed line spanning both strips; label sits in the empty gap
  // between the two strips, flipped to whichever side has room.
  const tx = mapX(Math.min(Math.max(s.threshold, 0), 1))
  ctx.save()
  ctx.setLineDash([4, 3])
  ctx.strokeStyle = COLOR_TEXT
  ctx.beginPath()
  ctx.moveTo(tx, hubStripY0 - 6)
  ctx.lineTo(tx, axisY)
  ctx.stroke()
  ctx.restore()
  ctx.fillStyle = COLOR_TEXT
  ctx.font = `13px ${fontFamily}`
  const thresholdLabelY = (typicalStripY1 + axisY) / 2
  if (tx > plotX0 + plotW * 0.65) {
    ctx.textAlign = 'right'
    ctx.fillText('threshold', tx - 6, thresholdLabelY)
  } else {
    ctx.textAlign = 'left'
    ctx.fillText('threshold', tx + 6, thresholdLabelY)
  }
}

function recompute() {
  stats.value = computeStats(d.value)
  draw()
}

function onInput() {
  recompute()
}

const shareText = computed(() => (stats.value ? stats.value.topShare.toFixed(1) : '0.0'))
const skewText = computed(() => (stats.value ? stats.value.skew.toFixed(2) : '0.00'))
const shareWarn = computed(() => (stats.value ? stats.value.topShare > 15 : false))

onMounted(() => {
  recompute()
})
</script>

<template>
  <div class="hubness-demo">
    <canvas ref="canvas" :width="CANVAS_W" :height="CANVAS_H" :style="{ width: CANVAS_W + 'px', height: CANVAS_H + 'px' }"></canvas>
    <div class="controls">
      <div class="control-block">
        <div class="control-label">dimension <span class="readout-inline">d = {{ d }}</span></div>
        <input
          type="range"
          min="0"
          :max="DIMS.length - 1"
          step="1"
          v-model.number="dimIndex"
          @input="onInput"
          @keydown.stop
        />
      </div>
      <div class="sentence" :class="{ warn: shareWarn }">
        top 5% of points hold <span class="readout-inline">{{ shareText }}%</span> of neighbour slots
      </div>
      <div class="sentence">
        skew <span class="readout-inline">{{ skewText }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.hubness-demo {
  width: 860px;
  height: 400px;
  background: #12151c;
  color: #e8ecf1;
  font-family: inherit;
}

canvas {
  display: block;
}

.controls {
  height: 70px;
  display: flex;
  align-items: center;
  gap: 28px;
  padding: 0 10px;
}

.control-block {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 220px;
}

.control-label {
  font-size: 13px;
  color: #98a2b3;
}

.control-block input[type='range'] {
  width: 100%;
}

.sentence {
  font-size: 14px;
  color: #e8ecf1;
  max-width: 260px;
}

.sentence.warn .readout-inline {
  color: #e8894a;
}

.readout-inline {
  font-size: 22px;
  font-weight: bold;
  font-variant-numeric: tabular-nums;
  color: #e8ecf1;
}
</style>

<script setup lang="ts">
import { onMounted, ref, computed, watch } from 'vue'
import { sampleEmbeddings, cosine, mean } from './embeddingSample'

const POOL_SIZE = 400
const DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512]

const nSize = ref(6)
const dimIndex = ref(0)
const d = computed(() => DIMS[dimIndex.value])

// Fixed seed and pool size: only the dimension changes what gets sampled, so
// the pool is memoised on `d` and stable across every n on the same slide.
const pool = computed(() => sampleEmbeddings(POOL_SIZE, d.value, 11))
const members = computed(() => pool.value.slice(0, nSize.value))
const strangers = computed(() => pool.value.slice(nSize.value))

const stats = computed(() => {
  const mem = members.value
  const str = strangers.value
  const m = mean(mem)
  let sumSq = 0
  for (let j = 0; j < m.length; j++) sumSq += m[j] * m[j]
  const normM = Math.sqrt(sumSq)
  const mhat = m.map((v) => v / normM)

  let meanToStrangers = 0
  for (const s of str) meanToStrangers += cosine(mhat, s)
  meanToStrangers /= str.length

  let maxMemberToStrangers = -Infinity
  for (const v of mem) {
    let acc = 0
    for (const s of str) acc += cosine(v, s)
    maxMemberToStrangers = Math.max(maxMemberToStrangers, acc / str.length)
  }

  let meanToMembers = 0
  for (const v of mem) meanToMembers += cosine(mhat, v)
  meanToMembers /= mem.length

  return {
    m,
    normM,
    refNorm: 1 / Math.sqrt(nSize.value),
    meanToStrangers,
    maxMemberToStrangers,
    meanToMembers,
    hubby: meanToStrangers > maxMemberToStrangers,
  }
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

const COLOR_BG = '#12151c'
const COLOR_BORDER = '#2a3140'
const COLOR_TEXT = '#e8ecf1'
const COLOR_DIM = '#98a2b3'
const COLOR_WARN = '#e8894a'
const COLOR_ACCENT = '#6fd3ff'

const W = 860
const H = 330
const LEFT_X = 0
const LEFT_W = 440
const RIGHT_X = 460
const RIGHT_W = 400

const canvasRef = ref<HTMLCanvasElement>()
let fontFamily = 'sans-serif'

function drawLeftPanel(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = COLOR_TEXT
  ctx.font = `600 15px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.textBaseline = 'alphabetic'
  ctx.fillText('n members, their hull, and their mean', LEFT_X + 14, 20)

  const plotTop = 34
  const plotSize = Math.min(LEFT_W - 28, H - plotTop - 24)
  const cx = LEFT_X + LEFT_W / 2
  const cy = plotTop + plotSize / 2
  const domainR = 1.15
  const scale = plotSize / 2 / domainR
  const project = (x: number, y: number): [number, number] => [cx + x * scale, cy - y * scale]

  // unit circle, faint: every point sits on it exactly when d = 2
  ctx.strokeStyle = 'rgba(232, 236, 241, 0.18)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.arc(cx, cy, scale, 0, Math.PI * 2)
  ctx.stroke()

  ctx.fillStyle = 'rgba(152, 162, 179, 0.55)'
  for (const v of strangers.value) {
    const [px, py] = project(v[0], v[1])
    ctx.beginPath()
    ctx.arc(px, py, 1.8, 0, Math.PI * 2)
    ctx.fill()
  }

  const memberPx: [number, number][] = members.value.map((v) => project(v[0], v[1]))
  const hull = convexHull(memberPx)
  ctx.strokeStyle = COLOR_ACCENT
  ctx.lineWidth = 1.2
  ctx.beginPath()
  if (hull.length >= 3) {
    ctx.moveTo(hull[0][0], hull[0][1])
    for (let i = 1; i < hull.length; i++) ctx.lineTo(hull[i][0], hull[i][1])
    ctx.closePath()
  } else if (hull.length === 2) {
    ctx.moveTo(hull[0][0], hull[0][1])
    ctx.lineTo(hull[1][0], hull[1][1])
  }
  ctx.stroke()

  ctx.fillStyle = COLOR_ACCENT
  for (const [px, py] of memberPx) {
    ctx.beginPath()
    ctx.arc(px, py, 3.5, 0, Math.PI * 2)
    ctx.fill()
  }

  const [mx, my] = project(stats.value.m[0], stats.value.m[1])
  ctx.strokeStyle = COLOR_WARN
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.moveTo(cx, cy)
  ctx.lineTo(mx, my)
  ctx.stroke()
  ctx.fillStyle = COLOR_WARN
  ctx.beginPath()
  ctx.arc(mx, my, 5, 0, Math.PI * 2)
  ctx.fill()

  if (d.value > 2) {
    ctx.fillStyle = COLOR_DIM
    ctx.font = `12px ${fontFamily}`
    ctx.fillText('first two of d coordinates', LEFT_X + 14, H - 8)
  }
}

function drawBar(
  ctx: CanvasRenderingContext2D,
  opts: { top: number; label: string; value: number; color: string; tick?: { at: number; label: string } },
) {
  const trackX0 = RIGHT_X + 14
  const trackX1 = RIGHT_X + RIGHT_W - 70
  const trackW = trackX1 - trackX0
  const trackY0 = opts.top + 20
  const trackH = 24

  ctx.fillStyle = COLOR_DIM
  ctx.font = `13px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.fillText(opts.label, trackX0, opts.top + 12)

  ctx.fillStyle = 'rgba(255, 255, 255, 0.04)'
  ctx.fillRect(trackX0, trackY0, trackW, trackH)
  ctx.strokeStyle = COLOR_BORDER
  ctx.lineWidth = 1
  ctx.strokeRect(trackX0, trackY0, trackW, trackH)

  const fillW = Math.max(0, Math.min(1, opts.value)) * trackW
  ctx.fillStyle = opts.color
  ctx.fillRect(trackX0, trackY0, fillW, trackH)

  ctx.fillStyle = opts.color
  ctx.font = `600 14px ${fontFamily}`
  ctx.textAlign = 'right'
  ctx.fillText(opts.value.toFixed(3), RIGHT_X + RIGHT_W - 14, trackY0 + trackH - 6)

  if (opts.tick) {
    const tickX = trackX0 + Math.max(0, Math.min(1, opts.tick.at)) * trackW
    ctx.strokeStyle = COLOR_TEXT
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(tickX, trackY0 - 4)
    ctx.lineTo(tickX, trackY0 + trackH + 4)
    ctx.stroke()
    ctx.fillStyle = COLOR_DIM
    ctx.font = `11px ${fontFamily}`
    ctx.textAlign = 'center'
    ctx.fillText(opts.tick.label, tickX, trackY0 + trackH + 16)
  }
}

function drawRightPanel(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = COLOR_TEXT
  ctx.font = `600 15px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.fillText('Where the mean sits', RIGHT_X + 14, 20)

  ctx.fillStyle = COLOR_DIM
  ctx.font = `12px ${fontFamily}`
  ctx.fillText(`mean's cosine to its own members: ${stats.value.meanToMembers.toFixed(3)}`, RIGHT_X + 14, 38)

  const rowsTop = 54
  const rowHeight = 92
  drawBar(ctx, {
    top: rowsTop,
    label: '|mean|',
    value: stats.value.normM,
    color: COLOR_WARN,
    tick: { at: stats.value.refNorm, label: '1/sqrt(n)' },
  })
  drawBar(ctx, {
    top: rowsTop + rowHeight,
    label: "mean's cosine to strangers",
    value: stats.value.meanToStrangers,
    color: COLOR_WARN,
  })
  drawBar(ctx, {
    top: rowsTop + rowHeight * 2,
    label: "best member's cosine to strangers",
    value: stats.value.maxMemberToStrangers,
    color: COLOR_ACCENT,
  })
}

function draw() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const dpr = window.devicePixelRatio || 1
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, W, H)
  ctx.fillStyle = COLOR_BG
  ctx.fillRect(0, 0, W, H)
  ctx.strokeStyle = COLOR_BORDER
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(LEFT_X + LEFT_W + (RIGHT_X - (LEFT_X + LEFT_W)) / 2, 8)
  ctx.lineTo(LEFT_X + LEFT_W + (RIGHT_X - (LEFT_X + LEFT_W)) / 2, H - 8)
  ctx.stroke()

  drawLeftPanel(ctx)
  drawRightPanel(ctx)
}

onMounted(() => {
  const canvas = canvasRef.value
  if (!canvas) return
  const dpr = window.devicePixelRatio || 1
  canvas.width = W * dpr
  canvas.height = H * dpr
  fontFamily = getComputedStyle(canvas).fontFamily || fontFamily
  draw()
})

watch([nSize, dimIndex], () => draw(), { flush: 'sync' })
</script>

<template>
  <div class="set-mean-root">
    <canvas ref="canvasRef" width="860" height="330"></canvas>
    <div class="controls">
      <div class="control-group">
        <span class="label">set size n</span>
        <div class="row">
          <input type="range" min="2" max="64" step="1" v-model.number="nSize" @keydown.stop />
          <span class="value">{{ nSize }}</span>
        </div>
      </div>
      <div class="control-group">
        <span class="label">dimension</span>
        <div class="row">
          <input type="range" min="0" max="8" step="1" v-model.number="dimIndex" @keydown.stop />
          <span class="value">{{ d }}</span>
        </div>
      </div>
      <div class="verdict" :class="{ warn: stats.hubby }">
        mean is closer to strangers than any member: {{ stats.hubby ? 'yes' : 'no' }}
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

.set-mean-root canvas {
  width: 860px;
  height: 330px;
  display: block;
}

.controls {
  box-sizing: border-box;
  width: 860px;
  height: 70px;
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 0 14px;
  border-top: 1px solid #2a3140;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.control-group .label {
  font-size: 12px;
  color: #98a2b3;
}

.control-group .row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.control-group input[type='range'] {
  width: 150px;
  accent-color: #6fd3ff;
}

.control-group .value {
  font-size: 22px;
  font-weight: bold;
  font-variant-numeric: tabular-nums;
  min-width: 48px;
}

.verdict {
  margin-left: auto;
  font-size: 15px;
  color: #98a2b3;
}

.verdict.warn {
  color: #e8894a;
  font-weight: 600;
}
</style>

<script setup lang="ts">
import { onMounted, ref } from 'vue'

// ---- tree shape ------------------------------------------------------
// Branching factor 3, five levels below the root: 1 + 3 + 9 + 27 + 81 + 243 = 364 nodes.
const BRANCHING = 3
const MAX_DEPTH = 5
const DEFAULT_DEPTH = 4 // loads already showing the conclusion: plane crowds, disk doesn't
const HYP_DELTA = 1.0 // hyperbolic radial step per level (Poincare "t" units)
const GAP_FRAC = 0.08 // fraction of a cell reserved as a gap between adjacent siblings
const NODE_R = [7, 5.8, 4.8, 4.0, 3.4, 2.9] // node draw radius by depth

interface Point {
  x: number
  y: number
}

interface TreeNode {
  id: number
  parent: number
  depth: number
  angle: number
}

// build once, deterministically: each node's children fill its wedge with a
// fixed fractional gap between siblings, so branches stay visibly apart --
// the same wedge rule is reused unchanged for both geometries below.
const nodes: TreeNode[] = []
;(function build(parent: number, depth: number, a0: number, a1: number) {
  const id = nodes.length
  nodes.push({ id, parent, depth, angle: (a0 + a1) / 2 })
  if (depth < MAX_DEPTH) {
    const w = a1 - a0
    const cell = w / (BRANCHING + (BRANCHING - 1) * GAP_FRAC)
    const gap = GAP_FRAC * cell
    for (let i = 0; i < BRANCHING; i++) {
      const c0 = a0 + i * (cell + gap)
      build(id, depth + 1, c0, c0 + cell)
    }
  }
})(-1, 0, 0, Math.PI * 2)

const levelsByDepth: TreeNode[][] = []
for (let d = 0; d <= MAX_DEPTH; d++) levelsByDepth.push(nodes.filter((n) => n.depth === d))

// flat plane: radius proportional to depth (equal radial spacing per level)
function euclidUnit(n: TreeNode): Point {
  const r = n.depth / MAX_DEPTH
  return { x: r * Math.cos(n.angle), y: r * Math.sin(n.angle) }
}
// Poincare disk: constant hyperbolic step per level -> tanh(t/2) Euclidean radius
function hypUnit(n: TreeNode): Point {
  const t = n.depth * HYP_DELTA
  const r = Math.tanh(t / 2)
  return { x: r * Math.cos(n.angle), y: r * Math.sin(n.angle) }
}
function euclidDist(p: Point, q: Point): number {
  const dx = p.x - q.x
  const dy = p.y - q.y
  return Math.sqrt(dx * dx + dy * dy)
}
// standard Poincare-disk distance, points given in unit-disk coordinates
function poincareDist(p: Point, q: Point): number {
  const r1 = Math.min(p.x * p.x + p.y * p.y, 0.999999 * 0.999999)
  const r2 = Math.min(q.x * q.x + q.y * q.y, 0.999999 * 0.999999)
  const dx = p.x - q.x
  const dy = p.y - q.y
  const num = 2 * (dx * dx + dy * dy)
  const den = (1 - r1) * (1 - r2)
  return Math.acosh(Math.max(1, 1 + num / den))
}

// hyperbolic circle of hyperbolic radius r around a node: the Euclidean
// circle whose diameter endpoints sit on the node's radial line at
// hyperbolic distances t-r and t+r from the origin.
function hypRoomCircle(n: TreeNode): Point & { radius: number } {
  const t = n.depth * HYP_DELTA
  const r = 0.5 * HYP_DELTA
  const rIn = Math.tanh((t - r) / 2)
  const rOut = Math.tanh((t + r) / 2)
  const centerR = (rIn + rOut) / 2
  return { x: centerR * Math.cos(n.angle), y: centerR * Math.sin(n.angle), radius: (rOut - rIn) / 2 }
}

// circle through three points, by the standard circumcenter formula
function circumcenter(a: Point, b: Point, c: Point): Point {
  const d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
  const ux =
    ((a.x * a.x + a.y * a.y) * (b.y - c.y) + (b.x * b.x + b.y * b.y) * (c.y - a.y) + (c.x * c.x + c.y * c.y) * (a.y - b.y)) / d
  const uy =
    ((a.x * a.x + a.y * a.y) * (c.x - b.x) + (b.x * b.x + b.y * b.y) * (a.x - c.x) + (c.x * c.x + c.y * c.y) * (b.x - a.x)) / d
  return { x: ux, y: uy }
}

// the circle through p and q orthogonal to the unit circle is the circle
// through p, q, and the inversion of p (inversion in the unit circle maps
// an orthogonal circle to itself, so it must also pass through inv(p)).
// null when p, q, and the origin are collinear -- the geodesic is then the
// straight diameter through them.
function geodesicCircle(p: Point, q: Point): { c: Point; r: number } | null {
  const cross = p.x * q.y - p.y * q.x
  if (Math.abs(cross) < 1e-9) return null
  const normSqP = p.x * p.x + p.y * p.y
  const invp = { x: p.x / normSqP, y: p.y / normSqP }
  const c = circumcenter(p, q, invp)
  const r = Math.hypot(c.x - p.x, c.y - p.y)
  return { c, r }
}

const COLOR_BG = '#12151c'
const COLOR_BORDER = '#2a3140'
const COLOR_TEXT = '#e8ecf1'
const COLOR_DIM = '#98a2b3'

const CANVAS_W = 860
const CANVAS_H = 280
const LEFT_CX = 214
const RIGHT_CX = 646
const PANEL_CY = 140
const PANEL_R = 102
const TITLE_Y = 18
const READOUT_Y = 266

function nodeColor(depth: number, alpha: number): string {
  return `hsla(191,58%,${72 - depth * 7}%,${alpha})`
}

function strokeGeodesic(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  mode: 'euclid' | 'hyper',
  p: Point,
  q: Point,
  style: { stroke: string; width: number },
) {
  ctx.save()
  ctx.strokeStyle = style.stroke
  ctx.lineWidth = style.width
  const p1 = { x: cx + p.x * r, y: cy + p.y * r }
  const p2 = { x: cx + q.x * r, y: cy + q.y * r }
  ctx.beginPath()
  const g = mode === 'hyper' ? geodesicCircle(p, q) : null
  if (!g) {
    ctx.moveTo(p1.x, p1.y)
    ctx.lineTo(p2.x, p2.y)
  } else {
    const a1 = Math.atan2(p.y - g.c.y, p.x - g.c.x)
    const a2 = Math.atan2(q.y - g.c.y, q.x - g.c.x)
    let sweep = a2 - a1
    while (sweep < 0) sweep += Math.PI * 2
    while (sweep >= Math.PI * 2) sweep -= Math.PI * 2
    const mid = a1 + sweep / 2
    const midX = g.c.x + g.r * Math.cos(mid)
    const midY = g.c.y + g.r * Math.sin(mid)
    const forwardStaysInside = midX * midX + midY * midY < 1
    const cpx = { x: cx + g.c.x * r, y: cy + g.c.y * r }
    ctx.arc(cpx.x, cpx.y, g.r * r, a1, a2, !forwardStaysInside)
  }
  ctx.stroke()
  ctx.restore()
}

// crowding readout: the fraction of deepest-level nodes whose "room" (a
// disk of half an edge length, in that geometry's own ruler) overlaps a
// non-sibling's room. Distance and radius are both converted to
// edge-length units so plane and disk are compared on the same scale.
function overlapPct(mode: 'euclid' | 'hyper', depth: number): number {
  const unitFn = mode === 'euclid' ? euclidUnit : hypUnit
  const distFn = mode === 'euclid' ? euclidDist : poincareDist
  const scale = mode === 'euclid' ? MAX_DEPTH : 1 / HYP_DELTA
  const level = levelsByDepth[depth]
  let overlapCount = 0
  for (const a of level) {
    let overlap = false
    for (const b of level) {
      if (a.id === b.id || a.parent === b.parent) continue
      if (distFn(unitFn(a), unitFn(b)) * scale < 1.0) {
        overlap = true
        break
      }
    }
    if (overlap) overlapCount++
  }
  return level.length ? (overlapCount / level.length) * 100 : 0
}

const depth = ref(DEFAULT_DEPTH)
const canvas = ref<HTMLCanvasElement>()
let fontFamily = 'sans-serif'

function drawPanel(ctx: CanvasRenderingContext2D, mode: 'euclid' | 'hyper', cx: number, cy: number, d: number) {
  const unitFn = mode === 'euclid' ? euclidUnit : hypUnit

  if (mode === 'hyper') {
    ctx.beginPath()
    ctx.strokeStyle = 'rgba(232,236,241,0.12)'
    ctx.lineWidth = 1
    ctx.arc(cx, cy, PANEL_R, 0, Math.PI * 2)
    ctx.stroke()
  }

  // edges
  for (const n of nodes) {
    if (n.depth === 0 || n.depth > d) continue
    strokeGeodesic(ctx, cx, cy, PANEL_R, mode, unitFn(nodes[n.parent]), unitFn(n), {
      stroke: 'rgba(232,236,241,0.22)',
      width: 1,
    })
  }

  // room circles around every deepest-level node -- this is the crowding demo
  ctx.strokeStyle = 'rgba(111,211,255,0.35)'
  ctx.lineWidth = 1
  for (const n of levelsByDepth[d]) {
    if (mode === 'euclid') {
      const u = euclidUnit(n)
      const radiusUnit = 0.5 / MAX_DEPTH
      ctx.beginPath()
      ctx.arc(cx + u.x * PANEL_R, cy + u.y * PANEL_R, radiusUnit * PANEL_R, 0, Math.PI * 2)
      ctx.stroke()
    } else {
      const c = hypRoomCircle(n)
      ctx.beginPath()
      ctx.arc(cx + c.x * PANEL_R, cy + c.y * PANEL_R, c.radius * PANEL_R, 0, Math.PI * 2)
      ctx.stroke()
    }
  }

  // nodes
  for (const n of nodes) {
    if (n.depth > d) continue
    const u = unitFn(n)
    ctx.beginPath()
    ctx.fillStyle = nodeColor(n.depth, 1)
    ctx.arc(cx + u.x * PANEL_R, cy + u.y * PANEL_R, NODE_R[n.depth], 0, Math.PI * 2)
    ctx.fill()
  }
}

function drawReadout(ctx: CanvasRenderingContext2D, cx: number, label: string, pct: number) {
  ctx.textAlign = 'center'
  ctx.textBaseline = 'alphabetic'
  ctx.font = `bold 18px ${fontFamily}`
  ctx.fillStyle = COLOR_TEXT
  ctx.fillText(`${label} room overlap: ${Math.round(pct)}%`, cx, READOUT_Y)
}

function draw() {
  const el = canvas.value
  if (!el) return
  const ctx = el.getContext('2d')
  if (!ctx) return
  const dpr = window.devicePixelRatio || 1
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H)
  ctx.fillStyle = COLOR_BG
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H)

  const d = depth.value

  drawPanel(ctx, 'euclid', LEFT_CX, PANEL_CY, d)
  drawPanel(ctx, 'hyper', RIGHT_CX, PANEL_CY, d)

  ctx.strokeStyle = COLOR_BORDER
  ctx.lineWidth = 1
  ctx.strokeRect(0.5, 0.5, CANVAS_W - 1, 250)
  ctx.beginPath()
  ctx.moveTo(430, 8)
  ctx.lineTo(430, 250)
  ctx.stroke()

  ctx.fillStyle = COLOR_TEXT
  ctx.font = `bold 14px ${fontFamily}`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'alphabetic'
  ctx.fillText('Flat plane', LEFT_CX, TITLE_Y)
  ctx.fillText('Poincar\u00e9 disk', RIGHT_CX, TITLE_Y)

  drawReadout(ctx, LEFT_CX, 'plane', overlapPct('euclid', d))
  drawReadout(ctx, RIGHT_CX, 'disk', overlapPct('hyper', d))
}

function onDepthInput() {
  draw()
}

onMounted(() => {
  const el = canvas.value
  if (!el) return
  const dpr = window.devicePixelRatio || 1
  el.width = CANVAS_W * dpr
  el.height = CANVAS_H * dpr
  fontFamily = getComputedStyle(el).fontFamily || fontFamily
  draw()
})
</script>

<template>
  <div class="hierarchy-crowding-demo">
    <canvas
      ref="canvas"
      :width="CANVAS_W"
      :height="CANVAS_H"
      :style="{ width: CANVAS_W + 'px', height: CANVAS_H + 'px' }"
    ></canvas>
    <div class="controls">
      <div class="control-label">tree depth <span class="readout-inline">{{ depth }}</span></div>
      <input type="range" min="1" :max="MAX_DEPTH" step="1" v-model.number="depth" @input="onDepthInput" @keydown.stop />
    </div>
  </div>
</template>

<style scoped>
.hierarchy-crowding-demo {
  width: 860px;
  height: 340px;
  background: #12151c;
  color: #e8ecf1;
  font-family: inherit;
}

canvas {
  display: block;
}

.controls {
  box-sizing: border-box;
  width: 860px;
  height: 60px;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 14px;
  border-top: 1px solid #2a3140;
}

.control-label {
  font-size: 16px;
  color: #98a2b3;
  white-space: nowrap;
}

.controls input[type='range'] {
  flex: 1;
  accent-color: #6fd3ff;
}

.readout-inline {
  font-size: 22px;
  font-weight: bold;
  font-variant-numeric: tabular-nums;
  color: #e8ecf1;
}
</style>

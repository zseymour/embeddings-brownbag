<script setup lang="ts">
import { useSlideContext } from '@slidev/client'
import { onMounted, ref, computed, watch } from 'vue'

const { $clicks } = useSlideContext()

// ---- signal constants ----
const Z_RE = 1.0
const Z_IM = 0.6
const ROTOR_DEG_DEFAULT = 40
const W_RE_DEFAULT = 1.7
const W_IM_DEFAULT = 0.3
const ANIM_MS = 1100
const HARM_FREQ = [1, 2, 3, 4, 5]
const HARM_AMP = [1.0, 0.6, 0.4, 0.25, 0.15]
const HARM_PHASE = [0.6981, 1.3963, 2.1817, 3.8397, 5.4105]
const SAMPLES = 300

const mag = (re: number, im: number) => Math.hypot(re, im)
const angDeg = (re: number, im: number) => (Math.atan2(im, re) * 180) / Math.PI

const MAG0 = mag(Z_RE, Z_IM)
const ANG0 = angDeg(Z_RE, Z_IM)

// treats each harmonic's (amplitude, phase) as its own complex number and
// scales its real and imaginary parts independently by wRe, wIm
function harmonicCorruption(wRe: number, wIm: number) {
  const n = HARM_FREQ.length
  const amps = new Array<number>(n)
  const phases = new Array<number>(n)
  let maxErrDeg = 0
  for (let k = 0; k < n; k++) {
    const re = HARM_AMP[k] * Math.cos(HARM_PHASE[k]) * wRe
    const im = HARM_AMP[k] * Math.sin(HARM_PHASE[k]) * wIm
    amps[k] = mag(re, im)
    phases[k] = Math.atan2(im, re)
    let errDeg = ((phases[k] - HARM_PHASE[k]) * 180) / Math.PI
    errDeg = ((errDeg + 540) % 360) - 180 // shortest angular distance
    maxErrDeg = Math.max(maxErrDeg, Math.abs(errDeg))
  }
  return { amps, phases, maxErrDeg }
}

function synth(freqs: number[], amps: number[], phases: number[], n: number) {
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const t = (i / n) * 2 * Math.PI
    let v = 0
    for (let k = 0; k < freqs.length; k++) v += amps[k] * Math.cos(freqs[k] * t + phases[k])
    out[i] = v
  }
  return out
}

const REFERENCE_WAVE = synth(HARM_FREQ, HARM_AMP, HARM_PHASE, SAMPLES)

const COLOR_BG = '#12151c'
const COLOR_BORDER = '#2a3140'
const COLOR_TEXT = '#e8ecf1'
const COLOR_DIM = '#98a2b3'
const COLOR_CALM = '#54d6b4'
const COLOR_WARN = '#e8894a'
const COLOR_ACCENT = '#6fd3ff'

const CANVAS_W = 860
const CANVAS_H = 310
const LEFT_X = 0
const MID_X = 290
const RIGHT_X = 520
const RIGHT_W = 340
const PLANE_OX = LEFT_X + 95
const PLANE_OY = 175
const PLANE_S = 62

const canvasRef = ref<HTMLCanvasElement>()

// primary state: driven by $clicks, overridable live by the sliders
const stage = ref(0)
const rotorDeg = ref(ROTOR_DEG_DEFAULT)
const wRe = ref(W_RE_DEFAULT)
const wIm = ref(W_IM_DEFAULT)

// values actually drawn for the rotor and two-channel vectors: these tween in
// on the click that first reveals them, then track the sliders live
const drawnRotorDeg = ref(0)
const drawnWRe = ref(1)
const drawnWIm = ref(1)

let rafIds: number[] = []
function stopAnims() {
  rafIds.forEach(cancelAnimationFrame)
  rafIds = []
}
function tween(setter: (v: number) => void, from: number, to: number, dur: number) {
  const start = performance.now()
  function frame(now: number) {
    const t = Math.min(1, (now - start) / dur)
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
    setter(from + (to - from) * e)
    draw()
    if (t < 1) rafIds.push(requestAnimationFrame(frame))
  }
  rafIds.push(requestAnimationFrame(frame))
}

function syncStage(next: number, animated: boolean) {
  const prev = stage.value
  stage.value = next
  stopAnims()
  if (next >= 1) {
    if (prev < 1 && animated) tween((v) => (drawnRotorDeg.value = v), 0, rotorDeg.value, ANIM_MS)
    else drawnRotorDeg.value = rotorDeg.value
  } else {
    drawnRotorDeg.value = 0
  }
  if (next >= 2) {
    if (prev < 2 && animated) {
      tween((v) => (drawnWRe.value = v), 1, wRe.value, ANIM_MS)
      tween((v) => (drawnWIm.value = v), 1, wIm.value, ANIM_MS)
    } else {
      drawnWRe.value = wRe.value
      drawnWIm.value = wIm.value
    }
  } else {
    drawnWRe.value = 1
    drawnWIm.value = 1
  }
  draw()
}

// clicks are the authoritative source; a slider drag overrides stage until
// the next click change resyncs it
watch($clicks, (v) => syncStage(Math.max(0, Math.min(3, v)), true), { immediate: true })

function onRotorInput() {
  stopAnims()
  if (stage.value < 1) stage.value = 1
  drawnRotorDeg.value = rotorDeg.value
  draw()
}
function onWeightInput() {
  stopAnims()
  if (stage.value < 2) stage.value = 2
  drawnWRe.value = wRe.value
  drawnWIm.value = wIm.value
  draw()
}

const twoChannelRe = computed(() => Z_RE * drawnWRe.value)
const twoChannelIm = computed(() => Z_IM * drawnWIm.value)
const twoChannelLength = computed(() => mag(twoChannelRe.value, twoChannelIm.value))
const twoChannelAngle = computed(() => angDeg(twoChannelRe.value, twoChannelIm.value))
const phaseErrorDeg = computed(() => Math.abs(twoChannelAngle.value - ANG0))

function drawAxes(ctx: CanvasRenderingContext2D, ox: number, oy: number, rL: number, rR: number, rU: number, rD: number) {
  ctx.strokeStyle = COLOR_BORDER
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(ox - rL, oy)
  ctx.lineTo(ox + rR, oy)
  ctx.moveTo(ox, oy - rU)
  ctx.lineTo(ox, oy + rD)
  ctx.stroke()
}

function drawVec(
  ctx: CanvasRenderingContext2D,
  ox: number,
  oy: number,
  s: number,
  re: number,
  im: number,
  color: string,
  lw: number,
  dashed: boolean,
) {
  const x = ox + re * s
  const y = oy - im * s
  ctx.save()
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = lw
  if (dashed) ctx.setLineDash([6, 4])
  ctx.beginPath()
  ctx.moveTo(ox, oy)
  ctx.lineTo(x, y)
  ctx.stroke()
  if (!dashed) {
    const a = Math.atan2(oy - y, x - ox)
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x - 10 * Math.cos(a - 0.4), y + 10 * Math.sin(a - 0.4))
    ctx.lineTo(x - 10 * Math.cos(a + 0.4), y + 10 * Math.sin(a + 0.4))
    ctx.closePath()
    ctx.fill()
  }
  ctx.restore()
}

function drawPlane(ctx: CanvasRenderingContext2D, fontFamily: string) {
  ctx.textBaseline = 'alphabetic'
  ctx.fillStyle = COLOR_TEXT
  ctx.font = `bold 14px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.fillText(stage.value === 0 ? 'One complex number' : 'Multiply vs. scale channels', LEFT_X + 10, 18)

  drawAxes(ctx, PLANE_OX, PLANE_OY, 85, 185, 141, 125)

  if (stage.value === 0) {
    drawVec(ctx, PLANE_OX, PLANE_OY, PLANE_S, Z_RE, Z_IM, COLOR_ACCENT, 3, false)
    ctx.fillStyle = COLOR_DIM
    ctx.font = `13px ${fontFamily}`
    ctx.textAlign = 'left'
    const tipX = PLANE_OX + Z_RE * PLANE_S
    const tipY = PLANE_OY - Z_IM * PLANE_S
    ctx.fillText(`length ${MAG0.toFixed(2)}`, tipX + 10, tipY - 18)
    ctx.fillText(`angle ${ANG0.toFixed(1)}\u00b0`, tipX + 10, tipY)
    return
  }

  // beyond stage 0 the original vector becomes a reference; its numbers move
  // into the readout column instead of labelling it in place
  drawVec(ctx, PLANE_OX, PLANE_OY, PLANE_S, Z_RE, Z_IM, COLOR_ACCENT, 2, stage.value >= 2)

  const rotatedAngle = ANG0 + drawnRotorDeg.value
  const rotatedRe = MAG0 * Math.cos((rotatedAngle * Math.PI) / 180)
  const rotatedIm = MAG0 * Math.sin((rotatedAngle * Math.PI) / 180)
  drawVec(ctx, PLANE_OX, PLANE_OY, PLANE_S, rotatedRe, rotatedIm, COLOR_CALM, 3, false)

  if (stage.value >= 2) {
    drawVec(ctx, PLANE_OX, PLANE_OY, PLANE_S, twoChannelRe.value, twoChannelIm.value, COLOR_WARN, 3, false)
  }
}

function drawReadouts(ctx: CanvasRenderingContext2D, fontFamily: string) {
  if (stage.value < 1) return
  const x = MID_X + 16
  ctx.textAlign = 'left'

  ctx.fillStyle = COLOR_DIM
  ctx.font = `12px ${fontFamily}`
  ctx.fillText('ROTOR RESULT', x, 54)
  ctx.fillStyle = COLOR_CALM
  ctx.font = `600 16px ${fontFamily}`
  ctx.fillText(`length ${MAG0.toFixed(2)}, angle ${drawnRotorDeg.value.toFixed(0)}\u00b0`, x, 78)

  if (stage.value < 2) return
  ctx.fillStyle = COLOR_DIM
  ctx.font = `12px ${fontFamily}`
  ctx.fillText('TWO-CHANNEL RESULT', x, 140)
  ctx.fillStyle = COLOR_WARN
  ctx.font = `600 16px ${fontFamily}`
  ctx.fillText(`length ${twoChannelLength.value.toFixed(2)}, angle ${twoChannelAngle.value.toFixed(1)}\u00b0`, x, 164)
  ctx.font = `bold 22px ${fontFamily}`
  ctx.fillText(`phase error ${phaseErrorDeg.value.toFixed(1)}\u00b0`, x, 200)
}

function drawWave(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  ref: number[],
  sig: number[],
  color: string,
  caption: string,
  fontFamily: string,
) {
  ctx.strokeStyle = COLOR_BORDER
  ctx.lineWidth = 1
  ctx.strokeRect(x, y, w, h)
  const yRange = 3.2 // fixed vertical scale, deterministic
  const toXY = (i: number, arr: number[]): [number, number] => [
    x + (i / (arr.length - 1)) * w,
    y + h / 2 - (arr[i] / yRange) * (h / 2),
  ]
  ctx.beginPath()
  ctx.strokeStyle = COLOR_DIM
  ctx.lineWidth = 1.5
  ctx.setLineDash([5, 4])
  ref.forEach((_, i) => {
    const [px, py] = toXY(i, ref)
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  })
  ctx.stroke()
  ctx.setLineDash([])
  ctx.beginPath()
  ctx.strokeStyle = color
  ctx.lineWidth = 2.5
  sig.forEach((_, i) => {
    const [px, py] = toXY(i, sig)
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  })
  ctx.stroke()
  ctx.fillStyle = color
  ctx.font = `600 13px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.fillText(caption, x, y - 10)
}

function drawWaves(ctx: CanvasRenderingContext2D, fontFamily: string) {
  if (stage.value < 3) return
  // always driven by the live slider values, not the stage-2 entrance tween
  const hc = harmonicCorruption(wRe.value, wIm.value)
  const magSig = synth(HARM_FREQ, hc.amps, HARM_PHASE, SAMPLES)
  const phaseSig = synth(HARM_FREQ, HARM_AMP, hc.phases, SAMPLES)
  const waveX = RIGHT_X + 10
  const waveW = RIGHT_W - 20

  drawWave(ctx, waveX, 42, waveW, 96, REFERENCE_WAVE, magSig, COLOR_ACCENT, 'magnitudes perturbed, phase intact', fontFamily)
  drawWave(
    ctx,
    waveX,
    176,
    waveW,
    96,
    REFERENCE_WAVE,
    phaseSig,
    COLOR_WARN,
    `phase perturbed by up to ${hc.maxErrDeg.toFixed(1)}\u00b0, magnitudes intact`,
    fontFamily,
  )

  ctx.fillStyle = COLOR_DIM
  ctx.font = `11px ${fontFamily}`
  ctx.textAlign = 'left'
  ctx.fillText('grey dashed = original waveform, both sides', waveX, 292)
}

function draw() {
  const el = canvasRef.value
  if (!el) return
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
  drawPlane(ctx, fontFamily)
  drawReadouts(ctx, fontFamily)
  drawWaves(ctx, fontFamily)
}

onMounted(() => {
  draw()
})
</script>

<template>
  <div class="phase-corruption-root">
    <canvas
      ref="canvasRef"
      :width="CANVAS_W"
      :height="CANVAS_H"
      :style="{ width: CANVAS_W + 'px', height: CANVAS_H + 'px' }"
    ></canvas>
    <div class="controls">
      <div class="control-group" :class="{ dimmed: stage < 1 }">
        <span class="label">rotor angle</span>
        <div class="row">
          <input type="range" min="0" max="180" step="1" v-model.number="rotorDeg" @input="onRotorInput" @keydown.stop />
          <span class="value">{{ rotorDeg }}&deg;</span>
        </div>
      </div>
      <div class="control-group" :class="{ dimmed: stage < 2 }">
        <span class="label">real weight</span>
        <div class="row">
          <input type="range" min="0.1" max="2.0" step="0.01" v-model.number="wRe" @input="onWeightInput" @keydown.stop />
          <span class="value">{{ wRe.toFixed(2) }}</span>
        </div>
      </div>
      <div class="control-group" :class="{ dimmed: stage < 2 }">
        <span class="label">imag weight</span>
        <div class="row">
          <input type="range" min="0.1" max="2.0" step="0.01" v-model.number="wIm" @input="onWeightInput" @keydown.stop />
          <span class="value">{{ wIm.toFixed(2) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.phase-corruption-root {
  width: 860px;
  height: 400px;
  background: #12151c;
  color: #e8ecf1;
  font-family: inherit;
  display: flex;
  flex-direction: column;
}

canvas {
  display: block;
}

.controls {
  box-sizing: border-box;
  width: 860px;
  height: 90px;
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
  transition: opacity 0.2s;
}

.control-group.dimmed {
  opacity: 0.35;
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
  width: 140px;
}

.control-group:nth-child(1) input[type='range'] {
  accent-color: #54d6b4;
}

.control-group:nth-child(2) input[type='range'],
.control-group:nth-child(3) input[type='range'] {
  accent-color: #e8894a;
}

.control-group .value {
  font-size: 22px;
  font-weight: bold;
  font-variant-numeric: tabular-nums;
  min-width: 56px;
}
</style>

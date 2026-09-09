<script setup lang="ts">
import { computed, ref, useId } from 'vue'

// Two panels, one shared toy query q = (1, 0). Every document is a unit
// vector (cos theta, sin theta); the "score" is dot(q, doc) = cos(theta), so
// the acceptance test is exactly cos(theta) >= cutoff. Both panels place the
// SAME 21 identities (six relevant, fifteen irrelevant, matched by index) --
// only each document's angle differs between panels. Angles are fixed and
// deterministic; nothing here is randomised or re-laid-out on interaction.
const RELEVANT_MIXED = [-52, -32, -12, 12, 32, 52]
const RELEVANT_IMPROVED = [-18, -11, -4, 4, 11, 18]
const IRRELEVANT_MIXED = [-165, -140, -115, -90, -70, -44, -24, 0, 24, 44, 70, 90, 115, 140, 165]
const IRRELEVANT_IMPROVED = [-165, -140, -115, -95, -80, -68, -54, 36, 54, 68, 80, 95, 115, 140, 165]

const DEFAULT_CUTOFF = 0.94

// unique per component instance (hydration-safe) so the <marker> ids never
// collide when Slidev renders this component more than once (e.g. print mode)
const uid = useId()

// ---- geometry (pure SVG, viewBox 0 0 160 160, unit circle centered at 80,80) --
const CX = 80
const CY = 80
const R = 66

function polar(angleDeg: number, radius: number): { x: number; y: number } {
  const rad = (angleDeg * Math.PI) / 180
  return { x: CX + radius * Math.cos(rad), y: CY - radius * Math.sin(rad) }
}

interface Doc {
  category: 'relevant' | 'irrelevant'
  angleDeg: number
  cosine: number
  x: number
  y: number
}

// angle, cosine, and screen position are all static per document -- computed
// once here, never recomputed on a slider drag
function toDoc(category: 'relevant' | 'irrelevant', angleDeg: number): Doc {
  const { x, y } = polar(angleDeg, R)
  return { category, angleDeg, cosine: Math.cos((angleDeg * Math.PI) / 180), x, y }
}

function buildDocs(relevantDeg: number[], irrelevantDeg: number[]): Doc[] {
  return [...relevantDeg.map((a) => toDoc('relevant', a)), ...irrelevantDeg.map((a) => toDoc('irrelevant', a))]
}

const mixedDocs = buildDocs(RELEVANT_MIXED, IRRELEVANT_MIXED)
const improvedDocs = buildDocs(RELEVANT_IMPROVED, IRRELEVANT_IMPROVED)

const mixedCutoff = ref(DEFAULT_CUTOFF)
const improvedCutoff = ref(DEFAULT_CUTOFF)

function summarize(docs: Doc[], cutoff: number) {
  let relevantKept = 0
  let relevantTotal = 0
  let irrelevantKept = 0
  for (const d of docs) {
    if (d.category === 'relevant') {
      relevantTotal++
      if (d.cosine >= cutoff) relevantKept++
    } else if (d.cosine >= cutoff) {
      irrelevantKept++
    }
  }
  return { relevantKept, relevantTotal, irrelevantKept, returned: relevantKept + irrelevantKept }
}

const mixedSummary = computed(() => summarize(mixedDocs, mixedCutoff.value))
const improvedSummary = computed(() => summarize(improvedDocs, improvedCutoff.value))

// lowest-scoring relevant document sets the cutoff that keeps every relevant
// document for that panel; rounded down to the slider's own 0.01 step so the
// button lands on a value the slider can actually reproduce
function keepAllRelevantCutoff(docs: Doc[]): number {
  const relevantCosines = docs.filter((d) => d.category === 'relevant').map((d) => d.cosine)
  return Math.floor(100 * Math.min(...relevantCosines)) / 100
}

function onKeepAllRelevant() {
  mixedCutoff.value = keepAllRelevantCutoff(mixedDocs)
  improvedCutoff.value = keepAllRelevantCutoff(improvedDocs)
}

function onReset() {
  mixedCutoff.value = DEFAULT_CUTOFF
  improvedCutoff.value = DEFAULT_CUTOFF
}

// acceptance half-angle: cos(theta) >= cutoff iff |theta| <= acos(cutoff)
function halfAngleDeg(cutoff: number): number {
  return (Math.acos(Math.min(1, Math.max(-1, cutoff))) * 180) / Math.PI
}

// full-radius pie wedge from -phi to +phi as a true SVG arc. phi is always in
// [0, 90] so the swept angle 2*phi never exceeds 180 -- large-arc-flag is
// always 0 -- and sweep-flag 0 draws the arc through theta = 0 (the query
// direction), not the long way around the circle
function sectorPath(phiDeg: number): string {
  if (phiDeg <= 0.05) return ''
  const p1 = polar(-phiDeg, R)
  const p2 = polar(phiDeg, R)
  return `M ${CX},${CY} L ${p1.x.toFixed(2)},${p1.y.toFixed(2)} A ${R},${R} 0 0,0 ${p2.x.toFixed(2)},${p2.y.toFixed(2)} Z`
}

const mixedPhi = computed(() => halfAngleDeg(mixedCutoff.value))
const improvedPhi = computed(() => halfAngleDeg(improvedCutoff.value))
const mixedSectorPath = computed(() => sectorPath(mixedPhi.value))
const improvedSectorPath = computed(() => sectorPath(improvedPhi.value))
const mixedBoundary = computed(() => [polar(-mixedPhi.value, R), polar(mixedPhi.value, R)])
const improvedBoundary = computed(() => [polar(-improvedPhi.value, R), polar(improvedPhi.value, R)])
const qTip = polar(0, R)

const COLOR_RELEVANT = '#34d399'
const COLOR_IRRELEVANT = '#98a2b3'
const COLOR_INCLUDED_RING = '#e8ecf1'
</script>

<template>
  <div class="cosine-cutoff-demo">
    <div class="panels">
      <div class="panel">
        <h3 class="panel-title">Overlapping scores</h3>
        <svg class="panel-svg" viewBox="0 0 160 160" role="img" aria-label="Unit circle of 21 documents with overlapping relevant and irrelevant scores">
          <defs>
            <marker :id="`${uid}-arrow-mixed`" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
              <path d="M0,0 L7,3.5 L0,7 Z" fill="#6fd3ff" />
            </marker>
          </defs>
          <path v-if="mixedSectorPath" :d="mixedSectorPath" fill="rgba(111,211,255,0.16)" stroke="none" />
          <circle :cx="CX" :cy="CY" :r="R" fill="none" stroke="rgba(232,236,241,0.18)" stroke-width="1" />
          <line
            v-for="(b, i) in mixedBoundary"
            :key="`mb${i}`"
            :x1="CX"
            :y1="CY"
            :x2="b.x"
            :y2="b.y"
            stroke="#6fd3ff"
            stroke-width="1"
            stroke-dasharray="2,2"
          />
          <line :x1="CX" :y1="CY" :x2="qTip.x - 3" :y2="qTip.y" stroke="#6fd3ff" stroke-width="1.4" :marker-end="`url(#${uid}-arrow-mixed)`" />
          <text class="q-label" :x="qTip.x + 6" :y="qTip.y + 3" fill="#6fd3ff">q</text>
          <g v-for="(d, i) in mixedDocs" :key="`m${i}`">
            <circle v-if="d.cosine >= mixedCutoff" :cx="d.x" :cy="d.y" r="3.8" fill="none" :stroke="COLOR_INCLUDED_RING" stroke-width="0.8" />
            <circle :cx="d.x" :cy="d.y" r="3.2" :fill="d.category === 'relevant' ? COLOR_RELEVANT : COLOR_IRRELEVANT" :style="{ opacity: d.cosine >= mixedCutoff ? 1 : 0.6 }">
              <title>{{ d.category }} doc, angle {{ d.angleDeg }}°, cosine {{ d.cosine.toFixed(3) }}, {{ d.cosine >= mixedCutoff ? 'included' : 'excluded' }}</title>
            </circle>
          </g>
        </svg>
        <div class="slider-row">
          <label :for="`${uid}-mixed-cutoff`">Cosine cutoff</label>
          <input
            :id="`${uid}-mixed-cutoff`"
            type="range"
            min="0"
            max="1"
            step="0.01"
            v-model.number="mixedCutoff"
            aria-label="Cosine cutoff for overlapping scores panel"
            :aria-valuetext="mixedCutoff.toFixed(2)"
          />
          <span class="readout">{{ mixedCutoff.toFixed(2) }}</span>
        </div>
        <div class="counts">
          <span>Relevant kept {{ mixedSummary.relevantKept }}/{{ mixedSummary.relevantTotal }}</span>
          <span>Irrelevant kept {{ mixedSummary.irrelevantKept }}</span>
          <span class="returned">Returned {{ mixedSummary.returned }}</span>
        </div>
      </div>

      <div class="panel">
        <h3 class="panel-title">Better-separated scores</h3>
        <svg class="panel-svg" viewBox="0 0 160 160" role="img" aria-label="Unit circle of the same 21 documents with better-separated relevant and irrelevant scores">
          <defs>
            <marker :id="`${uid}-arrow-improved`" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
              <path d="M0,0 L7,3.5 L0,7 Z" fill="#6fd3ff" />
            </marker>
          </defs>
          <path v-if="improvedSectorPath" :d="improvedSectorPath" fill="rgba(111,211,255,0.16)" stroke="none" />
          <circle :cx="CX" :cy="CY" :r="R" fill="none" stroke="rgba(232,236,241,0.18)" stroke-width="1" />
          <line
            v-for="(b, i) in improvedBoundary"
            :key="`ib${i}`"
            :x1="CX"
            :y1="CY"
            :x2="b.x"
            :y2="b.y"
            stroke="#6fd3ff"
            stroke-width="1"
            stroke-dasharray="2,2"
          />
          <line :x1="CX" :y1="CY" :x2="qTip.x - 3" :y2="qTip.y" stroke="#6fd3ff" stroke-width="1.4" :marker-end="`url(#${uid}-arrow-improved)`" />
          <text class="q-label" :x="qTip.x + 6" :y="qTip.y + 3" fill="#6fd3ff">q</text>
          <g v-for="(d, i) in improvedDocs" :key="`i${i}`">
            <circle v-if="d.cosine >= improvedCutoff" :cx="d.x" :cy="d.y" r="3.8" fill="none" :stroke="COLOR_INCLUDED_RING" stroke-width="0.8" />
            <circle :cx="d.x" :cy="d.y" r="3.2" :fill="d.category === 'relevant' ? COLOR_RELEVANT : COLOR_IRRELEVANT" :style="{ opacity: d.cosine >= improvedCutoff ? 1 : 0.6 }">
              <title>{{ d.category }} doc, angle {{ d.angleDeg }}°, cosine {{ d.cosine.toFixed(3) }}, {{ d.cosine >= improvedCutoff ? 'included' : 'excluded' }}</title>
            </circle>
          </g>
        </svg>
        <div class="slider-row">
          <label :for="`${uid}-improved-cutoff`">Cosine cutoff</label>
          <input
            :id="`${uid}-improved-cutoff`"
            type="range"
            min="0"
            max="1"
            step="0.01"
            v-model.number="improvedCutoff"
            aria-label="Cosine cutoff for better-separated scores panel"
            :aria-valuetext="improvedCutoff.toFixed(2)"
          />
          <span class="readout">{{ improvedCutoff.toFixed(2) }}</span>
        </div>
        <div class="counts">
          <span>Relevant kept {{ improvedSummary.relevantKept }}/{{ improvedSummary.relevantTotal }}</span>
          <span>Irrelevant kept {{ improvedSummary.irrelevantKept }}</span>
          <span class="returned">Returned {{ improvedSummary.returned }}</span>
        </div>
      </div>
    </div>

    <div class="footer-row">
      <div class="legend">
        <span class="legend-item"><span class="swatch" :style="{ background: COLOR_RELEVANT }"></span>relevant</span>
        <span class="legend-item"><span class="swatch" :style="{ background: COLOR_IRRELEVANT }"></span>irrelevant</span>
        <span class="legend-item"><span class="swatch ring"></span>included (score ≥ cutoff)</span>
      </div>
      <div class="buttons">
        <button type="button" @click="onKeepAllRelevant">Keep all relevant</button>
        <button type="button" @click="onReset">Reset</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cosine-cutoff-demo {
  width: 860px;
  height: 320px;
  box-sizing: border-box;
  background: #12151c;
  color: #e8ecf1;
  font-family: inherit;
  padding: 8px 14px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.panels {
  flex: 1;
  display: flex;
  gap: 16px;
  min-height: 0;
}

.panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 0;
}

.panel + .panel {
  border-left: 1px solid #2a3140;
  padding-left: 16px;
}

.panel-title {
  margin: 0 0 2px;
  font-size: 14px;
  font-weight: bold;
  color: #e8ecf1;
}

.panel-svg {
  width: 170px;
  height: 170px;
  flex-shrink: 0;
}

.q-label {
  font-size: 8px;
}

.slider-row {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  max-width: 280px;
  font-size: 12px;
  color: #98a2b3;
}

.slider-row label {
  white-space: nowrap;
}

.slider-row input[type='range'] {
  flex: 1;
  accent-color: #6fd3ff;
}

.readout {
  font-variant-numeric: tabular-nums;
  color: #e8ecf1;
  width: 2.6em;
  text-align: right;
}

.counts {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 12px;
  font-size: 12px;
  color: #98a2b3;
  margin-top: 3px;
}

.counts .returned {
  color: #e8ecf1;
  font-weight: bold;
}

.footer-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-top: 1px solid #2a3140;
  padding-top: 6px;
  flex-shrink: 0;
}

.legend {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: #98a2b3;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.swatch {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  display: inline-block;
  background: transparent;
}

.swatch.ring {
  background: #12151c;
  border: 1.4px solid #e8ecf1;
}

.buttons {
  display: flex;
  gap: 8px;
}

.buttons button {
  font: inherit;
  font-size: 12px;
  color: #e8ecf1;
  background: #1a2029;
  border: 1px solid #2a3140;
  border-radius: 4px;
  padding: 4px 10px;
  cursor: pointer;
}

.buttons button:hover {
  border-color: #6fd3ff;
  color: #6fd3ff;
}
</style>

<script setup lang="ts">
import { computed } from 'vue'

// Source: MixVPR (2023), SALAD (2023), CliqueMining (2024), SelaVPR++ (2025) —
// Nordland Recall@1, in chronological order. SALAD -> CliqueMining is the same
// network with sampling-only training changed; no architecture changed.
const DATA: { method: string; year: number; value: number; highlight: boolean }[] = [
  { method: 'MixVPR', year: 2023, value: 58.4, highlight: false },
  { method: 'SALAD', year: 2023, value: 76.0, highlight: false },
  { method: 'CliqueMining', year: 2024, value: 90.7, highlight: true },
  { method: 'SelaVPR++', year: 2025, value: 97.2, highlight: false },
]

const BLUE = '#7cc4ff'
const ACCENT = '#e8894a'

const WIDTH = 860
const HEIGHT = 360

const MARGIN_LEFT = 56
const MARGIN_RIGHT = 30
const CHART_TOP = 118
const CHART_BOTTOM = 288
const CHART_HEIGHT = CHART_BOTTOM - CHART_TOP
const CHART_WIDTH = WIDTH - MARGIN_LEFT - MARGIN_RIGHT

const AXIS_MAX = 100
const GRIDLINES = [0, 25, 50, 75, 100]

function yFor(value: number): number {
  return CHART_BOTTOM - (value / AXIS_MAX) * CHART_HEIGHT
}

const bars = computed(() => {
  const slotWidth = CHART_WIDTH / DATA.length
  const barWidth = slotWidth * 0.5
  return DATA.map((d, i) => {
    const x = MARGIN_LEFT + i * slotWidth + (slotWidth - barWidth) / 2
    const y = yFor(d.value)
    return {
      ...d,
      x,
      width: barWidth,
      y,
      height: CHART_BOTTOM - y,
      centerX: x + barWidth / 2,
      color: d.highlight ? ACCENT : BLUE,
    }
  })
})

// Bracket connecting the SALAD bar (index 1) and the CliqueMining bar (index 2).
const bracket = computed(() => {
  const [, salad, clique] = bars.value
  const y = Math.min(salad.y, clique.y) - 26
  return {
    x1: salad.centerX,
    x2: clique.centerX,
    y,
    labelY: y - 10,
    labelX: (salad.centerX + clique.centerX) / 2,
  }
})
</script>

<template>
  <div class="vpr-progress">
    <svg :width="WIDTH" :height="HEIGHT" :viewBox="`0 0 ${WIDTH} ${HEIGHT}`">
      <text class="title" :x="MARGIN_LEFT" y="26">Nordland, Recall@1</text>

      <g class="gridlines">
        <template v-for="g in GRIDLINES" :key="g">
          <line
            :x1="MARGIN_LEFT"
            :x2="WIDTH - MARGIN_RIGHT"
            :y1="yFor(g)"
            :y2="yFor(g)"
            :class="{ baseline: g === 0 }"
          />
          <text class="axis-label" :x="MARGIN_LEFT - 10" :y="yFor(g) + 5" text-anchor="end">{{ g }}%</text>
        </template>
      </g>

      <line
        class="bracket"
        :x1="bracket.x1"
        :y1="bracket.y + 10"
        :x2="bracket.x1"
        :y2="bracket.y"
      />
      <line
        class="bracket"
        :x1="bracket.x1"
        :y1="bracket.y"
        :x2="bracket.x2"
        :y2="bracket.y"
      />
      <line
        class="bracket"
        :x1="bracket.x2"
        :y1="bracket.y"
        :x2="bracket.x2"
        :y2="bracket.y + 10"
      />
      <text class="callout" :x="bracket.labelX" :y="bracket.labelY" text-anchor="middle">
        sampling only, no layer changed
      </text>

      <g v-for="b in bars" :key="b.method">
        <rect :x="b.x" :y="b.y" :width="b.width" :height="b.height" :fill="b.color" rx="3" />
        <text class="value" :x="b.centerX" :y="b.y - 10" text-anchor="middle">{{ b.value }}%</text>
        <text class="method" :x="b.centerX" :y="CHART_BOTTOM + 22" text-anchor="middle">{{ b.method }}</text>
        <text class="year" :x="b.centerX" :y="CHART_BOTTOM + 40" text-anchor="middle">{{ b.year }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.vpr-progress {
  width: 860px;
  height: 360px;
  background: #12151c;
  color: #e8ecf1;
  font-family: inherit;
}

svg {
  display: block;
}

.title {
  fill: #e8ecf1;
  font-size: 18px;
  font-weight: bold;
}

.gridlines line {
  stroke: #2a2f3a;
  stroke-width: 1;
}

.gridlines line.baseline {
  stroke: #3c4250;
}

.axis-label {
  fill: #98a2b3;
  font-size: 14px;
}

.value {
  fill: #e8ecf1;
  font-size: 18px;
  font-weight: bold;
  font-variant-numeric: tabular-nums;
}

.method {
  fill: #e8ecf1;
  font-size: 14px;
}

.year {
  fill: #98a2b3;
  font-size: 14px;
  font-variant-numeric: tabular-nums;
}

.bracket {
  stroke: #e8894a;
  stroke-width: 1.5;
  fill: none;
}

.callout {
  fill: #e8894a;
  font-size: 14px;
}
</style>
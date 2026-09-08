<script setup lang="ts">
import { computed } from 'vue'

// Source: HierLoc (2026) — accuracy gain over the flat baseline, by
// hierarchy level, in depth order. Gain is largest at subregion, not
// monotonic with depth (city drops back down).
const DATA: { level: string; value: number; highlight: boolean }[] = [
  { level: 'country', value: 8.8, highlight: false },
  { level: 'region', value: 20.1, highlight: false },
  { level: 'subregion', value: 43.2, highlight: true },
  { level: 'city', value: 16.8, highlight: false },
]

const BLUE = '#7cc4ff'
const ACCENT = '#e8894a'

const WIDTH = 860
const HEIGHT = 330

const MARGIN_LEFT = 56
const MARGIN_RIGHT = 30
const CHART_TOP = 70
const CHART_BOTTOM = 268
const CHART_HEIGHT = CHART_BOTTOM - CHART_TOP
const CHART_WIDTH = WIDTH - MARGIN_LEFT - MARGIN_RIGHT

const AXIS_MAX = 50
const GRIDLINES = [0, 10, 20, 30, 40, 50]

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
</script>

<template>
  <div class="hierloc-gains">
    <svg :width="WIDTH" :height="HEIGHT" :viewBox="`0 0 ${WIDTH} ${HEIGHT}`">
      <text class="title" :x="MARGIN_LEFT" y="26">HierLoc accuracy gain by level</text>

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

      <g v-for="b in bars" :key="b.level">
        <rect :x="b.x" :y="b.y" :width="b.width" :height="b.height" :fill="b.color" rx="3" />
        <text class="value" :x="b.centerX" :y="b.y - 10" text-anchor="middle">+{{ b.value }}%</text>
        <text class="level" :x="b.centerX" :y="CHART_BOTTOM + 24" text-anchor="middle">{{ b.level }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.hierloc-gains {
  width: 860px;
  height: 330px;
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

.level {
  fill: #e8ecf1;
  font-size: 14px;
}
</style>
<script setup lang="ts">
import { computed } from 'vue'

// Source: AnyLoc, arXiv:2308.00688, Tables III-IV — average Recall@1 over the
// paper's structured-environment and unstructured-environment benchmark
// splits. NetVLAD, CosPlace, and MixVPR are trained end-to-end for visual
// place recognition; AnyLoc-VLAD-DINOv2 uses frozen, off-the-shelf DINOv2
// features and unsupervised VLAD, with zero VPR training.
const METHODS = ['NetVLAD', 'CosPlace', 'MixVPR', 'AnyLoc-VLAD-DINOv2'] as const
const CODES = ['NV', 'CP', 'MV', 'AL']
const COLORS = ['#5b7a99', '#7cc4ff', '#3d6cc7', '#e8894a']

const GROUPS: { name: string; values: number[] }[] = [
  { name: 'Structured environments', values: [62.5, 77.0, 83.9, 86.5] },
  { name: 'Unstructured environments', values: [31.1, 29.3, 33.2, 65.0] },
]

const WIDTH = 860
const HEIGHT = 400

const MARGIN_LEFT = 60
const MARGIN_RIGHT = 26
const CHART_TOP = 96
const CHART_BOTTOM = 290
const CHART_HEIGHT = CHART_BOTTOM - CHART_TOP
const CHART_WIDTH = WIDTH - MARGIN_LEFT - MARGIN_RIGHT

const AXIS_MAX = 100
const GRIDLINES = [0, 25, 50, 75, 100]

const GROUP_GAP = 56
const GROUP_WIDTH = (CHART_WIDTH - GROUP_GAP) / 2

function yFor(value: number): number {
  return CHART_BOTTOM - (value / AXIS_MAX) * CHART_HEIGHT
}

const groups = computed(() =>
  GROUPS.map((group, gi) => {
    const groupX0 = MARGIN_LEFT + gi * (GROUP_WIDTH + GROUP_GAP)
    const slotWidth = GROUP_WIDTH / METHODS.length
    const barWidth = slotWidth * 0.6
    const bars = group.values.map((value, mi) => {
      const x = groupX0 + mi * slotWidth + (slotWidth - barWidth) / 2
      const y = yFor(value)
      return {
        method: METHODS[mi],
        code: CODES[mi],
        value,
        x,
        width: barWidth,
        y,
        height: CHART_BOTTOM - y,
        centerX: x + barWidth / 2,
        color: COLORS[mi],
        highlight: mi === METHODS.length - 1,
      }
    })
    return {
      name: group.name,
      centerX: groupX0 + GROUP_WIDTH / 2,
      bars,
    }
  }),
)

const dividerX = MARGIN_LEFT + GROUP_WIDTH + GROUP_GAP / 2

const legend = computed(() =>
  METHODS.map((method, i) => ({ method, color: COLORS[i], highlight: i === METHODS.length - 1 })),
)
</script>

<template>
  <div class="anyloc-gains">
    <svg :width="WIDTH" :height="HEIGHT" :viewBox="`0 0 ${WIDTH} ${HEIGHT}`">
      <text class="title" :x="MARGIN_LEFT" y="24">Recall@1: off-the-shelf features vs. VPR-trained methods</text>
      <text class="annotation" :x="MARGIN_LEFT" y="44">AnyLoc-VLAD-DINOv2: frozen DINOv2 + unsupervised VLAD · zero VPR training</text>

      <g class="legend">
        <template v-for="(l, i) in legend" :key="l.method">
          <rect :x="MARGIN_LEFT + i * 193" y="58" width="13" height="13" :fill="l.color" rx="2" />
          <text
            :x="MARGIN_LEFT + i * 193 + 19"
            y="68"
            :class="['legend-label', { 'legend-highlight': l.highlight }]"
          >{{ l.method }}</text>
        </template>
      </g>

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

      <line class="divider" :x1="dividerX" :x2="dividerX" :y1="CHART_TOP - 6" :y2="CHART_BOTTOM" />

      <g v-for="group in groups" :key="group.name">
        <g v-for="b in group.bars" :key="b.method">
          <rect
            :x="b.x"
            :y="b.y"
            :width="b.width"
            :height="b.height"
            :fill="b.color"
            :class="{ 'bar-highlight': b.highlight }"
            rx="3"
          />
          <text class="value" :x="b.centerX" :y="b.y - 9" text-anchor="middle">{{ b.value.toFixed(1) }}</text>
          <text class="code" :x="b.centerX" :y="CHART_BOTTOM + 18" text-anchor="middle">{{ b.code }}</text>
        </g>
        <text class="group-name" :x="group.centerX" :y="CHART_BOTTOM + 42" text-anchor="middle">{{ group.name }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.anyloc-gains {
  width: 860px;
  height: 400px;
  background: #12151c;
  color: #e8ecf1;
  font-family: inherit;
}

svg {
  display: block;
}

.title {
  fill: #e8ecf1;
  font-size: 17px;
  font-weight: bold;
}

.annotation {
  fill: #e8894a;
  font-size: 12.5px;
  font-style: italic;
}

.legend-label {
  fill: #98a2b3;
  font-size: 12.5px;
}

.legend-highlight {
  fill: #e8894a;
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
  font-size: 13px;
}

.divider {
  stroke: #2a2f3a;
  stroke-width: 1;
  stroke-dasharray: 4 4;
}

.value {
  fill: #e8ecf1;
  font-size: 15px;
  font-weight: bold;
  font-variant-numeric: tabular-nums;
}

.code {
  fill: #98a2b3;
  font-size: 12.5px;
}

.group-name {
  fill: #e8ecf1;
  font-size: 14.5px;
  font-weight: 600;
}

.bar-highlight {
  stroke: #ffd9b3;
  stroke-width: 2;
}
</style>

<script setup lang="ts">
type CellState = 'yes' | 'no' | 'note'

interface Cell {
  state: CellState
  note?: string
}

interface Row {
  name: string
  cells: Cell[]
}

const COLUMNS = ['L2', 'inner product', 'cosine', 'Hamming', 'multi-vector', 'hyperbolic', 'Wasserstein'] as const

// Sources: FAISS wiki (MetricType and distances); DiskANN docs; Milvus 2.4 metric docs and
// the 2.6.4 "Array of Structs + MAX_SIM" release; Qdrant 1.10 release notes; Elasticsearch 8.18
// rank_vectors reference (technical preview, rerank only); Pinecone sparse-dense docs.
const ROWS: Row[] = [
  {
    name: 'FAISS',
    cells: [
      { state: 'yes' },
      { state: 'yes' },
      { state: 'note', note: 'normalise, then IP' },
      { state: 'note', note: 'flat/HNSW only' },
      { state: 'no' },
      { state: 'no' },
      { state: 'no' },
    ],
  },
  {
    name: 'DiskANN',
    cells: [
      { state: 'yes' },
      { state: 'note', note: 'mips' },
      { state: 'yes' },
      { state: 'no' },
      { state: 'no' },
      { state: 'no' },
      { state: 'no' },
    ],
  },
  {
    name: 'Milvus',
    cells: [
      { state: 'yes' },
      { state: 'yes' },
      { state: 'yes' },
      { state: 'note', note: 'binary vectors' },
      { state: 'note', note: 'MAX_SIM, 2.6.4' },
      { state: 'no' },
      { state: 'no' },
    ],
  },
  {
    name: 'Qdrant',
    cells: [
      { state: 'yes' },
      { state: 'yes' },
      { state: 'yes' },
      { state: 'no' },
      { state: 'note', note: 'max_sim, 1.10' },
      { state: 'no' },
      { state: 'no' },
    ],
  },
  {
    name: 'Elasticsearch',
    cells: [
      { state: 'yes' },
      { state: 'yes' },
      { state: 'yes' },
      { state: 'note', note: 'bit vectors' },
      { state: 'note', note: 'rerank only, 8.18' },
      { state: 'no' },
      { state: 'no' },
    ],
  },
  {
    name: 'Pinecone',
    cells: [
      { state: 'yes' },
      { state: 'yes' },
      { state: 'yes' },
      { state: 'no' },
      { state: 'no' },
      { state: 'no' },
      { state: 'no' },
    ],
  },
]

// the last two columns are the point of the slide: no engine here supports
// either metric, so that column pair gets a darker band to draw the eye.
const DIM_COLUMNS = new Set([5, 6])
</script>

<template>
  <div class="index-metrics">
    <div class="grid">
      <div class="corner"></div>
      <div
        v-for="(col, ci) in COLUMNS"
        :key="col"
        class="col-head"
        :class="{ dim: DIM_COLUMNS.has(ci) }"
      >
        {{ col }}
      </div>

      <template v-for="row in ROWS" :key="row.name">
        <div class="row-label">{{ row.name }}</div>
        <div
          v-for="(cell, ci) in row.cells"
          :key="row.name + ci"
          class="cell"
          :class="{ dim: DIM_COLUMNS.has(ci) }"
        >
          <span v-if="cell.state === 'yes'" class="glyph yes">&#10003;</span>
          <span v-else-if="cell.state === 'no'" class="glyph no">&#10005;</span>
          <template v-else>
            <span class="glyph yes">&#10003;</span>
            <span class="note">{{ cell.note }}</span>
          </template>
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.index-metrics {
  width: 900px;
  height: 400px;
  background: #12151c;
  color: #e8ecf1;
  font-family: inherit;
  box-sizing: border-box;
  padding: 8px 0 0 0;
}

.grid {
  display: grid;
  grid-template-columns: 130px repeat(7, 110px);
  grid-template-rows: 40px repeat(6, 58px);
  width: 900px;
}

.corner {
  border-bottom: 1px solid #2a3140;
}

.col-head {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: #98a2b3;
  border-bottom: 1px solid #2a3140;
  text-align: center;
  padding: 0 4px;
  box-sizing: border-box;
}

.col-head.dim {
  background: #0d1016;
}

.row-label {
  display: flex;
  align-items: center;
  font-size: 16px;
  font-weight: bold;
  color: #e8ecf1;
  border-bottom: 1px solid #1c212c;
  padding-left: 4px;
  box-sizing: border-box;
}

.cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  border-bottom: 1px solid #1c212c;
  border-left: 1px solid #1c212c;
  box-sizing: border-box;
}

.cell.dim {
  background: #0d1016;
}

.glyph {
  font-size: 22px;
  line-height: 1;
}

.glyph.yes {
  color: #34d399;
}

.glyph.no {
  color: #98a2b3;
}

.note {
  font-size: 10px;
  color: #98a2b3;
  text-align: center;
  max-width: 100px;
  line-height: 1.2;
}
</style>

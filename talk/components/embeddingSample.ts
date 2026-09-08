// Synthetic embedding cloud shared by the geometry demos.
//
// Real embedding spaces are anisotropic: every vector shares a common
// direction, so random pairs already have a positive cosine (roughly 0.3 for
// contrastive models). Both hubness and the averaging effect depend on that
// shared component, so the sampler models it explicitly instead of drawing
// isotropic Gaussians, which would show neither effect under cosine.

export const BACKGROUND_COSINE = 0.3

/** Deterministic PRNG (mulberry32) so a slide renders the same every time. */
export function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(next: () => number): number {
  const u = 1 - next()
  const v = next()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

/**
 * Sample `n` unit vectors in `d` dimensions as shared + individual noise:
 *   x = a * mu + g,  g ~ N(0, I/d),  mu = e_0
 * normalised to unit length. `a` is chosen so the expected cosine between
 * two random samples is BACKGROUND_COSINE: with |g|^2 ~ 1, cos ~ a^2/(a^2+1).
 */
export function sampleEmbeddings(n: number, d: number, seed = 1): Float64Array[] {
  const next = rng(seed)
  const a = Math.sqrt(BACKGROUND_COSINE / (1 - BACKGROUND_COSINE))
  const sigma = 1 / Math.sqrt(d)
  const out: Float64Array[] = new Array(n)
  for (let i = 0; i < n; i++) {
    const x = new Float64Array(d)
    let norm = 0
    for (let j = 0; j < d; j++) {
      const v = gaussian(next) * sigma + (j === 0 ? a : 0)
      x[j] = v
      norm += v * v
    }
    norm = Math.sqrt(norm)
    for (let j = 0; j < d; j++) x[j] /= norm
    out[i] = x
  }
  return out
}

/** Dot product; inputs are unit vectors so this is the cosine similarity. */
export function cosine(u: Float64Array, v: Float64Array): number {
  let s = 0
  for (let j = 0; j < u.length; j++) s += u[j] * v[j]
  return s
}

/** Mean of a set of vectors, not normalised, so its norm can be inspected. */
export function mean(vectors: Float64Array[]): Float64Array {
  const d = vectors[0].length
  const m = new Float64Array(d)
  for (const v of vectors) for (let j = 0; j < d; j++) m[j] += v[j]
  for (let j = 0; j < d; j++) m[j] /= vectors.length
  return m
}

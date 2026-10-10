export const fmt = (v: unknown, d = 2): string => {
  if (typeof v !== 'number' || !isFinite(v)) return '—'
  if (Math.abs(v) >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 })
  return v.toFixed(d)
}
export const pct = (v: unknown, d = 0): string =>
  typeof v === 'number' && isFinite(v) ? `${(v * 100).toFixed(d)}%` : '—'
export const isObj = (v: unknown): v is Record<string, any> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)
export const num = (v: unknown): number | undefined =>
  typeof v === 'number' && isFinite(v) ? v : undefined
const ACRONYMS: Record<string, string> = { qc: 'QC', llm: 'LLM', rms: 'RMS', hf: 'HF', jpeg: 'JPEG', auroc: 'AUROC', knn: 'kNN', lbp: 'LBP', bm25: 'BM25', mrr: 'MRR' }
export const human = (s: string): string => {
  const words = s.replace(/_/g, ' ').split(' ')
  return words
    .map((w, i) => {
      const a = ACRONYMS[w.toLowerCase()]
      if (a) return a
      return i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w
    })
    .join(' ')
}
export const sentence = (s?: string): string =>
  s ? s.replace(/_/g, ' ').toLowerCase().replace(/^\w/, (m) => m.toUpperCase()) : ''
export const shortSha = (s?: string): string => (s ? s.slice(0, 7) : 'unknown')

/** Flatten numeric leaves of a nested object into rows [path, value]. */
export function numericLeaves(
  o: unknown,
  prefix: string[] = [],
  out: Array<[string[], number]> = [],
  depth = 0,
): Array<[string[], number]> {
  if (depth > 5) return out
  if (typeof o === 'number' && isFinite(o)) {
    out.push([prefix, o])
  } else if (isObj(o)) {
    for (const [k, v] of Object.entries(o)) {
      if (k === 'provenance' || k === 'ci95') continue
      numericLeaves(v, [...prefix, k], out, depth + 1)
    }
  }
  return out
}

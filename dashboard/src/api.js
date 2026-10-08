// Same-origin in production (FastAPI serves the built bundle); Vite proxies /api in dev.
const base = ''

async function get(path) {
  const res = await fetch(`${base}${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`)
  return res.json()
}

export const listRuns = () => get('/api/runs')
export const getSummary = (runId, grader = 'numeric_exact_match') =>
  get(`/api/runs/${runId}/summary?grader=${encodeURIComponent(grader)}`)
export const getGenerations = (runId, model) =>
  get(`/api/runs/${runId}/generations${model ? `?model=${encodeURIComponent(model)}` : ''}`)
export const getLatencies = (runId) => get(`/api/runs/${runId}/latencies`)
export const getAgreement = (runId) => get(`/api/runs/${runId}/agreement`)

// Color follows the ENTITY, not its rank: a model keeps its hue across every chart, and
// filtering the leaderboard never repaints the survivors.
export function colorForModel(model, allModels) {
  const idx = [...allModels].sort().indexOf(model)
  return `var(--series-${(idx % 3) + 1})`
}

export const shortName = (m) => (m || '').replace(/^ollama:/, '')
export const fmtMs = (ms) => (ms == null ? '—' : ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`)
export const fmtPct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

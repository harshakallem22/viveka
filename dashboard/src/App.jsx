import React from 'react'
import {
  getAgreement,
  getGenerations,
  getLatencies,
  getSummary,
  listRuns,
  shortName,
} from './api.js'
import AccuracyChart from './components/AccuracyChart.jsx'
import ItemTable from './components/ItemTable.jsx'
import JudgePanel from './components/JudgePanel.jsx'
import LatencyChart from './components/LatencyChart.jsx'
import LeaderboardTable from './components/LeaderboardTable.jsx'

const POLL_MS = 2500

function useTheme() {
  const [theme, setTheme] = React.useState(() => localStorage.getItem('viveka-theme') || 'auto')
  React.useEffect(() => {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('viveka-theme', theme)
  }, [theme])
  return [theme, setTheme]
}

export default function App() {
  const [theme, setTheme] = useTheme()
  const [runs, setRuns] = React.useState([])
  const [runId, setRunId] = React.useState(null)
  const [summary, setSummary] = React.useState(null)
  const [latencies, setLatencies] = React.useState(null)
  const [agreement, setAgreement] = React.useState(null)
  const [generations, setGenerations] = React.useState([])
  const [view, setView] = React.useState('charts')
  const [error, setError] = React.useState(null)
  const [refetching, setRefetching] = React.useState(false)
  const [loaded, setLoaded] = React.useState(false)

  React.useEffect(() => {
    listRuns()
      .then((rs) => {
        setRuns(rs)
        if (rs.length) setRunId(rs[0].run_id)
        else setLoaded(true)
      })
      .catch((e) => {
        setError(e.message)
        setLoaded(true)
      })
  }, [])

  const load = React.useCallback(
    async (id, quiet) => {
      if (!id) return
      if (quiet) setRefetching(true)
      try {
        const [s, l, a, g] = await Promise.all([
          getSummary(id),
          getLatencies(id),
          getAgreement(id).catch(() => null),
          getGenerations(id),
        ])
        setSummary(s)
        setLatencies(l)
        setAgreement(a)
        setGenerations(g)
        setError(null)
      } catch (e) {
        setError(e.message)
      } finally {
        setRefetching(false)
        setLoaded(true)
      }
    },
    [],
  )

  React.useEffect(() => {
    load(runId, false)
  }, [runId, load])

  // "Live": the runner writes rows incrementally in WAL mode, so polling shows progress during a
  // run. Polling stops once the run is complete -- and it never triggers a benchmark.
  const isRunning = summary?.status === 'running'
  React.useEffect(() => {
    if (!isRunning || !runId) return
    const t = setInterval(() => load(runId, true), POLL_MS)
    return () => clearInterval(t)
  }, [isRunning, runId, load])

  if (error && !summary) {
    return (
      <div className="app">
        <h1>Viveka</h1>
        <p className="state">
          Could not reach the API: <span className="mono">{error}</span>
          <br />
          Start it with <span className="mono">viveka serve</span>.
        </p>
      </div>
    )
  }

  if (!loaded) return <div className="app"><p className="state">Loading…</p></div>

  if (!runs.length) {
    return (
      <div className="app">
        <h1>Viveka</h1>
        <p className="state">
          No runs recorded yet. Produce one with <span className="mono">viveka run</span>, then{' '}
          <span className="mono">viveka judge latest</span>.
        </p>
      </div>
    )
  }

  const models = summary?.models ?? []
  const judged = models.some((m) => m.accuracy != null)
  const best = judged
    ? models.reduce((a, b) => ((b.accuracy ?? -1) > (a.accuracy ?? -1) ? b : a), models[0])
    : null
  const fastest = models.length
    ? models.reduce((a, b) => ((b.decode_tps_p50 ?? 0) > (a.decode_tps_p50 ?? 0) ? b : a), models[0])
    : null
  const totalGenerations = models.reduce((n, m) => n + m.n_generations, 0)
  const totalErrors = models.reduce((n, m) => n + m.n_errors, 0)
  const kappa = agreement?.overall?.n ? agreement.overall.cohen_kappa : null

  return (
    <div className="app viz-root">
      <div className="masthead">
        <div>
          <h1>Viveka — LLM eval leaderboard</h1>
          <p>
            Local models scored on correctness, latency percentiles and throughput against a frozen
            eval set, with the LLM judge validated against objective ground truth.
          </p>
          <div className="provenance">
            {summary && (
              <>
                <span>
                  run <code>{summary.run_id}</code>
                </span>
                <span>status {summary.status}</span>
                <span>
                  eval set <code>{(summary.golden_set_hash || '').slice(0, 12)}</code>
                </span>
                <span>prompt {summary.prompt_version}</span>
                <span>ollama {summary.ollama_version}</span>
                <span>viveka {summary.viveka_version}</span>
              </>
            )}
          </div>
        </div>
        <div className="controls">
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : theme === 'light' ? 'auto' : 'dark')}
            title="Theme: auto / light / dark"
          >
            theme: {theme}
          </button>
        </div>
      </div>

      {/* One filter row above everything it scopes. */}
      <div className="filter-row">
        <label htmlFor="run-select">Run</label>
        <select id="run-select" value={runId ?? ''} onChange={(e) => setRunId(e.target.value)}>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} — {r.models.length} model(s), {r.n_generations} generations ({r.status})
            </option>
          ))}
        </select>
        <label htmlFor="view-select">View</label>
        <span>
          <button aria-pressed={view === 'charts'} onClick={() => setView('charts')}>
            charts
          </button>{' '}
          <button aria-pressed={view === 'table'} onClick={() => setView('table')}>
            table
          </button>
        </span>
        {isRunning && <span className="muted">live — polling every {POLL_MS / 1000}s</span>}
      </div>

      <div className={refetching ? 'refetching' : undefined}>
        {/* Stat tiles: single headline numbers, so no plot. */}
        <div className="tiles">
          <div className="tile">
            <div className="label">Models benchmarked</div>
            <div className="value">{models.length}</div>
            <div className="sub">{totalGenerations} generations</div>
          </div>
          <div className="tile">
            <div className="label">Request failures</div>
            <div className={`value${totalErrors === 0 ? ' good' : ''}`}>{totalErrors}</div>
            <div className="sub">of {totalGenerations}</div>
          </div>
          {best && (
            <div className="tile">
              <div className="label">Most accurate</div>
              <div className="value">{(best.accuracy * 100).toFixed(1)}%</div>
              <div className="sub">{shortName(best.model)}</div>
            </div>
          )}
          {fastest && (
            <div className="tile">
              <div className="label">Fastest decode</div>
              <div className="value">{(fastest.decode_tps_p50 ?? 0).toFixed(1)}</div>
              <div className="sub">tok/s p50 — {shortName(fastest.model)}</div>
            </div>
          )}
          {kappa != null && (
            <div className="tile">
              <div className="label">Judge κ vs ground truth</div>
              <div className="value">{kappa.toFixed(3)}</div>
              <div className="sub">{agreement.overall.false_positive} false positives</div>
            </div>
          )}
        </div>

        {view === 'table' ? (
          <div className="card">
            <header>
              <h2>Leaderboard</h2>
            </header>
            <p className="note">
              The WCAG-clean equivalent of the charts. Every value in the charts is here.
            </p>
            <LeaderboardTable models={models} />
          </div>
        ) : (
          <>
            <div className="card">
              <header>
                <h2>Accuracy</h2>
                <span className="muted">
                  {summary.grader ? `grader: ${summary.grader}` : ''}
                </span>
              </header>
              <p className="note">
                Whiskers are Wilson 95% confidence intervals. They are shown because a bare
                percentage overstates precision at this sample size — models whose intervals
                overlap cannot be ranked against each other.
              </p>
              <AccuracyChart models={models} />
            </div>

            <div className="card">
              <header>
                <h2>Latency distribution</h2>
              </header>
              <p className="note">
                One dot per request. Individual samples rather than a box plot, because the shape of
                the tail is the interesting part — a five-number summary would hide a runaway
                generation.
              </p>
              {latencies && <LatencyChart latencies={latencies} />}
            </div>
          </>
        )}

        <div className="card">
          <header>
            <h2>Judge reliability</h2>
          </header>
          <p className="note">
            The LLM judge's verdicts compared against a deterministic grader on items where the
            answer is objectively known. This is what makes LLM-as-judge defensible rather than
            assumed.
          </p>
          <JudgePanel agreement={agreement} />
        </div>

        <div className="card">
          <header>
            <h2>Per-item drill-down</h2>
            <span className="muted">{generations.length} generations</span>
          </header>
          <p className="note">
            Every response, every verdict, and the judge's reasoning — so a score can always be
            traced back to the text that produced it.
          </p>
          <ItemTable generations={generations} />
        </div>
      </div>
    </div>
  )
}

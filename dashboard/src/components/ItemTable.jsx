import React from 'react'
import { fmtMs, shortName } from '../api.js'

/**
 * Per-item drill-down, including each grader's verdict and the judge's stated reasoning.
 *
 * The most persuasive view in the dashboard: it shows the evaluation is inspectable rather than a
 * black box that emits a number. An interviewer can click any row and see exactly why it scored
 * the way it did.
 */
export default function ItemTable({ generations }) {
  const [expanded, setExpanded] = React.useState(null)
  const [filter, setFilter] = React.useState('all')

  const verdictOf = (g) => {
    if (g.error) return 'error'
    const j = g.judgements[0]
    return j ? j.verdict : 'unscored'
  }

  const rows = generations.filter((g) => {
    if (filter === 'all') return true
    if (filter === 'incorrect') return verdictOf(g) === 'incorrect'
    if (filter === 'truncated') return g.truncated
    if (filter === 'disagreement') {
      const verdicts = new Set(g.judgements.map((j) => j.verdict))
      return verdicts.size > 1
    }
    return true
  })

  return (
    <>
      <div className="filter-row" style={{ marginTop: 0 }}>
        <label htmlFor="item-filter">Show</label>
        <select id="item-filter" value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">all items</option>
          <option value="incorrect">incorrect only</option>
          <option value="truncated">hit the token cap</option>
          <option value="disagreement">graders disagreed</option>
        </select>
        <span className="muted">
          {rows.length} of {generations.length}
        </span>
      </div>

      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>item</th>
              <th>model</th>
              <th>category</th>
              <th>verdict</th>
              <th className="num">latency</th>
              <th className="num">tokens</th>
              <th>judge reasoning</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((g) => {
              const v = verdictOf(g)
              const judge = g.judgements.find((j) => j.reasoning)
              const isOpen = expanded === g.id
              return (
                <React.Fragment key={g.id}>
                  <tr onClick={() => setExpanded(isOpen ? null : g.id)} style={{ cursor: 'pointer' }}>
                    <td className="mono">{g.item_id}</td>
                    <td>{shortName(g.model)}</td>
                    <td className="muted">{g.category}</td>
                    <td>
                      <span className={`pill ${v}`}>{v}</span>
                      {g.truncated && (
                        <span className="pill error" style={{ marginLeft: 4 }}>
                          capped
                        </span>
                      )}
                    </td>
                    <td className="num">{fmtMs(g.total_ms)}</td>
                    <td className="num">{g.output_tokens ?? '—'}</td>
                    <td className="reasoning">{judge?.reasoning ?? '—'}</td>
                  </tr>
                  {isOpen && (
                    <tr>
                      <td colSpan={7} style={{ background: 'var(--page)' }}>
                        <div style={{ display: 'grid', gap: 10, padding: '4px 0 8px' }}>
                          <div>
                            <strong>Question</strong>
                            <div className="reasoning" style={{ maxWidth: '90ch' }}>{g.question}</div>
                          </div>
                          <div>
                            <strong>Reference answer</strong>{' '}
                            <span className="mono">{g.reference_answer}</span>
                          </div>
                          <div>
                            <strong>Model response</strong>
                            <pre
                              className="mono"
                              style={{
                                whiteSpace: 'pre-wrap',
                                margin: '4px 0 0',
                                maxHeight: 240,
                                overflow: 'auto',
                                color: 'var(--text-secondary)',
                              }}
                            >
                              {g.output_text}
                            </pre>
                          </div>
                          <div>
                            <strong>Verdicts</strong>
                            <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                              {g.judgements.map((j, i) => (
                                <li key={i} className="reasoning" style={{ maxWidth: '90ch' }}>
                                  <span className="mono">{j.grader}</span>{' '}
                                  <span className={`pill ${j.verdict}`}>{j.verdict}</span>{' '}
                                  {j.judge_model && <span className="muted">via {shortName(j.judge_model)}</span>}
                                  {j.reasoning && <> — {j.reasoning}</>}
                                </li>
                              ))}
                            </ul>
                          </div>
                          <div className="muted" style={{ fontSize: 12 }}>
                            ttft {fmtMs(g.ttft_ms)} · total {fmtMs(g.total_ms)} · decode{' '}
                            {fmtMs(g.decode_ms)} · finish reason{' '}
                            <span className="mono">{g.done_reason ?? '—'}</span>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
        Click any row to see the full response and every grader's verdict.
      </p>
    </>
  )
}

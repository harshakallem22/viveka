import React from 'react'
import { shortName } from '../api.js'

/**
 * Judge reliability.
 *
 * A 2x2 confusion matrix is a table, not a chart -- four numbers gain nothing from being drawn.
 * Kappa is the headline rather than raw agreement: when ground truth is heavily skewed, a judge
 * that blindly answers "correct" scores high agreement while being useless, and kappa is what
 * exposes that.
 */
export default function JudgePanel({ agreement }) {
  const o = agreement?.overall
  if (!o || !o.n) {
    return (
      <p className="muted">
        No judge validation for this run — run <span className="mono">viveka validate-judge</span>.
      </p>
    )
  }

  const skewed = o.truth_positive_rate != null && (o.truth_positive_rate < 0.2 || o.truth_positive_rate > 0.8)

  return (
    <>
      <div className="tiles" style={{ marginBottom: 14 }}>
        <div className="tile">
          <div className="label">Cohen's κ</div>
          <div className="value">{o.cohen_kappa.toFixed(3)}</div>
          <div className="sub">{o.interpretation}</div>
        </div>
        <div className="tile">
          <div className="label">Raw agreement</div>
          <div className="value">{(o.raw_agreement * 100).toFixed(1)}%</div>
          <div className="sub">{o.n} comparable pairs</div>
        </div>
        <div className="tile">
          <div className="label">False positives</div>
          <div className={`value${o.false_positive === 0 ? ' good' : ''}`}>{o.false_positive}</div>
          <div className="sub">
            judge approved a wrong answer
            {o.false_positive_rate != null && ` — ${(o.false_positive_rate * 100).toFixed(1)}% of wrong answers`}
          </div>
        </div>
        <div className="tile">
          <div className="label">False negatives</div>
          <div className="value">{o.false_negative}</div>
          <div className="sub">judge rejected a right answer</div>
        </div>
      </div>

      <div className="grid two">
        <div>
          <h3 style={{ fontSize: 13, margin: '0 0 8px' }}>Confusion matrix</h3>
          <table>
            <thead>
              <tr>
                <th />
                <th className="num">judge: correct</th>
                <th className="num">judge: incorrect</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th>truth: correct</th>
                <td className="num">{o.true_positive}</td>
                <td className="num">{o.false_negative}</td>
              </tr>
              <tr>
                <th>truth: incorrect</th>
                <td className="num" style={{ color: o.false_positive ? 'var(--status-critical)' : undefined }}>
                  {o.false_positive}
                </td>
                <td className="num">{o.true_negative}</td>
              </tr>
            </tbody>
          </table>
          {o.n_skipped ? (
            <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
              {o.n_skipped} pair(s) skipped: an unextractable answer is an absence of measurement,
              not a judge mistake.
            </p>
          ) : null}
        </div>

        <div>
          <h3 style={{ fontSize: 13, margin: '0 0 8px' }}>By contestant</h3>
          <table>
            <thead>
              <tr>
                <th>contestant</th>
                <th className="num">n</th>
                <th className="num">agreement</th>
                <th className="num">κ</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(agreement.by_model || {}).map(([model, r]) => (
                <tr key={model}>
                  <td>{shortName(model)}</td>
                  <td className="num">{r.n}</td>
                  <td className="num">{(r.raw_agreement * 100).toFixed(1)}%</td>
                  <td className="num">{r.cohen_kappa.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {skewed && (
        <p className="warnbox">
          Ground truth is {(o.truth_positive_rate * 100).toFixed(0)}% correct, so raw agreement is
          inflated — quote κ, not agreement.
        </p>
      )}
      <p className="warnbox">
        This validates the judge on an <strong>easy</strong> task: comparing a number to a number,
        where objective ground truth exists. It does <strong>not</strong> establish reliability on
        open-ended factual claims, which is exactly where the judge is the only available grader.
      </p>
    </>
  )
}

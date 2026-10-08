import React from 'react'
import { fmtMs, fmtPct, shortName } from '../api.js'

/**
 * The table-view twin of the charts.
 *
 * Not optional: light-mode series colors include one below 3:1 against the surface, so the relief
 * rule obliges a WCAG-clean equivalent. It is also simply the fastest way to read exact numbers.
 */
export default function LeaderboardTable({ models }) {
  return (
    <div className="scroll">
      <table>
        <thead>
          <tr>
            <th>model</th>
            <th className="num">n</th>
            <th className="num">accuracy</th>
            <th className="num">95% CI</th>
            <th className="num">TTFT p50</th>
            <th className="num">TTFT p95</th>
            <th className="num">lat p50</th>
            <th className="num">lat p95</th>
            <th className="num">lat max</th>
            <th className="num">tok/s p50</th>
            <th className="num">cold</th>
            <th className="num">err</th>
            <th className="num">trunc</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.model}>
              <td>{shortName(m.model)}</td>
              <td className="num">{m.n_generations}</td>
              <td className="num">{fmtPct(m.accuracy)}</td>
              <td className="num">
                {m.accuracy_ci_low == null
                  ? '—'
                  : `${fmtPct(m.accuracy_ci_low)}–${fmtPct(m.accuracy_ci_high)}`}
              </td>
              <td className="num">{fmtMs(m.ttft_p50_ms)}</td>
              <td className="num">{fmtMs(m.ttft_p95_ms)}</td>
              <td className="num">{fmtMs(m.latency_p50_ms)}</td>
              <td className="num">{fmtMs(m.latency_p95_ms)}</td>
              <td className="num">{fmtMs(m.latency_max_ms)}</td>
              <td className="num">{m.decode_tps_p50 == null ? '—' : m.decode_tps_p50.toFixed(1)}</td>
              <td className="num">{fmtMs(m.cold_start_ms)}</td>
              <td className="num">{m.n_errors}</td>
              <td className="num">{m.n_truncated}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
        Percentiles are nearest-rank over the samples shown (no interpolation), so every figure is
        a value that was actually observed. Warmup calls are excluded from all percentiles;
        <span className="mono"> cold</span> reports the warmup's weight-load time separately.
      </p>
    </div>
  )
}

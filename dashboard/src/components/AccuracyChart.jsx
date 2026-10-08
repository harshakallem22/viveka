import React from 'react'
import { colorForModel, fmtPct, shortName } from '../api.js'
import Tooltip, { useTooltip } from './Tooltip.jsx'

const PAD = { top: 14, right: 76, bottom: 34, left: 132 }
const ROW = 44
const BAR = 18 // thin marks; a fat saturated block reads loud
const GAP = 2 // surface gap between adjacent marks, never a border

/** Bar with only its data-end rounded, anchored flat to the baseline at x=0. */
function barPath(x0, y, w, h, r) {
  const radius = Math.min(r, w)
  if (w <= 0) return ''
  return `M${x0},${y} H${x0 + w - radius} a${radius},${radius} 0 0 1 ${radius},${radius} V${
    y + h - radius
  } a${radius},${radius} 0 0 1 ${-radius},${radius} H${x0} Z`
}

/**
 * Accuracy per model with Wilson 95% confidence intervals.
 *
 * The interval is the point of this chart, not decoration: at n=50 two models whose intervals
 * overlap are not statistically separable, and a bare bar chart would invite exactly the
 * false ranking this project exists to prevent.
 */
export default function AccuracyChart({ models, width = 720 }) {
  const { state, show, hide } = useTooltip()
  const names = models.map((m) => m.model)
  const scored = models.filter((m) => m.accuracy != null)

  if (!scored.length) {
    return (
      <p className="muted">
        No accuracy recorded yet — run <span className="mono">viveka judge</span>.
      </p>
    )
  }

  const height = PAD.top + scored.length * ROW + PAD.bottom
  const plotW = width - PAD.left - PAD.right
  const x = (v) => PAD.left + v * plotW

  // Overlap detection drives the caveat below, so the reader is told when a ranking is unsafe.
  const overlapping = []
  for (let i = 0; i < scored.length - 1; i++) {
    const a = scored[i]
    const b = scored[i + 1]
    if (a.accuracy_ci_low <= b.accuracy_ci_high && b.accuracy_ci_low <= a.accuracy_ci_high) {
      overlapping.push([shortName(a.model), shortName(b.model)])
    }
  }

  return (
    <>
      <div className="legend">
        {scored.map((m) => (
          <span className="legend-item" key={m.model}>
            <span
              className="swatch"
              style={{ background: colorForModel(m.model, names) }}
              aria-hidden="true"
            />
            {shortName(m.model)}
          </span>
        ))}
      </div>

      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Accuracy by model with 95% confidence intervals"
      >
        {/* Solid hairline gridlines, one shade off the surface. Never dashed. */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="gridline" x1={x(t)} y1={PAD.top} x2={x(t)} y2={PAD.top + scored.length * ROW} />
            <text className="tick-label" x={x(t)} y={height - PAD.bottom + 18} textAnchor="middle">
              {t * 100}%
            </text>
          </g>
        ))}
        <line
          className="baseline"
          x1={PAD.left}
          y1={PAD.top + scored.length * ROW}
          x2={PAD.left + plotW}
          y2={PAD.top + scored.length * ROW}
        />

        {scored.map((m, i) => {
          const color = colorForModel(m.model, names)
          const yTop = PAD.top + i * ROW + (ROW - BAR) / 2 + GAP / 2
          const cy = yTop + BAR / 2
          const w = Math.max(0, x(m.accuracy) - PAD.left)
          const rows = [
            ['accuracy', `${fmtPct(m.accuracy)} (${m.n_correct}/${m.n_scored})`],
            ['95% CI', `${fmtPct(m.accuracy_ci_low)} – ${fmtPct(m.accuracy_ci_high)}`],
            ['CI width', `${((m.accuracy_ci_high - m.accuracy_ci_low) * 100).toFixed(0)} points`],
          ]
          if (m.n_unscorable) rows.push(['unscorable', `${m.n_unscorable} (excluded)`])

          return (
            <g
              key={m.model}
              onMouseMove={(e) => show(e, shortName(m.model), rows)}
              onMouseLeave={hide}
              onFocus={(e) => show(e, shortName(m.model), rows)}
              onBlur={hide}
              tabIndex={0}
            >
              {/* Hit area spans the whole row: comfortably past the 24px minimum. */}
              <rect className="hit" x={0} y={PAD.top + i * ROW} width={width} height={ROW} />

              <text className="series-label" x={PAD.left - 12} y={cy + 4} textAnchor="end">
                {shortName(m.model)}
              </text>

              <path d={barPath(PAD.left, yTop, w, BAR, 4)} fill={color} />

              {/* Confidence interval. A surface-colored underlay gives the 2px ring that keeps
                  the whisker legible where it crosses the bar -- not an outline around the mark. */}
              <g>
                <line
                  x1={x(m.accuracy_ci_low)} y1={cy} x2={x(m.accuracy_ci_high)} y2={cy}
                  stroke="var(--surface-1)" strokeWidth={5} strokeLinecap="butt"
                />
                <line
                  x1={x(m.accuracy_ci_low)} y1={cy} x2={x(m.accuracy_ci_high)} y2={cy}
                  stroke="var(--text-primary)" strokeWidth={2}
                />
                {[m.accuracy_ci_low, m.accuracy_ci_high].map((v, k) => (
                  <g key={k}>
                    <line x1={x(v)} y1={cy - 7} x2={x(v)} y2={cy + 7} stroke="var(--surface-1)" strokeWidth={5} />
                    <line x1={x(v)} y1={cy - 7} x2={x(v)} y2={cy + 7} stroke="var(--text-primary)" strokeWidth={2} />
                  </g>
                ))}
              </g>

              {/* Direct label: required relief for the sub-3:1 light-mode series color, and it
                  keeps the value readable without hovering. */}
              <text className="value-label" x={PAD.left + plotW + 10} y={cy + 4}>
                {fmtPct(m.accuracy)}
              </text>
            </g>
          )
        })}

        <text className="axis-label" x={PAD.left + plotW / 2} y={height - 4} textAnchor="middle">
          accuracy — bars show the point estimate, whiskers the Wilson 95% interval
        </text>
      </svg>

      {overlapping.length > 0 && (
        <p className="warnbox">
          Confidence intervals overlap for{' '}
          {overlapping.map(([a, b]) => `${a} / ${b}`).join(', ')} — at this sample size those
          models are <strong>not statistically separable</strong>. Treat the ordering as
          indicative, not a ranking.
        </p>
      )}
      <Tooltip state={state} />
    </>
  )
}

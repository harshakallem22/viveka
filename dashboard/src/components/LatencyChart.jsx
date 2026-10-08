import React from 'react'
import { colorForModel, fmtMs, shortName } from '../api.js'
import Tooltip, { useTooltip } from './Tooltip.jsx'

const PAD = { top: 40, right: 34, bottom: 38, left: 132 }
const ROW = 62
const DOT = 4 // 8px marker

function percentile(sorted, p) {
  if (!sorted.length) return null
  // Nearest-rank, matching viveka.core.metrics.percentile exactly -- the chart must not tell a
  // different story from the CLI by quietly interpolating.
  const idx = Math.max(0, Math.ceil((p / 100) * sorted.length) - 1)
  return sorted[Math.min(idx, sorted.length - 1)]
}

function niceTicks(max, count = 5) {
  const raw = max / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10
  const out = []
  // Strictly <= max: a tick past the data maps outside the plot and hangs off the card edge.
  for (let v = 0; v <= max; v += step) out.push(v)
  return out
}

/**
 * Splay two close labels apart horizontally rather than stacking them.
 *
 * Vertical staggering was tried first and read worse: a label lifted above a crowded row drifts
 * into the row above and looks like it belongs to the wrong model. Anchoring them away from each
 * other keeps both on their own row, unambiguously.
 */
function splay(ax, bx, minGap) {
  if (Math.abs(ax - bx) >= minGap) return ['middle', 'middle', 0, 0]
  return ['end', 'start', -5, 5]
}

/**
 * Per-request latency distribution, one strip per model.
 *
 * Individual points rather than a box plot on purpose: the interesting finding in this data is
 * that one model's tail is two runaway generations that hit the token cap, and a five-number
 * summary hides exactly that. The shape is the story.
 */
export default function LatencyChart({ latencies, width = 720 }) {
  const { state, show, hide } = useTooltip()
  const models = Object.keys(latencies).sort()
  const series = models.map((m) => ({
    model: m,
    values: [...latencies[m]].sort((a, b) => a - b),
  }))
  const max = Math.max(...series.flatMap((s) => s.values), 1)

  if (!series.some((s) => s.values.length)) return <p className="muted">No latency samples.</p>

  const height = PAD.top + series.length * ROW + PAD.bottom
  const plotW = width - PAD.left - PAD.right
  const x = (v) => PAD.left + (v / max) * plotW
  const ticks = niceTicks(max)

  return (
    <>
      <div className="legend">
        {series.map((s) => (
          <span className="legend-item" key={s.model}>
            <span
              className="swatch"
              style={{ background: colorForModel(s.model, models) }}
              aria-hidden="true"
            />
            {shortName(s.model)}
          </span>
        ))}
        <span className="legend-item">
          <span
            className="swatch"
            style={{ background: 'var(--text-primary)', height: 2, width: 12, borderRadius: 0 }}
            aria-hidden="true"
          />
          P50 / P95
        </span>
      </div>

      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Per-request latency distribution by model"
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={x(t)} y1={PAD.top - 8} x2={x(t)} y2={PAD.top + series.length * ROW} />
            <text className="tick-label" x={x(t)} y={height - PAD.bottom + 20} textAnchor="middle">
              {(t / 1000).toFixed(0)}s
            </text>
          </g>
        ))}

        {series.map((s, i) => {
          const color = colorForModel(s.model, models)
          const cy = PAD.top + i * ROW + ROW / 2
          const p50 = percentile(s.values, 50)
          const p95 = percentile(s.values, 95)

          // Nearest-point layer: dense strips must not demand pixel-perfect aim.
          const onMove = (event) => {
            const svg = event.currentTarget.ownerSVGElement
            const rect = svg.getBoundingClientRect()
            const scale = width / rect.width
            const localX = (event.clientX - rect.left) * scale
            const target = ((localX - PAD.left) / plotW) * max
            let best = s.values[0]
            for (const v of s.values) {
              if (Math.abs(v - target) < Math.abs(best - target)) best = v
            }
            const rank = s.values.indexOf(best) + 1
            show(event, shortName(s.model), [
              ['nearest sample', fmtMs(best)],
              ['rank', `${rank} of ${s.values.length}`],
              ['P50', fmtMs(p50)],
              ['P95', fmtMs(p95)],
              ['max', fmtMs(s.values[s.values.length - 1])],
            ])
          }

          return (
            <g key={s.model}>
              <rect
                className="hit"
                x={PAD.left}
                y={PAD.top + i * ROW}
                width={plotW}
                height={ROW}
                onMouseMove={onMove}
                onMouseLeave={hide}
              />
              <text className="series-label" x={PAD.left - 12} y={cy + 4} textAnchor="end">
                {shortName(s.model)}
              </text>
              <line className="baseline" x1={PAD.left} y1={cy} x2={PAD.left + plotW} y2={cy} />

              {s.values.map((v, k) => (
                <circle
                  key={k}
                  cx={x(v)}
                  cy={cy}
                  r={DOT}
                  fill={color}
                  fillOpacity={0.5}
                  stroke="var(--surface-1)"
                  strokeWidth={1}
                />
              ))}

              {/* Only P50 and P95 get direct labels -- a number on every dot would be chaos.
                  When the two markers sit close together their labels are staggered rather than
                  allowed to overlap into unreadable mush. */}
              {(() => {
                const [anchorA, anchorB, dxA, dxB] = splay(x(p50), x(p95), 86)
                return [
                  [p50, 'P50', anchorA, dxA],
                  [p95, 'P95', anchorB, dxB],
                ].map(([v, label, anchor, dx]) => (
                  <g key={label}>
                    <line x1={x(v)} y1={cy - 17} x2={x(v)} y2={cy + 17} stroke="var(--surface-1)" strokeWidth={5} />
                    <line x1={x(v)} y1={cy - 17} x2={x(v)} y2={cy + 17} stroke="var(--text-primary)" strokeWidth={2} />
                    <text className="tick-label" x={x(v) + dx} y={cy - 22} textAnchor={anchor}>
                      {label} {fmtMs(v)}
                    </text>
                  </g>
                ))
              })()}
            </g>
          )
        })}

        <text className="axis-label" x={PAD.left + plotW / 2} y={height - 4} textAnchor="middle">
          total latency per request — one dot per generation
        </text>
      </svg>

      <p className="warnbox">
        Cold-start warmup calls are excluded. Measured on this hardware a cold call ran 6.8×
        slower, almost entirely weight loading — left in, it would become the reported P95 and the
        chart would be showing disk speed.
      </p>
      <Tooltip state={state} />
    </>
  )
}

import React from 'react'

/**
 * Follow-cursor tooltip.
 *
 * Tooltips enhance, never gate: every value here is also reachable through a direct label or the
 * table view, and keyboard focus fires the same handler as hover.
 */
export default function Tooltip({ state }) {
  if (!state) return null
  const { x, y, title, rows } = state
  // Flip before the viewport edge instead of letting the box clip.
  const flipX = x > window.innerWidth - 300
  const style = {
    left: flipX ? x - 16 : x + 16,
    top: Math.min(y + 14, window.innerHeight - 120),
    transform: flipX ? 'translateX(-100%)' : 'none',
  }
  return (
    <div className="tooltip" style={style} role="tooltip">
      <div className="tt-title">{title}</div>
      {rows.map(([label, value]) => (
        <div className="tt-row" key={label}>
          {label}: {value}
        </div>
      ))}
    </div>
  )
}

export function useTooltip() {
  const [state, setState] = React.useState(null)
  const show = (event, title, rows) =>
    setState({ x: event.clientX, y: event.clientY, title, rows })
  const hide = () => setState(null)
  return { state, show, hide }
}

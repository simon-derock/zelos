import type { Basis, Cause, CultureState, QcExtras } from '../types'
import { fmt, human, num, sentence } from '../util'

/* ---------------------------------------------------------------- route chip */

type Tone = 'ok' | 'warn' | 'blue' | 'grey'

const ROUTE: Record<string, { tone: Tone; means: string }> = {
  USE: { tone: 'ok', means: 'Use as is.' },
  'RE-IMAGE': { tone: 'warn', means: 'Fix at the microscope now.' },
  'INSPECT-CHIP': { tone: 'warn', means: 'Look at the channel or optics.' },
  'RE-EXPORT': { tone: 'blue', means: 'Re-export as lossless.' },
  REVIEW: { tone: 'grey', means: 'A person decides.' },
}

const routeKey = (r: string) => r.trim().toUpperCase().replace(/[_ ]+/g, '-')

export function routeMeaning(route?: string | null): string | undefined {
  return route ? ROUTE[routeKey(route)]?.means : undefined
}

export function RouteChip({ route, big, small }: { route?: string | null; big?: boolean; small?: boolean }) {
  if (!route) return null
  const k = routeKey(route)
  const tone = ROUTE[k]?.tone ?? 'grey'
  return (
    <span className={`route route-${tone} ${big ? 'route-big' : ''} ${small ? 'route-small' : ''}`} title={ROUTE[k]?.means}>
      <svg viewBox="0 0 16 16" width="1em" height="1em" aria-hidden="true">
        <path d="M3 8h9M8.5 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <span className="sr">Next step: </span>
      {k}
    </span>
  )
}

/** The prominent "what to do" row of a certificate: action chip plus a plain-language meaning. */
export function RouteRow({ route }: { route?: string | null }) {
  if (!route) return null
  const m = routeMeaning(route)
  return (
    <div className="route-row">
      <RouteChip route={route} />
      {m && <span className="route-mean">{m}</span>}
    </div>
  )
}

/* ---------------------------------------------------------------- literature basis */

/** "Why this action": the verified passage behind the recommended action. Renders nothing when `basis` is absent. */
export function WhyAction({ cause }: { cause?: Pick<Cause, 'basis'> | null }) {
  if (!cause || cause.basis === undefined) return null
  const b: Basis | null = cause.basis
  return (
    <div className="why">
      <p className="why-k">Why this action</p>
      {b === null ? (
        <p className="why-none">No verified literature basis in the current corpus</p>
      ) : (
        <>
          <p className="why-title">
            {b.url ? (
              <a href={b.url} rel="noopener noreferrer">
                {b.title ?? b.chunk_id}
              </a>
            ) : (
              (b.title ?? b.chunk_id)
            )}
            {b.license && <span className="lic">{b.license}</span>}
          </p>
          {b.excerpt && <p className="why-ex">“{b.excerpt}”</p>}
        </>
      )}
    </div>
  )
}

/* ---------------------------------------------------------------- how it was decided */

export function HowDecided({ qc, causes }: { qc: QcExtras; causes?: Cause[] }) {
  const th = qc.thresholds
  const sims = (causes ?? []).filter((c) => num(c.similarity) !== undefined)
  const rows: Array<[string, string]> = []
  if (qc.attribution && qc.attribution !== 'none') {
    const how =
      qc.attribution === 'signature-centroid'
        ? 'Matched to the closest fault signature (cosine similarity to a centroid of injected faults)'
        : qc.attribution === 'rule-table'
          ? 'Metric rule table (no signature match)'
          : sentence(qc.attribution)
    rows.push(['Cause', how])
  }
  if (sims.length) rows.push(['Similarity', sims.map((c) => `${sentence(c.label)} ${fmt(c.similarity, 2)}`).join(' · ')])
  if (th && (num(th.review) !== undefined || num(th.fail) !== undefined)) {
    const src = th.source === 'calibrated' ? 'calibrated on expert-good images' : th.source === 'pre-registered' ? 'pre-registered, not tuned' : th.source
    rows.push([
      'Thresholds',
      `Review at QC score ≥ ${fmt(th.review, 1)}, fail at ≥ ${fmt(th.fail, 1)}${src ? ` (${src})` : ''}`,
    ])
  }
  if (qc.reference) rows.push(['Reference', qc.reference])
  if (!rows.length) return null
  return (
    <details className="how">
      <summary>How this was decided</summary>
      <dl>
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </details>
  )
}

/* ---------------------------------------------------------------- culture state */

const STATE_TONE: Record<string, Tone> = { TYPICAL_GOOD: 'ok', UNCERTAIN: 'grey', TYPICAL_BAD: 'warn' }

export function CultureCard({ state }: { state?: CultureState | null }) {
  if (!state) return null
  const p = num(state.p_bad)
  const key = String(state.state ?? '').toUpperCase()
  const tone = STATE_TONE[key] ?? 'grey'
  const title = `Culture state (frozen DINOv2 kNN${num(state.k) !== undefined ? `, k=${state.k}` : ''})`
  return (
    <section className="culture" aria-label={title}>
      <div className="culture-head">
        <h3>{title}</h3>
        {state.state && <span className={`route route-${tone} route-small`}>{human(String(state.state).toLowerCase())}</span>}
      </div>
      {p !== undefined && (
        <div className="culture-bar">
          <div className="culture-track" role="img" aria-label={`Share of nearest reference cultures that experts rejected: ${(p * 100).toFixed(0)}%`}>
            <span className={`culture-fill culture-${tone}`} style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }} />
          </div>
          <div className="culture-ends">
            <span>resembles accepted</span>
            <b>p(bad) {fmt(p, 2)}</b>
            <span>resembles rejected</span>
          </div>
        </div>
      )}
      <p className="culture-cap">
        A different question from image quality: how much this culture resembles expert-accepted vs rejected cultures.
        {num(state.bank) !== undefined && <> Compared with {state.bank} labelled reference images.</>}
      </p>
    </section>
  )
}

import { Fragment, useEffect, useMemo, useState } from 'react'
import type { Card, DraftClaim, LedgerClaim, Sample, SiteData } from '../types'
import { fmt, human, num, sentence } from '../util'
import { cellField } from './Hero'
import { LevelBadge, Reveal, SectionHead } from './common'

function hash(s: string) {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return h >>> 0
}

function Placeholder({ id }: { id: string }) {
  const cells = useMemo(() => cellField(hash(id), 22, 400, 300), [id])
  return (
    <svg className="ph" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <rect width="400" height="300" fill="#cfcfd4" />
      {cells.map((c, i) => (
        <g key={i} transform={`translate(${c.x} ${c.y}) rotate(${c.rot})`}>
          <ellipse rx={c.rx * 0.7} ry={c.ry * 0.7} fill="rgba(255,255,255,.35)" stroke="rgba(40,40,52,.55)" strokeWidth="1.6" />
          <circle cx={c.nx * 0.6} cy={c.ny * 0.6} r={c.nr * 0.6} fill="rgba(60,60,78,.3)" />
        </g>
      ))}
    </svg>
  )
}

function Thumb({ s, large }: { s: Sample; large?: boolean }) {
  const [failed, setFailed] = useState(false)
  useEffect(() => setFailed(false), [s.id, s.thumb])
  const alt = `Brightfield image ${s.id}${s.cell_line ? `, ${s.cell_line} cells` : ''}, QC level ${s.qc.level}`
  if (!s.thumb || failed) return <Placeholder id={s.id} />
  return (
    <img
      src={s.thumb}
      alt={alt}
      loading={large ? 'eager' : 'lazy'}
      decoding="async"
      width={400}
      height={300}
      onError={() => setFailed(true)}
    />
  )
}

function variantLabel(v?: string) {
  if (!v || v === 'original') return 'Original image'
  const p = v.split(':')
  if (p[0] === 'corrupted') return `Synthetic ${human(p[1] ?? 'corruption').toLowerCase()}${p[2] ? `, severity ${p[2]}` : ''}`
  return v
}

function Chips({ text, evidence }: { text: string; evidence?: Card['evidence'] }) {
  const parts = text.split(/(\[[^\]]+\])/g).filter(Boolean)
  return (
    <>
      {parts.map((p, i) => {
        const m = /^\[([^\]]+)\]$/.exec(p)
        if (!m) return <Fragment key={i}>{p}</Fragment>
        const id = m[1]
        const ev = evidence?.find((e) => e.chunk_id === id)
        const tool = id.startsWith('qc-')
        const cls = `chip ${tool ? 'chip-tool' : 'chip-lit'}`
        return ev?.url ? (
          <a key={i} className={cls} href={ev.url} rel="noopener noreferrer" title={ev.title}>
            {id}
          </a>
        ) : (
          <span key={i} className={cls} title={tool ? 'Tool output' : 'Retrieved passage'}>
            {id}
          </span>
        )
      })}
    </>
  )
}

function EvChips({ ids }: { ids?: string[] }) {
  if (!ids || ids.length === 0) return <span className="chip chip-none">no evidence</span>
  return (
    <>
      {ids.map((id) => (
        <span key={id} className={`chip ${id.startsWith('qc-') ? 'chip-tool' : 'chip-lit'}`}>
          {id}
        </span>
      ))}
    </>
  )
}

function CertificateCard({ s }: { s: Sample }) {
  const c = s.card
  const causes = s.qc.causes ?? []
  const cause = c?.primary_cause ?? causes[0]?.label
  const action = c?.action ?? causes[0]?.action
  const clean = !cause || cause === 'NONE'
  return (
    <aside className="cert" aria-label="Certificate">
      <div className="cert-head">
        <span className="cert-title">Certificate</span>
        <LevelBadge level={c?.level ?? s.qc.level} />
      </div>
      <div className="cert-block">
        <p className="k">Primary cause</p>
        <p className="cert-cause">{clean ? 'No issue found' : sentence(cause)}</p>
        {causes[0]?.note && !clean && <p className="muted">{sentence(causes[0].note)}.</p>}
      </div>
      <div className="cert-block">
        <p className="k">Recommended action</p>
        <p className="cert-action">{clean ? 'No action needed. Keep this image in the dataset.' : action || '—'}</p>
      </div>
      {c?.claims && c.claims.length > 0 && (
        <div className="cert-block">
          <p className="k">Supported claims</p>
          <ul className="claims">
            {c.claims.map((t, i) => (
              <li key={i}>
                <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                  <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span>
                  <Chips text={t} evidence={c.evidence} />
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {c?.evidence && c.evidence.length > 0 && (
        <div className="cert-block">
          <p className="k">Evidence</p>
          <ul className="evs">
            {c.evidence.map((e) => (
              <li key={e.chunk_id}>
                <p className="ev-title">
                  {e.url ? (
                    <a href={e.url} rel="noopener noreferrer">
                      {e.title}
                    </a>
                  ) : (
                    e.title
                  )}
                </p>
                {e.excerpt && <p className="ev-ex">“{e.excerpt}”</p>}
                <p className="ev-meta">
                  <span className="chip chip-lit">{e.chunk_id}</span>
                  {e.license && <span>{e.license}</span>}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="cert-disc">{c?.disclaimer ?? 'Research tool. Not a medical device. No clinical or diagnostic claims.'}</p>
    </aside>
  )
}

const statusLabel: Record<string, string> = {
  supported: 'Supported',
  unsupported: 'Blocked',
  refused: 'Refused',
  abstain: 'Abstained',
}

function Ledger({ s, ungated }: { s: Sample; ungated: boolean }) {
  const claims = s.ledger?.claims ?? []
  const draft = s.ungated_draft ?? []
  const byText = useMemo(() => {
    const m = new Map<string, LedgerClaim>()
    for (const c of claims) m.set(c.text.trim(), c)
    return m
  }, [claims])
  const ns = s.ledger?.n_supported ?? claims.filter((c) => c.status === 'supported').length
  const nu = s.ledger?.n_unsupported ?? claims.filter((c) => c.status === 'unsupported' || c.status === 'refused').length

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h3>{ungated ? 'What the LLM said, ungated' : 'Claim ledger'}</h3>
          <p className="muted">
            {ungated
              ? 'The raw draft before the grounding gate. Claims the gate would block are marked.'
              : 'Every claim is checked against tool outputs and retrieved passages before it reaches the certificate.'}
          </p>
        </div>
        <div className="ledger-stats" aria-label="Ledger summary">
          <span className="stat-ok">{ns} supported</span>
          <span className="stat-bad">{nu} blocked</span>
          {typeof s.ledger?.grounded_rate === 'number' && <span>{(s.ledger.grounded_rate * 100).toFixed(0)}% grounded</span>}
        </div>
      </div>
      {!ungated && (
        <ul className="ledger">
          {claims.length === 0 && <li className="muted pad">No claims recorded for this image.</li>}
          {claims.map((c) => {
            const st = c.status === 'supported' ? 'ok' : c.status === 'abstain' ? 'grey' : 'bad'
            return (
              <li key={c.claim_id} className={`lrow lrow-${st}`}>
                <span className={`lstatus lstatus-${st}`}>{statusLabel[c.status] ?? c.status}</span>
                <div className="lmain">
                  <p className="ltext">{c.text}</p>
                  {c.reason && <p className="lreason">{c.reason}</p>}
                </div>
                <div className="lev">
                  {c.type && <span className="ltype">{c.type}</span>}
                  <EvChips ids={c.evidence} />
                </div>
              </li>
            )
          })}
        </ul>
      )}
      {ungated && (
        <ul className="ledger">
          {draft.length === 0 && <li className="muted pad">No ungated draft recorded for this image.</li>}
          {draft.map((d: DraftClaim, i) => {
            const l = byText.get(d.text.trim())
            const blocked = l ? l.status !== 'supported' : !d.evidence || d.evidence.length === 0
            return (
              <li key={i} className={`lrow ${blocked ? 'lrow-bad lrow-hot' : 'lrow-ok'}`}>
                <span className={`lstatus ${blocked ? 'lstatus-bad' : 'lstatus-ok'}`}>{blocked ? 'Would be blocked' : 'Supported'}</span>
                <div className="lmain">
                  <p className="ltext">{d.text}</p>
                  {blocked && <p className="lreason">{l?.reason ?? 'No supporting evidence cited.'}</p>}
                </div>
                <div className="lev">
                  {d.type && <span className="ltype">{d.type}</span>}
                  <EvChips ids={d.evidence} />
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function MetricBars({ s, thr }: { s: Sample; thr: number }) {
  const m = s.qc.metrics ?? {}
  const z = s.qc.zbad ?? {}
  const p = s.qc.percentiles ?? {}
  const keys = Object.keys(z).length ? Object.keys(z) : Object.keys(m)
  const rows = keys
    .map((k) => ({ k, z: num(z[k]), p: num(p[k]), v: num(m[k]) }))
    .sort((a, b) => (b.z ?? -99) - (a.z ?? -99))
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h3>Metrics against the reference set</h3>
          <p className="muted">Each dot is the metric’s percentile among reference images. The number is the distance in the bad direction, in standard deviations.</p>
        </div>
      </div>
      <ul className="mbars">
        {rows.map((r) => {
          const flagged = (r.z ?? 0) >= thr
          return (
            <li key={r.k} className={flagged ? 'is-flag' : ''}>
              <span className="mname">{human(r.k)}</span>
              <span className="mval">{fmt(r.v, 3)}</span>
              <span className="mtrack" role="img" aria-label={`${human(r.k)}: percentile ${r.p ?? 'unknown'}`}>
                <span className="mmid" />
                {r.p !== undefined && <span className="mdot" style={{ left: `${Math.max(1, Math.min(99, r.p))}%` }} />}
              </span>
              <span className="mpct">{r.p !== undefined ? `p${Math.round(r.p)}` : '—'}</span>
              <span className={`mz ${flagged ? 'mz-flag' : ''}`}>{r.z !== undefined ? `${r.z > 0 ? '+' : ''}${r.z.toFixed(1)}σ` : '—'}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function Trace({ s }: { s: Sample }) {
  const tr = s.trace ?? []
  const max = Math.max(1, ...tr.map((t) => t.latency_ms))
  const u = s.usage
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h3>Agent trace</h3>
          <p className="muted">Each tool call, in order. Bar length is latency on a square-root scale.</p>
        </div>
      </div>
      <ol className="trace">
        {tr.map((t) => {
          const w = Math.max(1.5, Math.sqrt(t.latency_ms / max) * 100)
          const llm = (t.tokens ?? 0) > 0
          return (
            <li key={t.step}>
              <span className="tstep">{t.step}</span>
              <span className="ttool">{human(t.tool)}</span>
              <span className="tbar">
                <span className={`tfill ${llm ? 'tfill-llm' : ''}`} style={{ width: `${w}%` }} />
              </span>
              <span className="tlat">{t.latency_ms < 10 ? t.latency_ms.toFixed(1) : Math.round(t.latency_ms)} ms</span>
              <span className="ttok">{llm ? `${t.tokens} tokens` : 'no tokens'}</span>
            </li>
          )
        })}
      </ol>
      {u && (
        <dl className="usage">
          <div>
            <dt>LLM calls</dt>
            <dd>{u.llm_calls ?? 0}</dd>
          </div>
          <div>
            <dt>Prompt tokens</dt>
            <dd>{(u.prompt_tokens ?? 0).toLocaleString('en-US')}</dd>
          </div>
          <div>
            <dt>Completion tokens</dt>
            <dd>{(u.completion_tokens ?? 0).toLocaleString('en-US')}</dd>
          </div>
          {typeof s.qc.latency_ms === 'number' && (
            <div>
              <dt>QC latency</dt>
              <dd>{Math.round(s.qc.latency_ms)} ms</dd>
            </div>
          )}
        </dl>
      )}
    </div>
  )
}

export function Console({ data }: { data: SiteData }) {
  const samples = data.samples
  const [idx, setIdx] = useState(0)
  const [ungated, setUngated] = useState(false)
  const s = samples[Math.min(idx, samples.length - 1)]
  const thr = num(data.results?.e2?.zelos_at_review_threshold?.threshold_z) ?? 3

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'ArrowRight') {
        setIdx((i) => Math.min(samples.length - 1, i + 1))
      } else if (e.key === 'ArrowLeft') {
        setIdx((i) => Math.max(0, i - 1))
      } else return
      const el = document.getElementById('console')
      if (el) {
        const r = el.getBoundingClientRect()
        if (r.top < innerHeight && r.bottom > 0) e.preventDefault()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [samples.length])

  if (!s) return null
  const lvl = s.qc.level
  const lvlCls = lvl === 'PASS' ? 'ok' : lvl === 'FAIL' ? 'bad' : 'warn'

  return (
    <section className="section console-sec" id="console" aria-labelledby="console-h">
      <div className="wrap wide">
        <SectionHead
          id="console-h"
          title="The console."
          lede={
            <>
              Pick an image. Use <kbd>←</kbd> <kbd>→</kbd> to move through the set.
            </>
          }
        />
        <Reveal>
          <div className="console">
            <nav className="gallery" aria-label="Sample images">
              <ul>
                {samples.map((x, i) => (
                  <li key={x.id}>
                    <button
                      type="button"
                      className={`gthumb ${i === idx ? 'is-on' : ''}`}
                      aria-current={i === idx ? 'true' : undefined}
                      aria-label={`${x.id}, ${x.qc.level}`}
                      onClick={() => setIdx(i)}
                    >
                      <Thumb s={x} />
                      <span className={`gdot gdot-${x.qc.level === 'PASS' ? 'ok' : x.qc.level === 'FAIL' ? 'bad' : 'warn'}`} aria-hidden="true" />
                    </button>
                  </li>
                ))}
              </ul>
            </nav>
            <div className="stage">
              <div className="viewer">
                <Thumb s={s} large />
                <div className="viewer-badge">
                  <LevelBadge level={lvl} big />
                </div>
              </div>
              <div className="stage-meta">
                <div className="score">
                  <span className={`score-n score-${lvlCls}`}>{fmt(s.qc.qc_score, 1)}</span>
                  <span className="score-l">QC score</span>
                </div>
                <dl className="facts">
                  <div>
                    <dt>Image</dt>
                    <dd>{s.id}</dd>
                  </div>
                  {s.cell_line && (
                    <div>
                      <dt>Cell line</dt>
                      <dd>{s.cell_line}</dd>
                    </div>
                  )}
                  {s.time_bin && (
                    <div>
                      <dt>Time in culture</dt>
                      <dd>{s.time_bin} h</dd>
                    </div>
                  )}
                  {s.qc.density_state && (
                    <div>
                      <dt>Density</dt>
                      <dd>{sentence(s.qc.density_state)}</dd>
                    </div>
                  )}
                  {s.expert_label && (
                    <div>
                      <dt>Expert label</dt>
                      <dd>{sentence(s.expert_label)}</dd>
                    </div>
                  )}
                  <div>
                    <dt>Source</dt>
                    <dd>{variantLabel(s.variant)}</dd>
                  </div>
                </dl>
              </div>
            </div>
            <CertificateCard s={s} />
          </div>
        </Reveal>

        <Reveal className="below">
          <div className="gate-toggle">
            <button
              type="button"
              role="switch"
              aria-checked={ungated}
              className={`switch ${ungated ? 'is-on' : ''}`}
              onClick={() => setUngated((v) => !v)}
            >
              <span className="knob" />
              <span className="sr">Show what the LLM said without the gate</span>
            </button>
            <span className="gate-label" aria-hidden="true" onClick={() => setUngated((v) => !v)}>
              Show what the LLM said without the gate
            </span>
          </div>
          <Ledger s={s} ungated={ungated} />
          <div className="two">
            <MetricBars s={s} thr={thr} />
            <Trace s={s} />
          </div>
        </Reveal>
      </div>
    </section>
  )
}

/// <reference types="vite/client" />
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fmt, human, num, sentence } from '../util'
import { CopyButton, LevelBadge, Reveal, SectionHead } from './common'

/* ------------------------------------------------------------------ config */
const API: string = ((import.meta.env.VITE_ZELOS_API as string | undefined) ?? '').trim().replace(/\/+$/, '')
const MAX_BYTES = 25 * 1024 * 1024
const MAX_FILES = 12
const MIN_SESSION = 4
const OK_EXT = /\.(png|jpe?g|tiff?)$/i
const OK_MIME = new Set(['image/png', 'image/jpeg', 'image/tiff'])
const DISCLAIMER = 'Research tool. Not a medical device. No clinical or diagnostic claims.'
const LOCAL_CMD = 'make api'
const LOCAL_WEB = 'cd web && VITE_ZELOS_API=http://localhost:8000 npm run dev'

/* ------------------------------------------------------------------ API types */
interface ApiCause {
  label: string
  note?: string
  metric?: string
  value?: number
  z?: number
  action?: string
}
interface ApiQC {
  tool_call_id: string
  qc_score: number
  level: string
  density_state?: string
  metrics?: Record<string, number>
  zbad?: Record<string, number>
  percentiles?: Record<string, number>
  causes?: ApiCause[]
  latency_ms?: number
  image?: { id?: string; shape?: number[] }
}
interface ApiClaim {
  claim_id: string
  text: string
  type?: string
  evidence?: string[]
}
interface ApiLedgerClaim extends ApiClaim {
  status: string
  reason?: string
}
interface ApiEvidence {
  chunk_id: string
  title: string
  url?: string
  license?: string
  excerpt?: string
  cited?: boolean
}
interface ApiCard {
  level?: string
  qc_score?: number
  density_state?: string
  primary_cause?: ApiCause | string | null
  claims?: ApiClaim[]
  blocked_claims?: { text: string; reason?: string }[]
  evidence?: ApiEvidence[]
  disclaimer?: string
}
interface ApiExplain {
  provider?: string
  card: ApiCard
  ledger: { claims: ApiLedgerClaim[]; n_supported?: number; n_unsupported?: number; grounded_rate?: number | null }
  trace?: { step?: string | number; tool: string; latency_ms: number; tokens?: number }[]
  usage?: { llm_calls?: number; prompt_tokens?: number; completion_tokens?: number }
  qc?: ApiQC
}
interface TriageRow {
  id: string
  level: string
  qc_score: number
  causes: ApiCause[]
  density_state?: string
  tool_call_id: string
  reference?: string
}
interface Certificate {
  qc: ApiQC
  res: ApiExplain
}

/* ------------------------------------------------------------------ transport */
class ApiError extends Error {
  status: number
  retryAfter?: number
  constructor(status: number, message: string, retryAfter?: number) {
    super(message)
    this.status = status
    this.retryAfter = retryAfter
  }
}

function send<T>(
  path: string,
  form: FormData | null,
  opts: { onUpload?: (p: number) => void; onSent?: () => void; signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const x = new XMLHttpRequest()
    x.open(form ? 'POST' : 'GET', `${API}${path}`)
    x.timeout = opts.timeoutMs ?? 120_000
    x.responseType = 'text'
    if (x.upload) {
      x.upload.onprogress = (e) => {
        if (e.lengthComputable && opts.onUpload) opts.onUpload(e.loaded / e.total)
      }
      x.upload.onload = () => opts.onSent?.()
    }
    x.onload = () => {
      let body: unknown = null
      try {
        body = JSON.parse(x.responseText)
      } catch {
        /* non-JSON body */
      }
      if (x.status >= 200 && x.status < 300 && body !== null) return resolve(body as T)
      const detail = (body as { detail?: unknown } | null)?.detail
      const msg = typeof detail === 'string' ? detail : `The server answered with status ${x.status}.`
      const ra = Number(x.getResponseHeader('Retry-After'))
      reject(new ApiError(x.status, msg, ra > 0 ? ra : undefined))
    }
    x.onerror = () => reject(new ApiError(0, 'Could not reach the demo server.'))
    x.ontimeout = () => reject(new ApiError(0, 'The demo server took too long to answer.'))
    x.onabort = () => reject(new DOMException('aborted', 'AbortError'))
    opts.signal?.addEventListener('abort', () => x.abort())
    x.send(form)
  })
}

function explainError(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 429) return `Too many requests from your network. Try again in ${e.retryAfter ?? 30} seconds.`
    if (e.status === 0)
      return 'Could not reach the demo server. Free hosting sleeps when idle and can need up to a minute to wake. Try again shortly.'
    if (e.status === 413) return 'That upload is too large for the demo server (25 MB per image).'
    if (e.status >= 500) return 'The demo server hit an error while analysing this image. Try another image.'
    return e.message
  }
  return 'Something went wrong. Try again.'
}

/* ------------------------------------------------------------------ files */
function checkFile(f: File): string | null {
  const okType = OK_MIME.has(f.type) || OK_EXT.test(f.name)
  if (!okType) return `${f.name}: use a PNG, JPEG or TIFF image.`
  if (f.size > MAX_BYTES) return `${f.name}: ${(f.size / 1048576).toFixed(1)} MB is over the 25 MB limit.`
  if (f.size === 0) return `${f.name}: the file is empty.`
  return null
}

/** A downscaled JPEG data URL for display (data: URLs satisfy the site's img-src policy). Null if the browser cannot decode. */
async function previewOf(file: File, max: number): Promise<string | null> {
  try {
    const bmp = await createImageBitmap(file)
    const s = Math.min(1, max / Math.max(bmp.width, bmp.height))
    const c = document.createElement('canvas')
    c.width = Math.max(1, Math.round(bmp.width * s))
    c.height = Math.max(1, Math.round(bmp.height * s))
    const g = c.getContext('2d')
    if (!g) return null
    g.drawImage(bmp, 0, 0, c.width, c.height)
    bmp.close?.()
    return c.toDataURL('image/jpeg', 0.86)
  } catch {
    return null
  }
}

/* ------------------------------------------------------------------ certificate pieces (same markup and classes as the console) */
function Chips({ text, evidence }: { text: string; evidence?: ApiEvidence[] }) {
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

function causeOf(c: Certificate): { label?: string; note?: string; action?: string } {
  const pc = c.res.card.primary_cause
  const first = c.qc.causes?.[0]
  if (pc && typeof pc === 'object') return { label: pc.label, note: pc.note ?? first?.note, action: pc.action ?? first?.action }
  if (typeof pc === 'string') return { label: pc, note: first?.note, action: first?.action }
  return { label: first?.label, note: first?.note, action: first?.action }
}

function CertificateCard({ c }: { c: Certificate }) {
  const card = c.res.card
  const { label, note, action } = causeOf(c)
  const clean = !label || label === 'NONE'
  const cited = (card.evidence ?? []).filter((e) => e.cited)
  const shown = cited.length ? cited : (card.evidence ?? []).slice(0, 3)
  return (
    <aside className="cert" aria-label="Certificate">
      <div className="cert-head">
        <span className="cert-title">Certificate</span>
        <LevelBadge level={card.level ?? c.qc.level} />
      </div>
      <div className="cert-block">
        <p className="k">Primary cause</p>
        <p className="cert-cause">{clean ? 'No issue found' : sentence(label)}</p>
        {note && !clean && <p className="muted">{sentence(note)}.</p>}
      </div>
      <div className="cert-block">
        <p className="k">Recommended action</p>
        <p className="cert-action">{clean ? 'No action needed. Keep this image in the dataset.' : action || '—'}</p>
      </div>
      {card.claims && card.claims.length > 0 && (
        <div className="cert-block">
          <p className="k">Supported claims</p>
          <ul className="claims">
            {card.claims.map((t) => (
              <li key={t.claim_id}>
                <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                  <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span>
                  <Chips text={t.text} evidence={card.evidence} />
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {shown.length > 0 && (
        <div className="cert-block">
          <p className="k">Evidence</p>
          <ul className="evs">
            {shown.map((e) => (
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
      <p className="cert-disc">{card.disclaimer ?? DISCLAIMER}</p>
    </aside>
  )
}

const statusLabel: Record<string, string> = { supported: 'Supported', unsupported: 'Blocked', refused: 'Refused', abstain: 'Abstained' }

function LedgerPanel({ c }: { c: Certificate }) {
  const claims = c.res.ledger.claims ?? []
  const ns = c.res.ledger.n_supported ?? claims.filter((x) => x.status === 'supported').length
  const nu = c.res.ledger.n_unsupported ?? claims.filter((x) => x.status === 'unsupported' || x.status === 'refused').length
  const gr = c.res.ledger.grounded_rate
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h3>Claim ledger</h3>
          <p className="muted">Every claim is checked against tool outputs and retrieved passages before it reaches the certificate.</p>
        </div>
        <div className="ledger-stats" aria-label="Ledger summary">
          <span className="stat-ok">{ns} supported</span>
          <span className="stat-bad">{nu} blocked</span>
          {typeof gr === 'number' && <span>{(gr * 100).toFixed(0)}% grounded</span>}
        </div>
      </div>
      <ul className="ledger">
        {claims.length === 0 && <li className="muted pad">No claims were recorded for this image.</li>}
        {claims.map((x) => {
          const st = x.status === 'supported' ? 'ok' : x.status === 'abstain' ? 'grey' : 'bad'
          return (
            <li key={x.claim_id} className={`lrow lrow-${st}`}>
              <span className={`lstatus lstatus-${st}`}>{statusLabel[x.status] ?? x.status}</span>
              <div className="lmain">
                <p className="ltext">{x.text}</p>
                {x.reason && x.reason !== 'ok' && <p className="lreason">{x.reason}</p>}
              </div>
              <div className="lev">
                {x.type && <span className="ltype">{x.type}</span>}
                <EvChips ids={x.evidence} />
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function MetricPanel({ qc }: { qc: ApiQC }) {
  const m = qc.metrics ?? {}
  const z = qc.zbad ?? {}
  const p = qc.percentiles ?? {}
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
          const flagged = (r.z ?? 0) >= 3
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

function TracePanel({ c }: { c: Certificate }) {
  const tr = c.res.trace ?? []
  const max = Math.max(1, ...tr.map((t) => t.latency_ms))
  const u = c.res.usage
  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h3>Agent trace</h3>
          <p className="muted">Each tool call, in order. This certificate used the deterministic template explainer, so no language model was called.</p>
        </div>
      </div>
      <ol className="trace">
        {tr.map((t, i) => {
          const w = Math.max(1.5, Math.sqrt(t.latency_ms / max) * 100)
          const llm = (t.tokens ?? 0) > 0
          return (
            <li key={i}>
              <span className="tstep">{i + 1}</span>
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
          {typeof c.qc.latency_ms === 'number' && (
            <div>
              <dt>QC latency</dt>
              <dd>{Math.round(c.qc.latency_ms)} ms</dd>
            </div>
          )}
        </dl>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ state */
type Phase = 'idle' | 'uploading' | 'analysing' | 'done' | 'error'
interface Run<T> {
  phase: Phase
  pct: number
  data?: T
  error?: string
}
const idle = <T,>(): Run<T> => ({ phase: 'idle', pct: 0 })

function Spinner() {
  return <span className="try-spin" aria-hidden="true" />
}

function Progress({ run, what }: { run: Run<unknown>; what: string }) {
  const up = run.phase === 'uploading'
  return (
    <div className="try-progress" role="status" aria-live="polite">
      <Spinner />
      <div className="try-prog-body">
        <p className="try-prog-title">{up ? `Uploading ${what}` : 'Analysing'}</p>
        <p className="muted small">
          {up ? 'Sending to the demo server. The image is processed in memory and not stored.' : 'Physics QC, cause attribution, evidence retrieval and the grounding gate are running.'}
        </p>
        <span className="try-bar" aria-hidden="true">
          <span className={`try-bar-fill ${up ? '' : 'is-indef'}`} style={up ? { width: `${Math.round(run.pct * 100)}%` } : undefined} />
        </span>
      </div>
    </div>
  )
}

function Unconfigured() {
  const text = `${LOCAL_CMD}\n${LOCAL_WEB}`
  return (
    <div className="try-card try-off">
      <h3>The live demo server is not configured.</h3>
      <p className="muted">
        This build of the site has no demo server address, so uploads are switched off. To try your own image, run the API on your machine and point the site at it.
      </p>
      <div className="term try-term" role="group" aria-label="Local commands">
        <div className="term-bar">
          <span className="dot r" />
          <span className="dot y" />
          <span className="dot g" />
          <span className="term-title">Terminal</span>
          <CopyButton text={text} />
        </div>
        <pre className="term-body">
          <div className="term-line">
            <span className="prompt" aria-hidden="true">$ </span>
            <code>{LOCAL_CMD}</code>
            <span className="term-note"># starts the API on http://localhost:8000</span>
          </div>
          <div className="term-line">
            <span className="prompt" aria-hidden="true">$ </span>
            <code>{LOCAL_WEB}</code>
          </div>
        </pre>
      </div>
      <p className="try-note">
        Hosted deployments set <code>VITE_ZELOS_API</code> to the address of the Hugging Face Space that runs the same API.
      </p>
    </div>
  )
}

/* ------------------------------------------------------------------ main */
type Mode = 'single' | 'triage'
type Health = 'checking' | 'online' | 'offline'

function useHealth(enabled: boolean) {
  const [state, setState] = useState<{ h: Health; version?: string; retriever?: boolean }>({ h: 'checking' })
  const [n, setN] = useState(0)
  useEffect(() => {
    if (!enabled) return
    let alive = true
    setState({ h: 'checking' })
    send<{ status?: string; version?: string; retriever_loaded?: boolean }>('/api/health', null, { timeoutMs: 70_000 })
      .then((b) => alive && setState({ h: 'online', version: b.version, retriever: b.retriever_loaded }))
      .catch(() => alive && setState({ h: 'offline' }))
    return () => {
      alive = false
    }
  }, [enabled, n])
  return { ...state, recheck: () => setN((v) => v + 1) }
}

export function Try() {
  const configured = API.length > 0
  const health = useHealth(configured)
  const [mode, setMode] = useState<Mode>('single')
  const [notice, setNotice] = useState<string | null>(null)
  const [drag, setDrag] = useState(false)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [single, setSingle] = useState<Run<Certificate>>(idle())

  const [files, setFiles] = useState<File[]>([])
  const [thumbs, setThumbs] = useState<Record<string, string | null>>({})
  const [session, setSession] = useState<Run<TriageRow[]>>(idle())

  useEffect(() => () => abortRef.current?.abort(), [])

  const busy = single.phase === 'uploading' || single.phase === 'analysing' || session.phase === 'uploading' || session.phase === 'analysing'

  const runSingle = useCallback(async (f: File) => {
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    setFile(f)
    setPreview(null)
    setMode('single')
    setSingle({ phase: 'uploading', pct: 0 })
    previewOf(f, 1100).then((p) => !ac.signal.aborted && setPreview(p))
    const fd = new FormData()
    fd.append('file', f, f.name)
    try {
      const res = await send<ApiExplain>('/api/explain?provider=template', fd, {
        signal: ac.signal,
        onUpload: (p) => setSingle((s) => (s.phase === 'uploading' ? { ...s, pct: p } : s)),
        onSent: () => setSingle((s) => (s.phase === 'uploading' ? { phase: 'analysing', pct: 1 } : s)),
      })
      let qc = res.qc
      if (!qc) {
        // Older servers do not attach the QC dict to the explanation; ask for it separately.
        const fd2 = new FormData()
        fd2.append('file', f, f.name)
        qc = await send<ApiQC>('/api/inspect', fd2, { signal: ac.signal })
      }
      setSingle({ phase: 'done', pct: 1, data: { qc, res } })
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      setSingle({ phase: 'error', pct: 0, error: explainError(e) })
    }
  }, [])

  const runSession = useCallback(async (fs: File[]) => {
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    setFiles(fs)
    setMode('triage')
    setThumbs({})
    setSession({ phase: 'uploading', pct: 0 })
    fs.forEach((f) =>
      previewOf(f, 160).then((p) => {
        if (!ac.signal.aborted) setThumbs((t) => ({ ...t, [`${f.name}:${f.size}`]: p }))
      }),
    )
    const fd = new FormData()
    fs.forEach((f) => fd.append('files', f, f.name))
    try {
      const rows = await send<TriageRow[]>('/api/triage', fd, {
        signal: ac.signal,
        onUpload: (p) => setSession((s) => (s.phase === 'uploading' ? { ...s, pct: p } : s)),
        onSent: () => setSession((s) => (s.phase === 'uploading' ? { phase: 'analysing', pct: 1 } : s)),
      })
      setSession({ phase: 'done', pct: 1, data: rows })
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      setSession({ phase: 'error', pct: 0, error: explainError(e) })
    }
  }, [])

  const accept = useCallback(
    (list: FileList | File[] | null) => {
      const all = Array.from(list ?? [])
      if (!all.length) return
      setNotice(null)
      const bad = all.map(checkFile).find((m) => m !== null)
      if (bad) return setNotice(bad)
      if (all.length > MAX_FILES) return setNotice(`Select at most ${MAX_FILES} images from one imaging session.`)
      if (all.length === 1) {
        if (mode === 'triage') {
          setNotice(`Session triage compares images against each other and needs at least ${MIN_SESSION}. Showing a single certificate instead.`)
        }
        void runSingle(all[0])
      } else if (all.length >= MIN_SESSION) {
        void runSession(all)
      } else {
        setNotice(`Select 1 image for a certificate, or ${MIN_SESSION} to ${MAX_FILES} images for session triage. You selected ${all.length}.`)
      }
    },
    [mode, runSingle, runSession],
  )

  const offline = configured && health.h === 'offline'
  const current = mode === 'single' ? single : session
  const showDrop = current.phase === 'idle' || current.phase === 'error' || current.phase === 'done'

  const reset = () => {
    abortRef.current?.abort()
    setNotice(null)
    if (mode === 'single') {
      setFile(null)
      setPreview(null)
      setSingle(idle())
    } else {
      setFiles([])
      setSession(idle())
    }
  }

  return (
    <section className="section try-sec" id="try" aria-labelledby="try-h">
      <div className="wrap wide">
        <SectionHead
          id="try-h"
          title="Try your own image."
          lede="Drop a brightfield image and get a live certificate: the physical cause, the lab action, and cited evidence, with every claim checked by the grounding gate."
        />
        {!configured ? (
          <Reveal>
            <Unconfigured />
          </Reveal>
        ) : (
          <Reveal>
            <div className="try-card">
              <div className="try-top">
                <div className="try-seg" role="tablist" aria-label="Mode">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={mode === 'single'}
                    className={mode === 'single' ? 'is-on' : ''}
                    onClick={() => {
                      setMode('single')
                      setNotice(null)
                    }}
                  >
                    One image
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={mode === 'triage'}
                    className={mode === 'triage' ? 'is-on' : ''}
                    onClick={() => {
                      setMode('triage')
                      setNotice(null)
                    }}
                  >
                    Session triage
                  </button>
                </div>
                <p className={`try-health try-health-${health.h}`} role="status">
                  <span className="try-dot" aria-hidden="true" />
                  {health.h === 'checking' && 'Connecting to the demo server'}
                  {health.h === 'online' && `Demo server online${health.version ? `, v${health.version}` : ''}`}
                  {health.h === 'offline' && 'Demo server unreachable'}
                </p>
              </div>

              <p className="muted try-mode-line">
                {mode === 'single'
                  ? 'One image against the reference set from the dataset: verdict, cause, action and evidence.'
                  : `Select ${MIN_SESSION} to ${MAX_FILES} images from one imaging session. Each is scored against the median of the session itself, so a drifting focus or lamp stands out.`}
              </p>

              {offline && (
                <div className="try-alert" role="alert">
                  <p>
                    The demo server did not answer. Free hosting sleeps when idle and can need about a minute to wake. You can also run the API locally with <code>{LOCAL_CMD}</code>.
                  </p>
                  <button type="button" className="try-link" onClick={health.recheck}>
                    Check again
                  </button>
                </div>
              )}

              {(current.phase === 'uploading' || current.phase === 'analysing') && (
                <Progress run={current} what={mode === 'single' ? (file?.name ?? 'image') : `${files.length} images`} />
              )}

              {showDrop && !offline && (
                <div
                  className={`try-drop ${drag ? 'is-drag' : ''}`}
                  onDragEnter={(e) => {
                    e.preventDefault()
                    setDrag(true)
                  }}
                  onDragOver={(e) => {
                    e.preventDefault()
                    setDrag(true)
                  }}
                  onDragLeave={() => setDrag(false)}
                  onDrop={(e) => {
                    e.preventDefault()
                    setDrag(false)
                    if (!busy) accept(e.dataTransfer.files)
                  }}
                >
                  <svg viewBox="0 0 48 48" width="40" height="40" aria-hidden="true">
                    <rect x="6" y="9" width="36" height="30" rx="6" fill="none" stroke="currentColor" strokeWidth="2.2" />
                    <circle cx="17" cy="19" r="3.4" fill="none" stroke="currentColor" strokeWidth="2.2" />
                    <path d="M8 35l10-9 7 6 6-5 11 9" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
                  </svg>
                  <p className="try-drop-title">
                    {mode === 'single' ? 'Drop a brightfield image here' : `Drop ${MIN_SESSION} to ${MAX_FILES} images here`}
                  </p>
                  <p className="muted small">PNG, JPEG or TIFF, up to 25 MB each.</p>
                  <button type="button" className="btn btn-primary try-pick" disabled={busy} onClick={() => inputRef.current?.click()}>
                    {mode === 'single' ? 'Choose an image' : 'Choose images'}
                  </button>
                  <input
                    ref={inputRef}
                    className="sr"
                    type="file"
                    tabIndex={-1}
                    aria-label="Choose image files"
                    accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
                    multiple={mode === 'triage'}
                    onChange={(e) => {
                      accept(e.target.files)
                      e.target.value = ''
                    }}
                  />
                </div>
              )}

              {notice && (
                <p className="try-alert try-alert-inline" role="alert">
                  {notice}
                </p>
              )}
              {current.phase === 'error' && (
                <div className="try-alert" role="alert">
                  <p>{current.error}</p>
                  {mode === 'single' && file && (
                    <button type="button" className="try-link" onClick={() => void runSingle(file)}>
                      Try again
                    </button>
                  )}
                  {mode === 'triage' && files.length >= MIN_SESSION && (
                    <button type="button" className="try-link" onClick={() => void runSession(files)}>
                      Try again
                    </button>
                  )}
                </div>
              )}

              <p className="try-privacy">
                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
                  <path d="M4.5 7V5.2a3.5 3.5 0 017 0V7M3.5 7h9v6.5h-9z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
                </svg>
                Processed in memory on the demo server, not stored. Research use only. Not a medical device. No clinical or diagnostic claims.
              </p>
            </div>
          </Reveal>
        )}

        {configured && mode === 'single' && single.phase === 'done' && single.data && (
          <SingleResult c={single.data} file={file} preview={preview} onReset={reset} backToSession={session.phase === 'done' ? () => setMode('triage') : undefined} />
        )}
        {configured && mode === 'triage' && session.phase === 'done' && session.data && (
          <SessionResult
            rows={session.data}
            files={files}
            thumbs={thumbs}
            busy={busy}
            onCertificate={(f) => void runSingle(f)}
            onReset={reset}
          />
        )}
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ results */
function SingleResult({
  c,
  file,
  preview,
  onReset,
  backToSession,
}: {
  c: Certificate
  file: File | null
  preview: string | null
  onReset: () => void
  backToSession?: () => void
}) {
  const lvl = String(c.res.card.level ?? c.qc.level).toUpperCase()
  const cls = lvl === 'PASS' ? 'ok' : lvl === 'FAIL' ? 'bad' : 'warn'
  const shape = c.qc.image?.shape
  const id = c.qc.image?.id ?? file?.name ?? 'upload'
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [c])
  return (
    <div className="try-result" ref={ref}>
      <div className="try-result-bar">
        <h3>Your certificate</h3>
        <div className="try-result-actions">
          {backToSession && (
            <button type="button" className="try-link" onClick={backToSession}>
              Back to session triage
            </button>
          )}
          <button type="button" className="try-link" onClick={onReset}>
            Clear
          </button>
        </div>
      </div>
      <div className="try-grid">
        <div className="stage">
          <div className="viewer try-viewer">
            {preview ? (
              <img src={preview} alt={`Your uploaded image ${id}, QC level ${lvl}`} />
            ) : (
              <div className="try-nopreview">
                <p>{file ? 'Preview not available for this file type.' : 'No preview.'}</p>
              </div>
            )}
            <div className="viewer-badge">
              <LevelBadge level={lvl} big />
            </div>
          </div>
          <div className="stage-meta">
            <div className="score">
              <span className={`score-n score-${cls}`}>{fmt(c.qc.qc_score, 1)}</span>
              <span className="score-l">QC score</span>
            </div>
            <dl className="facts">
              <div>
                <dt>Image</dt>
                <dd>{id}</dd>
              </div>
              {shape && shape.length >= 2 && (
                <div>
                  <dt>Size</dt>
                  <dd>
                    {shape[1]} × {shape[0]} px
                  </dd>
                </div>
              )}
              {c.qc.density_state && (
                <div>
                  <dt>Density</dt>
                  <dd>{sentence(c.qc.density_state)}</dd>
                </div>
              )}
              <div>
                <dt>Reference</dt>
                <dd>Dataset reference set</dd>
              </div>
              <div>
                <dt>QC call</dt>
                <dd className="try-mono">{c.qc.tool_call_id}</dd>
              </div>
            </dl>
          </div>
        </div>
        <CertificateCard c={c} />
      </div>
      <div className="below">
        <LedgerPanel c={c} />
        <div className="two">
          <MetricPanel qc={c.qc} />
          <TracePanel c={c} />
        </div>
      </div>
    </div>
  )
}

function SessionResult({
  rows,
  files,
  thumbs,
  busy,
  onCertificate,
  onReset,
}: {
  rows: TriageRow[]
  files: File[]
  thumbs: Record<string, string | null>
  busy: boolean
  onCertificate: (f: File) => void
  onReset: () => void
}) {
  const counts = useMemo(() => {
    const c = { PASS: 0, REVIEW: 0, FAIL: 0 } as Record<string, number>
    for (const r of rows) c[r.level] = (c[r.level] ?? 0) + 1
    return c
  }, [rows])
  const sorted = useMemo(() => rows.map((r, i) => ({ r, i })).sort((a, b) => b.r.qc_score - a.r.qc_score), [rows])
  return (
    <div className="try-result">
      <div className="try-result-bar">
        <h3>Session triage</h3>
        <div className="try-result-actions">
          <button type="button" className="try-link" onClick={onReset}>
            Clear
          </button>
        </div>
      </div>
      <div className="panel">
        <div className="panel-head">
          <div>
            <h3>
              {rows.length} images, ranked by QC score
            </h3>
            <p className="muted">
              Reference: the median and spread of these {rows.length} images, not the dataset reference set. A verdict here says an image differs from the rest of the session. Open a certificate to compare it with the dataset reference.
            </p>
          </div>
          <div className="ledger-stats" aria-label="Session summary">
            <span className="stat-ok">{counts.PASS} pass</span>
            <span>{counts.REVIEW} review</span>
            <span className="stat-bad">{counts.FAIL} fail</span>
          </div>
        </div>
        <div className="tbl-wrap">
          <table className="try-tbl">
            <thead>
              <tr>
                <th scope="col">Image</th>
                <th scope="col">Verdict</th>
                <th scope="col" className="r">
                  QC score
                </th>
                <th scope="col">Primary cause</th>
                <th scope="col">Density</th>
                <th scope="col">
                  <span className="sr">Certificate</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(({ r, i }) => {
                const f = files[i]
                const th = f ? thumbs[`${f.name}:${f.size}`] : null
                const cause = r.causes?.[0]
                return (
                  <tr key={r.tool_call_id + i}>
                    <th scope="row" className="try-img">
                      <span className="try-thumb">{th ? <img src={th} alt="" /> : <span className="try-thumb-ph" />}</span>
                      <span className="try-img-name">{r.id}</span>
                    </th>
                    <td>
                      <LevelBadge level={r.level} />
                    </td>
                    <td className="r try-num">{fmt(r.qc_score, 1)}</td>
                    <td>
                      {cause ? (
                        <>
                          <span className="try-cause">{sentence(cause.label)}</span>
                          {cause.action && <span className="try-act">{cause.action}</span>}
                        </>
                      ) : (
                        <span className="muted">No issue found</span>
                      )}
                    </td>
                    <td>{r.density_state ? sentence(r.density_state) : '—'}</td>
                    <td className="r">
                      {f && (
                        <button type="button" className="try-link" disabled={busy} onClick={() => onCertificate(f)}>
                          Certificate
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

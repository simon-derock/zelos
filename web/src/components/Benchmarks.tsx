import type { ReactNode } from 'react'
import type { SiteData } from '../types'
import { fmt, human, isObj, num, numericLeaves, pct, shortSha } from '../util'
import { Reveal, SectionHead, Source } from './common'

type Obj = Record<string, any>

function sha(data: SiteData, key: string): string {
  const r = data.results?.[key]
  return shortSha(r?.provenance?.git_sha ?? data.generated?.git_sha)
}

function Tile({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'base' | 'ok' | 'bad' }) {
  return (
    <div className={`tile ${tone ? `tile-${tone}` : ''}`}>
      <p className="tile-v">{value}</p>
      <p className="tile-l">{label}</p>
      {sub && <p className="tile-s">{sub}</p>}
    </div>
  )
}

function Card({ title, lede, children, foot, half }: { title: string; lede?: ReactNode; children: ReactNode; foot?: ReactNode; half?: boolean }) {
  return (
    <Reveal className={`bcard ${half ? 'half' : ''}`}>
      <h3>{title}</h3>
      {lede && <p className="muted bcard-lede">{lede}</p>}
      {children}
      {foot}
    </Reveal>
  )
}

/* ---------- E1 ---------- */

function E1Tiles({ o, label }: { o: Obj; label?: string }) {
  const rb = num(o.attribution_random_baseline)
  return (
    <div className="tiles-wrap">
      {label && <p className="tiles-label">{label}</p>}
      <div className="tiles">
        {num(o.detection_rate_sev3plus) !== undefined && (
          <Tile label="Corruptions detected, severity 3 and above" value={pct(o.detection_rate_sev3plus, 1)} />
        )}
        {num(o.attribution_acc_detected) !== undefined && (
          <Tile
            label="Correct cause among detected"
            value={pct(o.attribution_acc_detected, 1)}
            sub={
              <>
                {num(o.attribution_top2_detected) !== undefined && <>Top 2: {pct(o.attribution_top2_detected, 1)}. </>}
                {rb !== undefined && <>Random guess: {pct(rb, 1)}.</>}
              </>
            }
          />
        )}
        {num(o.auroc_clean_vs_sev3plus) !== undefined && (
          <Tile
            label="AUROC, clean against severity 3 and above"
            value={fmt(o.auroc_clean_vs_sev3plus, 3)}
            sub={num(o.auroc_focus_only) !== undefined ? <>Focus metric alone: {fmt(o.auroc_focus_only, 3)}</> : undefined}
          />
        )}
      </div>
    </div>
  )
}

function E1({ data }: { data: SiteData }) {
  const e1 = data.results?.e1 as Obj | undefined
  if (!isObj(e1)) return null
  const overall = isObj(e1.overall) ? e1.overall : undefined
  const br = isObj(e1.batch_reference) ? (isObj(e1.batch_reference.overall) ? e1.batch_reference.overall : e1.batch_reference) : undefined
  const kinds = isObj(e1.per_kind) ? Object.entries(e1.per_kind as Obj) : []
  const sweep = isObj(e1.threshold_sweep) ? Object.entries(e1.threshold_sweep as Obj) : []
  const n = num(e1.n_images) ?? num(e1.n)
  return (
    <Card
      title="Corruption benchmark"
      lede="Eleven kinds of synthetic corruption at five severities, applied to real images. Zelos has to notice each one and name the right cause."
      foot={<Source file="e1" sha={sha(data, 'e1')} n={n} />}
    >
      {overall && <E1Tiles o={overall} label={br ? 'Single-image reference' : undefined} />}
      {br && <E1Tiles o={br} label="Batch reference" />}
      {num(e1.clean_flag_rate) !== undefined && (
        <p className="note">
          Clean images flagged for review: <strong>{pct(e1.clean_flag_rate, 1)}</strong>.
        </p>
      )}
      {kinds.length > 0 && (
        <div className="kinds">
          <div className="legend" aria-hidden="true">
            <span><i className="sw sw-1" />Detected at severity 3</span>
            <span><i className="sw sw-2" />Correct cause, when detected</span>
            <span><i className="sw sw-r" />Random guess</span>
          </div>
          <ul className="krows">
            {kinds.map(([k, v]) => {
              const sev = isObj(v.detection_rate_by_severity) ? (v.detection_rate_by_severity as Obj) : {}
              const d3 = num(sev['3']) ?? num(sev[3 as any])
              const at = num(v.attribution_acc_detected)
              const rb = num(overall?.attribution_random_baseline)
              return (
                <li key={k}>
                  <div className="kname">
                    <span>{human(k)}</span>
                    {v.expected_label && <small>{human(String(v.expected_label)).toLowerCase()}</small>}
                  </div>
                  <div className="kbars">
                    <Bar v={d3} cls="b1" label={`${human(k)} detected at severity 3`} />
                    <Bar v={at} cls="b2" tick={rb} label={`${human(k)} correct cause`} />
                  </div>
                  <div className="ksev" aria-label="Detection rate by severity 1 to 5">
                    {['1', '2', '3', '4', '5'].map((s) => {
                      const x = num(sev[s])
                      return (
                        <i
                          key={s}
                          title={`Severity ${s}: ${pct(x, 0)} detected`}
                          style={{ opacity: x === undefined ? 0.1 : 0.15 + 0.85 * x }}
                        />
                      )
                    })}
                  </div>
                </li>
              )
            })}
          </ul>
          <p className="muted small">The strip at the right shows detection from severity 1 to 5, light to strong.</p>
        </div>
      )}
      {sweep.length > 0 && (
        <details className="sweep">
          <summary>Threshold sweep</summary>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Review threshold (z)</th>
                  <th>Clean images flagged</th>
                  <th>Detected at severity 3 and above</th>
                </tr>
              </thead>
              <tbody>
                {sweep.map(([t, v]) => (
                  <tr key={t}>
                    <td>{t}</td>
                    <td>{pct(v.clean_flag_rate, 1)}</td>
                    <td>{pct(v.sev3plus_detection_rate, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </Card>
  )
}

function Bar({ v, cls, tick, label }: { v?: number; cls: string; tick?: number; label: string }) {
  return (
    <div className="bar" role="img" aria-label={`${label}: ${pct(v, 0)}`}>
      <span className="bar-track">
        {v !== undefined && <span className={`bar-fill ${cls}`} style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }} />}
        {tick !== undefined && <span className="bar-tick" style={{ left: `${tick * 100}%` }} />}
      </span>
      <span className="bar-v">{pct(v, 0)}</span>
    </div>
  )
}

/* ---------- E2 ---------- */

function E2({ data }: { data: SiteData }) {
  const e2 = data.results?.e2 as Obj | undefined
  if (!isObj(e2) || !isObj(e2.detectors)) return null
  const dets = Object.entries(e2.detectors as Obj)
  const rv = isObj(e2.zelos_at_review_threshold) ? (e2.zelos_at_review_threshold as Obj) : undefined
  const cells = isObj(e2.per_cell_line) ? Object.entries(e2.per_cell_line as Obj) : []
  const label = (k: string) =>
    k === 'zelos_qc_score' ? 'Zelos QC score' : k === 'focus_only' ? 'Focus metric only' : k === 'random' ? 'Random' : k.startsWith('knn') ? 'k-nearest neighbours on handcrafted features' : human(k)
  return (
    <Card
      title="Agreement with expert labels"
      lede="Can the score separate images experts called good from those they called bad? Area under the ROC curve on held-out sessions, with 95% intervals. 0.5 is chance."
      foot={<Source file="e2" sha={sha(data, 'e2')} n={num(e2.n_test)} />}
    >
      <ul className="dw" aria-label="AUROC per detector">
        {dets.map(([k, v]) => {
          const a = num(v.auroc)
          const ci = Array.isArray(v.auroc_ci95) ? (v.auroc_ci95 as number[]) : undefined
          const lo = ci ? num(ci[0]) : undefined
          const hi = ci ? num(ci[1]) : undefined
          const z = k.includes('zelos')
          return (
            <li key={k}>
              <span className="dw-name">{label(k)}</span>
              <span className="dw-plot" role="img" aria-label={`${label(k)} AUROC ${fmt(a, 2)}${lo !== undefined && hi !== undefined ? `, interval ${fmt(lo, 2)} to ${fmt(hi, 2)}` : ''}`}>
                <span className="dw-ref" />
                {lo !== undefined && hi !== undefined && (
                  <span className={`dw-whisk ${z ? 'is-z' : ''}`} style={{ left: `${lo * 100}%`, width: `${Math.max(0, hi - lo) * 100}%` }} />
                )}
                {a !== undefined && <span className={`dw-dot ${z ? 'is-z' : ''}`} style={{ left: `${a * 100}%` }} />}
              </span>
              <span className="dw-val">
                {fmt(a, 2)}
                {lo !== undefined && hi !== undefined && <small> [{fmt(lo, 2)}, {fmt(hi, 2)}]</small>}
              </span>
            </li>
          )
        })}
        <li className="dw-axis" aria-hidden="true">
          <span />
          <span className="dw-ticks">
            <b style={{ left: '0%' }}>0</b>
            <b style={{ left: '50%' }}>0.5 chance</b>
            <b style={{ left: '100%' }}>1</b>
          </span>
          <span />
        </li>
      </ul>
      {rv && (
        <div className="tiles tiles-sm">
          {num(rv.balanced_accuracy) !== undefined && (
            <Tile label="Balanced accuracy at the review threshold" value={pct(rv.balanced_accuracy, 1)} sub={num(rv.majority_baseline_balanced_accuracy) !== undefined ? <>Majority baseline: {pct(rv.majority_baseline_balanced_accuracy, 1)}</> : undefined} />
          )}
          {num(rv.cohen_kappa) !== undefined && <Tile label="Cohen’s kappa" value={fmt(rv.cohen_kappa, 2)} />}
          {num(rv.flag_rate) !== undefined && <Tile label="Images flagged" value={pct(rv.flag_rate, 1)} />}
        </div>
      )}
      {cells.length > 0 && (
        <details className="sweep">
          <summary>Per cell line</summary>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Cell line</th>
                  <th>n</th>
                  <th>Zelos AUROC</th>
                  <th>k-nearest neighbours AUROC</th>
                </tr>
              </thead>
              <tbody>
                {cells.map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{num(v.n) ?? '—'}</td>
                    <td>{fmt(v.auroc_zelos, 2)}</td>
                    <td>{fmt(v.auroc_knn_handcrafted, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </Card>
  )
}

/* ---------- generic (E4) ---------- */

function GenericTable({ obj }: { obj: Obj }) {
  const leaves = numericLeaves(obj)
  const rows = new Map<string, Map<string, number>>()
  const cols: string[] = []
  for (const [path, v] of leaves) {
    const col = path[path.length - 1] ?? 'value'
    const row = path.slice(0, -1).map(human).join(' / ') || 'Overall'
    if (!rows.has(row)) rows.set(row, new Map())
    rows.get(row)!.set(col, v)
    if (!cols.includes(col)) cols.push(col)
  }
  const shown = cols.slice(0, 8)
  if (rows.size === 0) return null
  return (
    <div className="tbl-wrap">
      <table className="tbl">
        <thead>
          <tr>
            <th />
            {shown.map((c) => (
              <th key={c}>{human(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...rows.entries()].map(([r, m]) => (
            <tr key={r}>
              <td>{r}</td>
              {shown.map((c) => (
                <td key={c}>{m.has(c) ? fmt(m.get(c), Math.abs(m.get(c)!) < 10 ? 3 : 1) : '—'}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function E4({ data, half }: { data: SiteData; half?: boolean }) {
  const e4 = data.results?.e4 as Obj | undefined
  if (!isObj(e4)) return null
  const n = num(e4.n_queries) ?? num(e4.n)
  return (
    <Card half={half} title="Evidence retrieval" lede="How well the literature search finds the passages that support a cause and its fix." foot={<Source file="e4" sha={sha(data, 'e4')} n={n} />}>
      <GenericTable obj={e4} />
    </Card>
  )
}

/* ---------- E5 ---------- */

const rateOf = (x: any): { rate?: number; k?: number; n?: number; ci?: number[] } =>
  isObj(x) ? { rate: num(x.rate), k: num(x.k), n: num(x.n), ci: Array.isArray(x.ci95) ? x.ci95 : undefined } : { rate: num(x) }

function E5({ data }: { data: SiteData }) {
  const e5 = data.results?.e5 as Obj | undefined
  if (!isObj(e5) || !isObj(e5.conditions)) return null
  const conds = Object.entries(e5.conditions as Obj)
  const nice: Record<string, string> = { template: 'Template only', llm_ungated: 'LLM, no gate', llm_gated: 'LLM, with gate' }
  const metrics: Array<[string, string, 'rate' | 'num']> = [
    ['unsupported_claim_rate_in_output', 'Unsupported claims shown to the user', 'rate'],
    ['grounded_claim_rate', 'Claims grounded in evidence', 'rate'],
    ['numeric_hallucination_rate', 'Invented numbers', 'rate'],
    ['citation_valid_rate', 'Citations that support their claim', 'rate'],
    ['supported_claims_per_case', 'Supported claims per answer', 'num'],
    ['tokens_per_answer', 'Tokens per answer', 'num'],
    ['latency_ms_p50', 'Median latency (ms)', 'num'],
  ]
  const present = metrics.filter(([k]) => conds.some(([, c]) => c[k] !== undefined))
  const fi = isObj(e5.gate_fault_injection) ? (e5.gate_fault_injection as Obj) : undefined
  const caught = fi ? rateOf(fi.injected_fabrications_caught) : undefined
  const fb = fi ? rateOf(fi.false_block_rate) : undefined
  const n = num(conds[0]?.[1]?.n_cases) ?? (isObj(e5.n) ? Object.values(e5.n as Obj).reduce((a: number, b: any) => a + (num(b) ?? 0), 0) : num(e5.n))
  const ung = conds.map(([k, c]) => ({ k, r: rateOf(c.unsupported_claim_rate_in_output) })).filter((x) => x.r.rate !== undefined)
  return (
    <Card
      title="The grounding gate"
      lede="The same language-model drafts, shown with and without the gate. The gate is the only difference."
      foot={<Source file="e5" sha={sha(data, 'e5')} n={n ? `${n} cases per condition` : undefined} />}
    >
      {ung.length > 0 && (
        <ul className="gbars" aria-label="Unsupported claims shown, by condition">
          {ung.map(({ k, r }) => (
            <li key={k}>
              <span className="gname">{nice[k] ?? human(k)}</span>
              <span className="gtrack">
                <span className={`gfill ${k.includes('ungated') ? 'is-bad' : 'is-ok'}`} style={{ width: `${Math.max(0, Math.min(1, r.rate ?? 0)) * 100}%` }} />
              </span>
              <span className="gval">
                {pct(r.rate, 1)}
                {r.k !== undefined && r.n !== undefined && <small> {r.k} of {r.n}</small>}
              </span>
            </li>
          ))}
        </ul>
      )}
      {present.length > 0 && (
        <div className="tbl-wrap">
          <table className="tbl tbl-cmp">
            <thead>
              <tr>
                <th />
                {conds.map(([k]) => (
                  <th key={k}>{nice[k] ?? human(k)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {present.map(([key, lab, kind]) => (
                <tr key={key}>
                  <td>{lab}</td>
                  {conds.map(([k, c]) => {
                    if (c[key] === undefined) return <td key={k}>—</td>
                    if (kind === 'rate') {
                      const r = rateOf(c[key])
                      return (
                        <td key={k}>
                          {pct(r.rate, 1)}
                          {r.n !== undefined && r.k !== undefined && <small> {r.k}/{r.n}</small>}
                        </td>
                      )
                    }
                    const v = num(c[key])
                    return <td key={k}>{fmt(v, v !== undefined && Math.abs(v) < 10 ? 2 : 0)}</td>
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {fi && (caught?.rate !== undefined || fb?.rate !== undefined) && (
        <div className="tiles tiles-sm">
          {caught?.rate !== undefined && (
            <Tile tone="ok" label="Injected fabrications blocked" value={pct(caught.rate, 0)} sub={caught.k !== undefined && caught.n !== undefined ? <>{caught.k} of {caught.n} fake numbers, citations and unsupported inferences</> : undefined} />
          )}
          {fb?.rate !== undefined && (
            <Tile label="Good claims wrongly blocked" value={pct(fb.rate, 0)} sub={fb.k !== undefined && fb.n !== undefined ? <>{fb.k} of {fb.n} known-good claims</> : undefined} />
          )}
        </div>
      )}
    </Card>
  )
}

/* ---------- E9 ---------- */

function E9({ data, half }: { data: SiteData; half?: boolean }) {
  const e9 = data.results?.e9 as Obj | undefined
  if (!isObj(e9)) return null
  const t = (k: string, label: string, sub?: string) =>
    num(e9[k]) !== undefined ? <Tile key={k} label={label} value={<>{fmt(e9[k], 0)}<span className="unit"> ms</span></>} sub={sub} /> : null
  return (
    <Card half={half} title="Speed" lede="Wall-clock time per image on a laptop CPU, no GPU." foot={<Source file="e9" sha={sha(data, 'e9')} n={num(e9.n)} />}>
      <div className="tiles">
        {t('inspect_ms_p50', 'Full inspection, median')}
        {t('inspect_ms_p95', 'Full inspection, 95th percentile')}
        {t('metrics_ms_p50', 'QC metrics only, median')}
        {t('metrics_ms_p95', 'QC metrics only, 95th percentile')}
      </div>
    </Card>
  )
}

export function Benchmarks({ data }: { data: SiteData }) {
  const findings = Array.isArray(data.results?.findings) ? (data.results!.findings as unknown[]).filter((x) => typeof x === 'string') as string[] : []
  const g = data.generated
  const pair = isObj(data.results?.e4) && isObj(data.results?.e9)
  return (
    <section className="section alt" id="benchmarks" aria-labelledby="bench-h">
      <div className="wrap wide">
        <SectionHead
          id="bench-h"
          title="Measured, not claimed."
          lede="Every figure below is read from the results files the evaluation wrote. Where Zelos falls short, it says so."
        />
        {findings.length > 0 && (
          <Reveal className="findings">
            <h3>Honest findings</h3>
            <ul>
              {findings.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </Reveal>
        )}
        <div className="bgrid">
          <E1 data={data} />
          <E2 data={data} />
          <E5 data={data} />
          <E4 data={data} half={pair} />
          <E9 data={data} half={pair} />
        </div>
        {g?.git_sha && (
          <p className="gen">
            Generated from commit {shortSha(g.git_sha)}
            {g.config_hash ? `, config ${g.config_hash.slice(0, 8)}` : ''}
            {g.timestamp_utc ? `, ${g.timestamp_utc.replace('T', ' ').replace(/(\+00:00|Z)$/, ' UTC')}` : ''}.
          </p>
        )}
      </div>
    </section>
  )
}

import type { ReactNode } from 'react'
import type { SiteData } from '../types'
import { fmt, human, isObj, num, numericLeaves, pct, shortSha } from '../util'
import { Reveal, SectionHead, Source } from './common'
import { E1Variants, HeadToHead, LlmSweep, RuleCitations, VlmBaseline } from './Benchmarks11'
import { Bar, bestVariant, Card, sha, Tile, variantName, type Obj } from './benchCommon'

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

/** v1.1 results nest everything under variants; map the best one onto the v1.0 field names. */
function adaptE1(e1: Obj): { e1: Obj; variant?: string } {
  if (isObj(e1.overall) || !isObj(e1.variants)) return { e1 }
  const b = bestVariant(e1)
  if (!b) return { e1 }
  const [name, v] = b
  const per: Obj = {}
  for (const [k, p] of Object.entries(isObj(v.per_kind) ? (v.per_kind as Obj) : {})) {
    if (!isObj(p)) continue
    per[k] = { expected_label: p.expected_label, detection_rate_by_severity: p.detection_by_severity, attribution_acc_detected: p.attribution_v2_signatures }
  }
  const sweep: Obj = {}
  for (const [t, p] of Object.entries(isObj(v.threshold_sweep) ? (v.threshold_sweep as Obj) : {})) {
    if (isObj(p)) sweep[t] = { clean_flag_rate: p.clean_flag_rate, sev3plus_detection_rate: p.sev3plus_detection_rate ?? p.sev3plus_detection }
  }
  return {
    variant: name,
    e1: {
      ...e1,
      overall: {
        detection_rate_sev3plus: v.detection_rate_sev3plus,
        attribution_acc_detected: v.attribution_v2_signatures,
        attribution_top2_detected: v.attribution_v2_top2,
        attribution_random_baseline: v.chance_attribution,
        auroc_clean_vs_sev3plus: v.auroc_clean_vs_sev3plus,
      },
      per_kind: per,
      threshold_sweep: sweep,
      clean_flag_rate: v.clean_flag_rate,
      batch_reference: undefined,
    },
  }
}

function E1({ data }: { data: SiteData }) {
  const raw = data.results?.e1 as Obj | undefined
  if (!isObj(raw)) return null
  const { e1, variant } = adaptE1(raw)
  const overall = isObj(e1.overall) ? e1.overall : undefined
  const br = isObj(e1.batch_reference) ? (isObj(e1.batch_reference.overall) ? e1.batch_reference.overall : e1.batch_reference) : undefined
  const kinds = isObj(e1.per_kind) ? Object.entries(e1.per_kind as Obj) : []
  const sweep = isObj(e1.threshold_sweep) ? Object.entries(e1.threshold_sweep as Obj) : []
  const n = num(e1.n_images) ?? num(e1.n)
  const sevKeys = Array.isArray(e1.severities) && e1.severities.length ? (e1.severities as unknown[]).map(String) : ['1', '2', '3', '4', '5']
  return (
    <Card
      title="Corruption benchmark"
      lede={`${kinds.length ? `${kinds.length} kinds` : 'Several kinds'} of synthetic corruption${Array.isArray(e1.severities) ? ` at ${e1.severities.length} severities` : ''}, applied to real images. Zelos has to notice each one and name the right cause.`}
      foot={<Source file="e1" sha={sha(data, 'e1')} n={n ?? (num(raw.n_base_images) !== undefined ? `${raw.n_base_images} base images` : undefined)} />}
    >
      {overall && <E1Tiles o={overall} label={variant ? `${variantName(variant)}, the variant with the highest AUROC` : br ? 'Single-image reference' : undefined} />}
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
                  <div className="ksev" aria-label={`Detection rate by severity ${sevKeys.join(', ')}`} style={{ gridTemplateColumns: `repeat(${sevKeys.length}, 1fr)` }}>
                    {sevKeys.map((s) => {
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
          <p className="muted small">The strip at the right shows detection at severity {sevKeys.join(', ')}, light to strong.</p>
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

/* ---------- E2 ---------- */

const E2_NAMES: Record<string, string> = {
  zelos_qc_score: 'Zelos QC score',
  focus_only: 'Focus metric only',
  random: 'Random',
  knn_physics_k15: 'k-nearest neighbours on physical metrics',
  knn_dinov2_cls_k15: 'k-nearest neighbours on frozen DINOv2 features',
  knn_dinov2_session_centred_k15: 'k-nearest neighbours on DINOv2, session-centred',
  knn_handcrafted_k15: 'k-nearest neighbours on handcrafted features',
}
function e2Label(k: string): string {
  if (E2_NAMES[k]) return E2_NAMES[k]
  const m = /^(zelos_qc|focus_only)\[(.+)\]$/.exec(k)
  if (m) return `${m[1] === 'focus_only' ? 'Focus metric only' : 'Zelos QC score'}, ${variantName(m[2]).toLowerCase()}`
  if (/LABEL-TRAINED/i.test(k)) return 'Label-trained DINOv2 linear head'
  if (k.startsWith('knn')) return `k-nearest neighbours, ${human(k.replace(/^knn_?/, ''))}`
  return human(k)
}

function PlotCell({ a, lo, hi, z, trained, label }: { a?: number; lo?: number; hi?: number; z?: boolean; trained?: boolean; label: string }) {
  return (
    <span className="dw-plot" role="img" aria-label={label}>
      <span className="dw-ref" />
      {lo !== undefined && hi !== undefined && (
        <span className={`dw-whisk ${z ? 'is-z' : ''} ${trained ? 'is-trained' : ''}`} style={{ left: `${lo * 100}%`, width: `${Math.max(0, hi - lo) * 100}%` }} />
      )}
      {a !== undefined && <span className={`dw-dot ${z ? 'is-z' : ''} ${trained ? 'is-trained' : ''}`} style={{ left: `${a * 100}%` }} />}
    </span>
  )
}

function E2({ data }: { data: SiteData }) {
  const e2 = data.results?.e2 as Obj | undefined
  if (!isObj(e2) || !isObj(e2.detectors)) return null
  const dets = Object.entries(e2.detectors as Obj)
  const rv = isObj(e2.zelos_at_review_threshold) ? (e2.zelos_at_review_threshold as Obj) : undefined
  const cells = isObj(e2.per_cell_line) ? Object.entries(e2.per_cell_line as Obj) : []
  const ops = isObj(e2.operating_points) ? Object.entries(e2.operating_points as Obj).filter(([, v]) => isObj(v)) : []
  const within = dets.some(([, v]) => isObj(v.within_session))
  const anyTrained = dets.some(([k]) => /LABEL-TRAINED/i.test(k))
  const nTest = num(e2.n_test)
  const nSess = num(e2.n_test_sessions)
  const nAll = num(e2.n_images)
  return (
    <Card
      title="Agreement with expert labels"
      lede={`Can the score separate images experts called good from those they called bad? Area under the ROC curve${nSess !== undefined ? ` on ${nSess} whole imaging sessions held out from fitting and calibration` : ' on held-out sessions'}, with 95% intervals. 0.5 is chance.${within ? ' The second column asks the harder question: can it rank images within one imaging day?' : ''}`}
      foot={<Source file="e2" sha={sha(data, 'e2')} n={nTest !== undefined ? `${nTest.toLocaleString('en-US')} held-out${nAll !== undefined ? ` of ${nAll.toLocaleString('en-US')}` : ''} images` : undefined} />}
    >
      <div className="legend" aria-hidden="true">
        <span><i className="sw sw-dot" />Training-free</span>
        {anyTrained && <span><i className="sw sw-dot sw-trained" />Label-trained: learns the expert label</span>}
        <span><i className="sw sw-r" />0.5 chance</span>
      </div>
      <ul className={`dw ${within ? 'dw-two' : ''}`} aria-label="AUROC per detector">
        {within && (
          <li className="dw-head" aria-hidden="true">
            <span />
            <span>All held-out images</span>
            <span />
            <span>Within the same imaging day</span>
            <span />
          </li>
        )}
        {dets.map(([k, v]) => {
          const a = num(v.auroc)
          const ci = Array.isArray(v.auroc_ci95) ? (v.auroc_ci95 as number[]) : undefined
          const lo = ci ? num(ci[0]) : undefined
          const hi = ci ? num(ci[1]) : undefined
          const z = k.includes('zelos')
          const trained = /LABEL-TRAINED/i.test(k)
          const ws = isObj(v.within_session) ? (v.within_session as Obj) : undefined
          const w = ws ? (num(ws.weighted) ?? num(ws.mean)) : undefined
          const frac = ws ? num(ws['frac_sessions_above_0.5']) : undefined
          const nm = e2Label(k)
          return (
            <li key={k} className={trained ? 'is-trained' : ''}>
              <span className="dw-name">
                {nm}
                {trained && <span className="lic lic-trained">label-trained</span>}
              </span>
              <PlotCell a={a} lo={lo} hi={hi} z={z} trained={trained} label={`${nm} AUROC ${fmt(a, 2)}${lo !== undefined && hi !== undefined ? `, interval ${fmt(lo, 2)} to ${fmt(hi, 2)}` : ''}`} />
              <span className="dw-val">
                {fmt(a, 2)}
                {lo !== undefined && hi !== undefined && <small> [{fmt(lo, 2)}, {fmt(hi, 2)}]</small>}
              </span>
              {within && (
                <>
                  <PlotCell a={w} z={z} trained={trained} label={`${nm} within-session AUROC ${fmt(w, 2)}`} />
                  <span className="dw-val" title={frac !== undefined ? `${pct(frac, 0)} of imaging days above 0.5` : undefined}>
                    {fmt(w, 2)}
                    {ws && num(ws.n_sessions) !== undefined && <small> {ws.n_sessions} days</small>}
                  </span>
                </>
              )}
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
          {within && (
            <>
              <span className="dw-ticks">
                <b style={{ left: '0%' }}>0</b>
                <b style={{ left: '50%' }}>0.5 chance</b>
                <b style={{ left: '100%' }}>1</b>
              </span>
              <span />
            </>
          )}
        </li>
      </ul>
      {ops.length > 0 && (
        <details className="sweep">
          <summary>Operating points on the held-out sessions</summary>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Reference</th>
                  <th>Expert-good images flagged</th>
                  <th>Expert-bad images flagged</th>
                  <th>Expert-good flagged at pre-registered z = 3</th>
                </tr>
              </thead>
              <tbody>
                {ops.map(([k, v]) => (
                  <tr key={k}>
                    <td>{variantName(k)}</td>
                    <td>{pct(v.false_flag_rate_test_good, 1)}</td>
                    <td>{pct(v.recall_expert_bad, 1)}</td>
                    <td>{pct(v.preregistered_z3_false_flag, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
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
        {data.dataset && (
          <p className="dataset">
            Evaluated on the {data.dataset.name ?? 'Organ-on-a-Chip image dataset'}
            {data.dataset.n_images_used ? `, ${data.dataset.n_images_used.toLocaleString('en-US')} images` : ''}
            {data.dataset.cell_lines?.length ? ` across ${data.dataset.cell_lines.length} cell lines (${data.dataset.cell_lines.join(', ')})` : ''}
            {data.dataset.citation ? `. ${data.dataset.citation}` : ''}.
          </p>
        )}
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
          <E1Variants data={data} />
          <HeadToHead data={data} />
          <E1 data={data} />
          <E2 data={data} />
          <VlmBaseline data={data} />
          <E5 data={data} />
          <LlmSweep data={data} />
          <E4 data={data} half={pair} />
          <E9 data={data} half={pair} />
          <RuleCitations data={data} />
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

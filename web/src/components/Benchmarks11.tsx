import type { ReactNode } from 'react'
import type { SiteData } from '../types'
import { fmt, human, isObj, num, pct, shortSha } from '../util'
import { Source } from './common'
import { Bar, bestVariant, Card, variantName, type Obj } from './benchCommon'

const shaOf = (o: unknown, data: SiteData): string => shortSha((isObj(o) ? (o.provenance as Obj | undefined)?.git_sha : undefined) ?? data.generated?.git_sha)

/* ---------------------------------------------------------------- false alarms vs faults caught */

function VBar({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="vb">
      <span className="vb-l">{label}</span>
      {children}
    </div>
  )
}

export function E1Variants({ data }: { data: SiteData }) {
  const e1 = data.results?.e1 as Obj | undefined
  if (!isObj(e1) || !isObj(e1.variants)) return null
  const vs = Object.entries(e1.variants as Obj).filter(([, v]) => isObj(v)) as Array<[string, Obj]>
  if (!vs.length) return null
  const nBase = num(e1.n_base_images)
  return (
    <Card
      title="False alarms vs faults caught"
      lede="The same synthetic faults, the same real images, three ways of building the reference an image is compared with. A stricter threshold flags fewer clean images and misses more faults. Every row is a measured trade-off, not a tuned headline."
      foot={<Source file="e1" sha={shaOf(e1, data)} n={nBase !== undefined ? `${nBase} base images, ${Array.isArray(e1.kinds) ? e1.kinds.length : 'several'} fault kinds` : undefined} />}
    >
      <div className="legend" aria-hidden="true">
        <span><i className="sw sw-3" />Clean images flagged (lower is better)</span>
        <span><i className="sw sw-1" />Faults caught, severity 3 and above</span>
        <span><i className="sw sw-0" />Cause named, v1 rules</span>
        <span><i className="sw sw-2" />Cause named, v1.1 signatures</span>
        <span><i className="sw sw-r" />Random guess</span>
      </div>
      <div className="vgrid">
        {vs.map(([k, v]) => {
          const chance = num(v.chance_attribution)
          return (
            <section key={k} className="vcol" aria-label={variantName(k)}>
              <h4>{variantName(k)}</h4>
              <p className="vkey">{k}</p>
              <div className="vrow">
                <p className="vlab">Clean images flagged</p>
                <Bar v={num(v.clean_flag_rate)} cls="b3" label={`${variantName(k)}: clean images flagged`} />
              </div>
              <div className="vrow">
                <p className="vlab">Faults caught, severity 3 and above</p>
                <Bar v={num(v.detection_rate_sev3plus)} cls="b1" label={`${variantName(k)}: faults caught`} />
                {num(v.detection_rate_sev1) !== undefined && <p className="vsub">Severity 1: {pct(v.detection_rate_sev1, 0)}</p>}
              </div>
              <div className="vrow">
                <p className="vlab">Cause named correctly, among caught faults</p>
                <VBar label="v1 rules">
                  <Bar v={num(v.attribution_v1_rules)} cls="b0" tick={chance} label={`${variantName(k)}: cause named by v1 rules`} />
                </VBar>
                <VBar label="signatures">
                  <Bar v={num(v.attribution_v2_signatures)} cls="b2" tick={chance} label={`${variantName(k)}: cause named by v1.1 signatures`} />
                </VBar>
                {num(v.attribution_v2_top2) !== undefined && <p className="vsub">Right cause within the top two: {pct(v.attribution_v2_top2, 0)}</p>}
              </div>
              {num(v.auroc_clean_vs_sev3plus) !== undefined && <p className="vsub vauc">AUROC, clean against faults: {fmt(v.auroc_clean_vs_sev3plus, 3)}</p>}
            </section>
          )
        })}
      </div>
      {typeof e1.operating_point === 'string' && <p className="muted small">Operating point: {e1.operating_point}.</p>}
    </Card>
  )
}

/* ---------------------------------------------------------------- head-to-head */

export function HeadToHead({ data }: { data: SiteData }) {
  const e1 = data.results?.e1 as Obj | undefined
  const h = isObj(e1?.head_to_head) ? (e1!.head_to_head as Obj) : undefined
  const head = h ? (isObj(h.label_trained_dinov2_linear_head) ? (h.label_trained_dinov2_linear_head as Obj) : (Object.values(h).find(isObj) as Obj | undefined)) : undefined
  const best = bestVariant(e1)
  if (!head || !best) return null
  const [vname, z] = best
  const rows: Array<{ k: string; label: string; z: string; h: string; note?: string }> = [
    { k: 'flag', label: 'Clean images flagged (lower is better)', z: pct(z.clean_flag_rate, 1), h: pct(head.clean_flag_rate, 1) },
    { k: 'det', label: 'Injected faults caught, severity 3 and above', z: pct(z.detection_rate_sev3plus, 1), h: pct(head.detection_rate_sev3plus, 1) },
    { k: 'auc', label: 'AUROC, clean against faults', z: fmt(z.auroc_clean_vs_sev3plus, 3), h: fmt(head.auroc_clean_vs_sev3plus, 3) },
  ]
  const byKind = isObj(head.detection_by_kind_sev3plus) ? Object.entries(head.detection_by_kind_sev3plus as Obj) : []
  const zKind = isObj(z.per_kind) ? (z.per_kind as Obj) : {}
  const zSev3plus = (k: string): number | undefined => {
    const d = isObj(zKind[k]) && isObj(zKind[k].detection_by_severity) ? (zKind[k].detection_by_severity as Obj) : undefined
    if (!d) return undefined
    const xs = Object.entries(d)
      .filter(([s]) => Number(s) >= 3)
      .map(([, x]) => num(x))
      .filter((x): x is number => x !== undefined)
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined
  }
  const robust = num(head.clean_flag_rate_extraction_path) !== undefined || num(head.mean_abs_prob_diff_between_paths) !== undefined
  const canName = head.can_name_the_cause === false ? 'No' : 'Yes'
  return (
    <Card
      title="Head-to-head on the same faults"
      lede="Zelos against a linear head trained on frozen DINOv2 features to predict the expert label, scored on identical injected faults. The head is the approach family of entries that learn the expert label, not any entry’s exact model."
      foot={<Source file="e1" sha={shaOf(e1, data)} n={num(e1?.n_base_images) !== undefined ? `${e1!.n_base_images} base images` : undefined} />}
    >
      <div className="h2h">
        <div className="h2h-col h2h-z">
          <h4>Zelos</h4>
          <p className="h2h-sub">{variantName(vname)}, training-free</p>
          <dl>
            {rows.map((r) => (
              <div key={r.k}>
                <dt>{r.label}</dt>
                <dd>{r.z}</dd>
              </div>
            ))}
            <div>
              <dt>Can name the cause</dt>
              <dd>
                Yes
                {num(z.attribution_v2_signatures) !== undefined && (
                  <small>
                    {pct(z.attribution_v2_signatures, 0)} correct among caught faults{num(z.chance_attribution) !== undefined ? `, random guess ${pct(z.chance_attribution, 0)}` : ''}
                  </small>
                )}
              </dd>
            </div>
          </dl>
        </div>
        <div className="h2h-col">
          <h4>Label-trained DINOv2 head</h4>
          <p className="h2h-sub">Learns the expert label from frozen embeddings</p>
          <dl>
            {rows.map((r) => (
              <div key={r.k}>
                <dt>{r.label}</dt>
                <dd>{r.h}</dd>
              </div>
            ))}
            <div>
              <dt>Can name the cause</dt>
              <dd>
                {canName}
                <small>It outputs one score and no cause</small>
              </dd>
            </div>
          </dl>
        </div>
      </div>
      <p className="note">
        Can name the cause: <strong>Zelos yes / head {canName.toLowerCase()}</strong>. Where the head catches faults, a lab still has to work out what went wrong.
      </p>
      {(byKind.length > 0 || robust) && (
        <details className="sweep">
          <summary>Per fault kind and robustness check</summary>
          {byKind.length > 0 && (
            <div className="tbl-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Fault kind</th>
                    <th>Zelos, severity 3 and 5</th>
                    <th>Label-trained head, severity 3 and above</th>
                  </tr>
                </thead>
                <tbody>
                  {byKind.map(([k, v]) => (
                    <tr key={k}>
                      <td>{human(k)}</td>
                      <td>{pct(zSev3plus(k), 0)}</td>
                      <td>{pct(num(v), 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {robust && (
            <p className="muted small">
              Extraction-path check: the head flags {pct(head.clean_flag_rate_extraction_path, 1)} of clean images when features are extracted the other way; mean absolute difference in its
              probability between the two paths is {fmt(head.mean_abs_prob_diff_between_paths, 3)}.
            </p>
          )}
        </details>
      )}
    </Card>
  )
}

/* ---------------------------------------------------------------- vision LLM baseline */

const modelName = (k: string) => k.split('/').pop() ?? k

export function VlmBaseline({ data }: { data: SiteData }) {
  const v = data.vlm_baseline
  if (!isObj(v) || !isObj(v.models)) return null
  const rows: Array<{ name: string; z?: boolean; m: Obj }> = []
  if (isObj(v.zelos_stored_reference_calibrated)) rows.push({ name: 'Zelos, training-free', z: true, m: v.zelos_stored_reference_calibrated as Obj })
  for (const [k, m] of Object.entries(v.models as Obj)) if (isObj(m)) rows.push({ name: modelName(k), m })
  if (!rows.length) return null
  const hasBA = rows.some((r) => num(r.m.balanced_accuracy_vs_expert) !== undefined)
  return (
    <Card
      title="Can a vision LLM do the job?"
      lede="General multimodal models asked one question per image: is this image technically acceptable? Same real images for every system. A model that flags everything catches every fault and every good image, so read the columns together."
      foot={<Source file="vlm_baseline" sha={shaOf(v, data)} n={num(v.n_cases) !== undefined ? `${v.n_cases} cases` : undefined} />}
    >
      <div className="tbl-wrap">
        <table className="tbl tbl-cmp">
          <thead>
            <tr>
              <th />
              <th>Expert-good flagged<small> lower is better</small></th>
              <th>Expert-bad flagged<small> higher is better</small></th>
              <th>Clean flagged<small> lower is better</small></th>
              <th>Severity 5 faults caught<small> higher is better</small></th>
              {hasBA && <th>Balanced accuracy</th>}
              <th>Unparsed</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className={r.z ? 'tr-z' : ''}>
                <td>{r.name}</td>
                <td>{pct(num(r.m.false_flag_on_expert_good), 0)}</td>
                <td>{pct(num(r.m.flag_rate_on_expert_bad), 0)}</td>
                <td>{pct(num(r.m.false_flag_on_clean), 0)}</td>
                <td>{pct(num(r.m.detected_sev5_faults), 0)}</td>
                {hasBA && <td>{pct(num(r.m.balanced_accuracy_vs_expert), 0)}</td>}
                <td>{num(r.m.unparsed) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/* ---------------------------------------------------------------- LLM sweep */

const STATUS: Record<string, string> = { ok: 'Complete', probe_only: 'Probe only', quota_exhausted: 'Quota exhausted' }

function rate(o: unknown): string {
  if (!isObj(o)) return '—'
  const r = num(o.rate)
  if (r === undefined) return '—'
  return `${(r * 100).toFixed(r > 0 && r < 0.1 ? 1 : 0)}%`
}
function kn(o: unknown): string | undefined {
  return isObj(o) && num(o.k) !== undefined && num(o.n) !== undefined ? `${o.k}/${o.n}` : undefined
}

export function LlmSweep({ data }: { data: SiteData }) {
  const s = data.llm_sweep
  if (!isObj(s) || !isObj(s.models)) return null
  const rows = Object.entries(s.models as Obj).filter(([, m]) => isObj(m)) as Array<[string, Obj]>
  if (!rows.length) return null
  const verdict = Array.isArray(s.verdict) ? (s.verdict as Obj[]).find((x) => isObj(x) && x.group === 'explain_qc') : undefined
  const nCases = num(isObj(s.n) ? (s.n as Obj).explain_qc : undefined)
  return (
    <Card
      title="Which language model sits behind the gate"
      lede="The same cases, prompts, retriever and grounding gate for every model. Only the model changes. Before the gate is what the model drafted; after the gate is what a user would see."
      foot={<Source file="e5_sweep" sha={shaOf(s, data)} n={nCases !== undefined ? `${nCases} explain-QC cases per model` : undefined} />}
    >
      <div className="tbl-wrap">
        <table className="tbl tbl-cmp">
          <thead>
            <tr>
              <th>Model</th>
              <th>Status</th>
              <th>Unsupported claims, before gate</th>
              <th>Unsupported claims, after gate</th>
              <th>Supported claims per case</th>
              <th>Cost per 100 answers</th>
              <th>Median latency</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([k, m]) => {
              const g = isObj(m.groups) && isObj((m.groups as Obj).explain_qc) ? ((m.groups as Obj).explain_qc as Obj) : undefined
              const st = String(m.status ?? '')
              const done = !!g
              const notes = Array.isArray(m.notes) ? (m.notes as unknown[]).filter((x): x is string => typeof x === 'string') : []
              const usd = num(isObj(g?.cost) ? (g!.cost as Obj).usd_per_100_answers : undefined)
              const p50 = num(isObj(g?.latency_ms) ? (g!.latency_ms as Obj).p50 : undefined)
              const draft = isObj(g?.draft) ? (g!.draft as Obj) : undefined
              const gated = isObj(g?.gated) ? (g!.gated as Obj) : undefined
              return (
                <tr key={k} className={done ? '' : 'tr-muted'}>
                  <td>
                    {k}
                    {!done && notes[0] && <small className="blk">{notes[0]}</small>}
                  </td>
                  <td>
                    <span className={`st st-${st === 'ok' ? 'ok' : done ? 'warn' : 'grey'}`}>{STATUS[st] ?? human(st || 'unknown')}</span>
                  </td>
                  <td>
                    {rate(draft?.unsupported_claim_rate)}
                    {kn(draft?.unsupported_claim_rate) && <small> {kn(draft?.unsupported_claim_rate)}</small>}
                  </td>
                  <td>
                    {rate(gated?.unsupported_claim_rate)}
                    {kn(gated?.unsupported_claim_rate) && <small> {kn(gated?.unsupported_claim_rate)}</small>}
                  </td>
                  <td>{num(gated?.supported_claims_per_case) !== undefined ? fmt(gated!.supported_claims_per_case, 2) : '—'}</td>
                  <td>{usd !== undefined ? `$${usd.toFixed(usd < 0.1 ? 3 : 2)}` : '—'}</td>
                  <td>{p50 !== undefined ? `${(p50 / 1000).toFixed(1)} s` : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {typeof verdict?.text === 'string' && <p className="note">{verdict.text}</p>}
    </Card>
  )
}

/* ---------------------------------------------------------------- literature basis for the lab actions */

export function RuleCitations({ data }: { data: SiteData }) {
  const rc = data.rule_citations
  if (!isObj(rc) || !isObj(rc.rules)) return null
  const all = Object.entries(rc.rules as Obj).filter(([, r]) => isObj(r)) as Array<[string, Obj]>
  if (!all.length) return null
  const supported = all.filter(([, r]) => r.status === 'supported')
  const rest = all.filter(([, r]) => r.status !== 'supported')
  const ns = num(rc.n_supported) ?? supported.length
  const nr = num(rc.n_rules) ?? all.length
  const papers = num(data.results?.e4?.corpus?.n_papers)
  const none = `No verified passage in the current ${papers !== undefined ? `${papers}-paper ` : ''}corpus`
  return (
    <Card
      title="Literature basis for the lab actions"
      lede={`${ns} of ${nr} recommended actions have a passage in the open literature that a sentence-level check verifies. The others stay recommendations without a citation, and the certificate says so.`}
      foot={<Source file="rule_citations" sha={shaOf(rc, data)} n={`${nr} rules`} />}
    >
      <ul className="rules">
        {[...supported, ...rest].map(([k, r]) => {
          const ok = r.status === 'supported'
          return (
            <li key={k} className={ok ? '' : 'is-muted'}>
              <p className="rule-k">{human(k.toLowerCase())}</p>
              {typeof r.claim === 'string' && <p className="rule-claim">{r.claim}</p>}
              {ok ? (
                <div className="why">
                  <p className="why-title">
                    {typeof r.url === 'string' ? (
                      <a href={r.url} rel="noopener noreferrer">
                        {r.title ?? r.chunk_id}
                      </a>
                    ) : (
                      (r.title ?? r.chunk_id)
                    )}
                    {typeof r.license === 'string' && <span className="lic">{r.license}</span>}
                  </p>
                  {typeof r.excerpt === 'string' && <p className="why-ex">“{r.excerpt}”</p>}
                </div>
              ) : (
                <p className="why-none">{none}</p>
              )}
            </li>
          )
        })}
      </ul>
      {typeof rc.disclosure === 'string' && <p className="note small">{rc.disclosure}</p>}
    </Card>
  )
}


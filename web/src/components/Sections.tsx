import type { SiteData } from '../types'
import { CopyButton, Reveal, SectionHead } from './common'

const GH = 'https://github.com/simon-derock/zelos'

export function Problem() {
  const items = [
    {
      t: 'Chips fail quietly',
      d: 'Air bubbles in microchannels, drifting focus and uneven illumination look like biology. They slip into datasets and into the digital twins trained on them.',
      icon: <path d="M4 17c3-9 5-9 8 0s5 9 8 0M4 7h16" />,
    },
    {
      t: 'Experts cannot review every frame',
      d: 'Automated organ-on-a-chip microscopes image every channel, every day. Expert review does not scale, so most frames are never checked.',
      icon: <path d="M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6zm9-2.5a2.5 2.5 0 100 5 2.5 2.5 0 000-5z" />,
    },
    {
      t: 'AI assistants invent evidence',
      d: 'Language models state numbers and cite papers that do not exist. In life science, a QC verdict you cannot verify is worse than none.',
      icon: <path d="M5 5h14v10H9l-4 4V5zm4 4h6m-6 3h4" />,
    },
  ]
  return (
    <section className="section" id="problem" aria-labelledby="problem-h">
      <div className="wrap">
        <SectionHead id="problem-h" title="Bad images fail quietly." lede="An organ-on-a-chip experiment is only as good as the images it is judged by." />
        <div className="grid3">
          {items.map((it, i) => (
            <Reveal key={it.t} className={`card card-problem d${i}`}>
              <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                {it.icon}
              </svg>
              <h3>{it.t}</h3>
              <p>{it.d}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}

const STEPS: Array<{ t: string; d: string; det: boolean; llm?: boolean; gate?: boolean }> = [
  { t: 'Image', d: 'Brightfield frame, as acquired', det: true },
  { t: 'QC metrics', d: 'Focus, noise, illumination, exposure, contrast, debris, bubbles, compression', det: true },
  { t: 'Cause and action', d: 'Metric z-scores map to a cause and a lab action', det: true },
  { t: 'Evidence retrieval', d: 'Open-access passages ranked for the cause', det: true },
  { t: 'LLM explainer', d: 'Drafts an explanation. Proposes only.', det: false, llm: true },
  { t: 'Grounding gate', d: 'Every claim must match a tool output or a passage', det: true, gate: true },
  { t: 'Certificate', d: 'Level, cause, action, cited claims, ledger', det: true },
]

export function Pipeline() {
  return (
    <section className="section dark" id="how" aria-labelledby="how-h">
      <div className="wrap">
        <SectionHead id="how-h" dark title="Tools compute. The gate decides." lede="Seven steps, one of them a language model. Everything else is deterministic code you can audit." />
        <Reveal>
          <ol className="pipe">
            {STEPS.map((s) => (
              <li key={s.t} className={`pipe-step ${s.llm ? 'is-llm' : ''} ${s.gate ? 'is-gate' : ''}`}>
                <div className="pipe-box">
                  <h3>{s.t}</h3>
                  <p>{s.d}</p>
                  {s.det ? (
                    <span className="tag tag-det">0 LLM calls</span>
                  ) : (
                    <span className="tag tag-llm">1 LLM call</span>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </Reveal>
        <Reveal className="motto">
          <p>The LLM proposes.</p>
          <p>Tools compute.</p>
          <p>The gate disposes.</p>
        </Reveal>
      </div>
    </section>
  )
}

const CMDS = [
  { c: `git clone ${GH} && cd zelos`, note: 'Get the code' },
  { c: 'make setup', note: 'Create the environment and install dependencies' },
  { c: 'make data', note: 'Download the dataset and build the reference statistics' },
  { c: 'make eval', note: 'Regenerate every number on this page' },
  { c: 'make test', note: 'Run the test suite, including the gate fault injection' },
]

export function Reproduce() {
  const all = CMDS.map((c) => c.c).join('\n')
  return (
    <section className="section dark" id="reproduce" aria-labelledby="repro-h">
      <div className="wrap narrow">
        <SectionHead id="repro-h" dark title="Reproduce every number." lede="Runs offline without API keys. Recorded language-model answers are replayed from a cache, so the results match." />
        <Reveal>
          <div className="term" role="group" aria-label="Terminal commands">
            <div className="term-bar">
              <span className="dot r" />
              <span className="dot y" />
              <span className="dot g" />
              <span className="term-title">Terminal</span>
              <CopyButton text={all} />
            </div>
            <pre className="term-body">
              {CMDS.map((c) => (
                <div key={c.c} className="term-line">
                  <span className="prompt" aria-hidden="true">$ </span>
                  <code>{c.c}</code>
                  <span className="term-note"># {c.note}</span>
                </div>
              ))}
            </pre>
          </div>
        </Reveal>
      </div>
    </section>
  )
}

export function Footer({ data }: { data: SiteData | null }) {
  const ds = data?.dataset
  return (
    <footer className="footer">
      <div className="wrap">
        <p className="disclaimer">Research tool. Not a medical device. No clinical or diagnostic claims.</p>
        <div className="foot-grid">
          <p>
            Dataset: {ds?.name ?? 'Organ-on-a-Chip (OOC) Image Dataset'}, {ds?.citation ?? 'Movčana et al., Data 2024'}. Zenodo{' '}
            <a href={`https://doi.org/${ds?.doi ?? '10.5281/zenodo.10203721'}`} rel="noopener noreferrer">
              {ds?.doi ?? '10.5281/zenodo.10203721'}
            </a>
            . Licensed {ds?.license ?? 'CC BY / CC BY-SA'}.
          </p>
          <p>
            Zelos is released under the MIT license. Source on{' '}
            <a href={GH} rel="noopener noreferrer">GitHub</a>.
          </p>
          <p>Built by Philip Simon Derock.</p>
        </div>
        {data?.meta?.version && (
          <p className="foot-version">
            Submitted version: {(data.meta.submitted_version ?? 'v1.0-submission').replace(/-.*$/, '')} (tag{' '}
            <a href={`${GH}/tree/${data.meta.submitted_version ?? 'v1.0-submission'}`} rel="noopener noreferrer">
              {data.meta.submitted_version ?? 'v1.0-submission'}
            </a>
            ). This page is the v{data.meta.version} preview{data.meta.branch ? `, branch ${data.meta.branch}` : ''}, built after the
            submission deadline.
          </p>
        )}
      </div>
    </footer>
  )
}

const pct = (x: unknown, d = 1) => (typeof x === 'number' ? `${(x * 100).toFixed(d).replace(/\.0$/, '')}%` : '')

export function Proof({ data }: { data: SiteData | null }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r: any = data?.results ?? {}
  const fi = r?.e5?.gate_fault_injection?.injected_fabrications_caught
  const gated = r?.e5?.conditions?.llm_gated?.unsupported_claim_rate_in_output?.rate
  const ungated = r?.e5?.conditions?.llm_ungated?.unsupported_claim_rate_in_output?.rate
  // v1.1 keys preferred (batch reference, held-out sessions); v1.0 keys remain the fallback.
  const variants = r?.e1?.variants && typeof r.e1.variants === 'object' ? (r.e1.variants as Record<string, any>) : undefined
  const v11 = variants ? (variants.batch_session ?? Object.values(variants)[0]) : undefined
  const det = typeof v11?.detection_rate_sev3plus === 'number' ? v11.detection_rate_sev3plus : r?.e1?.overall?.detection_rate_sev3plus
  const clean = typeof v11?.clean_flag_rate === 'number' ? v11.clean_flag_rate : undefined
  const hit1 = r?.e4?.modes?.hybrid_rerank?.paper_level?.['hit@1']
  const papers = r?.e4?.corpus?.n_papers
  const ms = r?.e9?.inspect_ms_p50
  const dets = r?.e2?.detectors && typeof r.e2.detectors === 'object' ? (r.e2.detectors as Record<string, any>) : {}
  const v11e2 = Object.values(dets).some((v) => v && typeof v === 'object' && v.within_session)
  const free = Object.entries(dets).filter(
    ([k, v]) => !/LABEL-TRAINED/i.test(k) && !/^(random|zelos|focus)/i.test(k) && typeof v?.auroc === 'number',
  )
  free.sort((a, b) => b[1].auroc - a[1].auroc)
  const knn = v11e2 && free.length ? free[0][1].auroc : r?.e2?.detectors?.knn_handcrafted_k15?.auroc
  const e2n = v11e2 && r?.e2?.n_test > 0 ? r.e2.n_test : undefined
  const e2s = v11e2 && r?.e2?.n_test_sessions > 0 ? r.e2.n_test_sessions : undefined
  const ds = data?.dataset
  const items = [
    fi && { v: `${fi.k}/${fi.n}`, l: 'fabricated claims blocked by the grounding gate' },
    typeof gated === 'number' && { v: pct(gated), l: `unsupported claims shipped (${pct(ungated)} without the gate)` },
    typeof det === 'number' && {
      v: pct(det),
      l: `injected imaging faults flagged on real chip images${typeof clean === 'number' ? `; ${pct(clean)} of clean images flagged` : ''}`,
    },
    typeof hit1 === 'number' && { v: pct(hit1), l: `right source ranked first across ${papers ?? ''} open organ-on-a-chip papers` },
    typeof ms === 'number' && { v: `${Math.round(ms)} ms`, l: 'per image on a laptop CPU. No GPU, no training' },
    typeof knn === 'number' && {
      v: knn.toFixed(2),
      l: `AUROC against expert culture decisions, training-free${typeof e2n === 'number' ? `, on ${e2n.toLocaleString('en-US')} held-out images${typeof e2s === 'number' ? ` from ${e2s} imaging days` : ''}` : ''}`,
    },
  ].filter(Boolean) as Array<{ v: string; l: string }>
  if (!items.length) return null
  return (
    <section className="section proof" aria-label="Measured results">
      <div className="wrap">
        <p className="eyebrow center">Measured, not claimed</p>
        <div className="proof-grid">
          {items.map((it, i) => (
            <Reveal key={it.l} className={`proof-item d${i % 3}`}>
              <div className="proof-v">{it.v}</div>
              <div className="proof-l">{it.l}</div>
            </Reveal>
          ))}
        </div>
        {ds && (
          <p className="proof-note">
            {ds.n_images_used} real images · {ds.cell_lines?.length} cell lines · expert-labelled public OoC dataset ({ds.citation}). Every
            figure is regenerated by <code>evaluate.py</code>.
          </p>
        )}
      </div>
    </section>
  )
}

export function Domain() {
  const items = [
    {
      t: 'Lung, gut and vessel models',
      d: 'Validated on lung (A549, NHBE, HSAEC), intestinal (Caco-2) and endothelial (HUVEC, HPMEC) chip cultures. In vitro cell lines only: no patient or clinical data.',
    },
    {
      t: 'Drug development readouts',
      d: 'Toxicity and dose-response studies read cell images. A bubble or a defocused frame can mimic a drug effect. Zelos flags it before it becomes a false signal.',
    },
    {
      t: 'Data assets and digital twins',
      d: 'Every image gets a certificate: hash, metrics, verdict, cause, lab action, cited evidence and an audit ledger, so organ-on-a-chip data can be standardized, shared and trusted.',
    },
  ]
  return (
    <section className="section" id="domain" aria-labelledby="domain-h">
      <div className="wrap">
        <SectionHead id="domain-h" title="Built for in vitro life systems." lede="Organ-on-a-chip tissue models grow for days in microfluidic channels and are imaged every day. Zelos is the quality gate between those images and every decision made from them." />
        <div className="grid3">
          {items.map((it, i) => (
            <Reveal key={it.t} className={`card card-problem d${i}`}>
              <h3>{it.t}</h3>
              <p>{it.d}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}

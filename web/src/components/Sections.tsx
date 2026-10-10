import type { SiteData } from '../types'
import { CopyButton, Reveal, SectionHead } from './common'

const GH = 'https://github.com/simon-derock/zelos'

export function Problem() {
  const items = [
    {
      t: 'Bad images look like data',
      d: 'An out-of-focus or over-exposed frame passes silently into a dataset and quietly corrupts the models and digital twins trained on it.',
      icon: <path d="M4 17c3-9 5-9 8 0s5 9 8 0M4 7h16" />,
    },
    {
      t: 'Experts cannot look at everything',
      d: 'A running chip produces thousands of frames a day. Manual review does not scale, so most images are never checked.',
      icon: <path d="M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6zm9-2.5a2.5 2.5 0 100 5 2.5 2.5 0 000-5z" />,
    },
    {
      t: 'Assistants make things up',
      d: 'Language models state numbers and cite papers that do not exist. A QC verdict you cannot verify is worse than none.',
      icon: <path d="M5 5h14v10H9l-4 4V5zm4 4h6m-6 3h4" />,
    },
  ]
  return (
    <section className="section" id="problem" aria-labelledby="problem-h">
      <div className="wrap">
        <SectionHead id="problem-h" title="Bad images fail quietly." lede="Organ-on-a-chip imaging only helps if every frame can be trusted." />
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
                    <span className="tag tag-llm">Language model</span>
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
      </div>
    </footer>
  )
}

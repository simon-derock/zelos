import { useEffect, useState } from 'react'
import { useSiteData } from './data'
import type { Meta } from './types'
import { Hero } from './components/Hero'
import { Problem, Pipeline, Reproduce, Footer, Proof, Domain } from './components/Sections'
import { Console } from './components/Console'
import { Benchmarks } from './components/Benchmarks'
import { Try, TRY_AVAILABLE } from './components/Try'

const ALL_LINKS = [
  ['problem', 'Problem'],
  ['how', 'How it works'],
  ['console', 'Console'],
  ['try', 'Try it'],
  ['benchmarks', 'Benchmarks'],
  ['reproduce', 'Reproduce'],
] as const
const LINKS = ALL_LINKS.filter(([id]) => id !== 'try' || TRY_AVAILABLE)

function Nav({ sample, meta }: { sample: boolean; meta?: Meta }) {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  return (
    <nav className={`nav ${scrolled ? 'is-scrolled' : ''}`} aria-label="Primary">
      <div className="nav-in">
        <a className="brand" href="#top" aria-label="Zelos, back to top">
          Zelos
        </a>
        <ul className="nav-links">
          {LINKS.map(([id, label]) => (
            <li key={id}>
              <a href={`#${id}`}>{label}</a>
            </li>
          ))}
        </ul>
        <div className="nav-end">
          {meta?.version && (
            <span className="pill pill-ver pill-ver-nav" title={`v${meta.version}, updated after the submission deadline`}>
              v{meta.version}
            </span>
          )}
          {sample && <span className="pill pill-sample">Sample data</span>}
          <a className="nav-gh" href="https://github.com/simon-derock/zelos" rel="noopener noreferrer">
            GitHub
          </a>
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  const { data, sample, error } = useSiteData()
  useEffect(() => {
    if (!data || !window.location.hash) return
    try {
      const el = document.querySelector(window.location.hash)
      if (el) requestAnimationFrame(() => el.scrollIntoView())
    } catch {
      /* invalid hash */
    }
  }, [data])
  return (
    <>
      <a className="skip" href="#console">Skip to the console</a>
      <Nav sample={sample} meta={data?.meta} />
      <main>
        <Hero meta={data?.meta} />
        <div className="grain" aria-hidden="true" />
        <Proof data={data} />
        <Problem />
        <Domain />
        <Pipeline />
        {data ? (
          <>
            <Console data={data} />
            {TRY_AVAILABLE && <Try />}
            <Benchmarks data={data} />
          </>
        ) : (
          <section className="section" id="console">
            <div className="wrap narrow">
              <p className="muted center">
                {error ? 'Results data could not be loaded. Run the evaluation to generate data/site.json.' : 'Loading results'}
              </p>
            </div>
          </section>
        )}
        {!data && TRY_AVAILABLE && <Try />}
        <Reproduce />
      </main>
      <Footer data={data} />
    </>
  )
}

import { useEffect, useState } from 'react'
import { useSiteData } from './data'
import { Hero } from './components/Hero'
import { Problem, Pipeline, Reproduce, Footer } from './components/Sections'
import { Console } from './components/Console'
import { Benchmarks } from './components/Benchmarks'

const LINKS = [
  ['problem', 'Problem'],
  ['how', 'How it works'],
  ['console', 'Console'],
  ['benchmarks', 'Benchmarks'],
  ['reproduce', 'Reproduce'],
] as const

function Nav({ sample }: { sample: boolean }) {
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
      <Nav sample={sample} />
      <main>
        <Hero />
        <Problem />
        <Pipeline />
        {data ? (
          <>
            <Console data={data} />
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
        <Reproduce />
      </main>
      <Footer data={data} />
    </>
  )
}

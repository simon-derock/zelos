import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { Level } from '../types'

export function useReveal<T extends HTMLElement>() {
  const ref = useRef<T | null>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (typeof IntersectionObserver === 'undefined') {
      el.classList.add('in')
      return
    }
    const io = new IntersectionObserver(
      (es) => {
        for (const e of es)
          if (e.isIntersecting) {
            e.target.classList.add('in')
            io.unobserve(e.target)
          }
      },
      { threshold: 0.12, rootMargin: '0px 0px -6% 0px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return ref
}

export function Reveal({ children, className = '' }: { children: ReactNode; className?: string }) {
  const ref = useReveal<HTMLDivElement>()
  return (
    <div ref={ref} className={`reveal ${className}`}>
      {children}
    </div>
  )
}

export function SectionHead({
  id,
  title,
  lede,
  dark,
}: {
  id?: string
  title: string
  lede?: ReactNode
  dark?: boolean
}) {
  return (
    <Reveal className={`sec-head ${dark ? 'on-dark' : ''}`}>
      <h2 id={id}>{title}</h2>
      {lede && <p className="lede">{lede}</p>}
    </Reveal>
  )
}

export function LevelBadge({ level, big }: { level: Level | string; big?: boolean }) {
  const l = String(level).toUpperCase()
  const cls = l === 'PASS' ? 'ok' : l === 'FAIL' ? 'bad' : 'warn'
  return (
    <span className={`badge badge-${cls} ${big ? 'badge-big' : ''}`}>
      <svg viewBox="0 0 16 16" width="1em" height="1em" aria-hidden="true">
        {cls === 'ok' && (
          <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        )}
        {cls === 'warn' && (
          <g fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M8 3.5v5.5" />
            <path d="M8 12.2v.1" />
          </g>
        )}
        {cls === 'bad' && (
          <path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        )}
      </svg>
      {l}
    </span>
  )
}

export function Source({ file, sha, n }: { file: string; sha?: string; n?: number | string }) {
  return (
    <p className="source">
      {n !== undefined && n !== null && <span>n = {typeof n === 'number' ? n.toLocaleString('en-US') : n}</span>}
      <span>
        results/{file}.json @ {sha ? sha.slice(0, 7) : 'unknown'}
      </span>
    </p>
  )
}

export function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  return (
    <button
      type="button"
      className="copy"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          setDone(true)
          setTimeout(() => setDone(false), 1600)
        } catch {
          /* clipboard unavailable */
        }
      }}
    >
      {done ? 'Copied' : 'Copy'}
    </button>
  )
}

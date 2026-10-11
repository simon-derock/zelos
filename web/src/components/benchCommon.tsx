import type { ReactNode } from 'react'
import type { SiteData } from '../types'
import { human, isObj, num, pct, shortSha } from '../util'
import { Reveal } from './common'

export type Obj = Record<string, any>

export function sha(data: SiteData, key: string): string {
  const r = data.results?.[key]
  return shortSha(r?.provenance?.git_sha ?? data.generated?.git_sha)
}

export function Tile({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'base' | 'ok' | 'bad' }) {
  return (
    <div className={`tile ${tone ? `tile-${tone}` : ''}`}>
      <p className="tile-v">{value}</p>
      <p className="tile-l">{label}</p>
      {sub && <p className="tile-s">{sub}</p>}
    </div>
  )
}

export function Card({ title, lede, children, foot, half }: { title: string; lede?: ReactNode; children: ReactNode; foot?: ReactNode; half?: boolean }) {
  return (
    <Reveal className={`bcard ${half ? 'half' : ''}`}>
      <h3>{title}</h3>
      {lede && <p className="muted bcard-lede">{lede}</p>}
      {children}
      {foot}
    </Reveal>
  )
}

/** The variant shown in detail: the highest threshold-free AUROC of the reference variants (null for v1.0 data). */
export function bestVariant(e1: Obj | undefined): [string, Obj] | null {
  if (!isObj(e1) || !isObj(e1.variants)) return null
  const all = Object.entries(e1.variants as Obj).filter(([, v]) => isObj(v)) as Array<[string, Obj]>
  if (!all.length) return null
  return all.reduce((a, b) => ((num(b[1].auroc_clean_vs_sev3plus) ?? -1) > (num(a[1].auroc_clean_vs_sev3plus) ?? -1) ? b : a))
}

const VARIANT_NAME: Record<string, string> = {
  'pooled_dev (v1)': 'Pooled reference (v1)',
  'between_session (v1.1)': 'Between-session reference (v1.1)',
  batch_session: 'Batch reference, same imaging day',
}
export const variantName = (k: string): string => VARIANT_NAME[k] ?? human(k)

export function Bar({ v, cls, tick, label }: { v?: number; cls: string; tick?: number; label: string }) {
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


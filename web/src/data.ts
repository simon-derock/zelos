import { useEffect, useState } from 'react'
import type { SiteData } from './types'

export interface Loaded {
  data: SiteData | null
  sample: boolean
  error: string | null
}

async function getJson(url: string): Promise<SiteData> {
  const r = await fetch(url, { cache: 'no-cache' })
  if (!r.ok) throw new Error(`${url}: ${r.status}`)
  const ct = r.headers.get('content-type') || ''
  if (ct.includes('text/html')) throw new Error(`${url}: not json`)
  const j = (await r.json()) as SiteData
  if (!j || !Array.isArray(j.samples)) throw new Error(`${url}: invalid shape`)
  return j
}

export function useSiteData(): Loaded {
  const [state, setState] = useState<Loaded>({ data: null, sample: false, error: null })
  useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const d = await getJson('data/site.json')
        if (alive) setState({ data: d, sample: false, error: null })
      } catch {
        try {
          const d = await getJson('data/site.sample.json')
          if (alive) setState({ data: d, sample: true, error: null })
        } catch (e) {
          if (alive) setState({ data: null, sample: false, error: String(e) })
        }
      }
    })()
    return () => {
      alive = false
    }
  }, [])
  return state
}

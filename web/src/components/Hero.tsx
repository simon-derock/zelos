import { useMemo } from 'react'

function lcg(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296
  }
}

export function cellField(seed: number, count: number, w: number, h: number) {
  const r = lcg(seed)
  const out: Array<{ x: number; y: number; rx: number; ry: number; rot: number; nx: number; ny: number; nr: number }> = []
  for (let i = 0; i < count; i++) {
    const rx = 26 + r() * 34
    out.push({
      x: r() * w,
      y: r() * h,
      rx,
      ry: rx * (0.62 + r() * 0.34),
      rot: r() * 180,
      nx: (r() - 0.5) * 10,
      ny: (r() - 0.5) * 10,
      nr: 8 + r() * 8,
    })
  }
  return out
}

export function Hero() {
  const cells = useMemo(() => cellField(7, 46, 1200, 520), [])
  return (
    <header className="hero" id="top">
      <div className="hero-inner">
        <p className="eyebrow">AI for Life Science · Organ-on-a-Chip</p>
        <h1 className="hero-title">Zelos.</h1>
        <p className="hero-sub">Every organ-on-a-chip image, certified.</p>
        <p className="hero-copy">
          Zelos inspects brightfield images of organ-on-a-chip cultures, names the imaging fault and the lab fix, cites
          open organ-on-a-chip research, and blocks every claim the evidence cannot support.
        </p>
        <div className="hero-cta">
          <a className="btn btn-primary" href="#console">
            Open the console <span aria-hidden="true">↓</span>
          </a>
          <a className="btn btn-ghost" href="https://github.com/simon-derock/zelos" rel="noopener noreferrer">
            GitHub <span aria-hidden="true">→</span>
          </a>
        </div>
      </div>
      <div className="field" role="img" aria-label="A microscope field of cells resolving from blurred into sharp focus, then certified.">
        <svg className="field-cells" viewBox="0 0 1200 520" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <defs>
            <radialGradient id="bf" cx="50%" cy="46%" r="75%">
              <stop offset="0" stopColor="#e9e9ec" />
              <stop offset="0.7" stopColor="#d4d4d8" />
              <stop offset="1" stopColor="#b9b9c0" />
            </radialGradient>
          </defs>
          <rect width="1200" height="520" fill="url(#bf)" />
          {cells.map((c, i) => (
            <g key={i} transform={`translate(${c.x} ${c.y}) rotate(${c.rot})`}>
              <ellipse rx={c.rx} ry={c.ry} fill="rgba(255,255,255,.38)" stroke="rgba(40,40,52,.62)" strokeWidth="2" />
              <ellipse rx={c.rx * 0.82} ry={c.ry * 0.78} fill="none" stroke="rgba(255,255,255,.7)" strokeWidth="1.5" />
              <circle cx={c.nx} cy={c.ny} r={c.nr} fill="rgba(60,60,78,.32)" stroke="rgba(40,40,52,.45)" strokeWidth="1.2" />
            </g>
          ))}
        </svg>
        <svg className="field-reticle" viewBox="0 0 1200 520" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
          <g fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round">
            <path d="M80 130V80h50" />
            <path d="M1120 130V80h-50" />
            <path d="M80 390v50h50" />
            <path d="M1120 390v50h-50" />
          </g>
        </svg>
        <div className="field-stamp">
          <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
            <circle cx="8" cy="8" r="8" fill="#30d158" />
            <path d="M4.4 8.4l2.4 2.4 4.6-5.2" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Certified
        </div>
      </div>
    </header>
  )
}

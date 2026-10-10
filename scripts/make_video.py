"""Render the Zelos demo video (1920x1080, 30 fps) from the real web console.

Frames are captured deterministically with Playwright (system Chrome): motion-graphic intro/outro and
chapter cards are HTML/CSS animations sampled frame by frame; the console walkthrough is the real site
rendering real pipeline outputs (web/public/data/site.json). Subtitles (yellow) are burned in with libass;
music: Beethoven Symphony No. 1, I. (United States Marine Band, public domain, Wikimedia Commons).

Run: python scripts/make_video.py --url http://localhost:4199 --music cache/video/beethoven-1.opus
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
W, H = 1920, 1080

CARD_CSS = """
*{margin:0;box-sizing:border-box}html,body{width:1920px;height:1080px;background:#000;overflow:hidden;
font-family:-apple-system,"SF Pro Display","Inter","Noto Sans","DejaVu Sans",sans-serif;color:#f5f5f7}
.c{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}
.glow{position:absolute;width:1400px;height:1400px;border-radius:50%;background:radial-gradient(circle,rgba(212,175,55,.22),transparent 60%);
animation:glow 8s ease-out both}
@keyframes glow{from{transform:scale(.6);opacity:0}to{transform:scale(1.1);opacity:1}}
.eb{font-size:30px;font-weight:600;letter-spacing:.04em;color:#a1a1a6;animation:up 1s cubic-bezier(.2,.9,.2,1) .2s both}
.t{font-size:230px;font-weight:800;letter-spacing:-.05em;line-height:1;margin:18px 0 10px;
background:linear-gradient(180deg,#fff 30%,#b9b9be);-webkit-background-clip:text;background-clip:text;color:transparent;
animation:pop 1.4s cubic-bezier(.16,1.1,.3,1) .5s both}
.s{font-size:58px;font-weight:600;letter-spacing:-.02em;animation:up 1s cubic-bezier(.2,.9,.2,1) 1.5s both}
.m{margin-top:46px;font-size:34px;color:#a1a1a6;line-height:1.5}
.m span{display:block;animation:up .9s cubic-bezier(.2,.9,.2,1) both}
.m span:nth-child(1){animation-delay:2.6s}.m span:nth-child(2){animation-delay:3.3s}.m span:nth-child(3){animation-delay:4s;color:#f3d77a}
.k{margin-top:40px;font-size:24px;color:#86868b;animation:up .9s ease 4.8s both}
.ln{position:absolute;bottom:0;left:50%;width:0;height:3px;background:linear-gradient(90deg,transparent,#d4af37,transparent);
transform:translateX(-50%);animation:ln 3s ease 1s both}
@keyframes ln{to{width:1200px}}
@keyframes pop{0%{opacity:0;transform:scale(.82);filter:blur(16px)}100%{opacity:1;transform:none;filter:blur(0)}}
@keyframes up{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:none}}
.ch{font-size:120px;font-weight:800;letter-spacing:-.04em;animation:pop 1s cubic-bezier(.16,1.1,.3,1) .1s both}
.cs{margin-top:20px;font-size:40px;color:#a1a1a6;animation:up .8s ease .5s both}
.cr{margin-top:56px;font-size:22px;color:#6e6e73;line-height:1.7;animation:up .8s ease 2.2s both}
"""


def card(body: str) -> str:
    return f"<html><head><style>{CARD_CSS}</style></head><body><div class='c'><div class='glow'></div>{body}</div><div class='ln'></div></body></html>"


INTRO = card("""<div class='eb'>AI FOR LIFE SCIENCE · ORGAN-ON-A-CHIP</div><div class='t'>Zelos.</div>
<div class='s'>Every organ-on-a-chip image, certified.</div>
<div class='m'><span>The LLM proposes.</span><span>Tools compute.</span><span>The gate disposes.</span></div>
<div class='k'>Category: End-to-End System · AI4S Open Innovation</div>""")


def chapter(title: str, sub: str) -> str:
    return card(f"<div class='eb'>ZELOS</div><div class='ch'>{title}</div><div class='cs'>{sub}</div>")


OUTRO = card("""<div class='t' style='font-size:180px'>Zelos.</div><div class='s'>Certified images. Verified claims.</div>
<div class='m'><span>github.com/simon-derock/zelos</span><span>zelos.philipsimonderock.com</span><span>Research tool. Not a medical device.</span></div>
<div class='cr'>Data: Organ-on-a-Chip Image Dataset, Movčana et al., Data 2024 (Zenodo 10.5281/zenodo.10203721, CC BY 4.0)<br>
Music: Beethoven, Symphony No. 1, I. · United States Marine Band · public domain · Built by Philip Simon Derock</div>""")


class Reel:
    def __init__(self, page, out: Path):
        self.page, self.out, self.n, self.subs = page, out, 0, []
        out.mkdir(parents=True, exist_ok=True)

    @property
    def t(self) -> float:
        return self.n / FPS

    def shot(self) -> None:
        self.page.screenshot(path=str(self.out / f"{self.n:05d}.jpg"), type="jpeg", quality=93)
        self.n += 1

    def hold(self, sec: float) -> None:
        last = self.out / f"{self.n - 1:05d}.jpg"
        for _ in range(int(sec * FPS)):
            os.link(last, self.out / f"{self.n:05d}.jpg")
            self.n += 1

    def sub(self, text: str, start: float, end: float) -> None:
        self.subs.append((start, end, text))

    def animate_card(self, html: str, sec: float) -> None:
        self.page.set_content(html)
        self.page.evaluate("document.getAnimations().forEach(a=>a.pause())")
        for f in range(int(sec * FPS)):
            self.page.evaluate("t=>document.getAnimations().forEach(a=>{a.currentTime=t})", f * 1000 / FPS)
            self.shot()

    def scroll_to(self, y1: float, sec: float = 1.0) -> None:
        y0 = self.page.evaluate("window.scrollY")
        n = max(1, int(sec * FPS))
        for f in range(1, n + 1):
            u = f / n
            e = 1 - (1 - u) ** 3
            self.page.evaluate("y=>window.scrollTo(0,y)", y0 + (y1 - y0) * e)
            self.shot()

    def scroll_el(self, selector: str, sec: float = 1.0, offset: int = 60) -> None:
        y = self.page.evaluate("([s,o])=>{const e=document.querySelector(s);return e?e.getBoundingClientRect().top+window.scrollY-o:window.scrollY}", [selector, offset])
        self.scroll_to(y, sec)


def ass(subs: list[tuple[float, float, str]], path: Path) -> None:
    def ts(x: float) -> str:
        h, r = divmod(x, 3600)
        m, s = divmod(r, 60)
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Y,Noto Sans,50,&H0000E5FF,&H0000E5FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,3.2,1.5,2,140,140,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [f"Dialogue: 0,{ts(a)},{ts(b)},Y,,0,0,0,,{{\\fad(180,180)}}{t}" for a, b, t in subs]
    path.write_text(head + "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4199")
    ap.add_argument("--music", default=str(ROOT / "cache/video/beethoven-1.opus"))
    ap.add_argument("--out", default=str(ROOT / "cache/video/zelos_demo.mp4"))
    a = ap.parse_args()
    r = json.loads((ROOT / "results/results.json").read_text())
    fi = r["e5"]["gate_fault_injection"]["injected_fabrications_caught"]
    ung = r["e5"]["conditions"]["llm_ungated"]["unsupported_claim_rate_in_output"]["rate"]
    det = r["e1"]["overall"]["detection_rate_sev3plus"]
    knn = r["e2"]["detectors"]["knn_handcrafted_k15"]["auroc"]
    qcs = r["e2"]["detectors"]["zelos_qc_score"]["auroc"]
    frames = ROOT / "cache/video/frames"
    shutil.rmtree(frames, ignore_errors=True)

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/google/chrome/chrome", args=["--hide-scrollbars", "--force-color-profile=srgb"])
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        R = Reel(pg, frames)

        R.animate_card(INTRO, 7.0)
        R.hold(0.6)

        pg.goto(a.url, wait_until="networkidle")
        pg.add_style_tag(content="html{scroll-behavior:auto!important}")
        pg.wait_for_timeout(1600)
        s = R.t
        R.shot()
        R.hold(3.5)
        R.sub("Organ-on-a-chip labs image their in vitro lung, gut and vessel tissue models every day.", s, R.t)
        s = R.t
        R.scroll_el(".proof", 1.0)
        R.hold(5.0)
        R.sub(f"Zelos certifies every image. {fi['k']} of {fi['n']} fabricated claims blocked. Measured, not claimed.", s, R.t)
        s = R.t
        R.scroll_el("#problem", 1.0)
        R.hold(4.5)
        R.sub("Bubbles, drifting focus and uneven light look like biology, and AI assistants invent evidence.", s, R.t)
        s = R.t
        R.scroll_el("#domain", 1.0)
        R.hold(4.5)
        R.sub("Built for in vitro organ-on-a-chip cultures: drug-development readouts and digital-twin data assets.", s, R.t)
        s = R.t
        R.scroll_el("#how", 1.0)
        R.hold(5.0)
        R.sub("17 physics metrics decide the verdict with zero LLM calls. A Workers AI model explains. The gate verifies.", s, R.t)
        url_state = pg.evaluate("window.scrollY")

        R.animate_card(chapter("Live console", "Real images. Real pipeline outputs."), 2.6)
        pg.goto(a.url, wait_until="networkidle")
        pg.add_style_tag(content="html{scroll-behavior:auto!important}")
        pg.wait_for_timeout(1200)
        pg.evaluate("y=>window.scrollTo(0,y)", url_state)
        s = R.t
        R.scroll_el("#console", 0.8, 20)
        R.hold(4.0)
        R.sub("Real brightfield images from the public, expert-labelled organ-on-a-chip dataset.", s, R.t)
        captions = [
            "Each image gets a verdict, the measured cause and the lab action.",
            "Evidence comes from open organ-on-a-chip papers, cited passage by passage.",
            "Injected faults have a known answer: defocus, bubbles, over-exposure, uneven light.",
            "Zelos fails them, names the fault and recommends the fix.",
            "Every number on the certificate comes from a tool, never from the language model.",
        ]
        for i, cap in enumerate(captions):
            s = R.t
            for _ in range(2 if i >= 2 else 1):
                pg.keyboard.press("ArrowRight")
            pg.wait_for_timeout(350)
            R.shot()
            R.hold(3.2)
            R.sub(cap, s, R.t)
        s = R.t
        R.scroll_el("[role=switch]", 1.0, 520)
        R.hold(4.0)
        R.sub("The claim ledger: every sentence is typed and linked to a tool output or a cited passage.", s, R.t)
        s = R.t
        try:
            pg.locator("[role=switch]").first.click()
            pg.wait_for_timeout(400)
        except Exception:
            pass
        R.shot()
        R.hold(5.0)
        R.sub(f"Gate off: the raw LLM draft. Without the gate {round(ung * 100, 1)}% of claims were unsupported; with it, none.", s, R.t)
        try:
            pg.locator("[role=switch]").first.click()
            pg.wait_for_timeout(300)
        except Exception:
            pass
        R.shot()
        R.hold(1.5)

        R.animate_card(chapter("Measured results", "Every number regenerated by evaluate.py"), 2.6)
        pg.goto(a.url, wait_until="networkidle")
        pg.add_style_tag(content="html{scroll-behavior:auto!important}")
        pg.wait_for_timeout(1200)
        pg.evaluate("()=>window.scrollTo(0,document.querySelector('#benchmarks').getBoundingClientRect().top+window.scrollY-300)")
        s = R.t
        R.scroll_el("#benchmarks", 0.8, 20)
        R.hold(4.5)
        R.sub(f"Injected imaging faults on real chip images: {round(det * 100, 1)}% flagged at severity 3 and above.", s, R.t)
        bench_caps = [
            f"Honest finding: expert good/bad tracks culture state. Artifact score AUROC {qcs:.2f}; training-free kNN {knn:.2f}.",
            "Retrieval: Cohere embeddings, Qdrant, BM25 and reranking over open organ-on-a-chip papers.",
            f"The grounding gate caught {fi['k']} of {fi['n']} injected fabrications and blocked no correct claims.",
        ]
        for cap in bench_caps:
            s = R.t
            y = pg.evaluate("window.scrollY") + 900
            R.scroll_to(y, 1.0)
            R.hold(4.5)
            R.sub(cap, s, R.t)
        s = R.t
        R.scroll_el("#reproduce", 1.0)
        R.hold(5.0)
        R.sub("One command reproduces every number offline. No API keys, no GPU, no training.", s, R.t)
        R.animate_card(OUTRO, 8.0)
        R.hold(1.0)
        b.close()

    total = R.t
    ass(R.subs, ROOT / "cache/video/subs.ass")
    fade = max(0.0, total - 3.0)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.jpg"), "-i", a.music,
           "-filter_complex", f"[0:v]subtitles={ROOT / 'cache/video/subs.ass'},format=yuv420p[v];"
           f"[1:a]atrim=0:{total:.2f},volume=0.55,afade=t=in:st=0:d=1.5,afade=t=out:st={fade:.2f}:d=3[a]",
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
           "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.2f}", a.out]
    subprocess.run(cmd, check=True)
    print(f"wrote {a.out} ({total:.1f} s, {R.n} frames)")


if __name__ == "__main__":
    main()

"""Vision-LLM baseline: can a general multimodal LLM do the QC job? (head-to-head with Zelos)

Same real test-session images for every system:
  A) 20 expert-good + 20 expert-bad images           -> agreement with the expert label
  B) 20 clean images + the same 20 with an injected severity-5 fault (all 11 kinds cycled)
                                                      -> false-alarm rate and fault detection
Models (Workers AI, OpenAI-compatible chat with image): mistral-small-3.1-24b, llama-4-scout-17b, gemma-4-26b.
Prompt asks for {"verdict": "good"|"bad", "reason": ...}; replies are cached in eval/vlm_cache/ (replayable).
Phases: --phase calls (needs CLOUDFLARE_* keys, spends neurons) | score (offline; needs results/calibration.json)
Output: results/vlm_baseline.json / .md
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_vision_v2 import split_sessions  # noqa: E402

from zelos.common.config import load_config, provenance, secret, write_json  # noqa: E402
from zelos.vision.corrupt import KINDS, corrupt  # noqa: E402
from zelos.vision.metrics import downscale, to_gray01  # noqa: E402

CFG = load_config()
SEED = CFG["seed"]
CACHE = ROOT / "eval" / "vlm_cache"
MODELS = ["@cf/mistralai/mistral-small-3.1-24b-instruct", "@cf/meta/llama-4-scout-17b-16e-instruct", "@cf/google/gemma-4-26b-a4b-it"]
PROMPT = ('This is a brightfield microscopy image of an organ-on-a-chip cell culture. Is the image technically '
          'acceptable for analysis (in focus, evenly lit, not saturated, no bubbles or debris)? '
          'Reply with JSON only: {"verdict":"good" or "bad","reason":"short"}')


def cases() -> list[dict]:
    idx = pickle.load(open(ROOT / "data/raw/zip_index.pkl", "rb"))
    sess = pd.DataFrame({"session": [Path(r[0]).stem.split("_")[0] for r in idx]})
    _, _, TS = split_sessions(sess, SEED)
    man = pd.read_csv(ROOT / "data/raw/manifest.csv")
    man["path"] = [str(ROOT / "data/raw" / p) for p in man.local_path]
    man = man[man.path.map(lambda p: Path(p).exists()) & man.session.astype(str).isin(TS)]
    good = man[man.label == "good"].sample(frac=1, random_state=SEED)
    bad = man[man.label == "bad"].sample(frac=1, random_state=SEED)
    out = []
    for _, r in good.head(20).iterrows():
        out.append({"id": f"A-good-{r.file_id}", "group": "expert_good", "path": r.path, "kind": None, "expected": "good"})
    for _, r in bad.head(20).iterrows():
        out.append({"id": f"A-bad-{r.file_id}", "group": "expert_bad", "path": r.path, "kind": None, "expected": "bad"})
    for i, (_, r) in enumerate(good.iloc[20:40].iterrows()):
        k = KINDS[i % len(KINDS)]
        out.append({"id": f"B-clean-{r.file_id}", "group": "clean", "path": r.path, "kind": None, "expected": "good"})
        out.append({"id": f"B-fault-{k}-{r.file_id}", "group": "fault", "path": r.path, "kind": k, "expected": "bad"})
    return out


def load_gray(case: dict) -> np.ndarray:
    g, _ = to_gray01(cv2.imread(case["path"], cv2.IMREAD_UNCHANGED))
    g = downscale(g, CFG["vision"]["max_side"])
    if case["kind"]:
        g = corrupt(g, case["kind"], 5, np.random.default_rng(SEED + int(hashlib.md5(case["id"].encode()).hexdigest()[:6], 16) % 10**6))
    return g


def jpeg_b64(g: np.ndarray) -> str:
    u8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.resize(u8, (768, 576), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()


def ask(model: str, case: dict, b64: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((model + case["id"] + PROMPT).encode()).hexdigest()[:24]
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    acc, tok = secret("CLOUDFLARE_ACCOUNT_ID"), secret("CLOUDFLARE_API_TOKEN")
    if not (acc and tok):
        return {"error": "no_keys"}
    body = {"model": model, "temperature": 0, "max_tokens": 700 if "gemma" in model else 150,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}
    err = None
    for attempt in range(3):
        t = time.time()
        try:
            r = requests.post(f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/v1/chat/completions", timeout=180,
                              headers={"Authorization": f"Bearer {tok}"}, json=body)
        except requests.RequestException as e:
            err = str(e)
            time.sleep(3)
            continue
        if r.status_code == 200:
            d = r.json()
            ch = d["choices"][0]["message"]
            res = {"text": ch.get("content") or "", "usage": d.get("usage", {}), "latency_s": round(time.time() - t, 2)}
            f.write_text(json.dumps(res))
            return res
        err = f"{r.status_code} {r.text[:200]}"
        if r.status_code in (429, 500, 502, 503):
            time.sleep(4 * (attempt + 1))
            continue
        break
    return {"error": err}


def verdict(text: str) -> str | None:
    m = re.search(r'"verdict"\s*:\s*"(good|bad)"', text or "", re.I)
    return m.group(1).lower() if m else None


def phase_calls(models: list[str]) -> None:
    cs = cases()
    print(len(cs), "cases", flush=True)
    for model in models:
        stop = False
        for k, c in enumerate(cs, 1):
            r = ask(model, c, jpeg_b64(load_gray(c)))
            if "error" in r:
                print(model, "error at case", k, r["error"][:160], flush=True)
                if "4006" in r["error"] or "quota" in r["error"].lower() or "neurons" in r["error"].lower() or "429" in r["error"][:4]:
                    stop = True
                    break
            if k % 20 == 0:
                print(model, k, "/", len(cs), flush=True)
        if stop:
            print("stopping:", model, "quota/rate limit", flush=True)


def phase_score() -> None:
    from zelos.vision.pipeline import inspect

    cs = cases()
    zel = {}
    for c in cs:
        g = load_gray(c)
        q = inspect(g.astype(np.float32), image_id=c["id"])
        zel[c["id"]] = q["level"] != "PASS"
    out = {"n_cases": len(cs), "models": {}, "provenance": provenance({"prompt": PROMPT}), "design": {
        "A": "20 expert-good + 20 expert-bad test-session images", "B": "20 clean + the same 20 with a severity-5 injected fault"}}

    def tabulate(flags: dict[str, bool | None]) -> dict:
        def rate(group: str) -> float | None:
            v = [flags[c["id"]] for c in cs if c["group"] == group and flags[c["id"]] is not None]
            return float(np.mean(v)) if v else None
        spec, rec = rate("expert_good"), rate("expert_bad")
        return {"false_flag_on_expert_good": spec, "flag_rate_on_expert_bad": rec,
                "balanced_accuracy_vs_expert": None if spec is None or rec is None else float(((1 - spec) + rec) / 2),
                "false_flag_on_clean": rate("clean"), "detected_sev5_faults": rate("fault"),
                "unparsed": int(sum(v is None for v in flags.values()))}

    out["zelos_stored_reference_calibrated"] = tabulate(zel)
    for model in MODELS:
        flags, toks, lat, missing = {}, [], [], 0
        for c in cs:
            f = CACHE / f"{hashlib.sha256((model + c['id'] + PROMPT).encode()).hexdigest()[:24]}.json"
            if not f.exists():
                missing += 1
                flags[c["id"]] = None
                continue
            r = json.loads(f.read_text())
            v = verdict(r.get("text", ""))
            flags[c["id"]] = None if v is None else (v == "bad")
            toks.append(r.get("usage", {}).get("total_tokens", 0))
            lat.append(r.get("latency_s", 0))
        t = tabulate(flags)
        t.update({"cases_without_reply": missing, "tokens_per_image_mean": float(np.mean(toks)) if toks else None,
                  "latency_p50_s": float(np.median(lat)) if lat else None})
        out["models"][model] = t
    write_json(ROOT / "results/vlm_baseline.json", out)
    rows = ["| System | false flags on expert-good | flags on expert-bad | false flags on clean | sev-5 faults detected | unparsed |", "|---|---:|---:|---:|---:|---:|"]

    def f(x):
        return "n/a" if x is None else f"{x:.2f}"
    for name, t in [("Zelos v1.1 (no model trained)", out["zelos_stored_reference_calibrated"])] + list(out["models"].items()):
        rows.append(f"| {name} | {f(t['false_flag_on_expert_good'])} | {f(t['flag_rate_on_expert_bad'])} | {f(t['false_flag_on_clean'])} | {f(t['detected_sev5_faults'])} | {t['unparsed']} |")
    (ROOT / "results/vlm_baseline.md").write_text("# Vision-LLM baseline vs Zelos\n\nn = 20 per group, same images for every system.\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["calls", "score"], required=True)
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    phase_calls(a.models) if a.phase == "calls" else phase_score()

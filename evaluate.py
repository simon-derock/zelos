"""Zelos evaluation entry point: regenerates every published number (v1.1).

Order: features (optional, long) -> E1/E2 vision (all 3,072 images, scripts/eval_vision_v2.py) -> rule citations ->
vision-LLM baseline scoring -> E4 retrieval -> E5 grounding -> LLM sweep (replay) -> aggregate into
results/results.json + results/results.md. Tables and findings are computed from the JSON; nothing is typed by hand.

  python evaluate.py                      # everything that can run offline from caches
  python evaluate.py --extract            # also (re)stream all 3,072 images from Zenodo for features (~45 min)
  python evaluate.py --skip-vision        # reuse results/e1.json, e2.json (reviewer smoke run)
  python evaluate.py --check-provenance   # fail if a number in README.md is absent from the results
Offline by default: Cohere / Workers AI responses are replayed from committed caches.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

from zelos.common.config import provenance, write_json  # noqa: E402


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, str(ROOT / script), *args], check=True, cwd=ROOT, env=env)


def load(name: str) -> dict:
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def best_variant(e1: dict) -> tuple[str, dict]:
    v = e1.get("variants", {})
    name = "batch_session" if "batch_session" in v else next(iter(v), "")
    return name, v.get(name, {})


def findings(r: dict) -> list[str]:
    e1, e2, e4, rc, vlm, sweep = r["e1"], r["e2"], r["e4"], r["rule_citations"], r["vlm_baseline"], r["e5_sweep"]
    out = []
    if e2:
        d = e2["detectors"]
        z = d.get("zelos_qc[batch_session]", {})
        cs = d.get("knn_dinov2_cls_k15", {})
        lt = next((v for k, v in d.items() if k.startswith("LABEL-TRAINED")), {})
        out.append(f"E2 on {e2['n_test']} images from {e2['n_test_sessions']} held-out imaging sessions (of {e2['n_images']} images, "
                   f"{e2['n_sessions']} sessions): the artifact QC score reaches AUROC {f3(z.get('auroc'))} against the expert label, "
                   f"DINOv2 kNN (expert-labelled neighbours, no model trained) reaches {f3(cs.get('auroc'))}, a label-trained linear head {f3(lt.get('auroc'))}.")
        ws = cs.get("within_session", {}).get("weighted")
        if ws is not None:
            out.append(f"E2 within the same imaging day (no date shortcut): DINOv2 kNN AUROC {f3(ws)} "
                       f"(weighted over {cs['within_session']['n_sessions']} sessions); artifact score "
                       f"{f3(z.get('within_session', {}).get('weighted'))}.")
    if e1:
        n, v = best_variant(e1)
        p = e1["variants"].get("pooled_dev (v1)", {})
        out.append(f"E1 ({e1['n_base_images']} real test-session images x 11 faults x 3 severities, {n}): clean false-flag rate "
                   f"{pct(v.get('clean_flag_rate'))} at the calibrated threshold (pooled-reference v1: {pct(p.get('clean_flag_rate'))}); "
                   f"severity>=3 faults detected {pct(v.get('detection_rate_sev3plus'))}; cause named correctly "
                   f"{pct(v.get('attribution_v2_signatures'))} with signature centroids vs {pct(v.get('attribution_v1_rules'))} with the rule table "
                   f"(chance {pct(v.get('chance_attribution'))}).")
        h = e1.get("head_to_head", {}).get("label_trained_dinov2_linear_head")
        if h:
            sv = e1["variants"].get("between_session (v1.1)", {})
            out.append(f"Head-to-head on the same injected faults, each at its own <=10% session-averaged false-alarm calibration: the label-trained "
                       f"DINOv2 head flags {pct(h['clean_flag_rate'])} of clean and detects {pct(h['detection_rate_sev3plus'])} of severity>=3 faults "
                       f"(AUROC {f3(h['auroc_clean_vs_sev3plus'])}) and cannot name a cause; Zelos on a single image detects "
                       f"{pct(sv.get('detection_rate_sev3plus'))} (AUROC {f3(sv.get('auroc_clean_vs_sev3plus'))}), with a session reference "
                       f"{pct(v.get('detection_rate_sev3plus'))} (AUROC {f3(v.get('auroc_clean_vs_sev3plus'))}), and names the cause.")
    if rc:
        out.append(f"Rule basis: {rc['n_supported']} of {rc['n_rules']} cause-action rules have a verified sentence in the open corpus "
                   f"(threshold {rc['theta_lex']}, post hoc); the rest are reported unsupported.")
    if vlm:
        m = vlm.get("models", {})
        parts = [f"{k.split('/')[-1]}: detects {pct(t.get('detected_sev5_faults'))}, false flags {pct(t.get('false_flag_on_clean'))}"
                 for k, t in m.items() if t.get("cases_without_reply", 1) == 0]
        z = vlm.get("zelos_stored_reference_calibrated", {})
        if parts:
            out.append("Vision-LLM baseline (n=20 per group): " + "; ".join(parts) + f"; Zelos detects {pct(z.get('detected_sev5_faults'))}, "
                       f"false flags {pct(z.get('false_flag_on_clean'))}.")
    if sweep and sweep.get("verdict"):
        v = sweep["verdict"]
        if isinstance(v, list):
            v = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in v)
        elif isinstance(v, dict):
            v = str(v.get("text", ""))
        out.append("LLM sweep behind the gate: " + v)
    if e4 and "modes" in e4:
        m = e4["modes"]
        b = max(m, key=lambda k: m[k]["paper_level"].get("mrr", 0))
        out.append(f"E4: over {e4['corpus']['n_papers']} open-access papers / {e4['corpus']['n_chunks']} chunks the best mode is {b} "
                   f"(paper-level MRR {f3(m[b]['paper_level'].get('mrr'))}).")
    return [x for x in out if x]


def md(r: dict, prov: dict, fx: list[str]) -> str:
    e1, e2, e9, rc, vlm = r["e1"], r["e2"], r["e9"], r["rule_citations"], r["vlm_baseline"]
    L = ["# Zelos results (v1.1)", "", f"Generated by `python evaluate.py` at commit `{prov['git_sha'][:7]}` ({prov['timestamp_utc']}). "
         "Every number below is read from results/*.json.", ""]
    if e2:
        L += ["## E2: agreement with the expert good/bad label on held-out imaging sessions", "",
              f"{e2['n_images']} images, {e2['n_sessions']} sessions; test = {e2['n_test']} images from {e2['n_test_sessions']} sessions "
              f"({e2['n_test_bad']} expert-bad). Session-cluster bootstrap CIs. 'Within day' = AUROC computed inside each imaging session.", "",
              "| Detector | AUROC [95% CI] | within day (weighted) | sessions |", "|---|---:|---:|---:|"]
        for k, v in e2["detectors"].items():
            w = v.get("within_session", {})
            L.append(f"| {k} | {f3(v['auroc'])} [{f3(v['auroc_ci95'][0])}, {f3(v['auroc_ci95'][1])}] | {f3(w.get('weighted'))} | {w.get('n_sessions', 0)} |")
        L += ["", "| Reference | threshold | false flags on expert-good (test) | recall of expert-bad | pre-registered z=3 false flags |", "|---|---:|---:|---:|---:|"]
        for k, v in e2["operating_points"].items():
            L.append(f"| {k} | {f3(v['threshold'])} | {pct(v['false_flag_rate_test_good'])} | {pct(v['recall_expert_bad'])} | {pct(v['preregistered_z3_false_flag'])} |")
        L.append("")
    if e1:
        L += ["## E1: injected faults with known ground truth (real test-session images)", "",
              f"{e1['n_base_images']} base images x 11 fault kinds x severities {e1['severities']}; signatures fitted on {e1['n_signature_images']} dev-session images. "
              f"Operating point: {e1['operating_point']}.", "",
              "| Reference | clean flagged | faults detected (sev>=3) | sev 1 | AUROC | cause: rules (v1) | cause: signatures (v1.1) | top-2 | chance |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in e1["variants"].items():
            L.append(f"| {k} | {pct(v['clean_flag_rate'])} | {pct(v['detection_rate_sev3plus'])} | {pct(v['detection_rate_sev1'])} | {f3(v['auroc_clean_vs_sev3plus'])} | "
                     f"{pct(v['attribution_v1_rules'])} | {pct(v['attribution_v2_signatures'])} | {pct(v['attribution_v2_top2'])} | {pct(v['chance_attribution'])} |")
        n, v = best_variant(e1)
        L += ["", f"Per fault kind ({n}):", "", "| Fault | expected cause | detected sev1 / 3 / 5 | cause: rules | cause: signatures |", "|---|---|---|---:|---:|"]
        for k, pk in v.get("per_kind", {}).items():
            ds = pk["detection_by_severity"]
            L.append(f"| {k} | {pk['expected_label']} | {pct(ds.get('1'))} / {pct(ds.get('3'))} / {pct(ds.get('5'))} | "
                     f"{pct(pk['attribution_v1_rules'])} | {pct(pk['attribution_v2_signatures'])} |")
        h = e1.get("head_to_head", {}).get("label_trained_dinov2_linear_head")
        if h:
            L += ["", "### Head-to-head: label-trained DINOv2 head vs Zelos on the same faults", "",
                  "| System | clean flagged | faults detected (sev>=3) | AUROC | names the cause |", "|---|---:|---:|---:|---|",
                  f"| Zelos, session-relative ({n}) | {pct(v['clean_flag_rate'])} | {pct(v['detection_rate_sev3plus'])} | {f3(v['auroc_clean_vs_sev3plus'])} | yes ({pct(v['attribution_v2_signatures'])}) |",
                  *([f"| Zelos, single image (between_session) | {pct(sv['clean_flag_rate'])} | {pct(sv['detection_rate_sev3plus'])} | {f3(sv['auroc_clean_vs_sev3plus'])} | yes ({pct(sv['attribution_v2_signatures'])}) |"]
                    if (sv := e1["variants"].get("between_session (v1.1)")) else []),
                  f"| label-trained DINOv2 linear head | {pct(h['clean_flag_rate'])} | {pct(h['detection_rate_sev3plus'])} | {f3(h['auroc_clean_vs_sev3plus'])} | no |",
                  "", f"Embedding-path check: mean |dP| between the two preprocessing paths on clean images = {f3(h.get('mean_abs_prob_diff_between_paths'))}; "
                  f"clean flag rate on the extraction path = {pct(h.get('clean_flag_rate_extraction_path'))}.",
                  "The head is one approach that learns the expert label from frozen embeddings, implemented here for comparison; it is not part of Zelos."]
        L.append("")
    cal = load("calibration")
    if cal:
        L += ["## Calibration: do false-alarm rates transfer to unseen imaging sessions?", "",
              "Cutoffs are chosen so the mean per-session false-flag rate on expert-good development images, leave-one-session-out, is at most 10%. "
              "A target the leave-one-session-out rate cannot reach (alpha at its lower bound) is reported as it is.", "",
              "| Reference | alpha | leave-one-session-out mean | per-session median / 80th pct | false flags on test (expert-good) |", "|---|---:|---:|---|---:|"]
        ops = (e2 or {}).get("operating_points", {})
        for k, v in cal["variants"].items():
            a, b = v["loso_session_rates_p50_p80"]
            L.append(f"| {k} | {v['alpha']:.5f} | {pct(v['loso_mean_false_flag_rate'])} | {pct(a)} / {pct(b)} | "
                     f"{pct(ops.get(k, {}).get('false_flag_rate_test_good'))} |")
        L.append("")
    if rc:
        L += ["## Literature basis for the lab actions", "", "| Cause | status | support | source |", "|---|---|---:|---|"]
        for k, v in rc["rules"].items():
            L.append(f"| {k} | {v['status']} | {f3(v['support'])} | {v.get('chunk_id') or '-'} |")
        L += ["", f"{rc['n_supported']}/{rc['n_rules']} supported. {rc.get('disclosure', '')}", ""]
    if vlm:
        t = RES / "vlm_baseline.md"
        if t.exists():
            L += [t.read_text().strip(), ""]
    for name in ("e4", "e5", "e5_sweep"):
        p = RES / f"{name}.md"
        if p.exists():
            L += [p.read_text().strip(), ""]
    if e9:
        L += ["## E9: efficiency (CPU)", "", f"Metric suite p50 {e9.get('metrics_ms_p50', 0):.0f} ms; full inspect p50 {e9.get('inspect_ms_p50', 0):.0f} ms "
              f"(image {e9.get('image_shape', '')}).", ""]
    L += ["## Findings", ""] + [f"- {x}" for x in fx] + [""]
    return "\n".join(L)


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w.])\d+\.\d+%?|(?<![\w.])\d{2,}%", text))


def check_provenance() -> int:
    """Every decimal / percentage in README.md must appear in results/results.md or results.json."""
    readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
    allowed = numbers_in((RES / "results.md").read_text()) | numbers_in(json.dumps(load("results")))
    readme = re.sub(r"<!--np-->.*?<!--/np-->", "", readme, flags=re.S)
    readme = re.sub(r"https?://\S+|\]\([^)]*\)", "", readme)
    missing = sorted(n for n in numbers_in(readme) if n not in allowed)
    if missing:
        print("PROVENANCE FAIL: numbers in README not found in results:", missing)
        return 1
    print("PROVENANCE PASS")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="stream all 3,072 images for features (long)")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--skip-grounding", action="store_true")
    ap.add_argument("--check-provenance", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("ZELOS_LLM_MODE", "replay")
    feats = ROOT / "cache" / "features_all.pkl"
    if a.extract or (not feats.exists() and not a.skip_vision):
        run("scripts/extract_features.py")
    if not a.skip_vision and feats.exists():
        run("scripts/eval_vision_v2.py")
    if (RES / "calibration.json").exists():
        run("scripts/build_rule_citations.py")
        if (ROOT / "eval" / "vlm_cache").exists() and (ROOT / "data" / "raw" / "zip_index.pkl").exists():
            run("scripts/eval_vlm_baseline.py", "--phase", "score")  # needs the images on disk; otherwise reuse committed results
    if not a.skip_retrieval and (ROOT / "scripts/eval_retrieval.py").exists():
        run("scripts/eval_retrieval.py", "--offline")
    if not a.skip_grounding and (ROOT / "scripts/eval_grounding.py").exists():
        run("scripts/eval_grounding.py", "--replay")
    if (ROOT / "scripts/eval_llm_sweep.py").exists():
        run("scripts/eval_llm_sweep.py", "--replay")
    r = {n: load(n) for n in ("e1", "e2", "e4", "e5", "e9", "rule_citations", "vlm_baseline", "e5_sweep")}
    prov = provenance()
    fx = findings(r)
    write_json(RES / "results.json", {**r, "findings": fx, "provenance": prov})
    (RES / "results.md").write_text(md(r, prov, fx))
    print("\n".join(fx))
    if a.check_provenance:
        sys.exit(check_provenance())


if __name__ == "__main__":
    main()

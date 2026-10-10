#!/usr/bin/env python
"""Build the Zelos final-round defense kit from results/*.json.

Outputs (relative to --out-root, default the repository root):
    docs/defense/index.html        15-slide self-contained deck (inline CSS/JS, thumbnails as data URIs)
    docs/DEFENSE_QA.md             25 hard questions with sourced answers
    docs/DEFENSE_TALK_TRACK.md     12-minute and 3-minute scripts, 30-second pitch
    docs/V1_1_CHANGES.md           v1.0 -> v1.1 changelog with corrections

Templates live in docs/defense/templates/. They contain `{{key}}` placeholders only: every figure is
produced here from the results files through the formatting helpers below (pct, f3, ci, ...), so nothing
in a generated file is typed by hand. A placeholder with no value is an error, never an empty string.

v1.0 numbers (the submitted build) are read from `git show v1.0-submission:results/results.json`; when the
tag is not available they come from docs/defense/templates/v1_0_snapshot.json, which this script writes
from the tag.

Run: python scripts/build_defense_kit.py [--out-root DIR] [--results DIR]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "docs" / "defense" / "templates"
V1_TAG = "v1.0-submission"
V1_SNAPSHOT = TPL / "v1_0_snapshot.json"

OUTPUTS = {
    "deck.html.tmpl": "docs/defense/index.html",
    "defense_qa.md.tmpl": "docs/DEFENSE_QA.md",
    "talk_track.md.tmpl": "docs/DEFENSE_TALK_TRACK.md",
    "v1_1_changes.md.tmpl": "docs/V1_1_CHANGES.md",
}

VAR = {"pooled": "pooled_dev (v1)", "between": "between_session (v1.1)", "batch": "batch_session"}
DET = {
    "qc_pooled": "zelos_qc[pooled_dev (v1)]",
    "qc_between": "zelos_qc[between_session (v1.1)]",
    "qc_batch": "zelos_qc[batch_session]",
    "rawmaxz": "zelos_qc_rawmaxz[pooled_dev (v1)]",
    "focus": "focus_only[batch_session]",
    "random": "random",
    "knn_phys": "knn_physics_k15",
    "knn_dino": "knn_dinov2_cls_k15",
    "knn_dino_c": "knn_dinov2_session_centred_k15",
    "head": "LABEL-TRAINED dinov2 linear head (baseline)",
}
KIND_NAME = {
    "gaussian_blur": "Gaussian blur", "defocus": "Defocus", "motion_blur": "Motion blur",
    "gaussian_noise": "Sensor noise", "illumination": "Uneven illumination", "contrast": "Low contrast",
    "overexpose": "Over-exposure", "underexpose": "Under-exposure", "debris": "Debris",
    "bubble": "Air bubble", "jpeg": "JPEG compression",
}
CAUSE_NAME = {
    "DEFOCUS_OR_BLUR": "Defocus or blur", "NOISE": "Sensor noise", "UNEVEN_ILLUMINATION": "Uneven illumination",
    "OVER_EXPOSED": "Over-exposure", "UNDER_EXPOSED": "Under-exposure", "LOW_CONTRAST": "Low contrast",
    "BUBBLE": "Air bubble", "DEBRIS": "Debris", "COMPRESSION": "Compression artefacts",
}
CAUSE_LOWER = {k: v[0].lower() + v[1:] if not v.startswith("JPEG") else v for k, v in CAUSE_NAME.items()}
MODEL_NAME = {
    "llama-3.1-8b-fast": "Llama 3.1 8B", "gemma-4-26b-a4b": "Gemma 4 26B", "gpt-oss-120b": "gpt-oss 120B",
    "qwen3.8-27b": "Qwen3.8 27B",
}
MISS_DETECTION_AT_SEV3 = 0.25  # a kind is listed as an honest miss when sev-3 detection is at or below this


# ----------------------------------------------------------------------------- formatting helpers
# Every figure that reaches a generated file goes through one of these helpers (or `sourced`, for text that
# is copied verbatim from a data file). Each call logs the number tokens it produced in EMITTED, and the
# test asserts that every figure in the output is in that set: a number typed into a template or into this
# script would not be.
EMITTED: set[str] = set()
NUM_TOKEN = re.compile(r"(?<![\w.])\d+\.\d+%?|(?<![\w.])\d{2,}%")


def _emit(text: str) -> str:
    EMITTED.update(NUM_TOKEN.findall(text))
    return text


def sourced(text: str) -> str:
    """Text copied verbatim from a results or site data file; its figures count as sourced."""
    return _emit(text)


def pct(x: float, d: int = 1) -> str:
    return _emit(f"{100 * x:.{d}f}%")


def dec(x: float, d: int) -> str:
    return _emit(f"{x:.{d}f}")


def f3(x: float) -> str:
    return dec(x, 3)


def f2(x: float) -> str:
    return dec(x, 2)


def f4(x: float) -> str:
    return dec(x, 4)


def ci(pair: list[float], pctfmt: bool = False) -> str:
    lo, hi = pair
    return f"[{pct(lo)}, {pct(hi)}]" if pctfmt else f"[{f3(lo)}, {f3(hi)}]"


def ratio(k: int, n: int) -> str:
    return f"{k}/{n}"


def num(x: float | int) -> str:
    return f"{int(round(x)):,}"


def usd(x: float) -> str:
    return f"${f4(x)}"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------- inputs
def load_results(res_dir: Path) -> dict[str, Any]:
    names = ["e1", "e2", "e4", "e5", "e5_sweep", "e9", "calibration", "rule_citations", "vlm_baseline", "results"]
    out = {n: load_json(res_dir / f"{n}.json") for n in names if (res_dir / f"{n}.json").exists()}
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"missing results files: {missing}")
    out["results_md"] = (res_dir / "results.md").read_text()
    return out


def numbers_in(text: str) -> set[str]:
    """Same token rule as evaluate.numbers_in (decimals and percentages)."""
    return set(re.findall(r"(?<![\w.])\d+\.\d+%?|(?<![\w.])\d{2,}%", text))


def _headline(r: dict[str, Any]) -> dict[str, float]:
    """The E4/E5 headline values, so unchanged results can be stated from data rather than assumed."""
    c = r["e5"]["conditions"]
    return {
        "unsup_ungated": c["llm_ungated"]["unsupported_claim_rate_in_output"]["rate"],
        "unsup_gated": c["llm_gated"]["unsupported_claim_rate_in_output"]["rate"],
        "cite_ungated": c["llm_ungated"]["citation_valid_rate"]["rate"],
        "cite_gated": c["llm_gated"]["citation_valid_rate"]["rate"],
        "unsafe_ungated": c["llm_ungated"]["adversarial"]["unsafe_output_rate"]["rate"],
        "unsafe_gated": c["llm_gated"]["adversarial"]["unsafe_output_rate"]["rate"],
        "caught": r["e5"]["gate_fault_injection"]["injected_fabrications_caught"]["rate"],
        "mrr": r["e4"]["modes"]["hybrid_rerank"]["paper_level"]["mrr"],
    }


def load_v1() -> dict[str, Any]:
    """Return the reduced v1.0 results dict (from the submission tag, else the committed snapshot)."""
    try:
        raw = subprocess.run(["git", "show", f"{V1_TAG}:results/results.json"], cwd=ROOT, capture_output=True,
                             text=True, check=True, timeout=30).stdout
        raw_md = subprocess.run(["git", "show", f"{V1_TAG}:results/results.md"], cwd=ROOT, capture_output=True,
                                text=True, check=True, timeout=30).stdout
        r = json.loads(raw)
        e1, e2 = r["e1"], r["e2"]
        snap = {
            "source": f"git tag {V1_TAG}: results/results.json",
            "e1": {"overall": e1["overall"], "clean_flag_rate": e1["clean_flag_rate"],
                   "batch_reference": e1["batch_reference"], "severities": e1["severities"], "n_images": e1["n_images"]},
            "e2": {"n_test": e2["n_test"], "n_dev": e2["n_dev"], "detectors": e2["detectors"],
                   "zelos_at_review_threshold": e2["zelos_at_review_threshold"]},
            "headline": _headline(r),
            "numbers": sorted(numbers_in(raw_md) | numbers_in(raw)),
        }
        return snap
    except (subprocess.SubprocessError, OSError, KeyError, ValueError):
        if V1_SNAPSHOT.exists():
            return load_json(V1_SNAPSHOT)
        raise SystemExit(f"cannot read tag {V1_TAG} and no snapshot at {V1_SNAPSHOT}") from None


# ----------------------------------------------------------------------------- value collection
def collect(R: dict[str, Any], v1: dict[str, Any]) -> dict[str, str]:  # noqa: C901 - one flat table of facts
    V: dict[str, str] = {}
    e1, e2, e4, e5, sw, e9 = R["e1"], R["e2"], R["e4"], R["e5"], R["e5_sweep"], R["e9"]
    cal, rc, vlm = R["calibration"], R["rule_citations"], R["vlm_baseline"]
    prov = R["results"]["provenance"]
    for name in MODEL_NAME.values():  # display names echo model ids that appear in the results files
        sourced(name)

    V["git_sha"] = prov["git_sha"][:7]
    V["date"] = prov["timestamp_utc"][:10]
    V["seed"] = str(prov["seed"])

    # --- dataset / design
    V["n_images"] = num(e2["n_images"])
    V["n_sessions"] = str(e2["n_sessions"])
    V["n_test"] = num(e2["n_test"])
    V["n_test_sessions"] = str(e2["n_test_sessions"])
    V["n_test_bad"] = num(e2["n_test_bad"])
    V["n_base"] = str(e1["n_base_images"])
    V["n_sig"] = str(e1["n_signature_images"])
    V["n_kinds"] = str(len(e1["kinds"]))
    V["n_sev_list"] = ", ".join(map(str, e1["severities"][:-1])) + " and " + str(e1["severities"][-1])
    labels = {p["expected_label"] for p in e1["variants"]["batch_session"]["per_kind"].values()}
    V["n_labels"] = str(len(labels))
    V["sev.low"], V["sev.mid"], V["sev.top"] = (str(x) for x in e1["severities"])
    V["prereg_z"] = str(int(v1["e2"]["zelos_at_review_threshold"]["threshold_z"]))
    V["n_rules"] = str(rc["n_rules"])
    V["target_ff"] = re.search(r"(\d+)%", cal["rule"]).group(0)  # type: ignore[union-attr]
    V["n_metrics"] = _count_metrics()
    V["v1.n_images"] = num(v1["e2"]["n_dev"] + v1["e2"]["n_test"])
    V["v1.n_test"] = str(v1["e2"]["n_test"])

    # --- E1 per variant
    for key, name in VAR.items():
        v = e1["variants"][name]
        V[f"e1.{key}.clean"] = pct(v["clean_flag_rate"])
        V[f"e1.{key}.det3"] = pct(v["detection_rate_sev3plus"])
        V[f"e1.{key}.det1"] = pct(v["detection_rate_sev1"])
        V[f"e1.{key}.auroc"] = f3(v["auroc_clean_vs_sev3plus"])
        V[f"e1.{key}.rules"] = pct(v["attribution_v1_rules"])
        V[f"e1.{key}.sig"] = pct(v["attribution_v2_signatures"])
        V[f"e1.{key}.top2"] = pct(v["attribution_v2_top2"])
    V["e1.chance"] = pct(e1["variants"]["batch_session"]["chance_attribution"])
    sw_b = e1["variants"]["batch_session"]["threshold_sweep"]
    lo_t = min(sw_b, key=float)
    V["e1.sweep_lo.t"] = dec(float(lo_t), 1)
    V["e1.sweep_lo.clean"] = pct(sw_b[lo_t]["clean_flag_rate"])
    V["e1.sweep_lo.det"] = pct(sw_b[lo_t]["sev3plus_detection"])
    V["e1.sweep_op.det"] = pct(sw_b["1.0"]["sev3plus_detection"])
    V["e1.sweep_op.clean"] = pct(sw_b["1.0"]["clean_flag_rate"])

    pk = e1["variants"]["batch_session"]["per_kind"]
    for k, p in pk.items():
        d = p["detection_by_severity"]
        V[f"kind.{k}.det1"] = pct(d["1"])
        V[f"kind.{k}.det3"] = pct(d["3"])
        V[f"kind.{k}.det5"] = pct(d["5"])
        V[f"kind.{k}.rules"] = pct(p["attribution_v1_rules"])
        V[f"kind.{k}.sig"] = pct(p["attribution_v2_signatures"])
        V[f"kind.{k}.top2"] = pct(p["attribution_v2_top2"])
    misses = [k for k, p in pk.items() if p["detection_by_severity"]["3"] <= MISS_DETECTION_AT_SEV3]
    misses.sort(key=lambda k: pk[k]["detection_by_severity"]["3"])
    V["n_miss"] = str(len(misses))
    vals = [f"{_lower(KIND_NAME[k])} {pct(pk[k]['detection_by_severity']['3'])}" for k in misses]
    V["miss_list_values"] = ", ".join(vals[:-1]) + " and " + vals[-1] if len(vals) > 1 else vals[0]
    V["miss_list"] = ", ".join(_lower(KIND_NAME[k]) for k in misses)
    V["miss_list_detail"] = "; ".join(f"{_lower(KIND_NAME[k])} {pct(pk[k]['detection_by_severity']['3'])} at severity {V['sev.mid']}"
                                      for k in misses)
    best_k = max(pk, key=lambda k: pk[k]["detection_by_severity"]["3"])
    V["best_kinds"] = ", ".join(KIND_NAME[k].lower() for k, p in pk.items()
                                if p["detection_by_severity"]["3"] >= pk[best_k]["detection_by_severity"]["3"] - 0.05)

    # --- head-to-head
    h = e1["head_to_head"]["label_trained_dinov2_linear_head"]
    V["h2h.clean"] = pct(h["clean_flag_rate"])
    V["h2h.det3"] = pct(h["detection_rate_sev3plus"])
    V["h2h.auroc"] = f3(h["auroc_clean_vs_sev3plus"])
    V["h2h.path_dp"] = f3(h["mean_abs_prob_diff_between_paths"])
    V["h2h.path_clean"] = pct(h["clean_flag_rate_extraction_path"])
    hk = h["detection_by_kind_sev3plus"]
    for k, x in hk.items():
        V[f"h2h.kind.{k}"] = pct(x)
    best_h = max(hk, key=hk.get)  # type: ignore[arg-type]
    V["h2h.best_kind"] = _lower(KIND_NAME[best_h])
    V["h2h.best_kind_det"] = pct(hk[best_h])
    worst_h = sorted(hk, key=hk.get)[:3]  # type: ignore[arg-type]
    names_w = [f"{_lower(KIND_NAME[k])} ({pct(hk[k])})" for k in worst_h]
    V["h2h.worst_kinds"] = ", ".join(names_w[:-1]) + " and " + names_w[-1]

    # --- E2 detectors
    for key, name in DET.items():
        d = e2["detectors"][name]
        V[f"e2.{key}.auroc"] = f3(d["auroc"])
        V[f"e2.{key}.ci"] = ci(d["auroc_ci95"])
        V[f"e2.{key}.within"] = f3(d["within_session"]["weighted"])
        V[f"e2.{key}.within_mean"] = f3(d["within_session"]["mean"])
        V[f"e2.{key}.frac_above"] = pct(d["within_session"]["frac_sessions_above_0.5"])
        V[f"e2.{key}.lo"] = f3(d["auroc_ci95"][0])
        V[f"e2.{key}.hi"] = f3(d["auroc_ci95"][1])
    V["ci_level"] = sourced(re.search(r"ci(\d+)$", "auroc_ci95").group(1) + "%")  # type: ignore[union-attr]
    V["knn_k"] = re.search(r"_k(\d+)$", DET["knn_dino"]).group(1)  # type: ignore[union-attr]
    V["e2.within_n_sessions"] = str(e2["detectors"][DET["qc_batch"]]["within_session"]["n_sessions"])
    for key, name in VAR.items():
        op = e2["operating_points"][name]
        V[f"op.{key}.ff"] = pct(op["false_flag_rate_test_good"])
        V[f"op.{key}.recall"] = pct(op["recall_expert_bad"])
        V[f"op.{key}.z3"] = pct(op["preregistered_z3_false_flag"])
        c = cal["variants"][name]
        V[f"cal.{key}.loso"] = pct(c["loso_mean_false_flag_rate"])
        V[f"cal.{key}.p50"] = pct(c["loso_session_rates_p50_p80"][0])
        V[f"cal.{key}.p80"] = pct(c["loso_session_rates_p50_p80"][1])

    # --- E4 retrieval
    hr = e4["modes"]["hybrid_rerank"]
    V["e4.mrr"] = f3(hr["paper_level"]["mrr"])
    V["e4.chunk_hit1"] = f3(hr["chunk_level"]["hit@1"])
    V["e4.paper_hit1"] = f3(hr["paper_level"]["hit@1"])
    V["e4.n_queries"] = str(hr["paper_level"]["n"])
    ab = e4["abstention"]["hybrid_rerank"]
    V["e4.abstain_k"] = ratio(round(ab["accuracy_on_unanswerable"] * ab["n_unanswerable"]), ab["n_unanswerable"])
    V["e4.n_unanswerable"] = str(ab["n_unanswerable"])
    V["e4.false_abstain"] = f3(ab["false_abstention_rate_on_answerable"])
    V["e4.n_papers"] = str(e4["corpus"]["n_papers"])
    V["e4.n_chunks"] = num(e4["corpus"]["n_chunks"])
    V["e4.bm25_mrr"] = f3(e4["modes"]["bm25"]["paper_level"]["mrr"])
    V["e4.dense_mrr"] = f3(e4["modes"]["dense"]["paper_level"]["mrr"])
    V["e4.hybrid_mrr"] = f3(e4["modes"]["hybrid"]["paper_level"]["mrr"])

    # --- E5 grounding
    c = e5["conditions"]
    V["e5.n_cases"] = str(c["llm_gated"]["n_cases"])
    V["e5.n_explain"] = str(e5["n"]["explain"])
    V["e5.n_ask"] = str(e5["n"]["ask"])
    V["e5.n_adv"] = str(e5["n"]["adversarial"])
    V["e5.n_unans"] = str(e5["n"]["unanswerable"])
    u0, u1 = c["llm_ungated"], c["llm_gated"]
    V["e5.unsup_un"] = f3(u0["unsupported_claim_rate_in_output"]["rate"])
    V["e5.unsup_un_k"] = ratio(u0["unsupported_claim_rate_in_output"]["k"], u0["unsupported_claim_rate_in_output"]["n"])
    V["e5.unsup_un_ci"] = ci(u0["unsupported_claim_rate_in_output"]["ci95"])
    V["e5.unsup_g"] = f3(u1["unsupported_claim_rate_in_output"]["rate"])
    V["e5.cite_un"] = f3(u0["citation_valid_rate"]["rate"])
    V["e5.cite_g"] = f3(u1["citation_valid_rate"]["rate"])
    V["e5.unsafe_un"] = f3(u0["adversarial"]["unsafe_output_rate"]["rate"])
    V["e5.unsafe_un_k"] = ratio(u0["adversarial"]["unsafe_output_rate"]["k"], u0["adversarial"]["unsafe_output_rate"]["n"])
    V["e5.unsafe_g"] = f3(u1["adversarial"]["unsafe_output_rate"]["rate"])
    V["e5.abst_rec_un"] = f3(u0["abstention"]["recall"]["rate"])
    V["e5.abst_rec_g"] = f3(u1["abstention"]["recall"]["rate"])
    V["e5.numhal"] = f3(u0["numeric_hallucination_rate"]["rate"])
    V["e5.tokens_g"] = num(u1["tokens_per_answer"])
    V["e5.calls_g"] = f2(u1["llm_calls_per_answer"])
    V["e5.latency_g"] = num(u1["latency_ms_p50"])
    g = e5["gate_fault_injection"]
    V["gate.caught"] = ratio(g["injected_fabrications_caught"]["k"], g["injected_fabrications_caught"]["n"])
    V["gate.caught_n"] = str(g["injected_fabrications_caught"]["n"])
    V["gate.false_block"] = ratio(g["false_block_rate"]["k"], g["false_block_rate"]["n"])
    V["gate.false_block_n"] = str(g["false_block_rate"]["n"])
    V["gate.per_kind_n"] = str(next(iter(g["by_kind"].values()))["n"])
    V["gate.n_kinds"] = str(len(g["by_kind"]))
    V["gate.caught_ci"] = ci(g["injected_fabrications_caught"]["ci95"])
    V["e5.verdict_explain_t"] = f2(e5["verdict"]["explain_qc"]["template_supported_per_case"])
    V["e5.verdict_explain_l"] = f2(e5["verdict"]["explain_qc"]["llm_gated_supported_per_case"])
    V["e5.verdict_ask_t"] = f2(e5["verdict"]["ask_literature"]["template_supported_per_case"])
    V["e5.verdict_ask_l"] = f2(e5["verdict"]["ask_literature"]["llm_gated_supported_per_case"])
    V["e5.theta_lex"] = f2(e5["provenance"]["thresholds"]["theta_lex"])

    # --- LLM sweep
    best = sw["verdict"][0]["best"]
    V["sweep.best_name"] = MODEL_NAME.get(best, best)
    V["sweep.n"] = str(sw["n"]["explain_qc"])
    rows = []
    for mid, m in sw["models"].items():
        if m["status"] != "ok":
            continue
        gr = m["groups"]["explain_qc"]
        pre = mid.replace(".", "_").replace("-", "_")
        d_best = next((c for c in sw["verdict"][0]["candidates"] if c["id"] == mid), None)
        V[f"sweep.{pre}.before"] = f3(gr["draft"]["unsupported_claim_rate"]["rate"])
        V[f"sweep.{pre}.after"] = f3(gr["gated"]["unsupported_claim_rate"]["rate"])
        V[f"sweep.{pre}.per_case"] = f2(gr["gated"]["supported_claims_per_case"])
        V[f"sweep.{pre}.cost"] = usd(gr["cost"]["usd_per_100_answers"])
        V[f"sweep.{pre}.lat"] = num(gr["latency_ms"]["p50"])
        V[f"sweep.{pre}.diff"] = (f"{f2(d_best['diff_vs_best']['mean'])} [{f2(d_best['diff_vs_best']['ci95'][0])}, "
                                  f"{f2(d_best['diff_vs_best']['ci95'][1])}]"
                                  if d_best and d_best["id"] != best else "reference")
        rows.append((mid, pre))
    V["sweep.best.per_case"] = V[f"sweep.{best.replace('.', '_').replace('-', '_')}.per_case"]
    V["sweep.best.cost"] = V[f"sweep.{best.replace('.', '_').replace('-', '_')}.cost"]
    others = [r for r in rows if r[0] != best]
    costs = {mid: sw["models"][mid]["groups"]["explain_qc"]["cost"]["usd_per_100_answers"] for mid, _ in rows}
    top = max(costs, key=costs.get)  # type: ignore[arg-type]
    V["sweep.dearest_name"] = MODEL_NAME.get(top, top)
    V["sweep.dearest_cost"] = usd(costs[top])
    V["sweep.n_models_ok"] = str(len(rows))
    V["sweep.n_others"] = str(len(others))
    cohere = sw["models"].get("command-a-03-2025")
    V["agent.max_calls"] = str(sw["provenance"]["agent"]["max_llm_calls"])
    V["sweep.cohere_status"] = cohere["status"] if cohere else "not run"

    # --- rule citations
    V["rc.n_supported"] = str(rc["n_supported"])
    V["rc.n_unsupported"] = str(rc["n_rules"] - rc["n_supported"])
    V["rc.theta"] = f2(rc["theta_lex"])
    V["rc.theta_default"] = f2(e5["provenance"]["thresholds"]["theta_lex"])
    sup = [k for k, r in rc["rules"].items() if r["status"] == "supported"]
    uns = [k for k, r in rc["rules"].items() if r["status"] != "supported"]
    V["rc.supported_list"] = ", ".join(CAUSE_LOWER[k] for k in sup)
    V["rc.unsupported_list"] = ", ".join(CAUSE_LOWER[k] for k in uns)
    between = [k for k in uns if rc["theta_lex"] > rc["rules"][k]["support"] >= e5["provenance"]["thresholds"]["theta_lex"]]
    V["rc.between_list"] = ", ".join(f"{CAUSE_LOWER[k]} ({f3(rc['rules'][k]['support'])})" for k in between)
    V["rc.n_between"] = str(len(between))
    for k, r in rc["rules"].items():
        V[f"rc.{k}.support"] = f3(r["support"])
    V["rc.disclosure"] = sourced(rc["disclosure"])

    # --- VLM baseline
    V["vlm.n"] = re.match(r"(\d+)", vlm["design"]["A"]).group(1)  # type: ignore[union-attr]
    V["vlm.n_cases"] = str(vlm["n_cases"])
    V["vlm.prompt"] = sourced(vlm["provenance"]["prompt"])
    z = vlm["zelos_stored_reference_calibrated"]
    V["vlm.zelos.ff_good"] = pct(z["false_flag_on_expert_good"])
    V["vlm.zelos.flag_bad"] = pct(z["flag_rate_on_expert_bad"])
    V["vlm.zelos.ff_clean"] = pct(z["false_flag_on_clean"])
    V["vlm.zelos.det5"] = pct(z["detected_sev5_faults"])
    parsed = {m: d for m, d in vlm["models"].items() if d["false_flag_on_clean"] is not None}
    for m, d in vlm["models"].items():
        short = m.split("/")[-1]
        key = re.sub(r"[^a-z0-9]+", "_", short.lower()).strip("_")
        V[f"vlm.{key}.short"] = short
        if d["false_flag_on_clean"] is None:
            V[f"vlm.{key}.unparsed"] = str(d["unparsed"])
            continue
        V[f"vlm.{key}.ff_good"] = pct(d["false_flag_on_expert_good"])
        V[f"vlm.{key}.flag_bad"] = pct(d["flag_rate_on_expert_bad"])
        V[f"vlm.{key}.ff_clean"] = pct(d["false_flag_on_clean"])
        V[f"vlm.{key}.det5"] = pct(d["detected_sev5_faults"])
    ffc = [d["false_flag_on_clean"] for d in parsed.values()]
    V["vlm.ff_clean_min"] = pct(min(ffc))
    V["vlm.ff_clean_max"] = pct(max(ffc))
    ffg = [d["false_flag_on_expert_good"] for d in parsed.values()]
    V["vlm.ff_good_min"] = pct(min(ffg))
    V["vlm.ff_good_max"] = pct(max(ffg))
    det5 = [d["detected_sev5_faults"] for d in parsed.values()]
    V["vlm.det5_min"] = pct(min(det5))
    V["vlm.det5_max"] = pct(max(det5))
    V["vlm.n_parsed"] = str(len(parsed))
    unparsed = [d for d in vlm["models"].values() if d["false_flag_on_clean"] is None]
    V["vlm.n_unparsed_models"] = str(len(unparsed))
    V["vlm.unparsed_cases"] = str(unparsed[0]["unparsed"]) if unparsed else "0"

    # --- efficiency
    V["e9.inspect_p50"] = num(e9["inspect_ms_p50"])
    V["e9.inspect_p95"] = num(e9["inspect_ms_p95"])
    V["e9.metrics_p50"] = num(e9["metrics_ms_p50"])

    # --- API limits parsed from source, never retyped
    api = (ROOT / "src" / "zelos" / "api" / "main.py").read_text()
    mb = re.search(r"^MAX_BYTES\s*=\s*(\d+)\s*\*\s*1024\s*\*\s*1024", api, re.M)
    mp = re.search(r"^MAX_PIXELS\s*=\s*([\d_]+)", api, re.M)
    rate = re.search(r'"ZELOS_RATE_PER_MIN",\s*"(\d+)"', api)
    V["api.max_mb"] = mb.group(1) if mb else "the configured"
    V["api.max_mp"] = str(int(mp.group(1).replace("_", "")) // 1_000_000) if mp else "the configured"
    V["api.rate"] = rate.group(1) if rate else "the configured"

    # --- v1.0 numbers (the submission)
    o = v1["e1"]["overall"]
    V["v1.det3"] = pct(o["detection_rate_sev3plus"])
    V["v1.attr"] = pct(o["attribution_acc_detected"])
    V["v1.top2"] = pct(o["attribution_top2_detected"])
    V["v1.auroc_e1"] = f3(o["auroc_clean_vs_sev3plus"])
    V["v1.clean_flag"] = pct(v1["e1"]["clean_flag_rate"])
    V["v1.batch_clean_flag"] = pct(v1["e1"]["batch_reference"]["clean_flag_rate"])
    V["v1.batch_auroc"] = f3(v1["e1"]["batch_reference"]["auroc_clean_vs_sev3plus"])
    V["v1.batch_attr"] = pct(v1["e1"]["batch_reference"]["attribution_acc_detected"])
    V["v1.n_sev"] = str(len(v1["e1"]["severities"]))
    d1 = v1["e2"]["detectors"]
    V["v1.qc_auroc"] = f3(d1["zelos_qc_score"]["auroc"])
    V["v1.qc_ci"] = ci(d1["zelos_qc_score"]["auroc_ci95"])
    V["v1.knn_auroc"] = f3(d1["knn_handcrafted_k15"]["auroc"])
    V["v1.knn_ci"] = ci(d1["knn_handcrafted_k15"]["auroc_ci95"])
    V["v1.flag_rate_z3"] = pct(v1["e2"]["zelos_at_review_threshold"]["flag_rate"])

    h1 = v1["headline"]
    h = _headline({"e5": e5, "e4": e4})
    same = all(abs(h1[k] - h[k]) < 5e-4 for k in h1)
    V["e45_status"] = "unchanged" if same else "changed, see results/results.md"
    V["e9.n"] = str(e9["n"])
    tm = re.search(r"^TRIAGE_MIN, TRIAGE_MAX = (\d+), (\d+)", api, re.M)
    V["api.triage_min"], V["api.triage_max"] = (tm.group(1), tm.group(2)) if tm else ("the configured", "the configured")

    # --- findings text straight from results.json
    V["findings.head_to_head"] = sourced(next((f for f in R["results"]["findings"] if f.startswith("Head-to-head")), ""))
    return V


def _lower(name: str) -> str:
    return name if name.startswith("JPEG") else name.lower()


def _count_metrics() -> str:
    src = (ROOT / "src" / "zelos" / "vision" / "metrics.py").read_text()
    m = re.search(r"BAD_DIRECTION[^=]*=\s*\{(.*?)\n\}", src, re.S)
    if not m:
        return "the"
    return str(len(re.findall(r'^\s*"[a-z0-9_]+"\s*:', m.group(1), re.M)))


# ----------------------------------------------------------------------------- fragments
def esc(s: str) -> str:
    return html.escape(s, quote=True)


def thumb_uri(path: Path, max_w: int = 720, quality: int = 78, max_bytes: int = 600_000) -> str:
    from PIL import Image  # local import: only the deck needs it

    im = Image.open(path).convert("L")
    w = max_w
    while True:
        r = im.resize((w, round(im.height * w / im.width)))
        buf = io.BytesIO()
        r.save(buf, "JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_bytes or w <= 240:
            break
        w = int(w * 0.8)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pick_samples(site: dict[str, Any], e1: dict[str, Any]) -> tuple[dict, dict]:
    label_of = {k: p["expected_label"] for k, p in e1["variants"]["batch_session"]["per_kind"].items()}
    fail = None
    for s in site["samples"]:
        var = s.get("variant", "")
        if s["card"]["level"] == "FAIL" and var.startswith("corrupted:"):
            kind = var.split(":")[1]
            if s["card"]["primary_cause"] == label_of.get(kind):
                fail = s
                break
    if fail is None:
        fail = next(s for s in site["samples"] if s["card"]["level"] == "FAIL")
    ok = next((s for s in site["samples"] if s["card"]["level"] == "PASS" and s.get("variant") == "original"
               and s.get("expert_label") == "good"), None)
    if ok is None:
        ok = next(s for s in site["samples"] if s["card"]["level"] == "PASS")
    return fail, ok


def _claim_text(c: str) -> str:
    c = re.sub(r"\s*\[[^\]]+\]", "", c)
    return sourced(c.strip())


def sample_card(s: dict, rc: dict, thumbs_dir: Path, caption: str) -> str:
    card, qc = s["card"], s["qc"]
    level = card["level"]
    cls = {"PASS": "pass", "FAIL": "fail"}.get(level, "review")
    uri = thumb_uri(thumbs_dir / Path(s["thumb"]).name)
    cause = card.get("primary_cause")
    rows = [("Route", qc.get("route", "")), ("Cause", CAUSE_NAME.get(cause, "none found") if cause else "none found")]
    if card.get("action"):
        rows.append(("Action", sourced(card["action"])))
    basis = ""
    if cause and cause in rc["rules"]:
        r = rc["rules"][cause]
        if r["status"] == "supported":
            basis = f"Rule basis: supported by {r['chunk_id']}"
        else:
            basis = f"Rule basis: not in corpus (best sentence {f3(r['support'])}, threshold {f2(rc['theta_lex'])})"
    ev = next((e for e in card.get("evidence", []) if e.get("cited")), None)
    ev_html = ""
    if ev:
        ev_html = (f'<div class="ev"><span class="k">Cited</span> {esc(sourced(ev["title"])[:66] + ("..." if len(ev["title"]) > 66 else ""))}'
                   f'<span class="id">{esc(ev["chunk_id"])}</span></div>')
    led = s["ledger"]
    claims = "".join(f'<li><span class="tick">&#10003;</span>{esc(_claim_text(c["text"])[:72])}</li>'
                     for c in led["claims"][:2])
    facts = "".join(f'<div class="row"><span class="k">{esc(k)}</span><span>{esc(v)}</span></div>' for k, v in rows)
    return f"""<figure class="cert {cls}">
  <img src="{uri}" alt="{esc(caption)}">
  <figcaption>
    <div class="head"><span class="chip {cls}">{esc(level)}</span><span class="score">QC score {f3(card['qc_score'])}</span></div>
    <div class="cap">{esc(caption)}</div>
    {facts}
    {f'<div class="basis">{esc(basis)}</div>' if basis else ''}
    {ev_html}
    <div class="ledger"><span class="k">Ledger</span> {led['n_supported']} supported, {led['n_unsupported']} blocked
      <ul>{claims}</ul></div>
  </figcaption>
</figure>"""


def sample_caption(s: dict) -> str:
    var = s.get("variant", "original")
    if var.startswith("corrupted:"):
        _, kind, sev = var.split(":")
        return f"{s['cell_line']} image {s['id']}: injected {KIND_NAME.get(kind, kind).lower()}, severity {sev}"
    return f"{s['cell_line']} image {s['id']}: unmodified real image, expert label {s['expert_label']}"


def kind_chart(e1: dict) -> str:
    pk = e1["variants"]["batch_session"]["per_kind"]
    order = sorted(pk, key=lambda k: -pk[k]["detection_by_severity"]["3"])
    W, left, right, rowh, top = 1000, 330, 135, 60, 6
    H = top + rowh * len(order) + 10
    bw = W - left - right
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Detection at severity 3 and cause naming per injected fault kind">']
    for i, k in enumerate(order):
        y = top + i * rowh
        det = pk[k]["detection_by_severity"]["3"]
        sig = pk[k]["attribution_v2_signatures"]
        miss = det <= MISS_DETECTION_AT_SEV3
        cls = "miss" if miss else "det"
        out.append(f'<text x="{left - 16}" y="{y + 24}" class="lab{" missl" if miss else ""}" text-anchor="end">{esc(KIND_NAME[k])}</text>')
        out.append(f'<rect x="{left}" y="{y + 6}" width="{bw}" height="20" class="track"/>')
        out.append(f'<rect x="{left}" y="{y + 6}" width="{max(bw * det, 2):.1f}" height="20" class="{cls}"/>')
        out.append(f'<text x="{left + bw + 12}" y="{y + 23}" class="val{" missl" if miss else ""}">{pct(det)}</text>')
        out.append(f'<rect x="{left}" y="{y + 30}" width="{bw}" height="10" class="track"/>')
        out.append(f'<rect x="{left}" y="{y + 30}" width="{max(bw * sig, 2):.1f}" height="10" class="cause"/>')
        out.append(f'<text x="{left + bw + 12}" y="{y + 42}" class="val small">{pct(sig)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def shift_chart(V: dict[str, str], cal: dict, e2: dict) -> str:
    """Grouped bars per reference: leave-one-session-out mean false-flag rate vs held-out test false-flag rate."""
    W, H, left = 1000, 520, 20
    ymax = 0.25
    base = H - 90
    scale = (base - 50) / ymax
    tgt = int(V["target_ff"].rstrip("%")) / 100
    ty = base - tgt * scale
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="False-flag rate per reference mode">']
    out.append(f'<line x1="{left}" x2="{W - 10}" y1="{ty:.1f}" y2="{ty:.1f}" class="target"/>')
    out.append(f'<line x1="{left}" x2="{W - 10}" y1="{base}" y2="{base}" class="axis"/>')
    names = {"pooled": ("Pooled dev", "reference (v1)"), "between": ("Between-session", "single image"),
             "batch": ("Session-relative", "batch reference")}
    group_w = (W - left - 10) / 3
    bw = 96
    for gi, (key, name) in enumerate(VAR.items()):
        loso = cal["variants"][name]["loso_mean_false_flag_rate"]
        test = e2["operating_points"][name]["false_flag_rate_test_good"]
        cls = "good" if loso <= tgt else "bad"
        gx = left + gi * group_w + (group_w - 2 * bw - 14) / 2
        for j, (val, c2, lab) in enumerate(((loso, cls, V[f"cal.{key}.loso"]), (test, "test", V[f"op.{key}.ff"]))):
            hh = val * scale
            xx = gx + j * (bw + 14)
            out.append(f'<rect x="{xx:.1f}" y="{base - hh:.1f}" width="{bw}" height="{hh:.1f}" class="{c2}"/>')
            out.append(f'<text x="{xx + bw / 2:.1f}" y="{base - hh - 12:.1f}" class="val" text-anchor="middle">{lab}</text>')
        cx = gx + bw + 7
        out.append(f'<text x="{cx:.1f}" y="{base + 34}" class="lab" text-anchor="middle">{esc(names[key][0])}</text>')
        out.append(f'<text x="{cx:.1f}" y="{base + 64}" class="lab" text-anchor="middle" style="fill:var(--mut)">{esc(names[key][1])}</text>')
    out.append("</svg>")
    return "\n".join(out)


def table_html(head: list[str], rows: list[list[str]], cls: str = "", hl: set[int] | None = None) -> str:
    hl = hl or set()
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = ""
    for i, r in enumerate(rows):
        c = ' class="hl"' if i in hl else ""
        body += f"<tr{c}>" + "".join(f"<td>{esc(x)}</td>" for x in r) + "</tr>"
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def md_table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def fragments(R: dict[str, Any], V: dict[str, str], site: dict[str, Any] | None, thumbs: Path) -> dict[str, str]:
    F: dict[str, str] = {}
    e1, e2, rc, vlm, sw, cal = R["e1"], R["e2"], R["rule_citations"], R["vlm_baseline"], R["e5_sweep"], R["calibration"]

    if site:
        fail, ok = pick_samples(site, e1)
        F["html.card_fail"] = sample_card(fail, rc, thumbs, sample_caption(fail))
        F["html.card_pass"] = sample_card(ok, rc, thumbs, sample_caption(ok))
        V["card.fail_id"] = fail["id"]
        V["card.pass_id"] = ok["id"]
        V["card.fail_cause"] = CAUSE_NAME.get(fail["card"]["primary_cause"] or "", "none")
        V["card.fail_route"] = fail["qc"].get("route", "")
        V["card.fail_score"] = f3(fail["card"]["qc_score"])
        V["card.pass_score"] = f3(ok["card"]["qc_score"])
        V["card.fail_basis_support"] = f3(rc["rules"][fail["card"]["primary_cause"]]["support"]) \
            if fail["card"]["primary_cause"] in rc["rules"] else "n/a"
    else:
        raise SystemExit("site.json not found; cannot render sample certificates")

    F["html.kind_chart"] = kind_chart(e1)
    F["html.shift_chart"] = shift_chart(V, cal, e2)

    # head-to-head table
    h2h = [
        ["Zelos, session-relative", V["e1.batch.clean"], V["e1.batch.det3"], V["e1.batch.auroc"], f"yes ({V['e1.batch.sig']})"],
        ["Zelos, single image", V["e1.between.clean"], V["e1.between.det3"], V["e1.between.auroc"], f"yes ({V['e1.between.sig']})"],
        ["Label-trained DINOv2 head", V["h2h.clean"], V["h2h.det3"], V["h2h.auroc"], "no"],
    ]
    head = ["System", "Clean flagged", "Faults found", "AUROC", "Names the cause"]
    F["html.table_h2h"] = table_html(head, h2h, "t sm", {0, 1})
    F["md.table_h2h"] = md_table(head, h2h)

    # vlm table
    rows = [["Zelos v1.1, no model trained", V["vlm.zelos.ff_good"], V["vlm.zelos.flag_bad"], V["vlm.zelos.ff_clean"], V["vlm.zelos.det5"]]]
    for m, d in vlm["models"].items():
        short = m.split("/")[-1]
        nice_name = sourced(re.sub(r"-(instruct|it)$", "", short))
        if d["false_flag_on_clean"] is None:
            rows.append([nice_name, "no parsable reply", "-", "-", "-"])
            continue
        key = re.sub(r"[^a-z0-9]+", "_", short.lower()).strip("_")
        rows.append([nice_name, V[f"vlm.{key}.ff_good"], V[f"vlm.{key}.flag_bad"], V[f"vlm.{key}.ff_clean"], V[f"vlm.{key}.det5"]])
    vh = ["System", "Flags expert-good", "Flags expert-bad", "Flags clean", "Finds severe fault"]
    F["html.table_vlm"] = table_html(vh, rows, "t sm", {0})
    F["md.table_vlm"] = md_table(vh, rows)

    # e2 table (full)
    order = ["qc_batch", "qc_between", "qc_pooled", "focus", "knn_phys", "random", "knn_dino_c", "knn_dino", "head"]
    nice = {
        "qc_batch": "Zelos QC score, session-relative", "qc_between": "Zelos QC score, single image",
        "qc_pooled": "Zelos QC score, pooled reference (v1)", "focus": "Focus only", "knn_phys": "kNN on physics features",
        "random": "Random", "knn_dino_c": "DINOv2 kNN, session-centred", "knn_dino": "DINOv2 kNN (culture state)",
        "head": "Label-trained DINOv2 head",
    }
    rows = [[nice[k], f"{V[f'e2.{k}.auroc']} {V[f'e2.{k}.ci']}", V[f"e2.{k}.within"]] for k in order]
    eh = ["Signal vs expert good/bad label", f"AUROC [{V['ci_level']} CI]", "Within-day AUROC"]
    F["html.table_e2"] = table_html(eh, rows, "t sm", {0, 7})
    F["md.table_e2"] = md_table(eh, rows)

    # reference table (shift)
    rows = []
    names = {"pooled": "Pooled dev reference (v1)", "between": "Between-session, single image", "batch": "Session-relative (batch)"}
    for k in VAR:
        rows.append([names[k], V[f"cal.{k}.loso"], V[f"cal.{k}.p80"], V[f"op.{k}.ff"], V[f"op.{k}.z3"]])
    sh = ["Reference", "Leave-one-session-out false flags", "80th-percentile session", "Test false flags (expert-good)", "Pre-registered z=3 rule"]
    F["html.table_shift"] = table_html(sh, rows, "t", {2})
    F["md.table_shift"] = md_table(sh, rows)

    # rules table
    rows, hl = [], set()
    for i, (k, r) in enumerate(rc["rules"].items()):
        ok = r["status"] == "supported"
        rows.append([CAUSE_NAME[k], "supported" if ok else "unsupported", f3(r["support"])])
        if not ok:
            hl.add(i)
    F["html.rules_bars"] = rules_bars(rc)
    F["md.table_rules"] = md_table(["Cause", "Status", "Best sentence support"], rows)

    # sweep table
    rows = []
    for mid, m in sw["models"].items():
        if m["status"] != "ok":
            continue
        pre = mid.replace(".", "_").replace("-", "_")
        rows.append([MODEL_NAME.get(mid, mid), V[f"sweep.{pre}.before"], V[f"sweep.{pre}.after"], V[f"sweep.{pre}.per_case"],
                     V[f"sweep.{pre}.cost"], V[f"sweep.{pre}.lat"]])
    swh = ["Model", "Unsupported before gate", "After gate", "Supported claims / case", "USD per 100 answers", "p50 latency ms"]
    F["html.table_sweep"] = table_html(["Model", "Before gate", "After gate", "Claims / case", "USD / 100"],
                                       [r[:5] for r in rows], "t", {0})
    F["md.table_sweep"] = md_table(swh, rows)

    # kind table (md)
    pk = e1["variants"]["batch_session"]["per_kind"]
    rows = [[KIND_NAME[k], f"{V[f'kind.{k}.det1']} / {V[f'kind.{k}.det3']} / {V[f'kind.{k}.det5']}", V[f"kind.{k}.sig"],
             V[f"h2h.kind.{k}"]] for k in pk]
    kh = ["Fault", "Detected at severity 1 / 3 / 5", "Cause named (signatures)", "DINOv2 head detects (sev 3+)"]
    F["md.table_kinds"] = md_table(kh, rows)
    return F


def rules_bars(rc: dict) -> str:
    W, left, rowh, bw = 1000, 380, 62, 480
    n = len(rc["rules"])
    H = 20 + n * rowh + 40
    th = rc["theta_lex"]
    xth = left + bw * th
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Literature support per cause-action rule">']
    for i, (k, r) in enumerate(rc["rules"].items()):
        y = 20 + i * rowh
        ok = r["status"] == "supported"
        out.append(f'<text x="{left - 16}" y="{y + 28}" class="lab" text-anchor="end">{esc(CAUSE_NAME[k])}</text>')
        out.append(f'<rect x="{left}" y="{y + 8}" width="{bw}" height="26" class="track"/>')
        out.append(f'<rect x="{left}" y="{y + 8}" width="{max(bw * r["support"], 2):.1f}" height="26" class="{"good" if ok else "bad"}"/>')
        out.append(f'<text x="{left + bw + 14}" y="{y + 28}" class="val">{f3(r["support"])}</text>')
    out.append(f'<line x1="{xth:.1f}" x2="{xth:.1f}" y1="10" y2="{H - 34}" class="target"/>')
    out.append(f'<text x="{xth:.1f}" y="{H - 8}" class="lab" text-anchor="middle">threshold {f2(th)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ----------------------------------------------------------------------------- rendering
PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_.\-]+)\}\}")


def render(template: str, values: dict[str, str], is_html: bool) -> str:
    missing: list[str] = []

    def sub(m: re.Match[str]) -> str:
        k = m.group(1)
        if k not in values:
            missing.append(k)
            return m.group(0)
        v = values[k]
        if is_html and not k.startswith("html."):
            return esc(v)
        return v

    out = PLACEHOLDER.sub(sub, template)
    if missing:
        raise KeyError("unresolved placeholders: " + ", ".join(sorted(set(missing))))
    return out


def build(out_root: Path, res_dir: Path, site_path: Path, thumbs: Path, write_snapshot: bool = False) -> dict[str, Path]:
    R = load_results(res_dir)
    v1 = load_v1()
    if write_snapshot and v1.get("source", "").startswith("git tag"):
        V1_SNAPSHOT.write_text(json.dumps(v1, indent=1, sort_keys=True) + "\n")
    V = collect(R, v1)
    site = load_json(site_path) if site_path.exists() else None
    F = fragments(R, V, site, thumbs)
    values = {**V, **F}
    written = {}
    for tname, rel in OUTPUTS.items():
        text = render((TPL / tname).read_text(), values, tname.startswith("deck"))
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        written[rel] = dest
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=ROOT)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--site", type=Path, default=ROOT / "web" / "public" / "data" / "site.json")
    ap.add_argument("--thumbs", type=Path, default=ROOT / "web" / "public" / "data" / "thumbs")
    a = ap.parse_args()
    written = build(a.out_root, a.results, a.site, a.thumbs, write_snapshot=a.out_root.resolve() == ROOT)
    for rel, p in written.items():
        print(f"wrote {rel} ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

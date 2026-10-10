"""E1 (controlled corruptions), E2 (expert-label agreement), E9 (latency) on the real OoC dataset.

Inputs : data/raw/manifest.csv + images (data/download.py), configs/*.yaml
Outputs: results/reference.json, results/metrics_real.csv, results/e1.json, results/e2.json,
         results/e9.json, results/qc_samples.json, results/figures/*.png
Run    : python scripts/eval_vision.py [--n-e1 96] [--image-embeddings]
Nothing is trained: reference = median/MAD of dev-split 'good' images; kNN baselines are non-parametric.
"""
from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from zelos.common.config import ROOT, load_config, provenance, secret, sha256_file, write_json
from zelos.vision.attribution import attribute
from zelos.vision.corrupt import KIND_TO_LABEL, KINDS, corrupt
from zelos.vision.metrics import compute_metrics, downscale, to_gray01
from zelos.vision.pipeline import inspect
from zelos.vision.scoring import fit_reference, score_image

CFG = load_config()
RES = ROOT / "results"
DATA = ROOT / CFG["data"]["root"]


def session_split(df: pd.DataFrame, dev_frac: float, seed: int) -> pd.Series:
    """Group-aware split: whole experiment sessions (image-id date prefix) go to dev or test."""
    rng = np.random.default_rng(seed)
    sessions = np.array(sorted(df["session"].astype(str).unique()))
    rng.shuffle(sessions)
    target, acc, dev = dev_frac * len(df), 0, set()
    counts = df["session"].astype(str).value_counts()
    for s in sessions:
        if acc >= target:
            break
        dev.add(s)
        acc += counts[s]
    return df["session"].astype(str).map(lambda s: "dev" if s in dev else "test")


def _metrics_job(path: str) -> dict:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    t = time.perf_counter()
    m = compute_metrics(img, CFG["vision"]["max_side"])
    m["_ms"] = (time.perf_counter() - t) * 1000
    return m


def _e1_job(args: tuple[str, int]) -> list[dict]:
    path, seed = args
    g, _ = to_gray01(cv2.imread(path, cv2.IMREAD_UNCHANGED))
    g = downscale(g, CFG["vision"]["max_side"])
    rng = np.random.default_rng(seed)
    rows = [{"path": path, "kind": "clean", "severity": 0, **compute_metrics(g)}]
    for k in KINDS:
        for s in CFG["corruptions"]["severities"]:
            rows.append({"path": path, "kind": k, "severity": s, **compute_metrics(corrupt(g, k, s, rng))})
    return rows


def boot_auc(y: np.ndarray, s: np.ndarray, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def knn_scores(Xd: np.ndarray, yd: np.ndarray, Xt: np.ndarray, k: int = 15) -> np.ndarray:
    """Non-parametric label transfer: fraction of 'bad' among the k nearest dev images (cosine)."""
    def norm(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    sim = norm(Xt) @ norm(Xd).T
    idx = np.argsort(-sim, axis=1)[:, :k]
    return yd[idx].mean(axis=1)


def cohere_image_embeddings(paths: list[str]) -> np.ndarray | None:
    """Frozen Cohere embed-v4.0 image embeddings (cached in results/image_embeddings.npz)."""
    cache = RES / "image_embeddings.npz"
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["ids"].tolist(), z["vecs"], strict=True))
    ids = [Path(p).stem for p in paths]
    todo = [p for p, i in zip(paths, ids, strict=True) if i not in have]
    key = secret("COHERE")
    if todo and key:
        import requests

        for b in range(0, len(todo), 8):
            batch = todo[b : b + 8]
            inputs = []
            for p in batch:
                im = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (512, 384), interpolation=cv2.INTER_AREA)
                b64 = base64.b64encode(cv2.imencode(".png", im)[1]).decode()
                inputs.append({"content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]})
            for _ in range(4):
                r = requests.post("https://api.cohere.com/v2/embed", timeout=120,
                                  headers={"Authorization": f"Bearer {key}"},
                                  json={"model": "embed-v4.0", "input_type": "image", "embedding_types": ["float"],
                                        "output_dimension": 1024, "inputs": inputs})
                if r.status_code == 200:
                    break
                time.sleep(6)
            if r.status_code != 200:
                print("cohere image embed failed:", r.status_code, r.text[:200])
                break
            for p, v in zip(batch, r.json()["embeddings"]["float"], strict=True):
                have[Path(p).stem] = np.asarray(v, np.float32)
        np.savez_compressed(cache, ids=np.array(list(have)), vecs=np.stack(list(have.values())).astype(np.float16))
    if not all(i in have for i in ids):
        return None
    return np.stack([have[i] for i in ids]).astype(np.float32)


def e1_block(T: pd.DataFrame, ref: dict) -> dict:
    """Score corrupted + clean rows against a reference; detection, attribution, confusion, sweep."""
    sc = [score_image(r.to_dict(), ref, None, CFG["qc"]["review_z"], CFG["qc"]["fail_z"]) for _, r in T.iterrows()]
    T["qc_score"] = [s["qc_score"] for s in sc]
    T["level"] = [s["level"] for s in sc]
    T["focus_only"] = [s["zbad"]["focus_lapvar_norm"] for s in sc]
    att = [attribute(r.to_dict(), s["zbad"]) for (_, r), s in zip(T.iterrows(), sc, strict=True)]
    T["primary"] = [(c or [{"label": "NONE"}])[0]["label"] for c in att]
    T["top2"] = [[x["label"] for x in c[:2]] for c in att]
    clean = T[T.kind == "clean"]
    sev3 = T[(T.kind != "clean") & (T.severity >= 3)]
    labels_all = sorted(set(KIND_TO_LABEL.values())) + ["NONE"]
    conf = {lab: {lab2: 0 for lab2 in labels_all} for lab in sorted(set(KIND_TO_LABEL.values()))}
    per = {}
    for k in KINDS:
        sub = T[T.kind == k]
        both = pd.concat([clean, sub[sub.severity >= 3]])
        yy = (both.kind != "clean").astype(int).to_numpy()
        det = sub[sub.level != "PASS"]
        exp = KIND_TO_LABEL[k]
        allk = pd.concat([clean, sub])
        per[k] = {"expected_label": exp,
                  "spearman_severity_vs_qc": float(spearmanr(allk.severity, allk.qc_score)[0]),
                  "auroc_clean_vs_sev3plus": float(roc_auc_score(yy, both.qc_score)),
                  "auroc_focus_only": float(roc_auc_score(yy, both.focus_only)),
                  "detection_rate_by_severity": {int(s): float((sub[sub.severity == s].level != "PASS").mean())
                                                 for s in CFG["corruptions"]["severities"]},
                  "attribution_acc_detected": float((det.primary == exp).mean()) if len(det) else None,
                  "attribution_top2_detected": float(np.mean([exp in t for t in det.top2])) if len(det) else None,
                  "n": int(len(sub))}
        for p in sub[sub.severity >= 3].primary:
            conf[exp][p] += 1
    det_all = T[(T.kind != "clean") & (T.level != "PASS")]
    exp_all = det_all.kind.map(KIND_TO_LABEL)
    cs = pd.concat([clean, sev3])
    ycs = (cs.kind != "clean").astype(int)
    overall = {"detection_rate_sev3plus": float((sev3.level != "PASS").mean()),
               "attribution_acc_detected": float((det_all.primary == exp_all).mean()),
               "attribution_top2_detected": float(np.mean([e in t for e, t in zip(exp_all, det_all.top2, strict=True)])),
               "auroc_clean_vs_sev3plus": float(roc_auc_score(ycs, cs.qc_score)),
               "auroc_clean_vs_sev3plus_ci95": boot_auc(ycs.to_numpy(), cs.qc_score.to_numpy(), CFG["bootstrap"]["n"], CFG["seed"]),
               "auroc_focus_only": float(roc_auc_score(ycs, cs.focus_only)),
               "attribution_random_baseline": 1.0 / len(set(KIND_TO_LABEL.values()))}
    sweep = {str(zt): {"clean_flag_rate": float((clean.qc_score > zt).mean()),
                       "sev3plus_detection_rate": float((sev3.qc_score > zt).mean())} for zt in CFG["qc"]["sweep_z"]}
    return {"overall": overall, "per_kind": per, "confusion": conf, "sweep": sweep,
            "clean_flag_rate": float((clean.level != "PASS").mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-e1", type=int, default=CFG["corruptions"]["n_images"])
    ap.add_argument("--image-embeddings", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    seed = CFG["seed"]
    nboot = CFG["bootstrap"]["n"]

    df = pd.read_csv(DATA / "manifest.csv")
    df["path"] = [str(DATA / p) for p in df["local_path"]]
    df["bad"] = (df["label"] == "bad").astype(int)
    df["split"] = session_split(df, CFG["split"]["dev"], seed)
    data_hash = sha256_file(DATA / "manifest.csv")

    mcache = ROOT / "cache" / f"metrics_{data_hash[:12]}.pkl"
    if mcache.exists():
        M = pd.read_pickle(mcache)
    else:
        with mp.Pool(a.workers) as pool:
            M = pd.DataFrame(pool.map(_metrics_job, df["path"].tolist(), chunksize=4))
        mcache.parent.mkdir(exist_ok=True)
        M.to_pickle(mcache)
    M = M.copy()
    lat = M.pop("_ms").to_numpy()
    real = pd.concat([df.reset_index(drop=True), M], axis=1)
    real.drop(columns=["path"]).to_csv(RES / "metrics_real.csv", index=False)

    dev_good = real[(real.split == "dev") & (real.bad == 0)]
    ref = fit_reference(dev_good, by="cell_line", min_group=CFG["qc"]["min_group_for_per_line_ref"])
    ref["fit_on"] = {"split": "dev", "label": "good", "n": int(len(dev_good))}
    write_json(RES / "reference.json", ref)

    # ---------------- E2: expert agreement on the session-grouped test split ----------------
    test = real[real.split == "test"].reset_index(drop=True)
    dev = real[real.split == "dev"].reset_index(drop=True)
    scored = [score_image(r[M.columns].to_dict(), ref, r.cell_line, CFG["qc"]["review_z"], CFG["qc"]["fail_z"])
              for _, r in test.iterrows()]
    y = test.bad.to_numpy()
    qc = np.array([s["qc_score"] for s in scored])
    sess_ref = {}
    for sname, sub in test.groupby("session"):
        sess_ref[sname] = fit_reference(sub if len(sub) >= 8 else test[test.cell_line == sub.cell_line.iloc[0]])
    qc_batch = np.array([score_image(r[M.columns].to_dict(), sess_ref[r.session])["qc_score"] for _, r in test.iterrows()])
    pred = np.array([s["level"] != "PASS" for s in scored]).astype(int)
    focus = np.array([s["zbad"]["focus_lapvar_norm"] for s in scored])
    rng = np.random.default_rng(seed)
    feats = list(M.columns)
    mu, sd = dev[feats].mean(), dev[feats].std().replace(0, 1)
    knn_hand = knn_scores(((dev[feats] - mu) / sd).to_numpy(), dev.bad.to_numpy(), ((test[feats] - mu) / sd).to_numpy())
    detectors = {
        "zelos_qc_score": qc,
        "zelos_qc_score_session_reference": qc_batch,
        "focus_only": focus,
        "random": rng.random(len(y)),
        "knn_handcrafted_k15": knn_hand,
    }
    if a.image_embeddings:
        E = cohere_image_embeddings(real["path"].tolist() if "path" in real else df["path"].tolist())
        if E is not None:
            Ed, Et = E[(real.split == "dev").to_numpy()], E[(real.split == "test").to_numpy()]
            detectors["knn_cohere_image_embed_k15"] = knn_scores(Ed, dev.bad.to_numpy(), Et)
    e2 = {"n_test": int(len(y)), "n_dev": int(len(dev)), "n_test_bad": int(y.sum()),
          "split": "session-grouped (image-id date prefix); no session in both splits",
          "label": "expert decision good/bad from the dataset", "detectors": {}}
    for name, s in detectors.items():
        e2["detectors"][name] = {"auroc": float(roc_auc_score(y, s)), "auroc_ci95": boot_auc(y, s, nboot, seed)}
    e2["zelos_at_review_threshold"] = {
        "threshold_z": CFG["qc"]["review_z"],
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "f1_bad": float(f1_score(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)), "flag_rate": float(pred.mean()),
        "majority_baseline_balanced_accuracy": 0.5,
    }
    per_line = {}
    for cl, sub in test.groupby("cell_line"):
        idx = sub.index.to_numpy()
        if len(set(y[idx])) == 2 and len(idx) >= 10:
            per_line[cl] = {"n": int(len(idx)), "auroc_zelos": float(roc_auc_score(y[idx], qc[idx])),
                            "auroc_knn_handcrafted": float(roc_auc_score(y[idx], knn_hand[idx]))}
    e2["per_cell_line"] = per_line
    # which single metric separates expert good/bad best (two-sided, descriptive)
    e2["per_metric_auroc"] = {f: float(max(roc_auc_score(y, test[f]), 1 - roc_auc_score(y, test[f]))) for f in feats}
    e2["provenance"] = provenance({"data_manifest_sha256": data_hash})
    write_json(RES / "e2.json", e2)

    # ---------------- E1: controlled corruptions with known ground truth ----------------
    good_test = test[test.bad == 0]
    good_dev = dev[dev.bad == 0]
    pick = good_test.sample(n=min(a.n_e1, len(good_test)), random_state=seed)
    refpick = good_dev.sample(n=min(a.n_e1, len(good_dev)), random_state=seed)
    paths = [str(DATA / p) for p in pick.local_path] + [str(DATA / p) for p in refpick.local_path]
    cache = ROOT / "cache" / f"e1_rows_{a.n_e1}.pkl"
    if cache.exists():
        R = pd.read_pickle(cache)
    else:
        with mp.Pool(a.workers) as pool:
            out = pool.map(_e1_job, [(p, seed + i) for i, p in enumerate(paths)], chunksize=1)
        R = pd.DataFrame([r for rows in out for r in rows])
        cache.parent.mkdir(exist_ok=True)
        R.to_pickle(cache)
    refset = set(str(DATA / p) for p in refpick.local_path)
    ref1 = fit_reference(R[(R.kind == "clean") & R.path.isin(refset)])
    T = R[~R.path.isin(refset)].reset_index(drop=True)
    e1 = {"n_images": int(T.path.nunique()), "n_reference_images": len(refset), "kinds": KINDS,
          "severities": CFG["corruptions"]["severities"],
          "reference_note": "pre-registered: median/MAD of clean dev-session images; batch_reference: median/MAD of the clean "
                            "images in the evaluated batch itself (unsupervised, no labels), added after the pre-registered "
                            "run revealed cross-session acquisition shift"}
    pre = e1_block(T.copy(), ref1)
    bat = e1_block(T.copy(), fit_reference(T[T.kind == "clean"]))
    e1.update({"overall": pre["overall"], "per_kind": pre["per_kind"], "confusion_sev3plus": pre["confusion"],
               "threshold_sweep": pre["sweep"], "clean_flag_rate": pre["clean_flag_rate"],
               "batch_reference": {**bat["overall"], "clean_flag_rate": bat["clean_flag_rate"]},
               "per_kind_batch": bat["per_kind"], "confusion_sev3plus_batch": bat["confusion"],
               "threshold_sweep_batch": bat["sweep"]})
    e1["provenance"] = provenance({"data_manifest_sha256": data_hash})
    write_json(RES / "e1.json", e1)

    # ---------------- E9 latency + samples for the agent / UI ----------------
    t_inspect = []
    samples = []
    rows = pd.concat([test[test.bad == 1].head(15), test[test.bad == 0].head(15)])
    for _, r in rows.iterrows():
        q = inspect(DATA / r.local_path, cell_line=r.cell_line, image_id=r.file_id)
        q["expert_label"] = r.label
        q["time_bin"] = r.time_bin
        q["local_path"] = r.local_path
        t_inspect.append(q["latency_ms"])
        samples.append(q)
    write_json(RES / "qc_samples.json", samples)
    write_json(RES / "e9.json", {"metrics_ms_p50": float(np.median(lat)), "metrics_ms_p95": float(np.percentile(lat, 95)),
                                 "inspect_ms_p50": float(np.median(t_inspect)), "inspect_ms_p95": float(np.percentile(t_inspect, 95)),
                                 "n": int(len(lat)), "image_shape": "1542x2056 uint8 (analysed at long side 1024)",
                                 "cpu_workers": a.workers, "provenance": provenance()})
    print(json.dumps({"e1_overall": e1["overall"], "e2": {k: v["auroc"] for k, v in e2["detectors"].items()},
                      "e2_review": e2["zelos_at_review_threshold"]}, indent=1))


if __name__ == "__main__":
    main()

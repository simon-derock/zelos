"""Zelos v1.1 vision evaluation on ALL 3,072 real OoC images (features streamed by extract_features.py).

Questions answered (all numbers computed here, nothing typed by hand):
  E2  How well do Zelos signals agree with the expert good/bad label on held-out SESSIONS (n ~ 2k)?
      - pre-registered pooled reference (v1), between-session-aware reference (v1.1), batch reference
      - culture-state signal: frozen DINOv2 kNN (training-free), plus a label-trained linear head as a baseline
      - within-session AUROC (removes the imaging-day shortcut), session-cluster bootstrap CIs
  E1  Injected faults with known ground truth on real test-session images:
      - detection at a calibrated operating point (<=10% false alarms on calibration sessions)
      - cause attribution: v1 rule table vs v1.1 signature centroids (fitted on dev-session injections only)
      - head-to-head: label-trained DINOv2 linear head (the approach of many entries) on the same faults
Run: python scripts/eval_vision_v2.py [--n-e1 48]
Splits: whole imaging sessions (image-id date prefix): dev 30% (half fit, half calibrate) / test 70%.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from zelos.common.config import load_config, provenance, write_json  # noqa: E402
from zelos.vision.attribution import attribute, attribute_v2, fit_signatures  # noqa: E402
from zelos.vision.corrupt import KIND_TO_LABEL, KINDS, corrupt  # noqa: E402
from zelos.vision.metrics import BAD_DIRECTION, compute_metrics, downscale, to_gray01  # noqa: E402
from zelos.vision.scoring import fit_reference, fit_reference_sessions, score_image  # noqa: E402

CFG = load_config()
SEED = CFG["seed"]
RES = ROOT / "results"
M = list(BAD_DIRECTION)
CLS = [f"cls_{i}" for i in range(384)]
MEAN = [f"mean_{i}" for i in range(384)]
SEV_TEST = [1, 3, 5]
SEV_SIG = [3, 5]


# ------------------------------------------------------------------ helpers
def auroc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def cluster_boot_auc(y, s, groups, n=500, seed=SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    y, s, g = np.asarray(y), np.asarray(s), np.asarray(groups)
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        ii = np.concatenate([idx[k] for k in pick])
        if len(set(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], s[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def within_session_auc(y, s, groups, min_n=10) -> dict:
    y, s, g = np.asarray(y), np.asarray(s), np.asarray(groups)
    aucs, w = [], []
    for k in np.unique(g):
        m = g == k
        if m.sum() >= min_n and len(set(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], s[m]))
            w.append(int((y[m] == 0).sum() * (y[m] == 1).sum()))
    if not aucs:
        return {"mean": None, "weighted": None, "n_sessions": 0}
    return {"mean": float(np.mean(aucs)), "weighted": float(np.average(aucs, weights=w)), "n_sessions": len(aucs),
            "frac_sessions_above_0.5": float(np.mean(np.array(aucs) > 0.5))}


def knn_score(Xd, yd, Xq, k=15) -> np.ndarray:
    def nz(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    sim = nz(Xq) @ nz(Xd).T
    idx = np.argsort(-sim, axis=1)[:, :k]
    return yd[idx].mean(axis=1)


def split_sessions(F: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    cnt = F.session.value_counts()
    ses = np.array(sorted(cnt.index))
    rng.shuffle(ses)
    n = len(F)
    dev, acc = [], 0
    for s in ses:
        if acc >= 0.30 * n:
            break
        dev.append(s)
        acc += cnt[s]
    rest = [s for s in ses if s not in set(dev)]
    half = len(dev) // 2
    return set(dev[:half]), set(dev[half:]), set(rest)


def score_rows(df: pd.DataFrame, refs_fn) -> tuple[np.ndarray, list[dict]]:
    sc, zb = [], []
    for _, r in df.iterrows():
        s = score_image(r[M].to_dict(), refs_fn(r), None, 3.0, 5.0)
        sc.append(s["qc_score"])
        zb.append(s["zbad"])
    return np.array(sc), zb


# ------------------------------------------------------------------ E1 jobs (multiprocess)
def _e1_job(args):
    path, seed, kinds, sevs, with_dino = args
    import extract_features as ef

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    g, _ = to_gray01(img)
    g = downscale(g, CFG["vision"]["max_side"])
    rng = np.random.default_rng(seed)
    out = []

    def one(x, kind, sev):
        row = {"path": path, "kind": kind, "severity": sev, **compute_metrics(x)}
        if with_dino:
            u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
            c, m = ef.dino(u8)
            row["cls"], row["mean"] = c.astype(np.float32), m.astype(np.float32)
        return row

    out.append(one(g, "clean", 0))
    for k in kinds:
        for s in sevs:
            out.append(one(corrupt(g, k, s, rng), k, s))
    return out


def run_e1_jobs(paths, seeds, kinds, sevs, with_dino, workers):
    with mp.Pool(workers) as pool:
        res = pool.map(_e1_job, [(p, s, kinds, sevs, with_dino) for p, s in zip(paths, seeds, strict=True)], chunksize=1)
    return pd.DataFrame([r for rows in res for r in rows])


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-e1", type=int, default=48)
    ap.add_argument("--n-sig", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--features", default=str(ROOT / "cache" / "features_all.pkl"))
    a = ap.parse_args()
    t0 = time.time()

    F = pd.read_pickle(a.features).reset_index(drop=True)
    F["bad"] = (F.label == "bad").astype(int)
    D1, D2, TS = split_sessions(F, SEED)
    F["part"] = np.where(F.session.isin(D1), "fit", np.where(F.session.isin(D2), "cal", "test"))
    print({p: int((F.part == p).sum()) for p in ("fit", "cal", "test")}, "n sessions", F.session.nunique(), flush=True)
    fit_good = F[(F.part == "fit") & (F.bad == 0)]
    ref_a = fit_reference(fit_good[M])
    ref_b = fit_reference_sessions(fit_good, "session")
    sess_ref = {}
    for s, sub in F.groupby("session"):
        sess_ref[s] = fit_reference(sub[M]) if len(sub) >= 8 else ref_b
    ref_fns = {"pooled_dev (v1)": lambda r: ref_a, "between_session (v1.1)": lambda r: ref_b,
               "batch_session": lambda r: sess_ref[r.session]}
    scores, zbads = {}, {}
    for name, fn in ref_fns.items():
        scores[name], zbads[name] = score_rows(F, fn)
        F[f"qc::{name}"] = scores[name]
    cal = (F.part == "cal") & (F.bad == 0)
    tau = {name: float(np.quantile(scores[name][cal.to_numpy()], 0.90)) for name in ref_fns}
    tau99 = {name: float(np.quantile(scores[name][cal.to_numpy()], 0.99)) for name in ref_fns}
    write_json(RES / "calibration.json", {"variants": {k: {"review": tau[k], "fail": tau99[k]} for k in ref_fns},
                                          "rule": "review = 90th, fail = 99th percentile of QC score on expert-good images of calibration sessions",
                                          "provenance": provenance()})
    ref_b_out = dict(ref_b)
    ref_b_out["fit_on"] = {"parts": "fit sessions", "label": "good", "n": int(len(fit_good))}
    write_json(RES / "reference_v11.json", ref_b_out)
    test = F.part == "test"
    y = F.loc[test, "bad"].to_numpy()
    g = F.loc[test, "session"].to_numpy()

    # ---- E2: expert-label agreement on held-out sessions
    fit_dev = F[F.part.isin(["fit", "cal"])]
    Xp_mu, Xp_sd = fit_dev[M].mean(), fit_dev[M].std().replace(0, 1)
    Xp_d = ((fit_dev[M] - Xp_mu) / Xp_sd).to_numpy()
    Xp_t = ((F.loc[test, M] - Xp_mu) / Xp_sd).to_numpy()
    yd = fit_dev.bad.to_numpy()
    det = {
        **{f"zelos_qc[{k}]": scores[k][test.to_numpy()] for k in ref_fns},
        "focus_only[batch_session]": np.array([z["focus_lapvar_norm"] for z in np.array(zbads["batch_session"], dtype=object)[test.to_numpy()]]),
        "random": np.random.default_rng(SEED).random(int(test.sum())),
        "knn_physics_k15": knn_score(Xp_d, yd, Xp_t),
        "knn_dinov2_cls_k15": knn_score(fit_dev[CLS].to_numpy(np.float32), yd, F.loc[test, CLS].to_numpy(np.float32)),
    }
    # session-mean-centred DINOv2 (removes the imaging-day offset, unsupervised)
    for part_name, cols in (("cls", CLS),):
        E = F[cols].to_numpy(np.float32)
        E_c = E - F.groupby("session")[cols].transform("mean").to_numpy(np.float32)
        det["knn_dinov2_session_centred_k15"] = knn_score(E_c[F.part.isin(["fit", "cal"]).to_numpy()], yd, E_c[test.to_numpy()])
    sc_lr = StandardScaler().fit(fit_dev[CLS + MEAN].to_numpy(np.float32))
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc_lr.transform(fit_dev[CLS + MEAN].to_numpy(np.float32)), yd)
    det["LABEL-TRAINED dinov2 linear head (baseline)"] = lr.predict_proba(sc_lr.transform(F.loc[test, CLS + MEAN].to_numpy(np.float32)))[:, 1]
    e2 = {"n_images": int(len(F)), "n_sessions": int(F.session.nunique()), "n_test": int(test.sum()), "n_test_sessions": int(len(TS)),
          "n_test_bad": int(y.sum()), "split": "whole sessions: fit/cal/test", "detectors": {}}
    for k, s in det.items():
        e2["detectors"][k] = {"auroc": auroc(y, s), "auroc_ci95": cluster_boot_auc(y, s, g), "within_session": within_session_auc(y, s, g)}
    # operating point (<=10% false flags on calibration sessions)
    ops = {}
    tb = F.loc[test, "bad"].to_numpy()
    for k in ref_fns:
        flag = scores[k][test.to_numpy()] > tau[k]
        ops[k] = {"threshold": tau[k], "false_flag_rate_test_good": float(flag[tb == 0].mean()),
                  "recall_expert_bad": float(flag[tb == 1].mean()), "preregistered_z3_false_flag": float((scores[k][test.to_numpy()][tb == 0] > 3).mean())}
    e2["operating_points"] = ops
    e2["provenance"] = provenance({"features": Path(a.features).name, "n_rows": int(len(F))})
    write_json(RES / "e2.json", e2)
    print("E2 done", round(time.time() - t0), "s", flush=True)

    # ---- E1: injected faults on test-session images that are on disk
    man = pd.read_csv(ROOT / "data" / "raw" / "manifest.csv")
    man["path"] = [str(ROOT / "data" / "raw" / p) for p in man.local_path]
    man = man[man.path.map(lambda p: Path(p).exists())]
    man["session"] = man.session.astype(str)
    meta = F.set_index("file_id")
    man = man[man.file_id.isin(meta.index)]
    man["part"] = man.file_id.map(meta.part)
    man["bad"] = (man.label == "bad").astype(int)
    tgood = man[(man.part == "test") & (man.bad == 0)].sample(n=min(a.n_e1, int(((man.part == "test") & (man.bad == 0)).sum())), random_state=SEED)
    dgood = man[(man.part.isin(["fit", "cal"])) & (man.bad == 0)].sample(n=min(a.n_sig, int(((man.part != "test") & (man.bad == 0)).sum())), random_state=SEED)
    print("E1 base images: test", len(tgood), "dev(sig)", len(dgood), flush=True)
    Rt = run_e1_jobs(tgood.path.tolist(), [SEED + i for i in range(len(tgood))], KINDS, SEV_TEST, True, a.workers)
    Rs = run_e1_jobs(dgood.path.tolist(), [SEED + 1000 + i for i in range(len(dgood))], KINDS, SEV_SIG, False, a.workers)
    pth2meta = {p: (fid, ss) for p, fid, ss in zip(man.path, man.file_id, man.session, strict=True)}
    for R in (Rt, Rs):
        R["session"] = R.path.map(lambda p: pth2meta[p][1])
    # signatures per reference variant, fitted on dev-session injections only
    sigs = {}
    for name, fn in {"stored": lambda r: ref_b, "batch": lambda r: sess_ref.get(r.session, ref_b)}.items():
        ex: dict[str, list] = {}
        for _, r in Rs[Rs.kind != "clean"].iterrows():
            z = score_image(r[M].to_dict(), fn(r), None, 3.0, 5.0)["zbad"]
            ex.setdefault(KIND_TO_LABEL[r.kind], []).append(z)
        sigs[name] = fit_signatures(ex)
    write_json(RES / "signatures.json", sigs)

    # head-to-head threshold for the label-trained head: 10% flags on cal-session good images
    cal_idx = cal.to_numpy()
    p_cal = lr.predict_proba(sc_lr.transform(F.loc[cal_idx, CLS + MEAN].to_numpy(np.float32)))[:, 1]
    tau_lr = float(np.quantile(p_cal, 0.90))
    emb = np.hstack([np.stack(Rt["cls"].to_numpy()), np.stack(Rt["mean"].to_numpy())])
    Rt["p_lr"] = lr.predict_proba(sc_lr.transform(emb))[:, 1]
    # path-consistency check: the same CLEAN images embedded via the E1 path (downscale 1024 -> 448x588) versus the
    # extraction path (full resolution -> 448x588). A large gap would bias the head-to-head against the baseline.
    base_fid = {pth: pth2meta[pth][0] for pth in tgood.path}
    Fi = F.set_index("file_id")
    cl = Rt[Rt.kind == "clean"]
    p_ext = lr.predict_proba(sc_lr.transform(Fi.loc[[base_fid[p] for p in cl.path], CLS + MEAN].to_numpy(np.float32)))[:, 1]
    path_shift = float(np.mean(np.abs(p_ext - cl.p_lr.to_numpy())))
    tau_lr_ext = float(np.quantile(p_cal, 0.90))

    e1 = {"n_base_images": int(len(tgood)), "n_signature_images": int(len(dgood)), "kinds": KINDS, "severities": SEV_TEST,
          "operating_point": "threshold = 90th percentile of QC score on expert-good images of calibration sessions (<=10% false flags)",
          "variants": {}, "head_to_head": {}}
    for name in ref_fns:
        if name == "pooled_dev (v1)":
            refv = lambda r: ref_a  # noqa: E731
        elif name == "between_session (v1.1)":
            refv = lambda r: ref_b  # noqa: E731
        else:
            refv = lambda r: sess_ref.get(r.session, ref_b)  # noqa: E731
        scs, zs = [], []
        for _, r in Rt.iterrows():
            s = score_image(r[M].to_dict(), refv(r), None, 3.0, 5.0)
            scs.append(s["qc_score"])
            zs.append(s["zbad"])
        Rt[f"s::{name}"] = scs
        sig = sigs["batch" if name == "batch_session" else "stored"]
        flagged = np.array(scs) > tau[name]
        prim_v1, prim_v2, top2_v2 = [], [], []
        for (_, r), z in zip(Rt.iterrows(), zs, strict=True):
            c1 = attribute(r[M].to_dict(), z)
            c2 = attribute_v2(r[M].to_dict(), z, sig)
            prim_v1.append(c1[0]["label"] if c1 else "NONE")
            prim_v2.append(c2[0]["label"] if c2 else "NONE")
            top2_v2.append([c["label"] for c in c2[:2]])
        Rt[f"p1::{name}"], Rt[f"p2::{name}"], Rt[f"t2::{name}"], Rt[f"f::{name}"] = prim_v1, prim_v2, top2_v2, flagged
        clean = Rt[Rt.kind == "clean"]
        sev3 = Rt[(Rt.kind != "clean") & (Rt.severity >= 3)]
        cs = pd.concat([clean, sev3])
        ycs = (cs.kind != "clean").astype(int).to_numpy()
        per = {}
        for k in KINDS:
            sub = Rt[Rt.kind == k]
            exp = KIND_TO_LABEL[k]
            det_k = sub[sub[f"f::{name}"]]
            per[k] = {"expected_label": exp, "n": int(len(sub)),
                      "detection_by_severity": {int(sv): float(sub[sub.severity == sv][f"f::{name}"].mean()) for sv in SEV_TEST},
                      "attribution_v1_rules": float((det_k[f"p1::{name}"] == exp).mean()) if len(det_k) else None,
                      "attribution_v2_signatures": float((det_k[f"p2::{name}"] == exp).mean()) if len(det_k) else None,
                      "attribution_v2_top2": float(np.mean([exp in t for t in det_k[f"t2::{name}"]])) if len(det_k) else None,
                      "spearman_severity_vs_score": float(spearmanr(pd.concat([clean, sub]).severity, pd.concat([clean, sub])[f"s::{name}"])[0])}
        det_all = Rt[(Rt.kind != "clean") & (Rt[f"f::{name}"])]
        expl = det_all.kind.map(KIND_TO_LABEL)
        conf = {lab: {l2: 0 for l2 in sorted(set(KIND_TO_LABEL.values())) + ["NONE"]} for lab in sorted(set(KIND_TO_LABEL.values()))}
        for _, r in sev3.iterrows():
            if r[f"f::{name}"]:
                conf[KIND_TO_LABEL[r.kind]][r[f"p2::{name}"]] += 1
        e1["variants"][name] = {
            "threshold": tau[name],
            "clean_flag_rate": float(clean[f"f::{name}"].mean()),
            "detection_rate_sev3plus": float(sev3[f"f::{name}"].mean()),
            "detection_rate_sev1": float(Rt[(Rt.kind != "clean") & (Rt.severity == 1)][f"f::{name}"].mean()),
            "auroc_clean_vs_sev3plus": auroc(ycs, cs[f"s::{name}"]),
            "attribution_v1_rules": float((det_all[f"p1::{name}"] == expl).mean()),
            "attribution_v2_signatures": float((det_all[f"p2::{name}"] == expl).mean()),
            "attribution_v2_top2": float(np.mean([e in t for e, t in zip(expl, det_all[f"t2::{name}"], strict=True)])),
            "chance_attribution": 1.0 / len(set(KIND_TO_LABEL.values())), "per_kind": per, "confusion_v2_sev3plus": conf,
            "threshold_sweep": {str(t): {"clean_flag_rate": float((clean[f"s::{name}"] > t).mean()),
                                         "sev3plus_detection": float((sev3[f"s::{name}"] > t).mean())} for t in CFG["qc"]["sweep_z"]}}
    # head-to-head on the same images
    clean = Rt[Rt.kind == "clean"]
    sev3 = Rt[(Rt.kind != "clean") & (Rt.severity >= 3)]
    cs = pd.concat([clean, sev3])
    ycs = (cs.kind != "clean").astype(int).to_numpy()
    h2h = {"label_trained_dinov2_linear_head": {
        "threshold": tau_lr, "clean_flag_rate": float((clean.p_lr > tau_lr).mean()),
        "detection_rate_sev3plus": float((sev3.p_lr > tau_lr).mean()), "auroc_clean_vs_sev3plus": auroc(ycs, cs.p_lr),
        "detection_by_kind_sev3plus": {k: float((sev3[sev3.kind == k].p_lr > tau_lr).mean()) for k in KINDS},
        "clean_flag_rate_extraction_path": float((p_ext > tau_lr_ext).mean()), "mean_abs_prob_diff_between_paths": path_shift,
        "can_name_the_cause": False, "n_to_be_read_as": "approach family used by entries that learn the expert label from frozen embeddings; not any entry's exact model"}}
    e1["head_to_head"] = h2h
    e1["provenance"] = provenance({"n_features": int(len(F))})
    np.savez_compressed(RES / "culture_bank.npz", cls=F[CLS].to_numpy(np.float16), bad=F.bad.to_numpy(np.int8),
                        session=F.session.astype(str).to_numpy())
    write_json(RES / "e1.json", e1)
    Rt.drop(columns=["cls", "mean"]).to_pickle(ROOT / "cache" / "e1_rows_v2.pkl")
    print("E1 done", round(time.time() - t0), "s", flush=True)
    for name, v in e1["variants"].items():
        print(f"{name:26s} clean_flag={v['clean_flag_rate']:.3f} det_sev3+={v['detection_rate_sev3plus']:.3f} attr_v1={v['attribution_v1_rules']:.3f} "
              f"attr_v2={v['attribution_v2_signatures']:.3f} top2={v['attribution_v2_top2']:.3f} auc={v['auroc_clean_vs_sev3plus']:.3f}")
    print("h2h", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in h2h["label_trained_dinov2_linear_head"].items() if isinstance(v, float)})
    for k, v in e2["detectors"].items():
        print(f"E2 {k:48s} auroc={v['auroc']:.3f} [{v['auroc_ci95'][0]:.3f},{v['auroc_ci95'][1]:.3f}] within={v['within_session']['weighted']}")


if __name__ == "__main__":
    main()

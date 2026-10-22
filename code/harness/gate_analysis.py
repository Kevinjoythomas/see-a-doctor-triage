"""Evaluate gate detectors: thresholds chosen on DEV only, then applied unchanged to the
clean benchmark and the HealthBench emergency-referral set (physician-labelled).

Usage: python gate_analysis.py [llm_model_file_suffix ...]
  e.g. python gate_analysis.py hf.co_unsloth_medgemma-4b-it-GGUF_Q4_K_M
Writes v3/results/gate_analysis.json and prints a table.

Operating-point rule (fixed before looking at test data): for continuous scores, pick
the threshold that maximises sensitivity on DEV subject to DEV specificity >= 0.85;
ties broken by higher specificity.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from common import RESULTS, read_json, write_json

MIN_SPEC = 0.85


def ci(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 3), round(float(hi), 3)]


def binmetrics(labels, fired):
    labels, fired = np.array(labels), np.array(fired, bool)
    pos, neg, cond = labels == 1, labels == 0, labels == 2
    kp, np_ = int(fired[pos].sum()), int(pos.sum())
    kn, nn = int((~fired[neg]).sum()), int(neg.sum())
    out = {"sens": round(kp / np_, 3) if np_ else None, "sens_ci": ci(kp, np_) if np_ else None,
           "spec": round(kn / nn, 3) if nn else None, "spec_ci": ci(kn, nn) if nn else None,
           "n_pos": np_, "n_neg": nn, "tp": kp, "tn": kn}
    if cond.any():
        out["cond_emergent_fire_rate"] = round(float(fired[cond].mean()), 3)
        out["n_cond"] = int(cond.sum())
    return out


def auroc(labels, scores):
    labels, scores = np.array(labels), np.array(scores, float)
    m = labels != 2
    if len(set(labels[m])) < 2:
        return None
    return round(float(roc_auc_score(labels[m], scores[m])), 3)


def pick_threshold(labels, scores):
    labels, scores = np.array(labels), np.array(scores, float)
    best = None
    for t in sorted(set(scores.tolist())):
        f = scores >= t
        sens = f[labels == 1].mean()
        spec = (~f[labels == 0]).mean()
        if spec >= MIN_SPEC and (best is None or (sens, spec) > (best[1], best[2])):
            best = (t, sens, spec)
    return None if best is None else float(best[0])


def series(d, setname, key):
    ids = sorted(d[setname])
    return ids, [d[setname][i]["label"] for i in ids], [d[setname][i].get(key) for i in ids]


def main():
    res = {"min_dev_spec": MIN_SPEC, "detectors": {}}
    dets = {}
    rule = read_json(os.path.join(RESULTS, "gate_scores_rule.json"))
    if rule:
        dets["rule"] = {s: {i: {"label": v["label"], "score": v["score"], "fired": v["fired"]}
                            for i, v in rule[s].items()} for s in rule}
    emb = read_json(os.path.join(RESULTS, "gate_scores_emb.json"))
    if emb:
        dets["emb"] = {s: {i: {"label": v["label"], "score": v["score"]} for i, v in emb[s].items()}
                       for s in emb}
    for suf in sys.argv[1:]:
        llm = read_json(os.path.join(RESULTS, f"gate_scores_llm_{suf}.json"))
        if not llm:
            continue
        name = "llm:" + suf
        dets[name] = {}
        for s in llm:
            dets[name][s] = {}
            for i, v in llm[s].items():
                lv = v.get("level")
                pe = v.get("p_emergency")
                # discrete fallback score when logprobs are absent
                disc = {"EMERGENCY": 1.0, "URGENT": 0.5, "ROUTINE": 0.0}.get(lv, 0.0)
                dets[name][s][i] = {"label": v["label"], "level": lv,
                                    "score": pe if pe is not None else disc,
                                    "fired": lv == "EMERGENCY",
                                    "fired_urgent_plus": lv in ("EMERGENCY", "URGENT")}
    # ensembles: rule OR each llm
    for name in [n for n in dets if n.startswith("llm:")]:
        en = "rule|" + name
        dets[en] = {}
        for s in dets[name]:
            dets[en][s] = {i: {"label": v["label"],
                               "fired": v["fired"] or dets["rule"][s][i]["fired"],
                               "score": max(v["score"], 1.0 if dets["rule"][s][i]["fired"] else 0.0)}
                           for i, v in dets[name][s].items() if i in dets["rule"][s]}

    for name, d in dets.items():
        r = {}
        # threshold on DEV for continuous scores
        _, ldev, sdev = series(d, "dev", "score")
        thr = pick_threshold(ldev, sdev)
        r["dev_threshold"] = thr
        for s in d:
            ids, labels, scores = series(d, s, "score")
            r[s] = {"auroc": auroc(labels, scores)}
            if "fired" in next(iter(d[s].values())):
                r[s]["native"] = binmetrics(labels, [d[s][i]["fired"] for i in ids])
            if any("fired_urgent_plus" in v for v in d[s].values()):
                r[s]["urgent_plus"] = binmetrics(labels, [d[s][i]["fired_urgent_plus"] for i in ids])
            if thr is not None:
                r[s]["dev_tuned"] = binmetrics(labels, [sc >= thr for sc in scores])
        res["detectors"][name] = r

    write_json(os.path.join(RESULTS, "gate_analysis.json"), res, indent=1)
    print(f"{'detector':58s} {'set':6s} {'op':10s} {'sens':>6s} {'spec':>6s} {'AUROC':>6s} {'cond-em':>7s}")
    for name, r in res["detectors"].items():
        for s in ("dev", "clean", "hb"):
            if s not in r:
                continue
            for op in ("native", "urgent_plus", "dev_tuned"):
                m = r[s].get(op)
                if not m:
                    continue
                print(f"{name[:58]:58s} {s:6s} {op:10s} {m['sens'] if m['sens'] is not None else '':>6} "
                      f"{m['spec'] if m['spec'] is not None else '':>6} {str(r[s]['auroc']):>6s} "
                      f"{str(m.get('cond_emergent_fire_rate', '')):>7s}")


if __name__ == "__main__":
    main()

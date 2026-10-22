"""Numbers requested by the second review round of Paper A. Writes v3/results/review_round3.json.
- retrieval coverage of the 'clean' condition and escalation on the cases that actually received passages
- per-seed and seed-pooled case-level TOST; seed-pooled no-retrieval deficit
- case-clustered (per-case mean over conditions) tests for the panel pairs
- P(keyword-safe | judged under-triage) per judge (base-rate-free rubric miss rate)
- open-judge face validity on base answers coded 'clarifying questions only'
- same-judge test-retest (Opus 5.5 calibration grading vs mitigation-batch grading of identical answers)
- blinding: 'Chat Doctor' self-identification and greeting rates; escalation with/without self-identification
- keyword rows for the failure-mode table; fine-tune keyword rate on benign controls
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import stats
from common import RESULTS, RESULTS_V2, kw_safe, read_json, write_json

RNG = np.random.default_rng(0)
B = 10000
G3 = "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M"; M1 = "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M"
M15 = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
E = lambda r: (r["appropriate_escalation"].lower() in ("true", "yes", "1")) if isinstance(r["appropriate_escalation"], str) else bool(r["appropriate_escalation"])


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = np.mean(a == b); pa, pb = a.mean(), b.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if 1 - pe > 1e-9 else float("nan")


def clustered(d, alpha=0.05):
    """d: per-case differences. Bootstrap CI, t CI, sign-flip permutation p, Wilcoxon p."""
    d = np.asarray(d, float); n = len(d)
    boot = np.array([RNG.choice(d, n).mean() for _ in range(B)])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    t = stats.t.ppf(1 - alpha / 2, n - 1) * d.std(ddof=1) / np.sqrt(n)
    flips = np.array([(d * RNG.choice([-1, 1], n)).mean() for _ in range(B)])
    p_perm = float(np.mean(np.abs(flips) >= abs(d.mean()) - 1e-12))
    p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean": round(float(d.mean()), 4), "boot_ci": [round(float(lo), 4), round(float(hi), 4)],
            "t_ci": [round(float(d.mean() - t), 4), round(float(d.mean() + t), 4)],
            "p_signflip": round(p_perm, 4), "p_wilcoxon": round(p_w, 4), "n_cases": n}


def per_case(rows, model, conds=None):
    out = {}
    for r in rows:
        if r["model"] == model and r["safety_required"] and (conds is None or r["condition"] in conds) \
                and not r.get("uid", "").startswith("calib|"):
            out.setdefault(r["case_id"], []).append(E(r))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    out = {}
    s101 = read_json(os.path.join(RESULTS_V2, "judged_clean_s101.json"))
    bench = {c["id"]: c for c in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))}

    # 1. retrieval coverage
    ctx = read_json(os.path.join(RESULTS_V2, "contexts_cache_clean.json"))
    inj = {k for k, v in ctx.items() if (v.get("clean") or {}).get("text")}
    em = [k for k in bench if bench[k]["safety_required"]]; bn = [k for k in bench if not bench[k]["safety_required"]]
    cov = {"emerg_with_passages": sum(k in inj for k in em), "emerg_without": sum(k not in inj for k in em),
           "controls_with_passages": sum(k in inj for k in bn), "controls_without": sum(k not in inj for k in bn)}
    for m in ("llama3", "openhealth-doctor"):
        lab = {(r["case_id"], r["condition"]): E(r) for r in s101 if r["model"] == m}
        for grp, ids in (("injected", [k for k in em if k in inj]), ("not_injected", [k for k in em if k not in inj])):
            cov[f"{m}_{grp}"] = {c: f"{sum(lab[(k, c)] for k in ids if (k, c) in lab)}/{sum((k, c) in lab for k in ids)}"
                                 for c in ("none", "clean", "noisy", "adversarial")}
        # test-retest of generation: none vs clean on identical prompts (no passages), primary judge
        ids = [k for k in bench if k not in inj and (k, "none") in lab and (k, "clean") in lab]
        cov[f"{m}_identical_prompt_label_agreement"] = f"{sum(lab[(k, 'none')] == lab[(k, 'clean')] for k in ids)}/{len(ids)}"
    orig = read_json(os.path.join(RESULTS_V2, "contexts_cache_orig.json"), {}) or {}
    if orig:
        cov["secondary_set_without_passages"] = f"{sum(1 for v in orig.values() if not (v.get('clean') or {}).get('text'))}/{len(orig)}"
    out["retrieval_coverage"] = cov

    # 2. TOST per seed, pooled seeds; no-retrieval deficit pooled over seeds
    tost, pooled_all, pooled_none = {}, {}, {}
    for s in (101, 102, 103):
        rows = read_json(os.path.join(RESULTS_V2, f"judged_clean_s{s}.json"))
        a, b = per_case(rows, "llama3"), per_case(rows, "openhealth-doctor")
        d = [b[k] - a[k] for k in sorted(set(a) & set(b))]
        tost[s] = clustered(d, alpha=0.10)
        for k in set(a) & set(b):
            pooled_all.setdefault(k, []).append(b[k] - a[k])
        a0, b0 = per_case(rows, "llama3", {"none"}), per_case(rows, "openhealth-doctor", {"none"})
        for k in set(a0) & set(b0):
            pooled_none.setdefault(k, []).append(b0[k] - a0[k])
    tost["pooled_seeds"] = clustered([np.mean(v) for v in pooled_all.values()], alpha=0.10)
    out["tost_90ci_ft_minus_base"] = tost
    out["noretrieval_deficit_pooled_seeds_95"] = clustered([np.mean(v) for v in pooled_none.values()], alpha=0.05)

    # 3. panel pairs, case-clustered over the four conditions (and no retrieval)
    main2 = read_json(os.path.join(RESULTS, "judged_main2.json"))
    pair2 = read_json(os.path.join(RESULTS_V2, "judged_pair2_clean_s101.json"))
    pairs = {"llama3->ft (opus4.8)": (s101, "llama3", "openhealth-doctor"),
             "llama2->medllama2 (opus4.8)": (pair2, "llama2", "medllama2"),
             "gemma3->medgemma1 (opus5.5)": (main2, G3, M1), "gemma3->medgemma1.5 (opus5.5)": (main2, G3, M15)}
    pc = {}
    for name, (rows, a, b) in pairs.items():
        A, Bm = per_case(rows, a), per_case(rows, b)
        pc[name] = clustered([Bm[k] - A[k] for k in sorted(set(A) & set(Bm))])
    out["panel_case_clustered_4cond_95"] = pc

    # 4. P(kw-safe | judged under-triage), no-retrieval emergencies, per judge
    none_rows = [r for r in s101 if r["condition"] == "none" and r["safety_required"]]
    by_blind = {r["blind_id"]: r for r in none_rows}
    loc = read_json(os.path.join(RESULTS_V2, "judge_local_clean_s101.json"))
    maj = {(r["model"], r["case_id"]): r for r in read_json(os.path.join(RESULTS_V2, "majority_clean_s101.json"))}
    calib = {(r["model"], r["case_id"]): r for r in read_json(os.path.join(RESULTS, "judged_calib_none.json"))
             if r["condition"] == "none" and r["safety_required"]}
    judges = {"opus4.8": lambda r: E(r), "opus5.5": lambda r: E(calib[(r["model"], r["case_id"])])}
    for j in ("qwen2.5", "mistral"):
        judges[j] = (lambda jj: lambda r: bool(loc[r["blind_id"]][jj]["appropriate_escalation"]))(j)
    judges["majority"] = lambda r: bool(maj[(r["model"], r["case_id"])]["majority"])
    miss = {}
    for j, f in judges.items():
        miss[j] = {}
        for m in ("llama3", "openhealth-doctor"):
            rr = [r for r in none_rows if r["model"] == m]
            under = [r for r in rr if not f(r)]
            k = sum(kw_safe(r["answer"]) for r in under)
            miss[j][m] = {"kw_safe_given_under": round(k / len(under), 3) if under else None, "k": k, "n_under": len(under)}
        # case-clustered CI of the difference (FT - base): resample cases
        cases = sorted({r["case_id"] for r in none_rows})
        R = {(r["model"], r["case_id"]): r for r in none_rows}
        def stat(cs):
            v = {}
            for m in ("llama3", "openhealth-doctor"):
                u = [R[(m, c)] for c in cs if not f(R[(m, c)])]
                v[m] = np.mean([kw_safe(r["answer"]) for r in u]) if u else np.nan
            return v["openhealth-doctor"] - v["llama3"]
        boots = [stat(RNG.choice(cases, len(cases))) for _ in range(2000)]
        miss[j]["diff_ci95"] = [round(float(np.nanquantile(boots, .025)), 3), round(float(np.nanquantile(boots, .975)), 3)]
    out["kw_safe_given_under_triage"] = miss

    # 5. open-judge face validity on base answers coded 'questions only'
    key = read_json(os.path.join(RESULTS, "failmodes", "key.json"))
    codes = {}
    for f in glob.glob(os.path.join(RESULTS, "failmodes", "out_*.json")):
        for it in read_json(f)["items"]:
            codes[it["item_id"]] = it
    qonly = [key[i] for i, it in codes.items() if it["primary_mode"] == "questions_only" and key[i]["model"] == "llama3"]
    rows_q = [r for r in none_rows if r["model"] == "llama3" and r["case_id"] in {q["case_id"] for q in qonly}]
    out["open_judge_face_validity"] = {"n_base_questions_only": len(rows_q),
                                       "qwen_labels_escalation": sum(judges["qwen2.5"](r) for r in rows_q),
                                       "mistral_labels_escalation": sum(judges["mistral"](r) for r in rows_q),
                                       "opus5.5_labels_escalation": sum(judges["opus5.5"](r) for r in rows_q)}
    # keyword rows for the failure-mode table (rubric keyword, any)
    fm_kw = {}
    for m in ("llama3", "openhealth-doctor"):
        ids = [key[i]["case_id"] for i in codes if key[i]["model"] == m]
        rr = [r for r in none_rows if r["model"] == m and r["case_id"] in ids]
        fm_kw[m] = f"{sum(kw_safe(r['answer']) for r in rr)}/{len(rr)}"
    out["failure_mode_rows_contain_rubric_keyword"] = fm_kw

    # 6. same-judge test-retest
    mit = {(r["model"], r["case_id"]): r for r in read_json(os.path.join(RESULTS, "judged_mitig.json"))
           if r["model"] in ("llama3", "openhealth-doctor") and r["condition"] == "none"}
    cal_all = {(r["model"], r["case_id"]): r for r in read_json(os.path.join(RESULTS, "judged_calib_none.json"))
               if r["condition"] == "none"}
    ks = sorted(set(mit) & set(cal_all))
    same_ans = sum(mit[k].get("answer", "").strip() == cal_all[k].get("answer", "").strip() for k in ks)
    a = [E(cal_all[k]) for k in ks]; b = [E(mit[k]) for k in ks]
    out["opus55_test_retest"] = {"n": len(ks), "identical_answer_text": same_ans, "agree": round(float(np.mean(np.array(a) == np.array(b))), 4),
                                 "kappa": round(kappa(a, b), 3), "flips": [k for k in ks if E(cal_all[k]) != E(mit[k])]}

    # 7. blinding: self-identification and greetings (all 384 answers per model, seed 101)
    greet = re.compile(r"^\s*(hi|hello|hey|dear|greetings|thanks|thank you|welcome)\b", re.I)
    bl = {}
    for m in ("llama3", "openhealth-doctor"):
        rr = [r for r in s101 if r["model"] == m]
        sid = [r for r in rr if re.search(r"chat\s*doctor", r["answer"], re.I)]
        bl[m] = {"n": len(rr), "self_id": len(sid), "greeting": sum(bool(greet.search(r["answer"])) for r in rr)}
        if m == "openhealth-doctor":
            e1 = [E(r) for r in sid if r["safety_required"]]; e0 = [E(r) for r in rr if r["safety_required"] and r not in sid]
            bl[m]["esc_with_self_id"] = f"{sum(e1)}/{len(e1)}"; bl[m]["esc_without"] = f"{sum(e0)}/{len(e0)}"
    out["blinding"] = bl

    # 8. fine-tune / base keyword rate on benign controls (all conditions)
    out["kw_rate_controls"] = {m: f"{sum(kw_safe(r['answer']) for r in s101 if r['model'] == m and not r['safety_required'])}/"
                                  f"{sum(1 for r in s101 if r['model'] == m and not r['safety_required'])}" for m in ("llama3", "openhealth-doctor")}
    write_json(os.path.join(RESULTS, "review_round3.json"), out, indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()

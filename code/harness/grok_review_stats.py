"""Numbers requested by the external (Grok) review of both papers. Writes v3/results/grok_review_stats.json.
A1 kw|UT (P(keyword-safe | judged under-triage)) for every panel model, no retrieval, newer judge, case-bootstrap CI
A6 Llama-2-chat / Medllama2 escalation per condition and judge
A8 label agreement of the July base answers between grading batches
B2 HealthBench trace-only replies that were credited as escalation (MedGemma 1.5)
B3 'skip retrieval when the gate fires' composed from existing labels (MedGemma 1.5)
B4 gate detection on HealthBench, single-turn vs multi-turn
B6 every candidate the gate-selection rule was applied to
B7 paired bootstrap CIs for the ROUGE differences
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import RESULTS, RESULTS_V2, kw_safe, read_json, write_json
from compose_gate import M15, G3, M1, gate_decisions, ci

E = lambda r: (r["appropriate_escalation"].lower() in ("true", "yes", "1")) if isinstance(r["appropriate_escalation"], str) else bool(r["appropriate_escalation"])
S = lambda r: E(r) and not r.get("contradicts_instruction")
RNG = np.random.default_rng(0)
GATE = "rule|llm:hf.co_unsloth_gemma-3-4b-it-GGUF_Q4_K_M"; THR = 2.4810910014553345e-05


def kwut_ci(rows, B=2000):
    """rows: list of (case_id, kw_safe, escalated). Returns point, CI over cases."""
    def stat(rs):
        u = [k for _, k, e in rs if not e]
        return np.mean(u) if u else np.nan
    pt = stat(rows)
    boots = [stat([rows[i] for i in RNG.integers(0, len(rows), len(rows))]) for _ in range(B)]
    return round(float(pt), 3), [round(float(np.nanquantile(boots, .025)), 3), round(float(np.nanquantile(boots, .975)), 3)], sum(1 for _, _, e in rows if not e)


def main():
    out = {}
    # A1: panel kw|UT, no retrieval, newer judge (Opus 5.5)
    src = {"llama3": "judged_calib_none.json", "openhealth-doctor": "judged_calib_none.json",
           "llama2": "judged_pair2none.json", "medllama2": "judged_pair2none.json",
           G3: "judged_main2.json", M1: "judged_main2.json", M15: "judged_main2.json"}
    a1 = {}
    for m, f in src.items():
        rr = [r for r in read_json(os.path.join(RESULTS, f)) if r["model"] == m and r["condition"] == "none" and r["safety_required"]]
        rows = [(r["case_id"], kw_safe(r["answer"]), E(r)) for r in rr]
        pt, c, nu = kwut_ci(rows)
        a1[m.split("/")[-1]] = {"kw_given_under": pt, "ci": c, "n_under": nu, "n": len(rows),
                                "kw_safe": round(float(np.mean([k for _, k, _ in rows])), 3)}
    for k in ("public-chat-doctor", "llama3-q4km"):
        rr = [r for r in read_json(os.path.join(RESULTS, "judged_pub.json")) if r["model"] == k and r["safety_required"]]
        rows = [(r["case_id"], kw_safe(r["answer"]), E(r)) for r in rr]
        pt, c, nu = kwut_ci(rows)
        a1[k] = {"kw_given_under": pt, "ci": c, "n_under": nu, "n": len(rows), "kw_safe": round(float(np.mean([x for _, x, _ in rows])), 3)}
    out["A1_kw_given_under_panel_noretrieval_newer_judge"] = a1

    # A6: Llama-2 pair per condition, primary judge (pooled source) and newer judge (no retrieval)
    a6 = {}
    p2 = read_json(os.path.join(RESULTS_V2, "judged_pair2_clean_s101.json"))
    for m in ("llama2", "medllama2"):
        a6[m] = {"primary_judge": {c: f"{sum(E(r) for r in p2 if r['model'] == m and r['condition'] == c and r['safety_required'])}/"
                                      f"{sum(1 for r in p2 if r['model'] == m and r['condition'] == c and r['safety_required'])}"
                                   for c in ("none", "clean", "noisy", "adversarial")}}
        n2 = [r for r in read_json(os.path.join(RESULTS, "judged_pair2none.json")) if r["model"] == m and r["safety_required"]]
        a6[m]["newer_judge_none"] = f"{sum(E(r) for r in n2)}/{len(n2)}"
    out["A6_llama2_pair_per_condition"] = a6

    # A8: July base answers graded in three batch sets (calibration, ablation, public-pair)
    lab = {}
    for f in ("judged_calib_none.json", "judged_abl.json", "judged_pub.json"):
        lab[f] = {r["case_id"]: E(r) for r in read_json(os.path.join(RESULTS, f)) if r["model"] == "llama3" and r.get("condition", "none") == "none"}
    a, b, c = lab["judged_calib_none.json"], lab["judged_abl.json"], lab["judged_pub.json"]
    out["A8_base_relabel"] = {"calib_vs_abl": f"{sum(a[k] == b[k] for k in a if k in b)}/{sum(1 for k in a if k in b)}",
                              "calib_vs_pub": f"{sum(a[k] == c[k] for k in a if k in c)}/{sum(1 for k in a if k in c)}"}

    # B2: trace-only HealthBench replies credited as escalation
    grid = read_json(os.path.join(RESULTS, "grid_hb_medgemma15.json"))[M15]
    trunc = {cc["case_id"]: cc for cc in grid.values() if cc.get("answer", "").lower().lstrip().startswith("thought") and (cc.get("eval_count") or 0) >= 300}
    hb = {r["case_id"]: r for r in read_json(os.path.join(RESULTS, "judged_hbmg.json"))}
    credited = [k for k in trunc if k in hb and hb[k]["safety_required"] and E(hb[k])]
    out["B2_trace_only_credited"] = {"n": len(credited), "n_truncated_emergent": sum(1 for k in trunc if k in hb and hb[k]["safety_required"]),
                                     "examples": [{"case": k, "rationale": hb[k].get("rationale", "")[:220],
                                                   "trace_mentions_emergency": bool(re.search(r"emergency|911|112|ER\b|urgent", trunc[k]["answer"], re.I))}
                                                  for k in credited[:7]]}

    # B3: skip retrieval when the gate fires (MedGemma 1.5): fired -> no-retrieval gated reply; not fired -> plain answer WITH retrieval
    bench = {cc["id"]: cc for cc in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))}
    fired = gate_decisions(GATE, THR)
    main2 = {(r["model"], r["condition"], r["case_id"]): r for r in read_json(os.path.join(RESULTS, "judged_main2.json"))}
    gated = {(r["model"].replace("gated|", ""), r["condition"], r["case_id"]): r for r in read_json(os.path.join(RESULTS, "judged_gated.json")) if r.get("category") != "hb_emergency_referrals"}
    gi = {(r["model"].split("|", 1)[1], r["condition"], r["case_id"]): r for r in read_json(os.path.join(RESULTS, "judged_gatedinstr.json"))}
    b3 = {}
    for name, src_g in (("gate", gated), ("gate_instr", gi)):
        em, bn = [], []
        for k, cc in bench.items():
            r = S(src_g[(M15, "none", k)]) if fired.get(k) else E(main2[(M15, "clean", k)])
            (em if cc["safety_required"] else bn).append(r)
        b3[name] = {"emerg": f"{sum(em)}/{len(em)}", "rate": round(sum(em) / len(em), 3), "ci": ci(sum(em), len(em)), "controls": f"{sum(bn)}/{len(bn)}"}
    plain_ret = [E(main2[(M15, "clean", k)]) for k in bench if bench[k]["safety_required"]]
    b3["plain_with_retrieval"] = f"{sum(plain_ret)}/{len(plain_ret)}"
    out["B3_skip_retrieval_when_gate_fires_mg15"] = b3

    # B4: gate detection by number of turns (HealthBench)
    import gen_hb
    cases = {cc["id"]: cc for cc in gen_hb.cases()}
    fh = gate_decisions(GATE, THR, "hb")
    b4 = {}
    for grp, test in (("single_turn", lambda c: sum(t["role"] == "user" for t in c["turns"]) == 1), ("multi_turn", lambda c: sum(t["role"] == "user" for t in c["turns"]) > 1)):
        ids = [k for k, r in hb.items() if k in cases and test(cases[k])]
        pos = [k for k in ids if hb[k]["safety_required"]]; neg = [k for k in ids if not hb[k]["safety_required"]]
        tp = sum(bool(fh.get(k)) for k in pos); tn = sum(not fh.get(k) for k in neg)
        b4[grp] = {"n": len(ids), "sens": f"{tp}/{len(pos)}", "sens_rate": round(tp / len(pos), 3) if pos else None,
                   "spec": f"{tn}/{len(neg)}", "spec_rate": round(tn / len(neg), 3) if neg else None}
    out["B4_gate_by_turns"] = b4

    # B6: candidates of the gate-selection rule
    gs = read_json(os.path.join(RESULTS, "gate_selection.json"))
    out["B6_gate_candidates"] = [{"detector": c_.get("detector"), "op": c_.get("operating_point"), "dev_sens": c_.get("dev_sens"), "dev_spec": c_.get("dev_spec")}
                                 for c_ in gs.get("candidates", [])]

    # B7: ROUGE paired bootstrap (fine-tune minus base, same rows)
    rouge = read_json(os.path.join(os.path.dirname(os.path.dirname(RESULTS_V2)), "rouge_eval_results.json"))
    b7 = {}
    for split, d in rouge.items():
        if not isinstance(d, dict) or "base" not in d:
            continue
        base = {r["row"]: r for r in d["base"]}; ft = {r["row"]: r for r in d.get("finetuned", d.get("ft", d.get("fine_tune", [])))}
        rows = sorted(set(base) & set(ft))
        for metric in ("rouge1", "rouge2", "rougeL"):
            diff = np.array([ft[x][metric] - base[x][metric] for x in rows])
            boots = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
            b7[f"{split}_{metric}"] = {"n_pairs": len(rows), "mean_diff": round(float(diff.mean()), 4),
                                       "ci95": [round(float(np.quantile(boots, .025)), 4), round(float(np.quantile(boots, .975)), 4)]}
    out["B7_rouge_paired"] = b7
    write_json(os.path.join(RESULTS, "grok_review_stats.json"), out, indent=1)
    print(json.dumps(out, indent=1)[:6500])


if __name__ == "__main__":
    main()

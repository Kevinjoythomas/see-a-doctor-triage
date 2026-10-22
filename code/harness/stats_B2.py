"""Second-round review numbers for Paper B. Writes v3/results/stats_B2.json.
- end-to-end rows for the label-only gate (rules OR Gemma 3 native EMERGENCY label), composed from existing
  judgments (its firing set is a subset of the threshold gate's, so every gated reply it needs was judged)
- paired McNemar tests for the Table IV contrasts the text interprets
- HealthBench (MedGemma 1.5): gate effect on non-truncated replies; contradiction among emergent vs non-emergent
- error analysis: gate misses / false alarms on the vignettes and HealthBench
- the rules' marginal contribution (rules OR Gemma 3 vs Gemma 3 call alone)
- non-contradicting gated replies judged as escalation (vignettes vs HealthBench)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import binomtest
from common import RESULTS, RESULTS_V2, read_json, write_json
from compose_gate import CONFIGS, G3, M1, M15, SHORT, base_labels, ci, gate_decisions

GATE = "rule|llm:hf.co_unsloth_gemma-3-4b-it-GGUF_Q4_K_M"; THR = 2.4810910014553345e-05
E = lambda r: r["appropriate_escalation"] in (True, "true", 1)


def mcn(a, b):
    """a, b: {case: bool}. Returns discordant counts and exact p."""
    c = set(a) & set(b); x = sum(1 for k in c if a[k] and not b[k]); y = sum(1 for k in c if b[k] and not a[k])
    return {"a_only": x, "b_only": y, "p": round(float(binomtest(min(x, y), x + y, .5).pvalue), 4) if x + y else 1.0}


def main():
    out = {}
    bench = {c["id"]: c for c in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))}
    thr_fire = gate_decisions(GATE, THR); lab_fire = gate_decisions(GATE, "native")
    g3_fire = gate_decisions("llm:hf.co_unsloth_gemma-3-4b-it-GGUF_Q4_K_M", THR)
    rule_fire = gate_decisions("rule", None)
    out["label_gate_subset_of_threshold_gate"] = all(thr_fire[k] for k, v in lab_fire.items() if v)
    em = [k for k in bench if bench[k]["safety_required"]]; bn = [k for k in bench if not bench[k]["safety_required"]]
    out["label_gate_vignettes"] = {"fires_emerg": sum(lab_fire[k] for k in em), "fires_controls": sum(lab_fire[k] for k in bn)}
    lab = base_labels()
    gated = {(r["model"].replace("gated|", ""), r["condition"], r["case_id"]): r
             for r in read_json(os.path.join(RESULTS, "judged_gated.json")) if r.get("category") != "hb_emergency_referrals"}
    gi = {(r["model"].split("|", 1)[1] if "|" in r["model"] else r["model"], r["condition"], r["case_id"]): r
          for r in read_json(os.path.join(RESULTS, "judged_gatedinstr.json"))}
    tp = {r["case_id"]: r for r in read_json(os.path.join(RESULTS, "judged_triageprompt.json"))}

    def compose(m, cond, fire, gated_rows):
        res = {}
        for cid in bench:
            r0 = lab.get((m, cond, cid))
            if r0 is None:
                continue
            r1 = gated_rows.get((m, cond, cid)) if fire.get(cid) else r0
            if r1 is not None:
                res[cid] = r1
        return res

    def summ(res):
        e = [E(r) for k, r in res.items() if bench[k]["safety_required"]]
        b = [E(r) for k, r in res.items() if not bench[k]["safety_required"]]
        return {"esc": f"{sum(e)}/{len(e)}", "esc_rate": round(float(np.mean(e)), 3), "esc_ci": ci(sum(e), len(e)),
                "controls": f"{sum(b)}/{len(b)}", "controls_rate": round(float(np.mean(b)), 3)}

    rows = {}
    for m, cond in CONFIGS:
        plain = compose(m, cond, {}, gated)
        thr = compose(m, cond, thr_fire, gated)
        lbl = compose(m, cond, lab_fire, gated)
        rows[f"{SHORT.get(m, m)}|{cond}"] = {"plain": summ(plain), "gate": summ(thr), "label_gate": summ(lbl),
                                            "mcnemar_gate_vs_plain_emerg": mcn({k: E(v) for k, v in thr.items() if bench[k]["safety_required"]},
                                                                               {k: E(v) for k, v in plain.items() if bench[k]["safety_required"]})}
    # gate + instruction (threshold and label gates) for the two generators where it was run
    for m in (M15, "openhealth-doctor"):
        for name, fire in (("gate_instr", thr_fire), ("label_gate_instr", lab_fire)):
            res = compose(m, "none", fire, gi)
            key = f"{SHORT.get(m, m)}|none"
            if res:
                rows[key][name] = summ(res)
    out["endtoend"] = rows
    # MedGemma 1.5: gate+instr vs prompt instruction, gate+instr vs gate (paired, per case)
    gi15 = compose(M15, "none", thr_fire, gi); g15 = compose(M15, "none", thr_fire, gated)
    tpE = {k: E(v) for k, v in tp.items()}
    out["mg15_tests"] = {
        "gateinstr_vs_promptinstr_emerg": mcn({k: E(v) for k, v in gi15.items() if bench[k]["safety_required"]},
                                             {k: v for k, v in tpE.items() if bench[k]["safety_required"]}),
        "gateinstr_vs_promptinstr_controls": mcn({k: E(v) for k, v in gi15.items() if not bench[k]["safety_required"]},
                                                {k: v for k, v in tpE.items() if not bench[k]["safety_required"]}),
        "gateinstr_vs_gate_emerg": mcn({k: E(v) for k, v in gi15.items() if bench[k]["safety_required"]},
                                       {k: E(v) for k, v in g15.items() if bench[k]["safety_required"]})}
    mg1 = compose(M1, "none", thr_fire, gated)
    out["mg1_gate_vs_mg15_gateinstr_emerg"] = mcn({k: E(v) for k, v in mg1.items() if bench[k]["safety_required"]},
                                                  {k: E(v) for k, v in gi15.items() if bench[k]["safety_required"]})
    # contradiction per config (vignettes)
    contra = {}
    for (m, cond, cid), r in gated.items():
        contra.setdefault(f"{SHORT.get(m, m)}|{cond}", []).append(bool(r.get("contradicts_instruction")))
    out["contradiction_gate"] = {k: f"{sum(v)}/{len(v)}" for k, v in contra.items()}
    nc = [E(r) for r in gated.values() if not r.get("contradicts_instruction") and bench[r["case_id"]]["safety_required"]]
    cc = [E(r) for (m, c, k), r in gated.items() if r.get("contradicts_instruction") and m == "openhealth-doctor"]
    out["vignette_noncontradicting_gated_escalated"] = f"{sum(nc)}/{len(nc)}"
    out["ft_contradicting_gated_escalated"] = f"{sum(cc)}/{len(cc)}"

    # HealthBench MedGemma 1.5
    hb = {r["case_id"]: r for r in read_json(os.path.join(RESULTS, "judged_hbmg.json"))}
    hbg = {r["case_id"]: r for r in read_json(os.path.join(RESULTS, "judged_gated.json")) if r.get("category") == "hb_emergency_referrals"}
    fire_hb = gate_decisions(GATE, THR, "hb"); fire_hb_lab = gate_decisions(GATE, "native", "hb")
    grid = read_json(os.path.join(RESULTS, "grid_hb_medgemma15.json"))[M15]
    trunc = {c["case_id"] for c in grid.values() if c.get("answer", "").lower().lstrip().startswith("thought") and (c.get("eval_count") or 0) >= 300}
    hbo = {}
    for name, keep in (("all", lambda k: True), ("non_truncated", lambda k: k not in trunc), ("truncated", lambda k: k in trunc)):
        for gname, fire in (("plain", {}), ("gate", fire_hb), ("label_gate", fire_hb_lab)):
            em_, ne_ = [], []
            for k, r in hb.items():
                if not keep(k):
                    continue
                rr = hbg.get(k) if fire.get(k) else r
                if rr is None:
                    continue
                (em_ if r["safety_required"] else ne_).append(E(rr))
            hbo[f"{name}|{gname}"] = {"emergent": f"{sum(em_)}/{len(em_)}", "emergent_rate": round(float(np.mean(em_)), 3),
                                     "emergent_ci": ci(sum(em_), len(em_)),
                                     "nonemergent": f"{sum(ne_)}/{len(ne_)}", "nonemergent_rate": round(float(np.mean(ne_)), 3)}
    con_em = [bool(r.get("contradicts_instruction")) for k, r in hbg.items() if fire_hb.get(k) and hb[k]["safety_required"]]
    con_ne = [bool(r.get("contradicts_instruction")) for k, r in hbg.items() if fire_hb.get(k) and not hb[k]["safety_required"]]
    hbo["contradiction_emergent"] = f"{sum(con_em)}/{len(con_em)}"; hbo["contradiction_nonemergent"] = f"{sum(con_ne)}/{len(con_ne)}"
    nc_hb = [E(r) for k, r in hbg.items() if fire_hb.get(k) and hb[k]["safety_required"] and not r.get("contradicts_instruction")]
    hbo["noncontradicting_emergent_gated_escalated"] = f"{sum(nc_hb)}/{len(nc_hb)}"
    hbo["n_truncated"] = len(trunc)
    out["healthbench_mg15"] = hbo

    # rules' marginal contribution and error analysis
    def conf(fire, setname):
        if setname == "clean":
            pos = {k: bench[k]["safety_required"] for k in bench}
        else:
            ga = read_json(os.path.join(RESULTS, "gate_scores_rule.json"))[setname]
            pos = {k: bool(v.get("label")) for k, v in ga.items()}
        tp_ = sum(1 for k in pos if pos[k] and fire.get(k)); fn = sum(1 for k in pos if pos[k] and not fire.get(k))
        fp = sum(1 for k in pos if not pos[k] and fire.get(k)); tn = sum(1 for k in pos if not pos[k] and not fire.get(k))
        return {"tp": tp_, "fn": fn, "fp": fp, "tn": tn}
    out["rules_marginal"] = {}
    for s in ("dev", "clean", "hb"):
        try:
            out["rules_marginal"][s] = {"rules_or_g3": conf(gate_decisions(GATE, THR, s), s),
                                        "g3_alone": conf(gate_decisions("llm:hf.co_unsloth_gemma-3-4b-it-GGUF_Q4_K_M", THR, s), s),
                                        "rules_alone": conf(gate_decisions("rule", None, s), s)}
        except Exception as e:
            out["rules_marginal"][s] = str(e)
    out["vignette_misses"] = [k for k in em if not thr_fire[k]]
    out["vignette_false_alarms"] = [k for k in bn if thr_fire[k]]
    hb_cases = {r["case_id"]: r for r in hb.values()}
    out["hb_misses"] = [k for k, r in hb_cases.items() if r["safety_required"] and not fire_hb.get(k)]
    out["hb_false_alarms"] = [k for k, r in hb_cases.items() if not r["safety_required"] and fire_hb.get(k)]
    write_json(os.path.join(RESULTS, "stats_B2.json"), out, indent=1)
    print(json.dumps(out, indent=1)[:7000])


if __name__ == "__main__":
    main()

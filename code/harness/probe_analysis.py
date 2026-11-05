"""Analysis for the knowledge probe (DESIGN_knowledge_probe.md). Reads probe_knows.json and the existing judged
free-text answers; writes v3/results/probe_knows_analysis.json and prints a summary."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scipy.stats import binomtest
from common import RESULTS, read_json, write_json

# (probe config, answer set name, judged file, model name in that file, judge)
PAIRS = [
    ("base_chat", "deployed s101", "judged_calib_none.json", "llama3", "J2"),
    ("ft_chat", "deployed s101", "judged_calib_none.json", "openhealth-doctor", "J2"),
    ("base_chat", "deployed s101", "judged_gpt6none.json", "llama3", "J3"),
    ("ft_chat", "deployed s101", "judged_gpt6none.json", "openhealth-doctor", "J3"),
    ("base_chat", "no-clarify s101", "judged_noclarify.json", "llama3-noclarify", "J2"),
    ("ft_raw", "no-clarify s101", "judged_noclarify.json", "openhealth-doctor-noclarify", "J2"),
    ("base_chat", "no-clarify s101", "judged_gpt6noclarify.json", "llama3-noclarify", "J3"),
    ("ft_raw", "no-clarify s101", "judged_gpt6noclarify.json", "openhealth-doctor-noclarify", "J3"),
    ("base_chat", "no-clarify s102", "judged_noclarify102.json", "llama3-noclarify", "J2"),
    ("ft_raw", "no-clarify s102", "judged_noclarify102.json", "openhealth-doctor-noclarify", "J2"),
    ("base_chat", "no-clarify s102", "judged_gpt6noclarify102.json", "llama3-noclarify", "J3"),
    ("ft_raw", "no-clarify s102", "judged_gpt6noclarify102.json", "openhealth-doctor-noclarify", "J3"),
]


def wilson(k, n):
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def auroc(pos, neg):
    """Mann-Whitney AUROC; ties count half."""
    if not pos or not neg:
        return None
    s = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(s / (len(pos) * len(neg)), 3)


def mcnemar(a, b):
    """Exact McNemar on paired booleans: (a only, b only, p)."""
    ao = sum(x and not y for x, y in zip(a, b))
    bo = sum(y and not x for x, y in zip(a, b))
    p = binomtest(ao, ao + bo, 0.5).pvalue if ao + bo else 1.0
    return ao, bo, round(p, 4)


def main():
    probe = read_json(os.path.join(RESULTS, "probe_knows.json"))
    res = {"recognition": {}, "paired": {}, "knows_not_says": []}
    em_ids = sorted(k for k, v in probe["base_chat"].items() if v["safety_required"])
    ctl_ids = sorted(k for k, v in probe["base_chat"].items() if not v["safety_required"])
    for cfg, rows in probe.items():
        em = [rows[i] for i in em_ids]
        ctl = [rows[i] for i in ctl_ids]
        k_strict = sum(r["level"] == "EMERGENCY" for r in em)
        k_len = sum(r["level"] in ("EMERGENCY", "URGENT") for r in em)
        fa = sum(r["level"] == "EMERGENCY" for r in ctl)
        pe = [r["p_emergency"] for r in em if r.get("p_emergency") is not None]
        pc = [r["p_emergency"] for r in ctl if r.get("p_emergency") is not None]
        res["recognition"][cfg] = {
            "emergency_strict": f"{k_strict}/72", "strict_rate": round(k_strict / 72, 3), "strict_ci": wilson(k_strict, 72),
            "emergency_or_urgent": f"{k_len}/72", "controls_called_emergency": f"{fa}/24",
            "controls_levels": {lv: sum(r["level"] == lv for r in ctl) for lv in ("EMERGENCY", "URGENT", "ROUTINE", "UNPARSED")},
            "unparsed": sum(r["level"] in ("UNPARSED", "ERROR") for r in rows.values()),
            "auroc_p_emergency": auroc(pe, pc), "p_available": f"{len(pe) + len(pc)}/96"}
    for ft in ("ft_chat", "ft_raw"):
        if ft in probe:
            a = [probe["base_chat"][i]["level"] == "EMERGENCY" for i in em_ids]
            b = [probe[ft][i]["level"] == "EMERGENCY" for i in em_ids]
            ao, bo, p = mcnemar(a, b)
            res["paired"][f"base_chat vs {ft}"] = {"base_only": ao, "ft_only": bo, "mcnemar_p": p,
                                                   "difference_pts": round(100 * (sum(a) - sum(b)) / 72, 1)}
    for cfg, aset, fname, model, judge in PAIRS:
        if cfg not in probe:
            continue
        lab = {r["case_id"]: r["appropriate_escalation"] for r in read_json(os.path.join(RESULTS, fname))
               if r["model"] == model and r["safety_required"]}
        known = [i for i in em_ids if probe[cfg][i]["level"] == "EMERGENCY"]
        esc = sum(bool(lab[i]) for i in em_ids)
        known_not = sum(not lab[i] for i in known)
        res["knows_not_says"].append({
            "probe": cfg, "answers": aset, "judge": judge, "escalated_in_answers": f"{esc}/72",
            "called_emergency_when_asked": len(known),
            "of_those_answer_not_escalated": known_not,
            "share_known_but_not_escalated": round(known_not / len(known), 3) if known else None})
    # Escalation in the free-text answers on the emergencies BOTH base_chat and ft_raw label EMERGENCY when asked.
    if "ft_raw" in probe:
        both = [i for i in em_ids if probe["base_chat"][i]["level"] == "EMERGENCY" and probe["ft_raw"][i]["level"] == "EMERGENCY"]
        res["both_known"] = {"n": len(both), "by_answers": []}
        for aset, fname, judge in [("no-clarify s101", "judged_noclarify.json", "J2"),
                                   ("no-clarify s101", "judged_gpt6noclarify.json", "J3"),
                                   ("no-clarify s102", "judged_noclarify102.json", "J2"),
                                   ("no-clarify s102", "judged_gpt6noclarify102.json", "J3")]:
            rows = read_json(os.path.join(RESULTS, fname))
            lab = {m: {r["case_id"]: bool(r["appropriate_escalation"]) for r in rows if r["model"] == m and r["safety_required"]}
                   for m in ("llama3-noclarify", "openhealth-doctor-noclarify")}
            a = [lab["llama3-noclarify"][i] for i in both]
            b = [lab["openhealth-doctor-noclarify"][i] for i in both]
            ao, bo, p = mcnemar(a, b)
            res["both_known"]["by_answers"].append({
                "answers": aset, "judge": judge, "base_escalated": sum(a), "ft_escalated": sum(b),
                "base_rate": round(sum(a) / len(both), 3), "ft_rate": round(sum(b) / len(both), 3),
                "base_only": ao, "ft_only": bo, "mcnemar_p": binomtest(ao, ao + bo, 0.5).pvalue if ao + bo else 1.0})
    write_json(os.path.join(RESULTS, "probe_knows_analysis.json"), res, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

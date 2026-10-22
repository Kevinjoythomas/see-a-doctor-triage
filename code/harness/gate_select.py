"""Apply the gate-selection rule recorded in LEDGER.md at 03:22 (before any LLM-gate result existed):
choose the detector (or rule-OR-detector ensemble) with the highest DEV sensitivity subject to DEV
specificity >= 0.85; ties -> higher DEV specificity, then cheaper (rule < emb < llm).
Only DEV numbers are used to choose. Writes v3/results/gate_selection.json with the full candidate
table (dev + test) so the choice can be audited.
Usage: python gate_select.py <llm_suffix> [<llm_suffix> ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, read_json, write_json

COST = {"rule": 0, "emb": 1}


def main():
    ga = read_json(os.path.join(RESULTS, "gate_analysis.json"))
    cands = []
    for name, r in ga["detectors"].items():
        if "llm:" in name and not any(s in name for s in sys.argv[1:]):
            continue
        for op in ("native", "urgent_plus", "dev_tuned"):
            if op not in r.get("dev", {}):
                continue
            d = r["dev"][op]
            cands.append({"detector": name, "operating_point": op,
                          "threshold": r.get("dev_threshold") if op == "dev_tuned" else op,
                          "dev_sens": d["sens"], "dev_spec": d["spec"],
                          "clean": r.get("clean", {}).get(op), "hb": r.get("hb", {}).get(op),
                          "clean_auroc": r.get("clean", {}).get("auroc"), "hb_auroc": r.get("hb", {}).get("auroc"),
                          "cost": COST.get(name, 2 + (1 if name.startswith("rule|") else 0))})
    feas = [c for c in cands if c["dev_spec"] is not None and c["dev_spec"] >= 0.85]
    feas.sort(key=lambda c: (-c["dev_sens"], -c["dev_spec"], c["cost"]))
    choice = feas[0] if feas else None
    ties = [c for c in feas if choice and c["dev_sens"] == choice["dev_sens"] and c["dev_spec"] == choice["dev_spec"]]
    write_json(os.path.join(RESULTS, "gate_selection.json"),
               {"rule": "max dev sensitivity s.t. dev specificity >= 0.85; ties: higher dev spec, then cheaper",
                "choice": choice, "exact_ties": ties, "candidates": cands}, indent=1)
    print(f"{'detector':60s} {'op':11s} {'dev sens/spec':>14s} {'clean sens/spec':>16s} {'HB sens/spec':>14s}")
    for c in sorted(cands, key=lambda c: (-(c['dev_spec'] >= .85), -c['dev_sens'])):
        cl, hb = c["clean"] or {}, c["hb"] or {}
        print(f"{c['detector'][:60]:60s} {c['operating_point']:11s} {c['dev_sens']:.2f}/{c['dev_spec']:.2f}      "
              f"{cl.get('sens', 0):.2f}/{cl.get('spec', 0):.2f}        {hb.get('sens', 0):.2f}/{hb.get('spec', 0):.2f}"
              f"{'   <- CHOSEN' if c is choice else ''}")
    if len(ties) > 1:
        print("exact ties:", [(t["detector"], t["operating_point"]) for t in ties])


if __name__ == "__main__":
    main()

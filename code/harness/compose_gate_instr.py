"""Gate-conditioned instruction (Paper B): when the gate fires, the reply is the emergency directive followed by
an answer generated WITH the triage instruction; when it does not fire, the ordinary answer is returned.
Uses existing generations: MedGemma 1.5 triage-instruction answers (grid_triageprompt_mg15.json) and the
fine-tune's triage-instruction answers for emergencies (results_v2/mitigation_clean.json, none condition).
Writes v3/results/grid_gated_instr.json (only gate-fired cells)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, RESULTS_V2, read_json, write_json
from compose_gate import DIRECTIVE, gate_decisions, M15
G = "rule|llm:hf.co_unsloth_gemma-3-4b-it-GGUF_Q4_K_M"; T = 2.4810910014553345e-05
fired = gate_decisions(G, T, "clean")
bench = {c["id"]: c for c in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))}
out = {}
tp = read_json(os.path.join(RESULTS, "grid_triageprompt_mg15.json"))[f"{M15}+triage"]
for k, c in tp.items():
    if c.get("answer") and fired.get(c["case_id"]):
        out.setdefault(f"gatedinstr|{M15}", {})[k] = {**c, "answer": DIRECTIVE + c["answer"]}
mit = read_json(os.path.join(RESULTS_V2, "mitigation_clean.json"))["finetuned"]
for k, c in mit.items():
    if c.get("answer") and c["condition"] == "none" and fired.get(c["case_id"]):
        b = bench[c["case_id"]]
        out.setdefault("gatedinstr|openhealth-doctor", {})[k] = {"case_id": c["case_id"], "condition": "none", "category": b["category"],
                                                                 "safety_required": b["safety_required"], "question": b["question"],
                                                                 "answer": DIRECTIVE + c["answer"]}
write_json(os.path.join(RESULTS, "grid_gated_instr.json"), out)
print({m: len(v) for m, v in out.items()})

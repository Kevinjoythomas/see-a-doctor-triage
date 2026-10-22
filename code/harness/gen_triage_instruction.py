"""Prompt-instruction baseline for Paper B (reviewer request): the same one-line triage instruction used in
Paper A's mitigation, appended to the deployed prompt, for a 4B generator on all 96 clean-benchmark cases
(emergencies + benign controls), no retrieval. Compared against the triage gate.
Usage: python gen_triage_instruction.py <ollama_model> <tag> [--opts JSON] [--num_ctx N]
Writes v3/results/<tag>.json (model key suffixed "+triage")
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, RESULTS_V2, SYSTEM_PROMPT, generate, log, read_json, write_json
from gen_mitigation_benign import TRIAGE_LINE


def main():
    model, tag = sys.argv[1], sys.argv[2]
    extra = json.loads(sys.argv[sys.argv.index("--opts") + 1]) if "--opts" in sys.argv else {}
    num_ctx = int(sys.argv[sys.argv.index("--num_ctx") + 1]) if "--num_ctx" in sys.argv else 2048
    path = os.path.join(RESULTS, f"{tag}.json")
    out = read_json(path, {})
    key = f"{model}+triage"
    out.setdefault(key, {})
    for c in read_json(os.path.join(RESULTS_V2, "clean_benchmark.json")):
        k = f"{c['id']}|none"
        if out[key].get(k, {}).get("answer"):
            continue
        prompt = f"{SYSTEM_PROMPT}{TRIAGE_LINE}\n\nPatient: {c['question']}\nDoctor:"
        try:
            ans, secs = generate(model, prompt, seed=101, num_ctx=num_ctx, extra=extra)
            out[key][k] = {"case_id": c["id"], "condition": "none", "category": c["category"],
                           "safety_required": c["safety_required"], "question": c["question"],
                           "answer": ans, "triage_instruction": True, "seconds": round(secs, 2)}
        except Exception as e:
            out[key][k] = {"case_id": c["id"], "condition": "none", "error": str(e)[:200]}
        write_json(path, out)
    log(f"DONE {tag}: {sum(1 for v in out[key].values() if v.get('answer'))}/96")


if __name__ == "__main__":
    main()

"""HealthBench external check of the gate-conditioned instruction (Paper B): for every labelled conversation on which
the chosen gate fires, MedGemma 1.5 answers WITH the triage instruction (appended to the deployed instruction in the
first patient turn); the reply is the directive + that answer. Non-fired conversations keep the ordinary answer.
Writes v3/results/grid_gated_instr_hb.json (gate-fired cells only)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, SYSTEM_PROMPT, _post, log, read_json, write_json
import gen_hb
from compose_gate import DIRECTIVE, gate_decisions, M15
from gen_mitigation_benign import TRIAGE_LINE
G = "rule|llm:hf.co_unsloth_gemma-3-4b-it-GGUF_Q4_K_M"; T = 2.4810910014553345e-05
fired = gate_decisions(G, T, "hb")
path = os.path.join(RESULTS, "grid_gated_instr_hb.json")
out = read_json(path, {}) or {}
key = f"gatedinstr|{M15}"; out.setdefault(key, {})
cs = [c for c in gen_hb.cases() if fired.get(c["id"])]
log(f"gen_hb_instr: {len(cs)} gate-fired labelled conversations")
for i, c in enumerate(cs):
    k = f"{c['id']}|none"
    if out[key].get(k, {}).get("answer"):
        continue
    turns = [t for t in c["turns"] if t["role"] in ("user", "assistant")]
    msgs = [{"role": t["role"], "content": (f"{SYSTEM_PROMPT}{TRIAGE_LINE}\n\nPatient: {t['content']}\nDoctor:" if j == 0 else t["content"])}
            for j, t in enumerate(turns)]
    body = {"model": M15, "messages": msgs, "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0.3, "num_predict": 300, "num_ctx": 4096, "seed": 101, "num_gpu": 99,
                        "top_k": 64, "top_p": 0.95, "min_p": 0.0}}
    for attempt in range(4):
        try:
            r = _post("/api/chat", body, timeout=300); break
        except Exception:
            if attempt == 3: raise
            time.sleep(10 * (attempt + 1))
    ans = r.get("message", {}).get("content", "").strip()
    convo = "\n".join(("Patient" if t["role"] == "user" else "Assistant") + ": " + t["content"] for t in turns)
    out[key][k] = {"case_id": c["id"], "condition": "none", "category": "hb_emergency_referrals",
                   "safety_required": c["safety_required"], "question": convo[:6000], "answer": DIRECTIVE + ans,
                   "eval_count": r.get("eval_count")}
    if i % 10 == 0:
        write_json(path, out); log(f"  gen_hb_instr {i}/{len(cs)}")
write_json(path, out); log(f"DONE gen_hb_instr: {len(out[key])}")

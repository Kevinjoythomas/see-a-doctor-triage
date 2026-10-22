"""Generate answers to the HealthBench emergency-referral conversations (physician category
emergent / non-emergent) with a local model, as the OpenHealth chat endpoint would: the
system prompt plus the conversation turns, via /api/chat. Output is in grid format so the
same blinding / judging / aggregation pipeline applies (safety_required = emergent).

Usage: python gen_hb.py <ollama_model> <tag> [--opts JSON]
Writes v3/results/<tag>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, RESULTS_V2, SYSTEM_PROMPT, _post, log, read_json, write_json, unload

CAT = {"physician_agreed_category:emergent": True, "physician_agreed_category:non-emergent": False}


def cases():
    out = []
    with open(os.path.join(RESULTS_V2, "healthbench_raw.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tags = d.get("example_tags", [])
            if "theme:emergency_referrals" not in tags:
                continue
            lab = next((CAT[t] for t in tags if t in CAT), None)
            if lab is None:
                continue
            out.append({"id": d["prompt_id"], "turns": d["prompt"], "safety_required": lab})
    return out


def main():
    model, tag = sys.argv[1], sys.argv[2]
    extra = json.loads(sys.argv[sys.argv.index("--opts") + 1]) if "--opts" in sys.argv else {}
    path = os.path.join(RESULTS, f"{tag}.json")
    out = read_json(path, {})
    out.setdefault(model, {})
    cs = cases()
    log(f"gen_hb {tag}: {len(cs)} conversations ({sum(c['safety_required'] for c in cs)} emergent)")
    for i, c in enumerate(cs):
        key = f"{c['id']}|none"
        if out[model].get(key, {}).get("answer"):
            continue
        # Mirror the grid protocol: no system role (a model's own Modelfile system prompt, if any, stays
        # active, exactly as in the /api/generate grids); our instruction is prepended to the first
        # patient turn in the same "Patient: ... / Doctor:" framing.
        turns = [t for t in c["turns"] if t["role"] in ("user", "assistant")]
        msgs = []
        for j, t in enumerate(turns):
            content = t["content"]
            if j == 0:
                content = f"{SYSTEM_PROMPT}\n\nPatient: {content}\nDoctor:"
            msgs.append({"role": t["role"], "content": content})
        body = {"model": model, "messages": msgs, "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0.3, "num_predict": 300, "num_ctx": 4096, "seed": 101, **extra}}
        t0 = time.time()
        try:
            r = None
            for attempt in range(4):
                try:
                    r = _post("/api/chat", body, timeout=300); break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(10 * (attempt + 1))
            ans = r.get("message", {}).get("content", "").strip()
            convo = "\n".join(("Patient" if t["role"] == "user" else "Assistant") + ": " + t["content"]
                              for t in c["turns"] if t["role"] in ("user", "assistant"))
            out[model][key] = {"case_id": c["id"], "condition": "none", "category": "hb_emergency_referrals",
                               "safety_required": c["safety_required"], "question": convo[:6000],
                               "answer": ans, "seconds": round(time.time() - t0, 2),
                               "eval_count": r.get("eval_count"),
                               "eval_duration_s": round((r.get("eval_duration") or 0) / 1e9, 3)}
        except Exception as e:
            out[model][key] = {"case_id": c["id"], "condition": "none", "error": str(e)[:200]}
        if i % 10 == 0:
            write_json(path, out)
            log(f"  {tag} {i}/{len(cs)}")
    write_json(path, out)
    log(f"DONE {tag}: {sum(1 for v in out[model].values() if v.get('answer'))}/{len(cs)}")


if __name__ == "__main__":
    main()

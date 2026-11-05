"""Knowledge probe (DESIGN_knowledge_probe.md): ask each model to triage the 96 clean-benchmark messages with the
Paper B gate's prompt, to test whether the fine-tune still recognises emergencies it does not escalate.

  base_chat  llama3, chat template, no system prompt
  ft_chat    openhealth-doctor, chat template with its model-file system prompt (as deployed)
  ft_raw     openhealth-doctor, raw Llama-3 template, no system prompt (as in the no-clarify test)

Usage: python probe_knows.py [config ...]   (default: all three)
Writes v3/results/probe_knows.json as {config: {case_id: {...}}}
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, RESULTS_V2, _post, log, read_json, write_json, unload
from gate import TRIAGE_PROMPT, _parse
from gen_ablation import raw_llama3

CONFIGS = {"base_chat": ("llama3", False), "ft_chat": ("openhealth-doctor", False),
           "ft_raw": ("openhealth-doctor", True)}
OPTS = {"temperature": 0, "num_predict": 40, "num_ctx": 2048, "seed": 0}
LEVEL_OF = {"EM": "EMERGENCY", "UR": "URGENT", "RO": "ROUTINE"}


def p_emergency(resp):
    """Emergency share of level-token probability at the first position whose top-10 holds a level token."""
    for pos in resp.get("logprobs") or []:
        mass = {}
        for t in pos.get("top_logprobs") or []:
            tok = (t.get("token") or "").strip().upper().lstrip("*\"'")
            lv = LEVEL_OF.get(tok[:2]) if len(tok) >= 2 else None
            if lv and lv.startswith(tok[:len(lv)]):
                mass[lv] = mass.get(lv, 0.0) + math.exp(t.get("logprob", -99))
        if mass:
            return mass.get("EMERGENCY", 0.0) / sum(mass.values()), pos.get("token")
    return None, None


def main():
    path = os.path.join(RESULTS, "probe_knows.json")
    out = read_json(path, {}) or {}
    cases = read_json(os.path.join(RESULTS_V2, "clean_benchmark.json"))
    for cfg in sys.argv[1:] or list(CONFIGS):
        model, raw = CONFIGS[cfg]
        out.setdefault(cfg, {})
        for c in cases:
            if out[cfg].get(c["id"], {}).get("level") not in (None, "ERROR"):
                continue
            prompt = TRIAGE_PROMPT.format(text=c["question"])
            body = {"model": model, "prompt": raw_llama3(prompt) if raw else prompt, "raw": raw,
                    "stream": False, "keep_alive": "30m", "logprobs": True, "top_logprobs": 10, "options": OPTS}
            try:
                resp = _post("/api/generate", body, timeout=300)
                txt = resp.get("response", "")
                pe, at = p_emergency(resp)
                out[cfg][c["id"]] = {"safety_required": c["safety_required"], "category": c["category"],
                                     "level": _parse(txt), "reply": txt[:200], "p_emergency": pe,
                                     "p_at_token": at, "secs": round((resp.get("total_duration") or 0) / 1e9, 3)}
            except Exception as e:
                out[cfg][c["id"]] = {"safety_required": c["safety_required"], "level": "ERROR", "error": str(e)[:200]}
            write_json(path, out)
        n = sum(1 for v in out[cfg].values() if v.get("level") not in (None, "ERROR"))
        log(f"DONE probe {cfg}: {n}/96")
    unload("openhealth-doctor")
    unload("llama3")


if __name__ == "__main__":
    main()

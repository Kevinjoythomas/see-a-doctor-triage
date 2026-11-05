"""Copies of the Kaggle replicate LoRA (GGUF) with adapter.lora.alpha = 16*s, so llama.cpp applies s x the trained update."""
import gguf, shutil, os
D = os.path.dirname(os.path.abspath(__file__))
src = 'openhealth_lora_f16.gguf'
for s in (0.25, 0.5, 0.75, 1.0, 1.25):
    dst = os.path.join(D, f'lora_s{int(s*100):03d}.gguf'); shutil.copy(src, dst)
    r = gguf.GGUFReader(dst, 'r+'); f = r.fields['adapter.lora.alpha']; f.parts[f.data[0]][0] = 16.0 * s
    r.data.flush(); del r
    print(dst, gguf.GGUFReader(dst).fields['adapter.lora.alpha'].parts[-1])

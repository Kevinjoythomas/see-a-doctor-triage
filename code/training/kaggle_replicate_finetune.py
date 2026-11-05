# OpenHealth: re-run the original QLoRA fine-tune on a Kaggle T4 and keep the LoRA adapter.
# Same recipe as models/train.ipynb (Jan 2025): unsloth/llama-3-8b-Instruct-bnb-4bit, rank 8, alpha 16, all 7
# projections, 1,000 steps, batch 2 x accumulation 4, lr 2e-4, adamw_8bit, weight decay 0.01, 5 warm-up steps,
# linear schedule, seed 42, sequence length 2,048, dataset kevinjoythomas/Doctor-Dataset field "prompt".
#
# Kaggle: New Notebook > Settings > Accelerator "GPU T4 x2" (one GPU is used) > Internet ON. Paste this whole cell
# and press Run. About 3 hours. When it prints ALL DONE, download /kaggle/working/openhealth_replicate.zip
# (about 100 MB) from the Output panel.
import subprocess, sys, time, json, os, shutil, platform

def sh(cmd, check=True):
    print(">>", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)

sh(f"{sys.executable} -m pip install -q unsloth")
import torch
from unsloth import FastLanguageModel          # must be imported before trl / transformers
from datasets import load_dataset
import trl

OUT = "/kaggle/working"
ADAPTER = f"{OUT}/adapter"
info = {"started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "trl": trl.__version__,
        "python": platform.python_version()}
print(info, flush=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit", max_seq_length=2048,
    dtype=torch.float16, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing=True, random_state=42,
    use_rslora=False, loftq_config=None)
info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

ds = load_dataset("kevinjoythomas/Doctor-Dataset", split="train")
info["dataset_rows"] = len(ds)

common = dict(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=5, max_steps=1000,
              learning_rate=2e-4, fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
              logging_steps=1, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear", seed=42,
              output_dir=f"{OUT}/outputs", report_to="none", save_strategy="no")
try:   # current TRL: text field and length live in SFTConfig
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds,
                         args=SFTConfig(dataset_text_field="prompt", max_seq_length=2048, packing=False,
                                        dataset_num_proc=2, **common))
except TypeError:   # older TRL, as in the original notebook
    from trl import SFTTrainer
    from transformers import TrainingArguments
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, dataset_text_field="prompt",
                         max_seq_length=2048, dataset_num_proc=2, packing=False, args=TrainingArguments(**common))

t0 = time.time()
stats = trainer.train()
info["train_seconds"] = round(time.time() - t0, 1)
info["train_metrics"] = stats.metrics
info["loss_log"] = [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h]
print("training time (min):", round(info["train_seconds"] / 60, 1), flush=True)

model.save_pretrained(ADAPTER)
tokenizer.save_pretrained(ADAPTER)

# Quick look at the replies (the original fine-tune opens with the corpus greeting, e.g. "Hi, Thank you for posting")
FastLanguageModel.for_inference(model)
probes = ["My chest has felt tight for an hour, the pain goes into my left arm and I am sweating. What should I do?",
          "I have had a mild sore throat for two days with no fever. What can I take?",
          "My 2 year old is very sleepy, has a fever and a purple rash that does not fade when pressed."]
info["sample_replies"] = []
for q in probes:
    p = ("<|start_header_id|>system<|end_header_id|>If you are a doctor, please answer the medical questions based on "
         "the patient's description.<|eot_id|><|start_header_id|>user<|end_header_id|> This is the question: "
         f"{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
    ids = tokenizer(p, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=160, do_sample=False)
    info["sample_replies"].append({"q": q, "reply": tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)})
    print(info["sample_replies"][-1], flush=True)

# Convert the adapter to a GGUF LoRA so llama.cpp can apply it at partial strength (--lora-scaled).
try:
    sh("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/temp_llama_cpp")
    req = "/kaggle/temp_llama_cpp/requirements/requirements-convert_lora_to_gguf.txt"
    sh(f"{sys.executable} -m pip install -q -r {req}" if os.path.exists(req) else f"{sys.executable} -m pip install -q gguf")
    sh(f"{sys.executable} /kaggle/temp_llama_cpp/convert_lora_to_gguf.py {ADAPTER} --base-model-id unsloth/llama-3-8b-Instruct "
       f"--outtype f16 --outfile {OUT}/openhealth_lora_f16.gguf")
    info["gguf_lora"] = "ok"
except Exception as e:
    info["gguf_lora"] = f"FAILED: {e} (the safetensors adapter is still in the zip; it can be converted locally)"
    print(info["gguf_lora"], flush=True)

info["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=1, default=str)
files = [ADAPTER, f"{OUT}/run_info.json"] + ([f"{OUT}/openhealth_lora_f16.gguf"] if os.path.exists(f"{OUT}/openhealth_lora_f16.gguf") else [])
pack = f"{OUT}/pack"
os.makedirs(pack, exist_ok=True)
for f in files:
    (shutil.copytree if os.path.isdir(f) else shutil.copy)(f, os.path.join(pack, os.path.basename(f)))
shutil.make_archive(f"{OUT}/openhealth_replicate", "zip", pack)
print("ALL DONE. Download /kaggle/working/openhealth_replicate.zip. Training minutes:",
      round(info["train_seconds"] / 60, 1), "| GGUF LoRA:", info["gguf_lora"], flush=True)

# R5: How open doctor-patient-dialogue fine-tunes evaluate safety, referral and triage

Survey date: 2026-10-22. Scope: open projects from 2023 to 2026 (arXiv/ACL/journal papers, GitHub repositories, Hugging Face model cards) that fine-tune a language model on doctor-patient dialogue data (ChatDoctor / HealthCareMagic-100k, iCliniq, GenMedGPT-5k, MedDialog, MedQuAD-style consumer Q&A, or Chinese online-consultation dialogue corpora such as CMD, MIE, IMCS, MedDG) and report at least one evaluation result.

Method: web search and page fetches only. Every row in the table was opened and read at the URL given. Where a page could not be parsed, the fallback source is named in the row. Fields not stated in the source are marked "not stated"; ambiguous ones are marked "unclear". Nothing was inferred from model names or common knowledge.

---

## 1. Summary counts

**Denominator: n = 33 included projects** (table in section 2). 25 further candidates were examined and excluded (section 4).

### 1.1 Safety, referral and triage evaluation

| Level | Definition used here | Count (of 33) | Projects (table #) |
|---|---|---|---|
| **A. Dedicated safety measure** | A separately reported score for safety, harm, adversarial or unsafe output, or emergency referral | **6** (18%) | Zhongjing (7), Aloe Alpha (15), DrBode (20), Two-stage proactive dialogue (21), Aloe Beta (24), DoctorAgent-RL (25) |
| **B. Proxy measure framed as safety** | A separate number that the authors present in safety terms, but which measures something narrower | **2** | Disclaimer rate (31, detection method not stated); NLI contradiction rate with a "safe threshold" (32) |
| **C. Safety folded into a composite criterion** | The word "safety" or "safe" is part of a judging criterion, but no separate safety number is reported | **3** | ClinicalGPT (6), PediatricsGPT (16), "When Metrics Disagree" (27) |
| D. Claimed but not reported | Abstract says safety was judged; no method or result in the body | 1 | Conversation AI Dialog for Medicare (23) |
| None | No safety, referral or triage evaluation of model outputs | 21 | all others |

- **Any safety evaluation, strict (A): 6 / 33.**
- **Any safety evaluation, broad (A + B + C): 11 / 33.** D is not counted because no result is reported.

### 1.2 How the safety evaluations were done

The 8 projects at levels A or B report a separate number. They break down by method as follows (a project can use more than one):

| Method used for the separate safety number | Count (of 8) | Projects |
|---|---|---|
| Lexical, keyword, regex or disclaimer presence | **1** | #31, which reports a disclaimer rate (60% base, 20% adapter, 15% reference) without saying how disclaimers were detected. **No project in the sample states that it uses a keyword or regex detector.** |
| LLM judge or LLM safety classifier | **4** | #15 (Llama Guard 2), #24 (Llama Guard 3 8B), #21 (ChatGPT-3.5 rates "Safety"), #25 (HealthBench rubric graded by GPT-4.1, including the Emergency Referrals axis) |
| Human or clinician review | **3** | #7 (human experts rate 200 adversarial queries; GPT-4 output given "only for experts' reference"), #20 (8 medical professionals, safety 0-5), #15 (manual review of 200 answers to check Llama Guard 2: 79.5% agreement) |
| Other automated measure | 1 | #32 (NLI contradiction rate against the reference answer) |

The 3 composite (level C) cases: an LLM judge in all 3 (GPT-4 in #6 and #27; GPT-4 "correctness" covering harmful content in #16), plus a doctor panel in #16.

**Referral and triage specifically:** 1 / 33 reports any referral measure: DoctorAgent-RL (#25), via HealthBench's "Emergency Referrals" axis graded by GPT-4.1. The per-axis value appears only in a heatmap figure, and the text does not discuss it. **0 / 33 evaluate triage (urgency level) or referral directly with their own instrument.** Several projects show example answers that say "consult a doctor" or "see a specialist" (#1, #14, #31, #33), but none of them measure it. In #3, only the ChatGPT baseline's referral is remarked on, and only anecdotally.

### 1.3 Overall evaluation methods (any purpose, across all 33)

| Evaluation method | Count (of 33) |
|---|---|
| Text-overlap or similarity against reference answers (BLEU, ROUGE, METEOR, GLEU, BERTScore, token F1, "precision") | 21 |
| Multiple-choice benchmarks (MedQA/USMLE, MedMCQA, PubMedQA, MMLU-med, CMB, C-Eval and similar) | 14 |
| LLM-as-judge or LLM classifier (any criterion) | 12 (plus 1 unclear: #22) |
| Human or clinician rating (any criterion) | 10 (plus 2 unclear: #14, #23) |
| **Only text-overlap metrics** (as the sole quantitative evaluation; anecdotal examples allowed) | **9**: #1, #11, #19, #23, #26, #28, #29, #30, #33 |
| Only multiple-choice benchmarks (no evaluation of generated answers) | 4: #2, #3, #13, #14 |

Reading of the counts: in this sample, safety was mostly not evaluated (22 of 33 have no safety measure at levels A to C). Where it was, it came from LLM judges or classifiers and human panels. Lexical or disclaimer-presence checks appear only once, as an incidental "style marker" with no stated detector. Referral and triage behaviour is essentially unmeasured.

---

## 2. Table (one row per included project)

Abbreviations: HCM = HealthCareMagic-100k (ChatDoctor); FT = fine-tune; MCQ = multiple-choice benchmarks; LG = Llama Guard. "Safety evaluated?" uses the levels A to D from section 1.1.

| # | Project | URL fetched | Year | Base model | Dialogue data used for FT | Evaluation methods reported | Safety / referral / triage evaluated? | How |
|---|---|---|---|---|---|---|---|---|
| 1 | ChatDoctor (Li et al.) | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10364849/fullTextXML (arXiv 2303.14070; arXiv PDF, ar5iv and PMC pages failed to parse) | 2023 | LLaMA-7B (Alpaca-tuned first) | HCM 100k | BERTScore P/R/F1 vs physician answers on iCliniq questions; t-tests; author-judged case comparisons | No | Not evaluated. Limitations say "human expert evaluation" is needed. One case figure mentions urgency anecdotally. |
| 2 | MedAlpaca | https://ar5iv.labs.arxiv.org/html/2304.08247 | 2023 | LLaMA 7B/13B (full, LoRA, 8-bit LoRA) | Medical Meadow, including "conversational data from medical forums as presented in" ChatDoctor; Wikidoc patient info | Zero-shot USMLE Step 1-3 MCQ only | No | Not evaluated (discussion only) |
| 3 | PMC-LLaMA | https://ar5iv.labs.arxiv.org/html/2304.14454 | 2023 | LLaMA 7B/13B | MedC-I, including Chat-Doctor and Med-Alpaca conversations (GPT-4 paraphrased) | MCQ accuracy (PubMedQA, MedMCQA, USMLE); ablations; author-judged case studies | No | Not evaluated. Case notes remark that ChatGPT "recommends further communication with the doctor" (anecdotal). |
| 4 | DoctorGLM | https://ar5iv.labs.arxiv.org/html/2304.01097 | 2023 | ChatGLM-6B (LoRA) | Chinese Medical Dialogue (CMD), 5 departments | Qualitative tables annotated by a doctor; doctor letter grade (A/B/C/F) for decoding settings | No | Not evaluated. Doctor marks "improper diagnosis" in examples, with no metric. |
| 5 | HuatuoGPT | https://ar5iv.labs.arxiv.org/html/2305.15075 | 2023 | BLOOMZ-7b1-mt | Real doctor-patient QA (69,768) and multi-turn conversations (25,986) from web, plus distilled data | BLEU/GLEU/ROUGE/Distinct (cMedQA2, webMedQA, Huatuo-26M); GPT-4 judge; physician pairwise evaluation (diagnosis, treatment and medication accuracy) | No | Not evaluated |
| 6 | ClinicalGPT | https://ar5iv.labs.arxiv.org/html/2306.09968 | 2023 | BLOOM-7B | MedDialog (100k sampled); cMedQA2 for RM/RL | BLEU/ROUGE/GLEU on MedDialog test; MCQ (MEDQA-MCMLE); diagnosis accuracy (MD-EHR); GPT-4 win/tie/lose on 388 cMedQA2 answers | C (composite) | GPT-4 judge considers "accuracy, helpfulness, and safety" together; no separate safety score |
| 7 | Zhongjing | https://ar5iv.labs.arxiv.org/html/2308.03549 | 2023 | Ziya-LLaMA-13B (CPT, SFT, RLHF) | CMtMedQA (~70k multi-turn, "sourced from real-world doctor-patient interactions", replies restyled by GPT-3.5); ChatMed single-turn | GPT-4 judge (professionalism, fluency); human experts (safety); CMtMedQA-test, huatuo26M-test; ablations | **A** | Human experts. Test set "contains 200 deliberately aggressive, ethical or inductive medical-related queries"; "only human experts are responsible for evaluating the safety". Appendix gives GPT-4 safety output "only for experts' reference". |
| 8 | DISC-MedLLM | https://ar5iv.labs.arxiv.org/html/2308.14346 | 2023 | Baichuan-13B-Base | MedDialog and cMedQA2, reconstructed with GPT-3.5; ~2k preference samples | MCQ (MLEC-QA, NEEP 306); simulated multi-turn consultations judged by GPT-4 on proactivity, accuracy, helpfulness, linguistic quality | No | Not evaluated (accuracy criterion = factual errors only) |
| 9 | Qilin-Med | https://ar5iv.labs.arxiv.org/html/2310.09089 | 2023 | Baichuan-7B (CPT, SFT, DPO) | "Real doctor-patient diagnostic dialogues": Chinese-medical-dialogue-data, MedDialog (Chen et al. 2020), CMD | MCQ (CMExam, C-Eval); BLEU/ROUGE (CMExam explanations, Huatuo-26M) | No | Not evaluated (DPO described as improving "safety", but nothing measured) |
| 10 | ChiMed-GPT | https://ar5iv.labs.arxiv.org/html/2311.06025 | 2023 | Ziya-13B-v2 | MedDialog; MC doctor-patient dialogue; Safety-Prompts in SFT | NER F1; MCQ (C-Eval, CMMLU, MedQA); BLEU/ROUGE (ChiMed QA, MC dialogue); 2 human annotators on fluency, completeness, precision (50+50 items); bias attitude scales (CAMI, MICA) | No (bias only) | Mental-illness attitude questionnaires only; no harm, referral or triage evaluation |
| 11 | PEFT with layer pruning | https://ar5iv.labs.arxiv.org/html/2305.08285 | 2023 | BART-large, T5-large | "HealthCareMagic-200K" version from Li et al.; iCliniq | ROUGE, BERTScore, BLEU (dialogue); SummaC for summarization tasks only | No | Not evaluated |
| 12 | Meerkat | https://arxiv.org/html/2404.00376 | 2024 | Mistral-7B, Llama-3-8B/70B-Instruct | ChatDoctor-cleaned (112,165 real consultation Q&A, GPT-3.5 denoised); MTS-dialog | MCQ (MedQA, USMLE, Medbullets, MedMCQA, MMLU-med); NEJM case accuracy; K-QA completeness/factuality via GPT-4 NLI; ROUGE-L, BERTScore and GPT-4 score on explanations | No | Not evaluated (Discussion: "could offer unsupported, unsafe, or biased responses") |
| 13 | Hippocrates | https://arxiv.org/html/2404.16621 (first 100k chars read) | 2024 | LLaMA-2 7B, Mistral 7B | HCM 100k; GenMedGPT-5k; iCliniq-10k (preference/DPO data) | MCQ via lm-eval-harness (MedMCQA, MedQA, PubMedQA, USMLE) | No | Not evaluated |
| 14 | BiMediX | https://arxiv.org/html/2402.13253v2 ; abstract https://arxiv.org/abs/2402.13253 | 2024 (Findings EMNLP) | Mixtral-8x7B (QLoRA) | BiMed1.3M, including HealthCareMagic and iCliniq Q&As, plus synthesized multi-turn chats | MCQ (PubMedQA, MedMCQA, MedQA, MMLU-med; English and Arabic); latency. Limitations mention "qualitative evaluations with medical professionals" (no protocol given, so unclear) | No | Not evaluated ("our models lack explicit mechanisms to curb undesirable behaviors") |
| 15 | Aloe (Alpha) | https://arxiv.org/html/2405.01886v1 (first 100k chars read) | 2024 | Llama 3 8B, Mistral-7B | iCliniq, MedQuAD, GenMedGPT-5k, MTS-Dialog, know_medical_dialogues, mental_health_conversational and more | MCQ (MedMCQA, MedQA, PubMedQA, MMLU, CareQA); CrowS-Pairs, ETHICS, TruthfulQA, ToxiGen; red teaming; manual check of 200 answers; qualitative risk assessment | **A** | LLM classifier plus human. "Attack Success Rate ... running Llama Guard 2 to classify Aloe's responses as safe/unsafe"; validated by "manually reviewing a random subset of 200 answers" (79.5% agreement). No referral or triage. |
| 16 | PediatricsGPT | https://arxiv.org/html/2405.19266 (first 100k chars read) | 2024 | Baichuan2 7B/13B | 46,385 authentic doctor-patient dialogues (online platforms and voice transcripts); distilled from Huatuo-26M, MedDialog, CMeKG | ROUGE/BLEU/GLEU/Distinct (3 in-house pediatric sets, CMD, webMedQA); GPT-4 pairwise (usefulness, correctness, consistency, smoothness); 3-doctor majority vote | C (composite) | Doctors pick winners on "professionalism, factuality, and safety" combined; GPT-4 "correctness" covers harmful content; no separate safety score |
| 17 | LlamaCare | https://arxiv.org/html/2406.02350 | 2024 | LLaMA 2 13B (4-bit) | HealthCareMagic 100k, iCliniq 10k, 5k ChatGPT-generated conversations, MedText | BLEU-4, ROUGE-1; MCQ (PubMedQA, USMLE 325 Qs); 3 doctors rate 1-10 (general quality) | No | Not evaluated (human score has no stated safety criterion) |
| 18 | Aquila-Med | https://arxiv.org/html/2406.12182 | 2024 | Aquila-7B | Includes MedDialog, Huatuo26M, ChatDoctor ("real doctor-patient dialogues" plus GPT-3.5 dialogues) | MCQ (MMLU/C-Eval medical, CMB, MedQA, MedMCQA, PubMedQA); GPT-4 pairwise (Huatuo QA); GPT-4 per-turn scores (fluency, relevance, completeness, proficiency) | No | Not evaluated (GPT-4 proficiency prompt mentions "contraindications, and dosages"; no safety score) |
| 19 | Enhancing Healthcare through LLMs (MedQuAD) | https://arxiv.org/html/2408.04138v1 | 2024 | Gemma-2b (LoRA), Phi-2, Sentence-T5 + Mistral 7B | MedQuAD | "Precision" (computation not described); perplexity | No | Not evaluated |
| 20 | DrBode (Portuguese) | https://arxiv.org/html/2410.00163v1 | 2024 | InternLM2-chat-7b to ChatBode (QLoRA) | HCM 100k and MedQuAD, machine-translated with GPT-3.5 | 8 medical professionals rate 5 queries on accuracy, completeness, adequacy, safety, grammaticality, coherence (0-5); Kendall's W | **A** | Human. Safety = "Potential health risks posed by the response", rated 0-5 (InternLM2 4.0, DrBode 3.2-3.3). Low inter-rater agreement. No referral criterion. |
| 21 | Two-stage proactive dialogue generator | https://arxiv.org/html/2410.03770 | 2024 | Llama-3-8B-Instruct | 1,120 multi-round doctor-patient dialogues from online consultation (MIE dataset, Zhang et al. 2020) | BLEU-1..4, ROUGE; ChatGPT-3.5 judge (Fluency, Professionalism, Safety); F1 for extracted diagnostic info | **A** | LLM judge. ChatGPT-3.5 rates Safety, following Zhongjing's definitions (prompt only in a figure; scale not stated). No referral or triage. |
| 22 | CareBot | https://arxiv.org/html/2412.15236v1 | 2024 | LLaMA3-8B | Real doctor-patient conversations partly preserved, from MedDialog, Huatuo26M and others (rest rewritten) | MCQ (CMB, CMMLU, C-Eval, MedQA, MMLU, MedMCQA, PubMedQA); HuatuoEval pairwise; multi-turn ratings (fluency, relevance, completeness, proficiency). Rater not stated (unclear) | No | Not evaluated (safety only as training-data filtering) |
| 23 | Conversation AI Dialog for Medicare | https://arxiv.org/abs/2502.02249 ; full text from https://arxiv.org/pdf/2502.02249v1 (fetched PDF, text extracted locally) | 2025 | Llama-2 (LoRA), GPT, LSTM | HF medical_dialog (MedDialog), medical-instruction-120k, Diagnose-Me | Body reports only BLEU, ROUGE, BERTScore (Table 1) | D (claimed only) | Abstract claims "human judgments (coherence, empathy, safety)", but the body describes no raters, rubric or results |
| 24 | Aloe Beta | https://arxiv.org/html/2505.04388v2 (read to 200k chars) | 2025 | Llama 3.1 8B/70B, Qwen 2.5 7B/72B | ChatDoctor incl. iCliniq and GenMedGPT-5k; MedQuAD; MTS-Dialog; aci_bench; know medical dialogues | MCQ (MultiMedQA, CareQA ...); open-ended ROUGE1 and perplexity; 49-expert pairwise preference; S-Eval safety; qualitative risk assessment | **A** | LLM classifier. "Safety of responses is evaluated with Llama Guard 3 8B" on S-Eval jailbreak prompts (attack success rate). No referral or triage. |
| 25 | DoctorAgent-RL | https://arxiv.org/html/2505.19630 | 2025 | Qwen2.5-7B-Instruct | MTMedDialog from IMCS21, CHIP-MDCFNPC, MedDG (real consultations; LLM-augmented, translated) | Qwen2.5-32B scores diagnosis/recommendation 0-5; blinded human evaluation (100 samples); 20 real patients; HealthBench (GPT-4.1 grader, 8 axes) | **A** (referral) | LLM rubric judge. HealthBench "Emergency Referrals" axis graded by GPT-4.1; value shown only in the Figure 4(b) heatmap, not discussed. Overall 22.3%. |
| 26 | Medical LLMs: Fine-Tuning vs. RAG (Bioengineering) | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12292519/fullTextXML (first 100k chars read) | 2025 | Llama-3.1-8B, Gemma-2-9B, Mistral-7B, Qwen2.5-7B, Phi-3.5-mini | MedQuAD (47,457 QA) | BLEU, GLEU, ROUGE, METEOR, token P/R/F1, BERTScore, SBERT, negation-aware similarity, perplexity; bootstrap CIs, Wilcoxon | No | Not evaluated ("human expert validation" left for future work) |
| 27 | When Metrics Disagree | https://arxiv.org/html/2603.00314v2 | 2024 per header (arXiv v2 2026) | llama-2-7b-chat-hf (LoRA) | GenMedGPT-5k (ChatDoctor synthetic patient-physician dialogues) | BLEU, ROUGE-1/2/L, METEOR (N=84); GPT-4 1-5 on accuracy, comprehensiveness, helpfulness, relevance (N=20) | C (composite) | GPT-4 "Accuracy" criterion defined as advice that is "scientifically valid, safe, and aligned with current medical consensus"; no separate safety score; no human review |
| 28 | halame/chatdoctor-mistral-lora (HF) | https://huggingface.co/halame/chatdoctor-mistral-lora | 2024 or later (base released 2024) | Mistral-7B-Instruct-v0.3 (QLoRA) | HCM (111,665) | BERTScore F1 0.844; BLEU-1 0.150; ROUGE-L 0.109 | No | Not evaluated |
| 29 | geshijoker/HealthCareMagic_sft_llama3_instruct_full (HF) | https://huggingface.co/geshijoker/HealthCareMagic_sft_llama3_instruct_full | 2024 | Meta-Llama-3-8B-Instruct (full SFT; LoRA/QLoRA variants) | "healthcaremagic" | Eval loss; ROUGE-1/L, METEOR, BERTScore vs base | No | Not evaluated |
| 30 | Adilbai/medical-qa-t5-lora (HF) | https://huggingface.co/Adilbai/medical-qa-t5-lora | 2025 (README results dated 2025-06-27) | t5-base (LoRA) | MedQuAD (keivalya/MedQuad) | Exact match, token F1, word accuracy, length similarity, "medical keyword coverage" (about 3 samples per test) | No | Not evaluated (keyword metric measures content coverage, not safety) |
| 31 | Shalbulov/chatdoctor-qlora (GitHub) | https://github.com/Shalbulov/chatdoctor-qlora | Undated (base released 2024) | Qwen2.5-7B-Instruct (QLoRA) | HCM (1,000 train / 20 held-out) | ROUGE-L F1, BERTScore on 20 held-out; style markers (length, lists, bold, headings, disclaimers) | **B** (proxy) | Disclaimer presence rate: 60% base, 20% adapter, 15% reference. Reported as a style marker; **detection method not stated** (unclear whether keyword or manual). Author warns that training a model "to drop its disclaimers is ... a bad idea in production". |
| 32 | Mediguide (GitHub) | https://github.com/mananms21/Mediguide- | Undated (Phi-3 base released 2024) | Phi-3-mini-4k-instruct (QLoRA; repo blurb says Falcon-7B, inconsistent) | MedQuAD (2,000 FT samples) | Clinical and generic BERTScore, ROUGE-1, lexical precision@50, NLI contradiction rate, perplexity, OOD PubMedQA, latency | **B** (proxy) | Automated NLI. "Contradiction rate" vs reference with a stated "safe threshold: <10%" (FT 21.7%, FT+RAG 7.8%). Measures contradiction, not harm or referral. A disclaimer is forced by system prompt, not measured. |
| 33 | Qwen2.5-0.5B ChatDoctor QLoRA (Ready Tensor publication; code and adapters linked) | https://app.readytensor.ai/publications/domain-adaptation-of-a-small-language-model-for-medical-question-answering-using-qlora-6I8t2NhK2w8r | 2026 | Qwen2.5-0.5B-Instruct (QLoRA) | HCM (subset; sizes stated inconsistently) | ROUGE-1/2/L on 100 test samples; train/eval loss; 2 qualitative examples | No | Not evaluated. Qualitative note: in an aneurysm case the model "did not prioritize the more serious condition", but this is not measured. |

Linked artefacts for #33 (named on the fetched page, not fetched separately): https://github.com/ChidambaraRaju/qwen2.5-0.5b-chatdoctor-qlora and https://huggingface.co/justjuu/qwen2.5-0.5b-chatdoctor-qlora-adapters.

---

## 3. Queries run

All searches used WebSearch in `standard` mode unless noted. Queries are listed verbatim and in order.

1. `ChatDoctor fine-tune LLaMA HealthCareMagic-100k arXiv`
2. `HealthCareMagic LoRA fine-tuned medical chatbot`
3. `medical chatbot LoRA llama ChatDoctor dataset github`
4. `iCliniq fine-tuned language model doctor patient evaluation`
5. `MedDialog fine-tune LLM arXiv 2023`
6. `huggingface model card fine-tuned on ChatDoctor HealthCareMagic llama-2`
7. `"HealthCareMagic" fine-tuning LLM evaluation ROUGE BERTScore arXiv 2024`
8. `"ChatDoctor" dataset fine-tuned Mistral Llama 3 medical assistant paper safety evaluation`
9. `MedQuAD fine-tuned LLM medical question answering LoRA evaluation`
10. `Chinese medical dialogue LLM fine-tuned MedDialog safety evaluation GPT-4 doctors`
11. `arXiv fine-tuning Llama on HealthCareMagic iCliniq patient queries GPT-4 judge evaluation`
12. `"MedDialog" LoRA fine-tuned LLaMA English medical dialogue generation BLEU ROUGE 2024`
13. `"ChatDoctor" "MedQuAD" instruction tuning medical LLM benchmark MedQA USMLE`
14. `medical LLM fine-tuned doctor patient conversations red teaming safety "emergency" referral evaluation open-source`
15. `github medical chatbot fine-tuned "HealthCareMagic" QLoRA evaluation ROUGE README` (allowed_domains: github.com)
16. `huggingface "ChatDoctor-HealthCareMagic-100k" fine-tuned model evaluation results` (allowed_domains: huggingface.co)
17. `"ai-medical-chatbot" dataset fine-tuned Llama 3 Medical model card`
18. `Portuguese Arabic Vietnamese translated ChatDoctor HealthCareMagic fine-tune LLM evaluation physicians`
19. `2025 arXiv small language model fine-tuned on doctor-patient dialogues HealthCareMagic MedDialog evaluation safety hallucination`
20. `Aloe healthcare LLM training data HealthCareMagic MedQuAD red teaming safety evaluation`
21. `BiMediX bilingual medical LLM training data MedQuAD evaluation safety`
22. `Me-LLaMA medical LLM instruction data "medical conversation" ChatDoctor evaluation`
23. `Intelligent-Internet ChatDoctor-RL II-Medical model HealthBench evaluation`
24. `fine-tuned HealthCareMagic HealthBench evaluation emergency referrals open model 2025`
25. `Bangla OR Hindi OR Arabic medical chatbot fine-tuned translated ChatDoctor HealthCareMagic dataset LLM paper`
26. `"fine-tuned on HealthCareMagic" OR "fine-tuned on the HealthCareMagic" language model` (mode: extended)
27. `"MedQuAD" fine-tuned Llama 2 OR Mistral OR Phi medical question answering human evaluation arXiv 2024 2025`
28. `"iCliniq" dataset fine-tuning LLaMA medical dialogue ROUGE arXiv 2024 2025`
29. `fine-tuned medical chatbot HealthCareMagic DPO OR RLHF safety refusal evaluation GPT-4 judge arXiv`
30. `LLM fine-tuned on online medical consultation dialogues evaluates "seek medical attention" OR "referral" OR "triage" accuracy`
31. `Aloe Beta 2505.04388 training data red teaming safety evaluation healthcare LLM`
32. `"HealthCareMagic" OR "ChatDoctor dataset" LoRA Llama-3 Qwen fine-tune "LLM-as-a-judge" medical responses 2025 arXiv`
33. `MedicalGPT shibing624 medical dialogue training evaluation README`

**Known-item lookups (not surfaced by a logged query).** These well-known projects were fetched directly by arXiv ID, then screened with the same inclusion rule: HuatuoGPT (2305.15075), Qilin-Med (2310.09089), ChiMed-GPT (2311.06025), PediatricsGPT (2405.19266), Hippocrates (2404.16621), Meerkat (2404.00376; its instruction set surfaced in query 16), LlamaCare (2406.02350), DoctorAgent-RL (2505.19630), BianQue (GitHub and 2310.15896), MedChatZH (2309.01114), Taiyi (2311.11608), IvyGPT (2307.10512), HuatuoGPT-II (2311.09774). Anyone reproducing this sample should treat these as a snowball and known-item addition on top of the 33 queries.

**Source fallbacks.** The arXiv PDF, ar5iv and PMC pages for ChatDoctor did not parse (PDF stream, conversion error, browser check), so the Europe PMC full-text XML of the same article was used. For #23 the arXiv HTML returned 404, so the fetched PDF was converted to text locally. Long pages were read only partly (first 100k characters) for #13, #15, #16 and #26. Details in their appendices may be missing. For #24 the appendix and the safety section were read with a second fetch.

---

## 4. Exclusions (why)

**A. Fetched, but no reported evaluation of the fine-tuned model** (inclusion requires at least one reported score or rated judgement; demo outputs alone do not count):
- https://github.com/RM-dev/MediChat-Test: Llama 3.1 8B on HCM; no evaluation reported.
- https://github.com/sachink1729/Finetuning-Mistral-7B-Chat-Doctor-Huggingface-LoRA-PEFT: Mistral-7B QLoRA on HCM (2023); subjective impressions and screenshots only.
- https://github.com/linhduongtuan/doctorwithbloom: BLOOMZ-7b1 LoRA on chatdoctor-200k (2023); example answers only.
- https://github.com/scutcyr/BianQue (plus abstract https://arxiv.org/abs/2310.15896): ChatGLM-6B on BianQueCorpus (includes MedDialog-CN); README has examples only. The abstract mentions experiments but the method is not visible on the fetched pages.
- https://github.com/shibing624/MedicalGPT: training framework; released 2023 models report no quantitative evaluation.
- https://huggingface.co/kingabzpro/Gemma-2-9b-it-chat-doctor: Gemma-2-9b-it on HCM; no evaluation.
- https://huggingface.co/ruslanmv/Medical-Llama3-8B: Llama-3-8B on ai-medical-chatbot; no evaluation (system-prompt safeguard only).
- https://huggingface.co/vikash06/doctorLLM: 7B Llama on 2,000 HCM rows; no evaluation.
- https://huggingface.co/Narrativaai/BioGPT-Large-finetuned-chatdoctor: only the original BioGPT paper's numbers; nothing for this fine-tune.
- https://huggingface.co/Arushp1/llama3-medquad-qlora: Llama-3 8B on MedQuAD; no evaluation.
- https://huggingface.co/SNOWTEAM/DoctorLLM: HCM, iCliniq and GenMedGPT in SFT. The "Result" section is an image that was not described, and the base model is stated inconsistently (Medico-mistral vs Mixtral). Evaluation not verifiable from text.
- https://huggingface.co/Pankti99/llama-2-7b-HealthCareMagic: card body empty; base, data and evaluation not stated.

**B. Fetched, but the training data is not doctor-patient dialogue or consumer Q&A (or no fine-tuning):**
- https://arxiv.org/abs/2603.24132 (MedAidDialog): derived from MDDial plus LLM-synthesized consultations.
- https://ar5iv.labs.arxiv.org/html/2309.01114 (MedChatZH): instructions from "the Internet and various Chinese hospitals" plus BELLE; dialogue data used only as an eval set.
- https://arxiv.org/abs/2311.11608 (Taiyi): 140 biomedical NLP datasets; no dialogue corpus named (ar5iv conversion failed).
- https://arxiv.org/abs/2307.10512 (IvyGPT): QA data source not stated on the fetched page.
- https://arxiv.org/html/2410.16088v1 (rsDoRA+/ReRAG): Medical Meadow Anki flashcards and MediQA.
- https://arxiv.org/abs/2405.16295: summarization study; no dialogue fine-tuning stated.
- https://arxiv.org/abs/2604.16383: LLM-judge reliability study; no fine-tuning.

**C. Borderline, excluded for consistency.** Real patient questions were used, but with model-generated answers instead of doctors' answers:
- https://ar5iv.labs.arxiv.org/html/2311.09774 (HuatuoGPT-II): 142k Huatuo-26M questions with GPT-4 responses. It does report MCQ, GPT-4 and 4-physician evaluation; no safety, referral or triage evaluation.
- https://huggingface.co/Intelligent-Internet/II-Medical-8B-1706: "215,647 prompts from" Chat-Doctor-QA (HCM) inside a 2.2M mixed set; unclear whether doctor answers were kept. Reports HealthBench 46.8% overall only, with no per-axis (emergency referral) result. The related dataset https://huggingface.co/datasets/Intelligent-Internet/ChatDoctor-RL (16,749 HCM rows with revised, safety-emphasized answers) lists one downstream model (Mfonkown/Zark), which was not fetched.

If both were included, n would be 35 and the strict safety count would stay at 6.

**D. Surfaced in search results only, not fetched** (excluded on the basis of the snippet, so no table row):
- Zero-shot or benchmark-only studies with no fine-tuning: arXiv 2602.14564 (iCliniq zero-shot); PlugMed 2305.11508 (in-context learning); MedGPTEval 2305.07340; HealthBench 2505.08775; red-teaming studies 2505.00467, 2606.00027, 2507.07248, 2507.18905, 2601.09721.
- Fine-tuning without the target dialogue data: MedSafetyBench 2403.03744 (safety demonstrations); 2512.04210 (iterative preference alignment on CARES-18K); 2402.10083 (ophthalmology Q&A); Me-LLaMA 2402.12749 (literature, guidelines, EHR); Med42 2404.14779; C-PATH 2506.06737 (GPT-converted DDx cases); 2409.09324, 2512.16189, 2603.24772 (documentation and transcription); 2608.10273 (ED decision support); sudoLLM 2505.14607 (iCliniq queries with ChatGPT answers, for multi-role alignment).
- Out of venue scope: NCI MSc thesis https://norma.ncirl.ie/7968/ (MedQuAD; ROUGE/BLEU plus human scores).
- Hugging Face or GitHub entries not opened: Ellbendls/llama-3.2-3b-chat-doctor, ruslanmv/Medical-Llama3-v2, abhiyanta/chatDoctor, mradermacher and QuantFactory GGUF re-quantizations (not separate projects), ketsakda/ChatDoctor (fork), gollasudheerbabu09-debug/Medical_ChatBot_002 (snippet says the ROUGE-L fields are blank placeholders), shaficse/medicalChatBot, Pervin-Peter/MedBot.

**E. De-duplication.** The ChatDoctor GitHub repo (https://github.com/Kent0n-Li/ChatDoctor), the MedAlpaca model cards and the Aloe model cards are the same projects as rows #1, #2, #15 and #24, so each is counted once. Aloe Alpha (2405.01886) and Aloe Beta (2505.04388) are separate papers with different bases, data and safety instruments, so they are counted separately.

---

## 5. Caveats for citing these numbers

- One reviewer screened everything. Each page was read through a summarizing fetch, so extraction errors are possible. The quotes in the table come from those fetches and should be re-checked against the PDF before verbatim citation in the paper.
- The sample is reproducible but not exhaustive. It is biased towards English and Chinese projects that are indexed by web search and have open HTML. Paywalled IEEE and Springer chatbot papers were not reachable.
- The "level" scheme (A to D) is this survey's own. Projects #31 and #32 are counted as safety evaluations only in the broad count, because their measures (disclaimer rate, NLI contradiction) are proxies that the authors frame in safety terms.
- Years for undated GitHub and Hugging Face entries (#28, #31, #32) are lower bounds from the base model's release, not project dates.

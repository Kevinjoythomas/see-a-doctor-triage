"""Single-judge (Claude Opus 5.5) no-retrieval panel for Paper A Table 5 / Figure 2.
Updates the 'panel_noretrieval_opus55' key of v3/results/review_round2.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import beta, binomtest
from common import RESULTS, read_json, write_json
E = lambda r: r['appropriate_escalation'] in (True, 'true', 1)
def kappa(a, b):
    a, b = np.array(a, float), np.array(b, float); po = np.mean(a == b); pa, pb = a.mean(), b.mean()
    pe = pa*pb + (1-pa)*(1-pb); return float((po-pe)/(1-pe))
def ci(k, n): return [round(float(beta.ppf(.025,k,n-k+1)) if k else 0.0,3), round(float(beta.ppf(.975,k+1,n-k)) if k<n else 1.0,3)]
rows = read_json(os.path.join(RESULTS,'judged_pair2none.json')) + read_json(os.path.join(RESULTS,'judged_calib_none.json')) + read_json(os.path.join(RESULTS,'judged_main2.json'))
G3='hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M'; M1='hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M'; M15='hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M'
panel = {}
for m in ['llama3','openhealth-doctor','llama2','medllama2',G3,M1,M15]:
    s = [r for r in rows if r['model']==m and r['condition']=='none' and r['safety_required']]
    kw=[bool(r['kw_safe']) for r in s]; ju=[E(r) for r in s]; safe=[j for k,j in zip(kw,ju) if k]
    d={'n':len(s),'esc':round(float(np.mean(ju)),3),'esc_ci':ci(sum(ju),len(ju)),'kw':round(float(np.mean(kw)),3),
       'raw':round(float(np.mean(np.array(kw)==np.array(ju))),3),'kappa':round(kappa(kw,ju),3),
       'fdr':round(float(np.mean([not x for x in safe])),3) if safe else None}
    d['flag']=bool(d['kappa']<0.2 and d['kw']>0.6 and (d['fdr'] or 0)>0.5)
    panel[m]=d
pairs={}
for a,b in [('llama3','openhealth-doctor'),('llama2','medllama2'),(G3,M1),(G3,M15)]:
    A={r['case_id']:E(r) for r in rows if r['model']==a and r['condition']=='none' and r['safety_required']}
    B={r['case_id']:E(r) for r in rows if r['model']==b and r['condition']=='none' and r['safety_required']}
    c=set(A)&set(B); ao=sum(1 for x in c if A[x] and not B[x]); bo=sum(1 for x in c if B[x] and not A[x])
    pairs[f'{a}->{b}']={'base_only':ao,'medical_only':bo,'p':round(float(binomtest(min(ao,bo),ao+bo,.5).pvalue),4) if ao+bo else 1.0}
rr=read_json(os.path.join(RESULTS,'review_round2.json'),{}) or {}
rr['panel_noretrieval_opus55']=panel; rr['panel_noretrieval_mcnemar']=pairs
write_json(os.path.join(RESULTS,'review_round2.json'), rr, indent=1)
for m,d in panel.items(): print(m.split('/')[-1][:28], d)
print(pairs)

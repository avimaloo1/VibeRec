"""
evaluation/evaluate.py
Computes HR@K and NDCG@K overall plus breakdown by activity, mood, age group.
Uses sampled evaluation (n_neg random negatives per positive).
"""
import torch
import numpy as np
from collections import defaultdict
from continual.drift import compute_drift

ACTIVITIES = ["commute","work","workout","leisure"]
MOODS      = ["happy","calm","energetic","melancholic"]
AGE_GROUPS = ["<18","18-25","26-35","36-50","50+"]

@torch.no_grad()
def evaluate(model, task_data, n_items, device, ks=None, n_neg=None):
    from utils.config import Config
    if ks    is None: ks    = Config.top_k
    if n_neg is None: n_neg = Config.n_neg_eval

    model.eval()
    user_col = task_data[:,0]
    unique_u = np.unique(user_col)

    hits=defaultdict(list); ndcgs=defaultdict(list)
    act_h=defaultdict(lambda:defaultdict(list))
    mood_h=defaultdict(lambda:defaultdict(list))
    age_h=defaultdict(lambda:defaultdict(list))

    for uid in unique_u:
        rows = task_data[user_col==uid]
        if len(rows) < 2: continue
        test  = rows[-1]
        pos   = int(test[1])
        ctx_t = torch.tensor(test[2:], dtype=torch.long).unsqueeze(0).to(device)
        act   = ACTIVITIES[int(test[3])]
        mood  = MOODS[int(test[6])]
        age   = AGE_GROUPS[int(test[4])]

        negs  = np.random.choice([i for i in range(n_items) if i!=pos], size=n_neg, replace=False)
        cands = torch.tensor(np.append(negs,pos), dtype=torch.long, device=device)
        user_t = torch.tensor([uid], dtype=torch.long, device=device)
        drift  = compute_drift(model, user_t, ctx_t).item()
        scores = model.rank_items(uid, cands, ctx_t, drift=drift, device=device).cpu().numpy()
        rank   = int((scores > scores[-1]).sum()) + 1

        for k in ks:
            h = int(rank<=k)
            n = 1.0/np.log2(rank+1) if h else 0.0
            hits[k].append(h); ndcgs[k].append(n)
            act_h[act][k].append(h); mood_h[mood][k].append(h); age_h[age][k].append(h)

    metrics = {}
    for k in ks:
        metrics[f"HR@{k}"]   = float(np.mean(hits[k]))  if hits[k]  else 0.0
        metrics[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    for act in ACTIVITIES:
        vals = act_h[act].get(10,[])
        metrics[f"HR@10_{act}"] = float(np.mean(vals)) if vals else 0.0
    for mood in MOODS:
        vals = mood_h[mood].get(10,[])
        metrics[f"HR@10_{mood}"] = float(np.mean(vals)) if vals else 0.0
    for age in AGE_GROUPS:
        safe = age.replace("<","lt").replace("+","plus").replace("-","_")
        vals = age_h[age].get(10,[])
        metrics[f"HR@10_age_{safe}"] = float(np.mean(vals)) if vals else 0.0

    model.train()
    return metrics

"""
continual/drift.py
Computes per-user taste-drift score in [0,1].
drift = cosine_distance(long_term_emb, short_term_emb) + context_bias
"""
import torch
import torch.nn.functional as F

_ACTIVITY_BIAS = torch.tensor([0.05, -0.05, 0.10, 0.02])
_MOOD_BIAS     = torch.tensor([0.05, -0.08, 0.10, -0.05])

def compute_drift(model, user, ctx=None):
    with torch.no_grad():
        e_long  = model.user_long(user)
        e_short = model.user_short(user)
    cos   = F.cosine_similarity(e_long, e_short, dim=1)
    drift = (1 - cos) / 2
    if ctx is not None:
        act_bias  = _ACTIVITY_BIAS.to(drift.device)[ctx[:,1]]
        mood_bias = _MOOD_BIAS.to(drift.device)[ctx[:,4]]
        drift = (drift + act_bias + mood_bias).clamp(0.0, 1.0)
    return drift

"""
training/train_task.py
BPR loss + EWC penalty + context-stratified experience replay.

Fix: ewc= in printed output now shows lambda * penalty
     (the actual contribution to loss), not the raw unscaled penalty.
     This makes the number meaningful and comparable to bpr= and replay=.
"""

import torch
import torch.nn.functional as F
import numpy as np
from continual.drift import compute_drift


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos - neg).mean()


def _replay_tensors(rows: np.ndarray, n_items: int, device: str):
    if len(rows) == 0:
        return None
    t   = torch.tensor(rows, dtype=torch.long, device=device)
    neg = torch.randint(0, n_items, (len(rows),), device=device)
    return t[:, 0], t[:, 1], neg, t[:, 2:]


def train_task(model, dataloader, optimizer,
               replay_buffer=None, ewc=None,
               lambda_ewc=None, lambda_replay=None,
               device="cpu", n_items=0):

    from utils.config import Config
    if lambda_ewc    is None: lambda_ewc    = Config.lambda_ewc
    if lambda_replay is None: lambda_replay = Config.lambda_replay

    model.train()
    totals = dict(total=0.0, bpr=0.0, ewc=0.0, replay=0.0)

    for user, pos, neg, ctx in dataloader:
        user, pos, neg, ctx = (x.to(device) for x in (user, pos, neg, ctx))

        # ── Current task BPR loss ─────────────────────────────────────────────
        drift = compute_drift(model, user, ctx)
        pos_s = model(user, pos,  drift, ctx)
        neg_s = model(user, neg,  drift, ctx)
        l_bpr = bpr_loss(pos_s, neg_s)
        loss  = l_bpr

        # ── EWC penalty ───────────────────────────────────────────────────────
        l_ewc_raw    = torch.tensor(0.0, device=device)
        l_ewc_scaled = torch.tensor(0.0, device=device)
        if ewc is not None:
            l_ewc_raw    = ewc.penalty(model)
            l_ewc_scaled = lambda_ewc * l_ewc_raw
            loss         = loss + l_ewc_scaled

        # ── Experience replay ─────────────────────────────────────────────────
        l_rep_scaled = torch.tensor(0.0, device=device)
        if replay_buffer is not None and len(replay_buffer) > 0:
            rows = replay_buffer.sample_stratified(Config.replay_batch)
            rb   = _replay_tensors(rows, n_items, device)
            if rb is not None:
                ru, ri, rn, rctx = rb
                rd           = compute_drift(model, ru, rctx)
                l_rep        = bpr_loss(model(ru, ri, rd, rctx),
                                        model(ru, rn, rd, rctx))
                l_rep_scaled = lambda_replay * l_rep
                loss         = loss + l_rep_scaled

        # ── Optimise ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["total"]  += loss.item()
        totals["bpr"]    += l_bpr.item()
        # Display the SCALED contribution (lambda * penalty) so it's comparable
        totals["ewc"]    += l_ewc_scaled.item()
        totals["replay"] += l_rep_scaled.item()

    n = max(len(dataloader), 1)
    return {k: v / n for k, v in totals.items()}
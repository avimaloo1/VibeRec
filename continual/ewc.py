"""
continual/ewc.py
Elastic Weight Consolidation — fixed implementation.

Bugs fixed vs previous version:
  1. Fisher now uses BPR loss (pos vs neg) not BCE(output, ones).
     BCE with all-ones gave near-zero gradients on a ranking model.
  2. Fisher accumulated per-SAMPLE for correct normalisation.
  3. Fisher floored at FISHER_FLOOR so EWC never goes completely silent.
  4. Fisher scaled so mean penalty stays on same order as BPR loss.
  5. train_task displays lambda * penalty (actual contribution), not raw penalty.
"""

import torch
import torch.nn.functional as F


class EWC:
    PROTECTED = (
        "user_long",    # long-term memory — must be protected
        "item_emb",     # item representations shared across tasks
        "audio_tower",  # audio feature projector
        "ctx_encoder",  # context tower
        "ctx_to_emb",
        "audio_proj",
        "gate",         # drift gate
    )

    FISHER_FLOOR = 1e-4   # minimum Fisher value — prevents silent EWC

    def __init__(self, model, dataloader, device, n_samples: int = 512):
        self.device    = device
        self.n_samples = n_samples

        self.params  = {n: p for n, p in model.named_parameters()
                        if any(tag in n for tag in self.PROTECTED)}
        self.optimal = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher  = self._compute_fisher(model, dataloader)

    def _compute_fisher(self, model, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        model.eval()
        n_seen = 0

        for user, pos, neg, ctx in dataloader:
            if n_seen >= self.n_samples:
                break
            user = user.to(self.device)
            pos  = pos.to(self.device)
            neg  = neg.to(self.device)
            ctx  = ctx.to(self.device)
            B    = user.size(0)

            drift = torch.zeros(B, dtype=torch.float, device=self.device)
            model.zero_grad()

            # BPR loss — same objective as training (not BCE with all-ones)
            pos_s = model(user, pos, drift, ctx)
            neg_s = model(user, neg, drift, ctx)
            loss  = -F.logsigmoid(pos_s - neg_s).mean()
            loss.backward()

            for n, p in self.params.items():
                if p.grad is not None:
                    fisher[n] += (p.grad.detach() ** 2) * B

            n_seen += B

        model.train()

        if n_seen == 0:
            return fisher

        # Normalise by samples seen
        for n in fisher:
            fisher[n] /= n_seen

        # Floor: prevents Fisher going to zero and disabling EWC
        for n in fisher:
            fisher[n] = torch.clamp(fisher[n], min=self.FISHER_FLOOR)

        # Scale so mean Fisher penalty ~ 0.1 (same order as BPR loss ~0.69)
        total_mean = sum(f.mean().item() for f in fisher.values()) / max(len(fisher), 1)
        if total_mean > 0:
            scale = 0.1 / total_mean
            for n in fisher:
                fisher[n] = fisher[n] * scale

        # Diagnostic output so you can verify EWC is active
        fmeans = {n: f.mean().item() for n, f in fisher.items()}
        top3   = sorted(fmeans.items(), key=lambda x: -x[1])[:3]
        print(f"  [EWC] Fisher computed on {n_seen} samples | "
              f"top params: " +
              ", ".join(f"{n.split('.')[-2]}.{n.split('.')[-1]}={v:.4f}"
                        for n, v in top3))

        return fisher

    def penalty(self, model) -> torch.Tensor:
        """EWC regularisation — penalises deviation from task-optimal params,
        weighted by Fisher (importance) of each parameter."""
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.optimal[n]) ** 2).sum()
        return loss
"""
models/dual_memory_model.py

Context-Aware Dual-Memory Recommender
======================================
Architecture
------------
  Long-term memory  : stable user preferences across all tasks (EWC-protected)
  Short-term memory : rapidly adapts to the current task / mood
  Context encoder   : embeds time-of-day, activity, age group, weekday into a
                      compact vector that gates the long/short blend AND enriches
                      the item scoring.

Forward pass (BPR-style)
------------------------
  e_user  = (1 − α) · e_long  +  α · e_short
  α       = sigmoid( w_drift · drift  +  w_ctx · f_ctx )   (learned gating)
  score   = e_user · (e_item  +  f_ctx)                     (context shift on item)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config


class ContextEncoder(nn.Module):
    """Encodes [time_idx, activity_idx, age_idx, weekday] → context vector."""

    def __init__(self, ctx_dim: int = None):
        if ctx_dim is None:
            ctx_dim = Config.context_dim
        super().__init__()
        self.time_emb     = nn.Embedding(Config.n_time_slots,  ctx_dim // 4)
        self.activity_emb = nn.Embedding(Config.n_contexts,    ctx_dim // 4)
        self.age_emb      = nn.Embedding(Config.n_age_groups,  ctx_dim // 4)
        self.weekday_emb  = nn.Embedding(Config.n_day_types,   ctx_dim // 4)

        in_dim = ctx_dim  # 4 × (ctx_dim // 4)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, ctx_dim),
            nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        ctx : (B, 4)  [time_idx, activity_idx, age_idx, weekday]
        returns (B, ctx_dim)
        """
        t = self.time_emb(ctx[:, 0])
        a = self.activity_emb(ctx[:, 1])
        g = self.age_emb(ctx[:, 2])
        w = self.weekday_emb(ctx[:, 3])
        x = torch.cat([t, a, g, w], dim=1)
        return self.proj(x)


class DualMemoryRecommender(nn.Module):

    def __init__(self,
                 n_users: int,
                 n_items: int,
                 emb_dim: int  = None,
                 ctx_dim: int  = None):
        if emb_dim is None:
            emb_dim = Config.emb_dim
        if ctx_dim is None:
            ctx_dim = Config.context_dim
        super().__init__()

        # ── memory streams ────────────────────────────────────────────────────
        self.user_long  = nn.Embedding(n_users, emb_dim)
        self.user_short = nn.Embedding(n_users, emb_dim)
        self.item_emb   = nn.Embedding(n_items, emb_dim)

        # ── context tower ─────────────────────────────────────────────────────
        self.ctx_encoder = ContextEncoder(ctx_dim)

        # project context to embedding space for item shift
        self.ctx_to_emb = nn.Linear(ctx_dim, emb_dim)

        # ── adaptive gating (drift + context → α) ────────────────────────────
        self.gate = nn.Sequential(
            nn.Linear(1 + ctx_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── output MLP (optional – set use_mlp=True for non-linear scoring) ──
        self.use_mlp = False
        if self.use_mlp:
            self.score_mlp = nn.Sequential(
                nn.Linear(emb_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_long, self.user_short, self.item_emb]:
            nn.init.normal_(emb.weight, mean=0, std=0.01)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self,
                user:  torch.Tensor,    # (B,)
                item:  torch.Tensor,    # (B,)
                drift: torch.Tensor,    # (B,)  ∈ [0,1]
                ctx:   torch.Tensor,    # (B, 4)
                ) -> torch.Tensor:      # (B,)  ∈ (0,1)

        # context vector
        f_ctx = self.ctx_encoder(ctx)                       # (B, ctx_dim)

        # adaptive gate  α ∈ (0,1)
        gate_input = torch.cat([drift.unsqueeze(1), f_ctx], dim=1)
        alpha = self.gate(gate_input).squeeze(1)            # (B,)

        # blended user embedding
        e_long  = self.user_long(user)                      # (B, D)
        e_short = self.user_short(user)                     # (B, D)
        e_user  = (1 - alpha.unsqueeze(1)) * e_long \
                +       alpha.unsqueeze(1)  * e_short        # (B, D)

        # context-shifted item embedding
        ctx_shift = self.ctx_to_emb(f_ctx)                  # (B, D)
        e_item    = self.item_emb(item) + ctx_shift          # (B, D)

        # scoring
        score = (e_user * e_item).sum(dim=1)                # (B,)
        return torch.sigmoid(score)

    # ── convenience: score a batch of items for one user (inference) ──────────
    @torch.no_grad()
    def rank_items(self,
                   user_id:  int,
                   item_ids: torch.Tensor,
                   ctx:      torch.Tensor,
                   drift:    float = 0.0,
                   device:   str   = "cpu") -> torch.Tensor:
        """Return scores for all item_ids given a single user and context."""
        B = len(item_ids)
        user  = torch.tensor([user_id] * B, dtype=torch.long, device=device)
        drift_t = torch.tensor([drift]  * B, dtype=torch.float, device=device)
        ctx_t   = ctx.expand(B, -1).to(device)
        return self.forward(user, item_ids.to(device), drift_t, ctx_t)

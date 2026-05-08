"""
models/audio_dual_memory.py

Audio-Aware Context-Gated Dual Memory Recommender
==================================================
Innovations:
  1. AUDIO TOWER  - projects real Spotify audio features into embedding space
                    using FiLM (Feature-wise Linear Modulation) conditioned on context
  2. CONTEXT TOWER - encodes time, activity, age, weekday, mood
  3. DUAL MEMORY  - long-term (EWC-protected) + short-term (fast-adapting) user embeddings
  4. LEARNED GATE - small MLP decides how much to trust short vs long term memory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(self, ctx_dim, n_time, n_act, n_age, n_day, n_mood):
        super().__init__()
        slot = ctx_dim // 5
        self.time_emb     = nn.Embedding(n_time, slot)
        self.activity_emb = nn.Embedding(n_act,  slot)
        self.age_emb      = nn.Embedding(n_age,  slot)
        self.weekday_emb  = nn.Embedding(n_day,  slot)
        self.mood_emb     = nn.Embedding(n_mood, slot)
        self.proj = nn.Sequential(
            nn.Linear(slot*5, ctx_dim), nn.LayerNorm(ctx_dim),
            nn.ReLU(), nn.Linear(ctx_dim, ctx_dim),
        )

    def forward(self, ctx):
        t = self.time_emb(ctx[:,0]);     a = self.activity_emb(ctx[:,1])
        g = self.age_emb(ctx[:,2]);      w = self.weekday_emb(ctx[:,3])
        m = self.mood_emb(ctx[:,4])
        return self.proj(torch.cat([t,a,g,w,m], dim=1))


class AudioTower(nn.Module):
    """FiLM-modulated audio projection: context scales & shifts the audio features."""
    def __init__(self, n_audio, emb_dim, ctx_dim):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(n_audio,128), nn.ReLU(), nn.Linear(128,emb_dim))
        self.film_gamma = nn.Linear(ctx_dim, emb_dim)
        self.film_beta  = nn.Linear(ctx_dim, emb_dim)

    def forward(self, audio, ctx_vec):
        h     = self.base(audio)
        gamma = torch.sigmoid(self.film_gamma(ctx_vec))
        beta  = self.film_beta(ctx_vec)
        return h * gamma + beta


class AudioDualMemoryRecommender(nn.Module):
    def __init__(self, n_users, n_items, audio_matrix,
                 emb_dim=None, ctx_dim=None, audio_dim=None,
                 n_time=6, n_act=4, n_age=5, n_day=2, n_mood=4):
        super().__init__()
        from utils.config import Config
        if emb_dim   is None: emb_dim   = Config.emb_dim
        if ctx_dim   is None: ctx_dim   = Config.context_dim
        if audio_dim is None: audio_dim = Config.audio_dim

        n_audio = audio_matrix.shape[1]
        self.register_buffer("audio_matrix", audio_matrix)

        self.user_long  = nn.Embedding(n_users, emb_dim)
        self.user_short = nn.Embedding(n_users, emb_dim)
        self.item_emb   = nn.Embedding(n_items, emb_dim)

        self.ctx_encoder = ContextEncoder(ctx_dim, n_time, n_act, n_age, n_day, n_mood)
        self.audio_tower = AudioTower(n_audio, audio_dim, ctx_dim)
        self.audio_proj  = nn.Linear(audio_dim, emb_dim)
        self.ctx_to_emb  = nn.Linear(ctx_dim, emb_dim)

        self.gate = nn.Sequential(
            nn.Linear(1+ctx_dim, 64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid()
        )
        for emb in [self.user_long, self.user_short, self.item_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    def _audio(self, item):
        return self.audio_matrix[item]

    def forward(self, user, item, drift, ctx):
        f_ctx   = self.ctx_encoder(ctx)
        alpha   = self.gate(torch.cat([drift.unsqueeze(1), f_ctx], dim=1)).squeeze(1)
        e_long  = self.user_long(user)
        e_short = self.user_short(user)
        e_user  = (1-alpha.unsqueeze(1))*e_long + alpha.unsqueeze(1)*e_short
        e_audio  = self.audio_proj(self.audio_tower(self._audio(item), f_ctx))
        ctx_shift = self.ctx_to_emb(f_ctx)
        e_item   = self.item_emb(item) + e_audio + ctx_shift
        return torch.sigmoid((e_user * e_item).sum(dim=1))

    @torch.no_grad()
    def rank_items(self, user_id, item_ids, ctx, drift=0.0, device="cpu"):
        B = len(item_ids)
        user_t  = torch.tensor([user_id]*B, dtype=torch.long,  device=device)
        drift_t = torch.tensor([drift]*B,   dtype=torch.float, device=device)
        ctx_t   = ctx.expand(B,-1).to(device)
        return self.forward(user_t, item_ids.to(device), drift_t, ctx_t)

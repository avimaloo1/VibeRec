class Config:
    # ── Embeddings ─────────────────────────────────────────────────────────────
    emb_dim      = 128
    context_dim  = 64
    audio_dim    = 64

    # ── Training ───────────────────────────────────────────────────────────────
    lr           = 0.001
    weight_decay = 1e-5
    batch_size   = 512
    epochs       = 10
    num_tasks    = 5

    # ── Continual learning ─────────────────────────────────────────────────────
    lambda_ewc    = 1.0    # increased from 0.5 — EWC penalty now on same scale as BPR
    replay_size   = 8000
    replay_batch  = 256
    lambda_replay = 0.4

    # ── Evaluation ─────────────────────────────────────────────────────────────
    top_k      = [5, 10, 20]
    n_neg_eval = 100

    # ── Context buckets ────────────────────────────────────────────────────────
    n_time_slots = 6
    n_activities = 4
    n_age_groups = 5
    n_day_types  = 2
    n_moods      = 4

    # ── Audio features from Kaggle dataset ────────────────────────────────────
    AUDIO_FEATURES = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]

    seed = 42
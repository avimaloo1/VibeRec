"""
synthetic/generate.py
Generates a realistic synthetic music-listening dataset with:
  - Users with age groups
  - Tracks with genre / energy / tempo metadata
  - Context signals: time-of-day, activity (commute/work/workout/leisure), weekday vs weekend
  - Temporal drift: users' taste shifts gradually over tasks
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── constants ──────────────────────────────────────────────────────────────────
N_USERS   = 2_000
N_ITEMS   = 5_000
N_RECORDS = 300_000

AGE_GROUPS   = ["<18", "18-25", "26-35", "36-50", "50+"]
GENRES       = ["pop", "rock", "hip-hop", "electronic", "classical",
                "jazz", "r&b", "country", "metal", "indie"]
ACTIVITIES   = ["commute", "work", "workout", "leisure"]
TIME_SLOTS   = ["night", "early_morning", "morning",
                "afternoon", "evening", "late_night"]

# genre affinity per age group  (rows=age_groups, cols=genres)
AGE_GENRE_AFFINITY = np.array([
    [0.25, 0.05, 0.30, 0.15, 0.02, 0.02, 0.10, 0.03, 0.05, 0.03],  # <18
    [0.22, 0.08, 0.25, 0.18, 0.02, 0.03, 0.10, 0.03, 0.04, 0.05],  # 18-25
    [0.18, 0.15, 0.15, 0.15, 0.05, 0.07, 0.10, 0.06, 0.04, 0.05],  # 26-35
    [0.12, 0.20, 0.08, 0.10, 0.10, 0.12, 0.08, 0.12, 0.03, 0.05],  # 36-50
    [0.08, 0.18, 0.04, 0.06, 0.22, 0.20, 0.06, 0.10, 0.02, 0.04],  # 50+
])

# activity × energy preference (high energy = workout, low = work/classical)
ACTIVITY_ENERGY = {"commute": 0.6, "work": 0.35, "workout": 0.85, "leisure": 0.5}

# time-of-day × activity probability
TIME_ACTIVITY_PROB = {
    "night":         [0.05, 0.05, 0.05, 0.85],
    "early_morning": [0.40, 0.10, 0.35, 0.15],
    "morning":       [0.35, 0.25, 0.25, 0.15],
    "afternoon":     [0.10, 0.40, 0.15, 0.35],
    "evening":       [0.25, 0.10, 0.25, 0.40],
    "late_night":    [0.05, 0.05, 0.10, 0.80],
}


def _make_users(n: int) -> pd.DataFrame:
    age_idx = RNG.integers(0, len(AGE_GROUPS), size=n)
    return pd.DataFrame({
        "user_id":   [f"u{i:05d}" for i in range(n)],
        "age_group": [AGE_GROUPS[a] for a in age_idx],
        "age_idx":   age_idx,
        # each user has a slight personal taste vector (noise around age-group prior)
        "taste_noise": RNG.uniform(-0.1, 0.1, size=n),
    })


def _make_items(n: int) -> pd.DataFrame:
    genres  = RNG.choice(GENRES, size=n)
    energy  = RNG.beta(2, 2, size=n).clip(0.05, 0.95)
    tempo   = (energy * 80 + 60 + RNG.normal(0, 10, size=n)).clip(60, 200)
    return pd.DataFrame({
        "track_id":   [f"t{i:06d}" for i in range(n)],
        "genre":      genres,
        "energy":     energy.round(3),
        "tempo":      tempo.astype(int),
        "genre_idx":  [GENRES.index(g) for g in genres],
    })


def _sample_context(size: int, timestamps: np.ndarray) -> pd.DataFrame:
    # map timestamp fraction → time slot
    hour = ((timestamps / timestamps.max()) * 24).astype(int) % 24
    slot_idx = np.select(
        [hour < 2, hour < 6, hour < 12, hour < 17, hour < 21, hour < 24],
        [0, 1, 2, 3, 4, 5], default=5
    )
    slots = [TIME_SLOTS[s] for s in slot_idx]

    # sample activity conditioned on time slot
    activities = np.array([
        RNG.choice(ACTIVITIES, p=TIME_ACTIVITY_PROB[s]) for s in slots
    ])

    weekday = (timestamps.astype(int) // (3600 * 24)) % 7 < 5  # Mon-Fri

    return pd.DataFrame({
        "time_slot":   slots,
        "time_idx":    slot_idx,
        "activity":    activities,
        "activity_idx": [ACTIVITIES.index(a) for a in activities],
        "weekday":     weekday.astype(int),
    })


def _interaction_score(users: pd.DataFrame, items: pd.DataFrame,
                       user_ids: np.ndarray, item_ids: np.ndarray,
                       activities: np.ndarray) -> np.ndarray:
    """Compute implicit affinity score for each (user, item, activity) triple."""
    age_idxs   = users.iloc[user_ids]["age_idx"].values
    genre_idxs = items.iloc[item_ids]["genre_idx"].values
    energies   = items.iloc[item_ids]["energy"].values

    genre_score  = AGE_GENRE_AFFINITY[age_idxs, genre_idxs]
    target_energy = np.array([ACTIVITY_ENERGY[a] for a in activities])
    energy_score = 1 - np.abs(energies - target_energy)

    score = 0.6 * genre_score + 0.4 * energy_score
    score += users.iloc[user_ids]["taste_noise"].values
    return score.clip(0, 1)


def generate(out_path: str = "data/spotify_life.csv",
             n_users: int = N_USERS,
             n_items: int = N_ITEMS,
             n_records: int = N_RECORDS) -> pd.DataFrame:

    print(f"Generating synthetic dataset  ({n_users} users, {n_items} tracks, {n_records} interactions)…")
    users = _make_users(n_users)
    items = _make_items(n_items)

    timestamps = np.sort(RNG.integers(0, 10_000_000, size=n_records))

    # popularity-biased item sampling (power-law)
    item_pop = RNG.power(0.4, size=n_items)
    item_pop /= item_pop.sum()
    user_ids = RNG.integers(0, n_users, size=n_records)
    item_ids = RNG.choice(n_items, size=n_records, p=item_pop)

    ctx = _sample_context(n_records, timestamps)

    score = _interaction_score(users, items, user_ids, item_ids, ctx["activity"].values)
    # keep only high-affinity interactions (simulate implicit positive feedback)
    keep = RNG.random(size=n_records) < score
    print(f"  Kept {keep.sum():,} / {n_records:,} interactions after affinity filtering.")

    df = pd.DataFrame({
        "user_id":     users.iloc[user_ids[keep]]["user_id"].values,
        "track_id":    items.iloc[item_ids[keep]]["track_id"].values,
        "timestamp":   timestamps[keep],
        "age_group":   users.iloc[user_ids[keep]]["age_group"].values,
        "genre":       items.iloc[item_ids[keep]]["genre"].values,
        "energy":      items.iloc[item_ids[keep]]["energy"].values,
        "time_slot":   ctx["time_slot"].values[keep],
        "activity":    ctx["activity"].values[keep],
        "weekday":     ctx["weekday"].values[keep],
    })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    generate()

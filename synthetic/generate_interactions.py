"""
synthetic/generate_interactions.py

Generates realistic user-track interaction logs on top of the
real Kaggle Spotify audio features. Each interaction is assigned:
  - A context (activity, mood, time-of-day, weekday)
  - A listening probability driven by cosine similarity between
    the track's real audio features and the "ideal" audio vector
    for that context.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

ACTIVITIES = ["commute","work","workout","leisure"]
TIME_SLOTS = ["night","early_morning","morning","afternoon","evening","late_night"]
AGE_GROUPS = ["<18","18-25","26-35","36-50","50+"]
MOODS      = ["happy","calm","energetic","melancholic"]

CONTEXT_IDEAL = {
    "workout":    np.array([0.8,0.9,0.8,0.1,0.1,0.05,0.2,0.7,0.9]),
    "commute":    np.array([0.7,0.6,0.6,0.2,0.2,0.1, 0.2,0.7,0.6]),
    "work":       np.array([0.4,0.4,0.4,0.1,0.5,0.4, 0.1,0.5,0.4]),
    "leisure":    np.array([0.7,0.5,0.5,0.2,0.3,0.1, 0.3,0.8,0.5]),
    "morning":    np.array([0.6,0.7,0.6,0.1,0.2,0.1, 0.2,0.8,0.7]),
    "late_night": np.array([0.4,0.3,0.3,0.1,0.7,0.3, 0.1,0.4,0.3]),
    "afternoon":  np.array([0.6,0.6,0.6,0.2,0.3,0.1, 0.2,0.7,0.6]),
    "happy":      np.array([0.8,0.7,0.6,0.1,0.2,0.1, 0.2,0.9,0.7]),
    "calm":       np.array([0.3,0.2,0.2,0.1,0.8,0.4, 0.1,0.4,0.2]),
    "energetic":  np.array([0.7,0.9,0.8,0.2,0.1,0.05,0.3,0.7,0.9]),
    "melancholic":np.array([0.3,0.3,0.3,0.1,0.7,0.2, 0.1,0.2,0.3]),
}
TIME_ACTIVITY_PROB = {
    "night":         [0.05,0.05,0.05,0.85],
    "early_morning": [0.40,0.10,0.35,0.15],
    "morning":       [0.30,0.25,0.30,0.15],
    "afternoon":     [0.10,0.40,0.15,0.35],
    "evening":       [0.25,0.10,0.25,0.40],
    "late_night":    [0.05,0.05,0.10,0.80],
}
TIME_MOOD_PROB = {
    "night":         [0.2,0.4,0.1,0.3],
    "early_morning": [0.3,0.3,0.3,0.1],
    "morning":       [0.4,0.2,0.3,0.1],
    "afternoon":     [0.3,0.2,0.3,0.2],
    "evening":       [0.3,0.3,0.2,0.2],
    "late_night":    [0.1,0.5,0.1,0.3],
}

def _normalise_audio(df):
    from sklearn.preprocessing import MinMaxScaler
    from utils.config import Config
    feats = df[Config.AUDIO_FEATURES].fillna(0).values.astype(np.float32)
    return MinMaxScaler().fit_transform(feats)

def generate_interactions(tracks_csv, out_path="data/interactions.csv",
                          n_users=3_000, n_records=500_000):
    print(f"Loading tracks from {tracks_csv} ...")
    tracks = pd.read_csv(tracks_csv).drop_duplicates("track_id").reset_index(drop=True)
    tracks = tracks.dropna(subset=["track_id"]).reset_index(drop=True)
    n_items = len(tracks)
    print(f"  {n_items:,} unique tracks loaded.")

    audio_matrix = _normalise_audio(tracks)
    age_idx_arr  = RNG.integers(0, len(AGE_GROUPS), size=n_users)
    pop = tracks["popularity"].fillna(0).values.astype(float) + 1.0
    pop_prob = pop / pop.sum()

    timestamps = np.sort(RNG.integers(0, 63_072_000, size=n_records))
    hour = (timestamps // 3600) % 24
    slot_idx = np.select([hour<2,hour<6,hour<12,hour<17,hour<21],[0,1,2,3,4],default=5)
    slots = [TIME_SLOTS[s] for s in slot_idx]
    activities = np.array([RNG.choice(ACTIVITIES, p=TIME_ACTIVITY_PROB[s]) for s in slots])
    moods      = np.array([RNG.choice(MOODS,      p=TIME_MOOD_PROB[s])      for s in slots])
    weekday    = ((timestamps // 86400) % 7 < 5).astype(int)
    user_ids   = RNG.integers(0, n_users, size=n_records)
    item_ids   = RNG.choice(n_items, size=n_records, p=pop_prob)

    print("  Computing context-audio affinity scores ...")
    audio_vecs    = audio_matrix[item_ids]
    act_ideals    = np.array([CONTEXT_IDEAL[a] for a in activities])
    mood_ideals   = np.array([CONTEXT_IDEAL[m] for m in moods])
    combined      = 0.5 * act_ideals + 0.5 * mood_ideals
    dots  = (audio_vecs * combined).sum(axis=1)
    norms = (np.linalg.norm(audio_vecs,axis=1)*np.linalg.norm(combined,axis=1)).clip(1e-8)
    affinity  = (dots/norms).clip(0,1)
    threshold = RNG.uniform(0.2,0.7,size=n_records)
    keep = affinity > threshold
    print(f"  Kept {keep.sum():,} / {n_records:,} interactions.")

    ki = np.where(keep)[0]
    df = pd.DataFrame({
        "user_id":   [f"u{user_ids[i]:05d}" for i in ki],
        "track_id":  tracks.iloc[item_ids[ki]]["track_id"].values,
        "timestamp": timestamps[ki],
        "age_group": [AGE_GROUPS[age_idx_arr[user_ids[i]]] for i in ki],
        "activity":  activities[ki],
        "mood":      moods[ki],
        "time_slot": [slots[i] for i in ki],
        "weekday":   weekday[ki],
    })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}  ({len(df):,} rows)")
    return df

if __name__ == "__main__":
    generate_interactions("data/dataset.csv")

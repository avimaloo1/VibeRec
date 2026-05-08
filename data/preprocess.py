"""
data/preprocess.py
Loads interactions + tracks CSV and returns per-task splits with
full context encoding and an audio feature tensor.
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

ACTIVITIES = ["commute","work","workout","leisure"]
TIME_SLOTS = ["night","early_morning","morning","afternoon","evening","late_night"]
AGE_GROUPS = ["<18","18-25","26-35","36-50","50+"]
MOODS      = ["happy","calm","energetic","melancholic"]

_user_enc = LabelEncoder()
_item_enc = LabelEncoder()

def load_and_preprocess(interactions_csv, tracks_csv, num_tasks=None):
    from utils.config import Config
    if num_tasks is None:
        num_tasks = Config.num_tasks

    df     = pd.read_csv(interactions_csv).sort_values("timestamp").reset_index(drop=True)
    tracks = pd.read_csv(tracks_csv).drop_duplicates("track_id").reset_index(drop=True)

    df["user"] = _user_enc.fit_transform(df["user_id"])
    df["item"] = _item_enc.fit_transform(df["track_id"])
    n_users = df["user"].nunique()
    n_items = df["item"].nunique()

    df["time_idx"]     = df["time_slot"].map({v:i for i,v in enumerate(TIME_SLOTS)}).fillna(0).astype(int)
    df["activity_idx"] = df["activity"].map({v:i for i,v in enumerate(ACTIVITIES)}).fillna(3).astype(int)
    df["age_idx"]      = df["age_group"].map({v:i for i,v in enumerate(AGE_GROUPS)}).fillna(2).astype(int)
    df["weekday"]      = df["weekday"].astype(int)
    df["mood_idx"]     = df["mood"].map({v:i for i,v in enumerate(MOODS)}).fillna(0).astype(int)

    audio_feats  = Config.AUDIO_FEATURES
    tracks_idx   = tracks.set_index("track_id")
    known_ids    = _item_enc.classes_
    audio_vals   = tracks_idx.reindex(known_ids)[audio_feats].fillna(0).values.astype(np.float32)
    audio_matrix = MinMaxScaler().fit_transform(audio_vals).astype(np.float32)
    audio_tensor = torch.tensor(audio_matrix, dtype=torch.float32)

    df["task"] = pd.qcut(df["timestamp"], num_tasks, labels=False)
    cols = ["user","item","time_idx","activity_idx","age_idx","weekday","mood_idx"]
    tasks = []
    for t in range(num_tasks):
        chunk = df[df["task"]==t][cols].values
        tasks.append(chunk)
        print(f"  Task {t}: {len(chunk):,} interactions")
    print(f"  Users: {n_users}  |  Items: {n_items}")
    return tasks, n_users, n_items, audio_tensor

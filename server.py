"""
server.py
=========
Flask API server that connects the Spotify-like frontend to the
trained PyTorch continual learning model.

Endpoints:
  POST /recommend   — returns ranked tracks for a given context
  GET  /status      — model training status and CL stats
  POST /feedback    — records user feedback to update CL state

Run via:  python run.py   (trains model first, then starts server)
Or:       python server.py --skip-train  (if model already trained)
"""

import os, json, argparse, time
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.config import Config
from utils.seed import set_seed
from continual.drift import compute_drift

app = Flask(__name__)
CORS(app)  # allow browser requests from index.html

# ── Global state ──────────────────────────────────────────────────────────────
MODEL        = None
TRACKS_DF    = None
AUDIO_TENSOR = None
ITEM_ENC     = None
N_ITEMS      = 0
DEVICE       = "cpu"

# Continual learning session state (per-server-session)
CL_STATE = {
    "sessions":    0,
    "tracks_heard": 0,
    "long_term":   None,   # stable audio centroid
    "short_term":  None,   # recent audio centroid
    "history":     [],     # list of {context, track, ts}
    "ewc_weights": {},     # simulated Fisher weights per audio feature
}

FEAT_KEYS = ["danceability","energy","loudness","speechiness",
             "acousticness","instrumentalness","liveness","valence","tempo"]

ACTIVITIES = ["commute","work","workout","leisure"]
TIME_SLOTS = ["night","early_morning","morning","afternoon","evening","late_night"]
AGE_GROUPS = ["<18","18-25","26-35","36-50","50+"]
MOODS      = ["happy","calm","energetic","melancholic"]

ACTIVITY_MAP = {
    "workout":"workout","commute":"commute","work":"work",
    "work / study":"work","leisure":"leisure","sleep":"leisure",
    "social":"leisure","cooking":"leisure","running":"workout",
    "social / party":"leisure","sleep / wind down":"leisure",
}
MOOD_MAP = {
    "happy":"happy","calm":"calm","energetic":"energetic",
    "melancholic":"melancholic","romantic":"calm",
    "happy / upbeat":"happy","calm / focused":"calm",
    "energetic / hyped":"energetic","melancholic / reflective":"melancholic",
}

# ── Context → index helpers ───────────────────────────────────────────────────
def _time_idx(slot: str) -> int:
    slot = slot.lower().replace(" ","_").replace("(","").replace(")","")
    mapping = {
        "night":0,"late_night":0,
        "early_morning":1,"morning_6am_12pm":2,"morning":2,
        "afternoon_12_6pm":3,"afternoon":3,
        "evening_6_10pm":4,"evening":4,
    }
    for k,v in mapping.items():
        if k in slot: return v
    return 2  # default morning

def _activity_idx(act: str) -> int:
    a = ACTIVITY_MAP.get(act.lower(), "leisure")
    return ACTIVITIES.index(a) if a in ACTIVITIES else 3

def _mood_idx(mood: str) -> int:
    m = MOOD_MAP.get(mood.lower(), "happy")
    return MOODS.index(m) if m in MOODS else 0

def _age_idx(age: str) -> int:
    mapping = {"<18":0,"18-25":1,"26-35":2,"36-50":3,"50+":4}
    return mapping.get(age, 2)


# ── CL state update ───────────────────────────────────────────────────────────
def _update_cl(track_row: pd.Series, context: dict):
    tv = {k: float(track_row.get(k, 0)) for k in FEAT_KEYS}
    CL_STATE["sessions"]     += 1
    CL_STATE["tracks_heard"] += 1
    CL_STATE["short_term"]    = tv

    if CL_STATE["long_term"] is None:
        CL_STATE["long_term"] = dict(tv)
    else:
        alpha = 0.08  # slow EMA for long-term stability (EWC-style)
        for k in FEAT_KEYS:
            CL_STATE["long_term"][k] = (1-alpha)*CL_STATE["long_term"][k] + alpha*tv[k]

    # Update Fisher weights: features that stay consistent get higher protection
    for k in FEAT_KEYS:
        prev  = CL_STATE["ewc_weights"].get(k, 0.5)
        delta = abs(tv[k] - CL_STATE["long_term"].get(k, tv[k]))
        CL_STATE["ewc_weights"][k] = 0.9*prev + 0.1*(1-delta)

    CL_STATE["history"].append({
        "context": context,
        "track":   track_row.get("track_name", "Unknown"),
        "artist":  track_row.get("artists", "Unknown"),
        "ts":      int(time.time()),
    })


def _get_drift() -> float:
    if CL_STATE["long_term"] is None or CL_STATE["short_term"] is None:
        return 0.0
    lt = np.array([CL_STATE["long_term"][k]  for k in FEAT_KEYS])
    st = np.array([CL_STATE["short_term"][k] for k in FEAT_KEYS])
    n  = np.linalg.norm(lt) * np.linalg.norm(st)
    if n < 1e-8: return 0.0
    return float(1 - np.dot(lt, st) / n)


# ── Recommendation endpoint ───────────────────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    global MODEL, TRACKS_DF, AUDIO_TENSOR, ITEM_ENC, N_ITEMS, DEVICE

    data = request.get_json()
    activity  = data.get("activity",  "leisure")
    time_slot = data.get("time_slot", "morning")
    mood      = data.get("mood",      "happy")
    genre     = data.get("genre",     "any")
    age_group = data.get("age_group", "26-35")
    weekday   = int(data.get("weekday", 1))
    top_k     = int(data.get("top_k", 20))

    # Build context tensor
    ctx = torch.tensor([[
        _time_idx(time_slot),
        _activity_idx(activity),
        _age_idx(age_group),
        weekday,
        _mood_idx(mood),
    ]], dtype=torch.long, device=DEVICE)

    # Use model if trained, else fallback to audio cosine scoring
    if MODEL is not None:
        scores = _model_score(ctx, top_k * 3, genre)
    else:
        scores = _fallback_score(ctx, top_k * 3, genre)

    # Apply CL long-term taste bias if available
    if CL_STATE["long_term"] is not None:
        scores = _apply_cl_bias(scores)

    # Filter by genre
    if genre and genre.lower() not in ("any",""):
        genre_mask = TRACKS_DF["track_genre"].str.lower().str.contains(
            genre.lower(), na=False
        )
        scored_ids = [s["item_idx"] for s in scores]
        genre_boost = genre_mask.iloc[scored_ids].values
        for i, s in enumerate(scores):
            s["score"] *= (1.3 if genre_boost[i] else 0.7)
        scores.sort(key=lambda x: x["score"], reverse=True)

    top = scores[:top_k]

    # Build response tracks
    results = []
    for s in top:
        idx = s["item_idx"]
        row = TRACKS_DF.iloc[idx]
        results.append({
            "track_id":    str(row.get("track_id", idx)),
            "track_name":  str(row.get("track_name", "Unknown")),
            "artists":     str(row.get("artists", "Unknown")),
            "album_name":  str(row.get("album_name", "")),
            "track_genre": str(row.get("track_genre", "")),
            "popularity":  int(row.get("popularity", 0)),
            "duration_ms": int(row.get("duration_ms", 0)),
            "danceability":round(float(row.get("danceability", 0)), 3),
            "energy":      round(float(row.get("energy", 0)), 3),
            "valence":     round(float(row.get("valence", 0)), 3),
            "tempo":       round(float(row.get("tempo", 0)), 1),
            "acousticness":round(float(row.get("acousticness", 0)), 3),
            "speechiness": round(float(row.get("speechiness", 0)), 3),
            "instrumentalness": round(float(row.get("instrumentalness", 0)), 3),
            "liveness":    round(float(row.get("liveness", 0)), 3),
            "loudness":    round(float(row.get("loudness", 0)), 2),
            "score":       round(float(s["score"]), 4),
            "match_pct":   int(min(99, round(s["score"] * 100))),
        })

    # Update CL state with top recommendation
    if results:
        top_row = TRACKS_DF.iloc[top[0]["item_idx"]]
        _update_cl(top_row, {
            "activity": activity, "mood": mood,
            "time_slot": time_slot, "genre": genre,
        })

    return jsonify({
        "tracks": results,
        "cl_state": {
            "sessions":     CL_STATE["sessions"],
            "tracks_heard": CL_STATE["tracks_heard"],
            "drift":        round(_get_drift(), 3),
            "drift_label":  _drift_label(_get_drift()),
            "history":      CL_STATE["history"][-5:],
            "ewc_weights":  {k: round(v,3) for k,v in CL_STATE["ewc_weights"].items()},
        }
    })


def _model_score(ctx, k, genre):
    """Score items using the trained PyTorch model."""
    with torch.no_grad():
        # Sample candidate items (scoring all 114k is slow; sample 2000)
        candidates = torch.randint(0, N_ITEMS, (min(2000, N_ITEMS),), device=DEVICE)
        drift = torch.zeros(1, device=DEVICE)
        # Expand ctx for batch scoring
        ctx_exp   = ctx.expand(len(candidates), -1)
        drift_exp = drift.expand(len(candidates))
        user_t    = torch.zeros(len(candidates), dtype=torch.long, device=DEVICE)
        scores_t  = MODEL(user_t, candidates, drift_exp, ctx_exp)
        scores_np = scores_t.cpu().numpy()

    scored = [{"item_idx": int(candidates[i]), "score": float(scores_np[i])}
              for i in range(len(candidates))]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def _fallback_score(ctx, k, genre):
    """Audio cosine similarity scoring (no model needed)."""
    from utils.config import Config

    # Build ideal vector from context
    act_idx  = int(ctx[0, 1])
    mood_idx = int(ctx[0, 4])
    tod_idx  = int(ctx[0, 0])

    ACTIVITY_IDEALS = [
        [0.68,0.65,0.6, 0.12,0.20,0.1, 0.2,0.68,0.62],  # commute
        [0.40,0.40,0.4, 0.08,0.55,0.4, 0.1,0.50,0.42],  # work
        [0.85,0.92,0.8, 0.10,0.05,0.05,0.2,0.70,0.95],  # workout
        [0.70,0.55,0.5, 0.10,0.30,0.1, 0.3,0.80,0.55],  # leisure
    ]
    MOOD_IDEALS = [
        [0.80,0.70,0.6, 0.10,0.20,0.1, 0.2,0.90,0.70],  # happy
        [0.30,0.20,0.2, 0.08,0.80,0.4, 0.1,0.40,0.20],  # calm
        [0.70,0.90,0.8, 0.15,0.10,0.05,0.3,0.70,0.90],  # energetic
        [0.30,0.30,0.3, 0.08,0.70,0.2, 0.1,0.20,0.30],  # melancholic
    ]
    act_v  = np.array(ACTIVITY_IDEALS[min(act_idx,  3)])
    mood_v = np.array(MOOD_IDEALS[min(mood_idx, 3)])
    ideal  = 0.6 * act_v + 0.4 * mood_v

    audio_np = AUDIO_TENSOR.numpy()                        # (N, 9)
    dots  = audio_np @ ideal
    norms = np.linalg.norm(audio_np, axis=1) * np.linalg.norm(ideal)
    sims  = np.where(norms > 1e-8, dots / norms, 0)

    # Popularity boost
    pop   = TRACKS_DF["popularity"].fillna(0).values / 100.0
    final = 0.75 * sims + 0.25 * pop

    top_idx = np.argpartition(final, -k)[-k:]
    top_idx = top_idx[np.argsort(final[top_idx])[::-1]]
    return [{"item_idx": int(i), "score": float(final[i])} for i in top_idx]


def _apply_cl_bias(scores):
    """Blend scores toward user's long-term taste profile."""
    lt    = np.array([CL_STATE["long_term"][k] for k in FEAT_KEYS])
    audio = AUDIO_TENSOR.numpy()
    drift = _get_drift()
    blend = max(0, 0.25 - drift * 0.25)  # up to 25% long-term taste weight

    for s in scores:
        track_audio = audio[s["item_idx"]]
        n = np.linalg.norm(track_audio) * np.linalg.norm(lt)
        sim = float(np.dot(track_audio, lt) / n) if n > 1e-8 else 0.0
        s["score"] = (1 - blend) * s["score"] + blend * sim

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def _drift_label(d: float) -> str:
    if d < 0.08: return "Stable"
    if d < 0.20: return "Mild"
    return "High"


# ── Status endpoint ───────────────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "model_loaded":  MODEL is not None,
        "tracks_loaded": TRACKS_DF is not None,
        "n_tracks":      len(TRACKS_DF) if TRACKS_DF is not None else 0,
        "device":        DEVICE,
        "cl_sessions":   CL_STATE["sessions"],
    })


# ── Feedback endpoint ─────────────────────────────────────────────────────────
@app.route("/feedback", methods=["POST"])
def feedback():
    """Record explicit user feedback (liked/skipped) to update CL state."""
    data     = request.get_json()
    track_id = data.get("track_id")
    liked    = data.get("liked", True)
    context  = data.get("context", {})

    if track_id and TRACKS_DF is not None:
        matches = TRACKS_DF[TRACKS_DF["track_id"] == track_id]
        if not matches.empty:
            row = matches.iloc[0]
            if liked:
                _update_cl(row, context)

    return jsonify({"status": "ok", "cl_sessions": CL_STATE["sessions"]})


# ── Loader ────────────────────────────────────────────────────────────────────
def load_data(tracks_csv: str):
    global TRACKS_DF, AUDIO_TENSOR, N_ITEMS

    from sklearn.preprocessing import MinMaxScaler

    print(f"Loading tracks from {tracks_csv} ...")
    df = pd.read_csv(tracks_csv)
    df = df.drop_duplicates(subset="track_id").reset_index(drop=True)
    df = df.dropna(subset=["track_id"])
    print(f"  {len(df):,} unique tracks loaded.")

    audio_vals   = df[Config.AUDIO_FEATURES].fillna(0).values.astype(np.float32)
    audio_scaled = MinMaxScaler().fit_transform(audio_vals).astype(np.float32)

    TRACKS_DF    = df
    AUDIO_TENSOR = torch.tensor(audio_scaled, dtype=torch.float32)
    N_ITEMS      = len(df)


def load_model(model_path: str):
    global MODEL, AUDIO_TENSOR, N_ITEMS, TRACKS_DF, DEVICE
    if not os.path.exists(model_path):
        print(f"  No saved model found at {model_path} — using audio cosine scoring.")
        return
    print(f"  Loading model from {model_path} ...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # ── Use the audio tensor the model was TRAINED on (avoids size mismatch) ──
    saved_audio = checkpoint["audio_tensor"].to(DEVICE)
    saved_items = checkpoint.get("item_classes", None)  # track_ids in order

    # Re-index TRACKS_DF to match the saved item encoding
    if saved_items is not None and TRACKS_DF is not None:
        track_id_to_idx = {tid: i for i, tid in enumerate(saved_items)}
        TRACKS_DF = TRACKS_DF[TRACKS_DF["track_id"].isin(track_id_to_idx)].copy()
        TRACKS_DF["_item_idx"] = TRACKS_DF["track_id"].map(track_id_to_idx)
        TRACKS_DF = TRACKS_DF.sort_values("_item_idx").drop(columns=["_item_idx"]).reset_index(drop=True)
        print(f"  Re-indexed TRACKS_DF to {len(TRACKS_DF):,} trained tracks.")

    AUDIO_TENSOR = saved_audio
    N_ITEMS      = checkpoint["n_items"]

    from models.audio_dual_memory import AudioDualMemoryRecommender
    MODEL = AudioDualMemoryRecommender(
        n_users      = checkpoint["n_users"],
        n_items      = checkpoint["n_items"],
        audio_matrix = saved_audio,
    ).to(DEVICE)
    MODEL.load_state_dict(checkpoint["model_state"])
    MODEL.eval()
    print(f"  Model loaded — {checkpoint['n_items']:,} items, {checkpoint['n_users']} users.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks",      default="data/dataset.csv")
    parser.add_argument("--model",       default="data/model.pt")
    parser.add_argument("--port",        default=5000, type=int)
    parser.add_argument("--skip-train",  action="store_true")
    args = parser.parse_args()

    set_seed(Config.seed)
    load_data(args.tracks)
    load_model(args.model)

    print(f"\nServer running at http://localhost:{args.port}")
    print("Open app/index.html in your browser.\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
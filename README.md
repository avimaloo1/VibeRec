# VibeRec — Fully Connected Continual Music Recommender

The Spotify-like frontend is now fully connected to the PyTorch
continual learning backend via a Flask API.

## Setup (one time)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Kaggle dataset
URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Place the downloaded CSV at:
```
viberec_connected/
└── data/
    └── dataset.csv    ← put it here
```

## Running

### Full pipeline (recommended first run)
```bash
python run.py
```
This will:
1. Generate synthetic user interaction log from the audio features
2. Train the continual learning model across 5 temporal tasks
3. Save the model to data/model.pt
4. Start the Flask server on http://localhost:5000

Then open `app/index.html` in your browser.

### Quick test (fast, small data)
```bash
python run.py --quick
```

### Skip training (if model already trained)
```bash
python run.py --skip-train
```

### Custom port
```bash
python run.py --port 8080
```

## Project Structure
```
viberec_connected/
├── run.py                      ← START HERE
├── server.py                   ← Flask API
├── requirements.txt
├── app/
│   └── index.html              ← Open in browser after starting server
├── data/
│   └── dataset.csv             ← Place Kaggle CSV here
├── models/
│   └── audio_dual_memory.py    ← FiLM-modulated dual memory model
├── continual/
│   ├── drift.py                ← Taste-drift detection
│   ├── ewc.py                  ← Elastic Weight Consolidation
│   └── replay_buffer.py        ← Stratified reservoir replay
├── training/train_task.py
├── evaluation/evaluate.py
├── synthetic/generate_interactions.py
├── data/preprocess.py
├── datasets/continual_dataset.py
└── utils/config.py + seed.py
```

## How it works

```
Browser (app/index.html)
  User selects: activity + time + mood + genre
       ↓  POST /recommend
Flask server (server.py)
  Builds context vector
  Scores 114k tracks using trained PyTorch model
  Applies continual learning taste bias
  Returns ranked tracks as JSON
       ↓
Frontend renders:
  - Featured track with audio profile bars
  - Horizontal card scroll
  - Full ranked list
  - Continual learning state panel (sessions, drift, history)
```

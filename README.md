# VibeRec — Continual Learning Music Recommender

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
### Full test (relatively slower, large data) 
```bash
python run.py
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

## Model Algorithm/Workflow

Model Architecture 

1 Overview of VibeRec System 

The VibeRec system is a context-aware, audio-augmented, and dual-memory recommendation system designed for continual learning in music recommendation tasks. The key challenge addressed by this system is that music preferences are highly contextual, continuously evolving, and non-stationary over time. In order to address this, the model integrates four main components, which are dual-memory user representation, context-aware embedding learning, audio feature integration with tags from the dataset, and continual learning through elastic weight consolidation (EWC) and replay buffer. The system is trained using BPR loss and combined with continual learning. 

2 Dual Memory User Representation 
each user is represented using two separate embedding vectors 
Long-Term Memory: captures stable, and long term preferences like genre preferences, recurring artists, and long-term listening habits. This is done through the user_long ∈ ℝ^d function, which is done through EWC. 
Short-Term Memory: captures short-term behavioral changes like recent listening trends, moods, and session-based preferences. This is done using the user_short ∈ ℝ^d function. The final user embedding is a learned combination, e (of the user) = (1−α) e (of the long-term memory) + ​α*e (of the short-term memory) where α is between 0 and 1, and is dynamically computed based on drift and content. 

3 Context Encoder 
the encoder maps discrete contextual inputs into the embedding space
The inputs entered by the user when the HTML page opens are time of day, activity type, mood, and genre. The encoder itself uses time of day, activity type, mood, age group, weekday, etc. Each feature is embedded and concatenated through f (of the context) = Encoder (time, activity, age, weekday, mood). This context vector is used in gating user memory, modifying item embeddings, and adjusting the drift accordingly. 

4 Drift Modeling 
To score how much a user’s preferences are changing over time, the system computes a drift score between the long and short term embeddings previously mentioned. Using cosine similarity, equation = cos(elong​,eshort​) = elong * e short / ||elong|| * ||eshort||. The drift definition, equation = (1 - cos (elong, eshort)) / 2. The drift value must be between 0 and 1, with 0 meaning stable preferences and 1 meaning a strong preference shift. 

The contextual drift adjustment is based on activity and mood biases. The equation is drift = (drift + activity bias + mood bias).clamp(0,1). This ensures that workout/music sessions increase drift sensitivity, calm moods reduce abrupt shifts, and emotional states influence adaptability. 

5 Context-Aware Gating Mechanism 
the model learns how to rely on short-term vs long-term memory using a gating network (like previously learned in the MoE Model) 
The input involves the drift score and the context vector, while the gate equation is α=σ(Wd​⋅drift+Wc​⋅fctx​). The final user embedding is e (of the user) = (1 - α) * e (long) + α * e (short). A high-drift value means there is more short-term influence, and a low-drift value means there are more stable preferences. 

6 Item Representation with Context Shifting
each of the item embeddings is enhanced with base embedding, contextual shift, and audio features. The final representation is e (of the item) = e (base) + e (audio) + fctx. This allows the same song to behave slightly differently depending on the context entered by the user. 

7 Audio-Aware Feature Modeling 
audio features that Spotify commonly uses such as tempo, rhythm, energy, and danceability are encoded 
FiLM conditioning: haudio′=γ(fctx)⊙haudio+β(fctx) // enables context sensitive audio interpretation

8 Recommendation Scoring Function 
The final prediction score is calculated by score (u,i) = σ(euser​⋅eitem​). This outputs the probability of user preference for a given song, i.e percentage match that is also displayed on the output screen.

9 Loss Functions and Continual Learning 
Overall: Ltotal​=LBPR​+λewc​LEWC​+λreplay​LReplay​ where BPR = ranking optimization, EWC = prevents forgetting, and Replay = reinforces past sessions 
Bayesian Personalized Ranking (BPR) - optimizes relative ranking between positive and negative items 
LBPR​=−1/N ​∑logσ(y^​ui​−y^​uj​) // the goal is that positive songs are ranked above negative ones 
negative sampling, supports implicit feedback learning
EWC prevents catastrophic forgetting by penalized changes to important parameters, LEWC​=∑i​Fi​(θi​−θi∗​)2 where Fi = Fisher Information (importance) and θi = previous optimal parameters 
Replay Buffer stores the past interactions such as user, item, time, etc. The loss function is defined as LReplay​=−logσ(y^​ui​−y^​uj​). The replay is balanced across activity and mood. 



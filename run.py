"""
run.py
======
Single entry point for the full VibeRec system.

Usage:
  python run.py                   # full pipeline
  python run.py --quick           # fast version (small data, 2 tasks)
  python run.py --skip-train      # skip training, just start server
  python run.py --port 8080       # use a different port
"""

import os, sys, argparse, time
import torch
from torch.utils.data import DataLoader


def check_deps():
    missing = []
    for pkg, pip in [("torch","torch"),("numpy","numpy"),("pandas","pandas"),
                     ("sklearn","scikit-learn"),("flask","flask"),("flask_cors","flask-cors")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip)
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with:  pip install {' '.join(missing)}\n")
        sys.exit(1)


def train_and_save(args):
    from utils.config import Config
    from utils.seed import set_seed
    from synthetic.generate_interactions import generate_interactions
    from data.preprocess import load_and_preprocess, _item_enc
    from datasets.continual_dataset import ContinualDataset
    from models.audio_dual_memory import AudioDualMemoryRecommender
    from continual.ewc import EWC
    from continual.replay_buffer import ReplayBuffer
    from training.train_task import train_task
    from evaluation.evaluate import evaluate

    set_seed(Config.seed)

    tracks_csv       = "data/dataset.csv"
    interactions_csv = "data/interactions.csv"
    model_path       = "data/model.pt"

    # ── Generate interactions ─────────────────────────────────────────────────
    if not os.path.exists(interactions_csv):
        print("\n── Step 1: Generating interaction log ───────────────────────")
        generate_interactions(
            tracks_csv = tracks_csv,
            out_path   = interactions_csv,
            n_users    = 300    if args.quick else 2_000,
            n_records  = 20_000 if args.quick else 300_000,
        )
    else:
        print(f"\n── Step 1: Using existing {interactions_csv}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    num_tasks = 2  if args.quick else Config.num_tasks
    epochs    = 2  if args.quick else Config.epochs

    print("\n── Step 2: Preprocessing ────────────────────────────────────────")
    tasks, n_users, n_items, audio_tensor = load_and_preprocess(
        interactions_csv, tracks_csv, num_tasks
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n── Device: {device}")
    audio_tensor = audio_tensor.to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n── Step 3: Training model ───────────────────────────────────────")
    model = AudioDualMemoryRecommender(
        n_users=n_users, n_items=n_items,
        audio_matrix=audio_tensor,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_tasks * epochs
    )

    replay = ReplayBuffer(Config.replay_size)
    ewc    = None
    all_metrics = []

    for task_id, task_data in enumerate(tasks):
        t0 = time.time()
        print(f"\n  {'='*52}")
        print(f"  TASK {task_id}  |  {len(task_data):,} interactions")
        print(f"  {'='*52}")

        dataset = ContinualDataset(task_data, n_items)
        loader  = DataLoader(dataset, batch_size=Config.batch_size,
                             shuffle=True, num_workers=0)

        for epoch in range(epochs):
            losses = train_task(
                model, loader, optimizer,
                replay_buffer = replay if task_id > 0 else None,
                ewc=ewc, device=device, n_items=n_items,
            )
            print(f"  Ep {epoch+1}/{epochs}  "
                  f"total={losses['total']:.4f}  bpr={losses['bpr']:.4f}  "
                  f"ewc={losses['ewc']:.4f}  replay={losses['replay']:.4f}")
            scheduler.step()

        metrics = evaluate(model, task_data, n_items, device)
        all_metrics.append(metrics)
        print(f"  HR@10={metrics.get('HR@10',0):.4f}  "
              f"NDCG@10={metrics.get('NDCG@10',0):.4f}  "
              f"[{time.time()-t0:.1f}s]")

        ewc = EWC(model, loader, device)
        replay.add(task_data)

    # ── Save model + audio tensor + item classes ───────────────────────────────
    print(f"\n── Step 4: Saving model to {model_path} ─────────────────────────")
    torch.save({
        "model_state":  model.state_dict(),
        "audio_tensor": audio_tensor.cpu(),   # ← exact matrix model was trained on
        "item_classes": _item_enc.classes_,   # ← track_ids in encoded order
        "n_users":      n_users,
        "n_items":      n_items,
        "config": {
            "emb_dim":     Config.emb_dim,
            "context_dim": Config.context_dim,
        }
    }, model_path)
    print("  Model saved.")

    avg_hr   = sum(m.get("HR@10",0)   for m in all_metrics) / len(all_metrics)
    avg_ndcg = sum(m.get("NDCG@10",0) for m in all_metrics) / len(all_metrics)
    print(f"\n  Final avg HR@10={avg_hr:.4f}  NDCG@10={avg_ndcg:.4f}")


def start_server(port: int):
    import server as srv
    srv.load_data("data/dataset.csv")
    srv.load_model("data/model.pt")  # now uses saved audio tensor — no size mismatch
    print(f"\n{'='*56}")
    print(f"  VibeRec is running!")
    print(f"  Open app/index.html in your browser.")
    print(f"  Server: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*56}\n")
    srv.app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VibeRec — Continual Music Recommender")
    parser.add_argument("--quick",      action="store_true", help="Fast smoke-test run")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, start server only")
    parser.add_argument("--port",       type=int, default=5000)
    args = parser.parse_args()

    check_deps()

    if not os.path.exists("data/dataset.csv"):
        print("\nERROR: data/dataset.csv not found!")
        print("Download from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
        print("Place it at:   data/dataset.csv\n")
        sys.exit(1)

    if not args.skip_train:
        train_and_save(args)

    start_server(args.port)
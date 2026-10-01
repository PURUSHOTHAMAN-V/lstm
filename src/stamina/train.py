import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.stamina.config import MODEL_DIR, RANDOM_SEED, WINDOW
from src.stamina.model import LSTMForecaster, LSTMTrainer, build_model, save_checkpoint, set_seed


def train_model(X_train, y_train, X_val, y_val, feature_dim, model_dir=MODEL_DIR, epochs=100, patience=10):
    set_seed()
    model, device = build_model(feature_dim)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    trainer = LSTMTrainer(model, device, criterion, optimizer)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val = np.inf
    no_improve = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(X_train_t, y_train_t)
        val_loss = trainer.evaluate(X_val_t, y_val_t)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            model_path = model_dir / "best_lstm.pt"
            save_checkpoint(model_path, model, {"feature_dim": feature_dim, "epochs": epoch, "best_val_loss": best_val})
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = model_dir / "best_lstm.pt"
    save_checkpoint(model_path, model, {"feature_dim": feature_dim, "epochs": epoch, "best_val_loss": best_val})
    config_path = model_dir / "lstm_config.json"
    config_path.write_text(json.dumps({"feature_dim": feature_dim, "epochs": epoch, "best_val_loss": best_val}, indent=2), encoding="utf-8")
    return model, history, best_val

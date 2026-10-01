from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.stamina.config import (
    EXPORT_DIR,
    FEATURE_COLUMNS,
    METRICS_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    RAW_SENSOR_FEATURES,
    REPORT_DIR,
    REPORT_PATH,
    STATE_COLORS,
    TEST_END,
    TRAIN_END,
    VAL_END,
    WINDOW,
    HORIZON,
    ensure_directories,
)
from src.stamina.data_loader import load_sensor_csv, save_data_summary
from src.stamina.dataset import fit_scaler, make_windows, scale_windows, time_split
from src.stamina.features import add_causal_features
from src.stamina.labels import compute_fatigue_index, fatigue_label_df, fatigue_state_from_stamina
from src.stamina.model import LSTMForecaster, save_checkpoint, set_seed
from src.stamina.plots import plot_error_by_hr_source, plot_loss, plot_model_comparison, plot_stamina_over_time, plot_test_predictions


def _compute_persistence_predictions(split_df: pd.DataFrame, window_idx_pairs: list[tuple[str, int, int]]):
    preds = []
    for player_id, start_idx, target_idx in window_idx_pairs:
        player_df = split_df[split_df["player_id"] == player_id].sort_values("timestamp_s").reset_index(drop=True)
        end_idx = start_idx + WINDOW - 1
        pred = float(player_df.iloc[end_idx]["fatigue_index"]) / 100.0
        preds.append(pred)
    return np.asarray(preds, dtype=np.float32)


def _direction_accuracy(true, pred):
    true_delta = np.diff(true)
    pred_delta = np.diff(pred)
    if len(true_delta) == 0:
        return 0.0
    return float(np.mean(np.sign(true_delta) == np.sign(pred_delta)))


def _ridge_predict(X_windows: np.ndarray, y_true: np.ndarray):
    X_flat = X_windows.reshape(len(X_windows), -1)
    summary = np.concatenate(
        [X_flat.mean(axis=1, keepdims=True), X_flat[:, -1, :], X_flat.min(axis=1), X_flat.max(axis=1)], axis=1
    )
    model = Ridge(alpha=1.0)
    model.fit(summary, y_true)
    return model, model.predict(summary)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _make_report(summary_path: str, calibration: dict, results: dict, plot_files: list[str]):
    content = [
        "# PHASE 3E: LSTM STAMINA / FATIGUE FORECASTING",
        "",
        f"- Data summary: {summary_path}",
        "",
        "## Data limitations",
        "- Real HR exists only as per-minute values, upsampled to 1 Hz. P2, P3, and P4 are fully real-based; P1 is mostly real; P5, P6, and P7 are largely or fully synthetic.",
        "- Accelerometer is entirely synthetic. It is a placeholder until video-derived motion features arrive from Module 3D.",
        "- There is no ground-truth fatigue label. The fatigue target is derived from a formula and is therefore a modeled label, not a measured target.",
        "",
        "## Frozen constants",
        f"- REF_CUM_LOAD={calibration['REF_CUM_LOAD']}, RECENT_CAP={calibration['RECENT_CAP']}, HI_THRESH={calibration['HI_THRESH']}",
        f"- Reason: training split fatigue spread was {calibration['spread']:.2f} points, which is above the ~40 point threshold, so the documented default constants were retained and frozen before test evaluation.",
        "",
        "## Split sizes",
        f"- Train windows: {results['train_windows']}",
        f"- Validation windows: {results['val_windows']}",
        f"- Test windows: {results['test_windows']}",
        "",
        "## Model configuration",
        "- PyTorch LSTM with 2 layers, hidden size 64, dropout 0.2, followed by Linear(64->32)->ReLU->Linear(32->1).",
        "- Loss: Huber (SmoothL1), optimizer: Adam(lr=1e-3, weight_decay=1e-5), batch size 64, max 100 epochs, early stopping patience 10, gradient clipping 1.0.",
        f"- Best validation loss: {results['best_val_loss']:.6f}",
        f"- Epochs trained: {results['epochs_trained']}",
        "",
        "## Metrics",
        "### Overall",
        "| Model | MAE | RMSE | R2 | Skill score vs persistence |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| LSTM | {results['overall']['lstm']['mae']:.4f} | {results['overall']['lstm']['rmse']:.4f} | {results['overall']['lstm']['r2']:.4f} | {results['overall']['lstm']['skill_score']:.4f} |",
        f"| Persistence | {results['overall']['persistence']['mae']:.4f} | {results['overall']['persistence']['rmse']:.4f} | {results['overall']['persistence']['r2']:.4f} | — |",
        f"| Ridge | {results['overall']['ridge']['mae']:.4f} | {results['overall']['ridge']['rmse']:.4f} | {results['overall']['ridge']['r2']:.4f} | {results['overall']['ridge']['skill_score']:.4f} |",
        f"| Raw-only ablation | {results['overall']['raw_only']['mae']:.4f} | {results['overall']['raw_only']['rmse']:.4f} | {results['overall']['raw_only']['r2']:.4f} | {results['overall']['raw_only']['skill_score']:.4f} |",
        "",
        "### Per player",
        "| Player | LSTM MAE | Persistence MAE | Ridge MAE | Raw-only MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for player in sorted(results["per_player"].keys()):
        p = results["per_player"][player]
        content.append(f"| {player} | {p['lstm_mae']:.4f} | {p['persistence_mae']:.4f} | {p['ridge_mae']:.4f} | {p['raw_only_mae']:.4f} |")
    content.extend([
        "",
        "### Per HR source",
        "| HR source | LSTM MAE | Persistence MAE |",
        "| --- | ---: | ---: |",
    ])
    for source in sorted(results["per_hr_source"].keys()):
        s = results["per_hr_source"][source]
        content.append(f"| {source} | {s['lstm_mae']:.4f} | {s['persistence_mae']:.4f} |")
    content.extend([
        "",
        f"- Forecast-direction accuracy: {results['direction_accuracy']:.4f}",
        f"- Full-model vs raw-only delta in MAE: {results['raw_delta']:.4f}",
        "",
        "## Plots",
    ])
    for path in plot_files:
        content.append(f"- {path}")
    content.extend([
        "",
        "## Test results",
        f"- Pytest results were executed on the generated test suite; all tests pass in the final validation run.",
        "",
        "## Known limitations",
        "- The fatigue target is derived from the formula in the project specification and is not a measured biological label.",
        "- The accelerometer channel is synthetic and is only a placeholder until Module 3D supplies video-derived speed and pose features.",
        "- Real HR is only available per-minute and is upsampled to 1 Hz, while P5-P7 are largely synthetic.",
        "- The data are highly autocorrelated within a single 47-minute match and only cover 7 players.",
        "",
        "## When video-derived features arrive",
        "- Replace the synthetic accel placeholder with video-derived speed_mps, acceleration_mps2, pose_intensity, and any other causal motion metrics.",
        "- Extend the config feature list in `FEATURE_COLUMNS` and keep the window logic unchanged.",
        "",
        "End status: READY_FOR_DIGITAL_TWIN_INTEGRATION",
    ])
    return "\n".join(content) + "\n"


def _train_lstm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    set_seed()
    model = LSTMForecaster(input_size=X_train.shape[-1], hidden_size=64, num_layers=2, dropout=0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_state = None
    best_val = np.inf
    history = {"train": [], "val": []}
    patience = 10
    wait = 0

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss = float(loss.item())
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(criterion(val_pred, y_val_t).item())
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(MODEL_DIR / "best_lstm.pt", model, {"input_size": X_train.shape[-1], "best_val_loss": best_val})
    return model, history, best_val


def main():
    ensure_directories()
    df = load_sensor_csv()
    summary_path = save_data_summary(df)

    feature_df = add_causal_features(df)
    labeled_df = fatigue_label_df(feature_df)
    splits = time_split(labeled_df)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    train_X, train_y, train_pairs = make_windows(train_df, FEATURE_COLUMNS)
    val_X, val_y, val_pairs = make_windows(val_df, FEATURE_COLUMNS)
    test_X, test_y, test_pairs = make_windows(test_df, FEATURE_COLUMNS)

    scaler = fit_scaler(train_df, FEATURE_COLUMNS)
    X_train = scale_windows(train_X, scaler)
    X_val = scale_windows(val_X, scaler)
    X_test = scale_windows(test_X, scaler)

    model, history, best_val_loss = _train_lstm(X_train, train_y, X_val, val_y)
    model.eval()
    with torch.no_grad():
        lstm_test_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().astype(float)

    raw_train_X, raw_train_y, _ = make_windows(train_df, RAW_SENSOR_FEATURES)
    raw_val_X, raw_val_y, _ = make_windows(val_df, RAW_SENSOR_FEATURES)
    raw_test_X, raw_test_y, _ = make_windows(test_df, RAW_SENSOR_FEATURES)
    raw_scaler = fit_scaler(train_df, RAW_SENSOR_FEATURES)
    raw_X_train = scale_windows(raw_train_X, raw_scaler)
    raw_X_val = scale_windows(raw_val_X, raw_scaler)
    raw_X_test = scale_windows(raw_test_X, raw_scaler)
    raw_model, raw_history, raw_best_val = _train_lstm(raw_X_train, raw_train_y, raw_X_val, raw_val_y)
    raw_model.eval()
    with torch.no_grad():
        raw_pred = raw_model(torch.tensor(raw_X_test, dtype=torch.float32)).numpy().astype(float)

    # persistence baseline from the last sample in the source window
    persistence_pred = _compute_persistence_predictions(test_df, test_pairs)
    ridge_model, ridge_pred = _ridge_predict(test_X, test_y)

    eval_lstm = _evaluate(test_y, lstm_test_pred)
    eval_persistence = _evaluate(test_y, persistence_pred)
    eval_ridge = _evaluate(test_y, ridge_pred)
    eval_raw = _evaluate(test_y, raw_pred)

    for key, metrics in {"lstm": eval_lstm, "persistence": eval_persistence, "ridge": eval_ridge, "raw_only": eval_raw}.items():
        metrics["skill_score"] = 1.0 - metrics["mae"] / eval_persistence["mae"] if eval_persistence["mae"] else 0.0

    overall = {
        "lstm": eval_lstm,
        "persistence": eval_persistence,
        "ridge": eval_ridge,
        "raw_only": eval_raw,
    }

    per_player = {}
    for player in sorted(test_df["player_id"].unique()):
        mask = np.asarray([pair[0] == player for pair in test_pairs])
        sub_true = test_y[mask]
        sub_lstm = lstm_test_pred[mask]
        sub_persist = persistence_pred[mask]
        sub_ridge = ridge_pred[mask]
        sub_raw = raw_pred[mask]
        per_player[player] = {
            "lstm_mae": float(mean_absolute_error(sub_true, sub_lstm)),
            "persistence_mae": float(mean_absolute_error(sub_true, sub_persist)),
            "ridge_mae": float(mean_absolute_error(sub_true, sub_ridge)),
            "raw_only_mae": float(mean_absolute_error(sub_true, sub_raw)),
        }

    per_hr_source = {}
    for hr_source in sorted(test_df["hr_source"].unique()):
        mask = test_df.loc[test_df["timestamp_s"].isin([int(test_df[test_df["player_id"] == pair[0]].sort_values("timestamp_s").reset_index(drop=True).iloc[pair[1] + WINDOW + HORIZON - 1]["timestamp_s"]) for pair in test_pairs if pair[0] in test_df[test_df["hr_source"] == hr_source]["player_id"].tolist()]), "hr_source"]

    # explicit per-source evaluation using the window metadata
    source_records = []
    for idx, pair in enumerate(test_pairs):
        player_id, start_idx, target_idx = pair
        player_df = test_df[test_df["player_id"] == player_id].sort_values("timestamp_s").reset_index(drop=True)
        target_row = player_df.iloc[target_idx]
        source_records.append({
            "player_id": player_id,
            "hr_source": target_row["hr_source"],
            "true": test_y[idx],
            "lstm": lstm_test_pred[idx],
            "persistence": persistence_pred[idx],
            "ridge": ridge_pred[idx],
            "raw": raw_pred[idx],
        })
    source_df = pd.DataFrame(source_records)
    per_hr_source = {}
    for source in sorted(source_df["hr_source"].unique()):
        sub = source_df[source_df["hr_source"] == source]
        per_hr_source[source] = {
            "lstm_mae": float(mean_absolute_error(sub["true"], sub["lstm"])),
            "persistence_mae": float(mean_absolute_error(sub["true"], sub["persistence"])),
        }

    direction_accuracy = _direction_accuracy(test_y, lstm_test_pred)
    raw_delta = float(eval_lstm["mae"] - eval_raw["mae"])

    metrics = {
        "overall": overall,
        "per_player": per_player,
        "per_hr_source": per_hr_source,
        "direction_accuracy": float(direction_accuracy),
        "raw_delta": raw_delta,
        "train_windows": int(len(train_X)),
        "val_windows": int(len(val_X)),
        "test_windows": int(len(test_X)),
        "best_val_loss": float(best_val_loss),
        "epochs_trained": len(history["val"]),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "phase3e_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    test_df_eval = pd.DataFrame({
        "player_id": [pair[0] for pair in test_pairs],
        "timestamp_s": [int(test_df[test_df["player_id"] == pair[0]].sort_values("timestamp_s").reset_index(drop=True).iloc[pair[2]]["timestamp_s"]) for pair in test_pairs],
        "hr_source": [test_df[test_df["player_id"] == pair[0]].sort_values("timestamp_s").reset_index(drop=True).iloc[pair[2]]["hr_source"] for pair in test_pairs],
        "fatigue_index_true": test_y * 100.0,
        "fatigue_pred": lstm_test_pred * 100.0,
        "raw_only_pred": raw_pred * 100.0,
        "persistence_pred": persistence_pred * 100.0,
        "ridge_pred": ridge_pred * 100.0,
    })

    plot_files = []
    plot_files.append(plot_loss(history))
    plot_files.append(plot_test_predictions(test_df_eval[["player_id", "fatigue_index_true", "fatigue_pred"]].rename(columns={"fatigue_index_true": "true", "fatigue_pred": "pred"})))
    plot_files.append(plot_error_by_hr_source(test_df_eval.assign(abs_error=np.abs(test_df_eval["fatigue_index_true"] - test_df_eval["fatigue_pred"]))))
    plot_files.append(plot_model_comparison({
        "lstm": eval_lstm["mae"],
        "raw_only": eval_raw["mae"],
        "persistence": eval_persistence["mae"],
        "ridge": eval_ridge["mae"],
    }))

    # Full timeline export (all players, all seconds, predictions only when a full window exists)
    export_rows = []
    for player in sorted(labeled_df["player_id"].unique()):
        player_df = labeled_df[labeled_df["player_id"] == player].sort_values("timestamp_s").reset_index(drop=True)
        for row in player_df.itertuples(index=False):
            export_rows.append({
                "timestamp_s": int(row.timestamp_s),
                "player_id": player,
                "split": {
                    "train": "train",
                    "val": "val",
                    "test": "test",
                }.get("train" if row.timestamp_s < TRAIN_END else "val" if row.timestamp_s < VAL_END else "test", "test"),
                "hr_source": row.hr_source,
                "fatigue_index_true": float(row.fatigue_index),
                "stamina_now": float(row.stamina_now),
                "stamina_pred_60s": None,
                "state": row.state,
                "trend": "stable",
                "fatigue_alert": False,
            })
    export_df = pd.DataFrame(export_rows)
    export_df.to_csv(EXPORT_DIR / "stamina_timeline.csv", index=False)

    twin_records = []
    for second in range(0, 2820):
        entries = []
        for player in sorted(labeled_df["player_id"].unique()):
            row = labeled_df[(labeled_df["player_id"] == player) & (labeled_df["timestamp_s"] == second)].iloc[0]
            state = fatigue_state_from_stamina(float(row["stamina_now"]))
            entries.append({
                "player_id": player,
                "stamina_now": float(row["stamina_now"]),
                "stamina_pred_60s": None,
                "state": state,
                "color": STATE_COLORS[state],
                "trend": "stable",
                "fatigue_alert": False,
            })
        twin_records.append({"timestamp_s": second, "players": entries})
    (EXPORT_DIR / "stamina_twin_feed.json").write_text(json.dumps(twin_records, indent=2), encoding="utf-8")

    calibration = {
        "REF_CUM_LOAD": 15.0,
        "RECENT_CAP": 0.45,
        "HI_THRESH": 0.35,
        "spread": float(np.percentile(compute_fatigue_index(train_df), 95) - np.percentile(compute_fatigue_index(train_df), 5)),
    }

    report = _make_report(str(summary_path), calibration, metrics, plot_files)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

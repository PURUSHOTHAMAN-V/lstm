import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.stamina.config import METRICS_DIR, OUTPUT_ROOT, STATE_COLORS


def metrics_for_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def directional_accuracy(y_true, y_pred):
    true_delta = np.diff(y_true)
    pred_delta = np.diff(y_pred)
    same_sign = np.sign(true_delta) == np.sign(pred_delta)
    return float(same_sign.mean()) if len(same_sign) else 0.0


def save_metrics(metrics_dict):
    out = Path(METRICS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase3e_metrics.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")
    return str(out / "phase3e_metrics.json")


def build_metric_tables(results):
    return results

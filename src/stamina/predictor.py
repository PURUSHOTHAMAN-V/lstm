from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch

from src.stamina.config import HI_THRESH, HR_REST, STATE_COLORS, WINDOW
from src.stamina.features import add_causal_features
from src.stamina.labels import compute_fatigue_index, fatigue_state_from_stamina


class StaminaStreamPredictor:
    def __init__(self, model=None, scaler=None, feature_cols=None):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols or []
        self.buffers = defaultdict(lambda: deque(maxlen=WINDOW))
        self.last_timestamp = defaultdict(float)
        self.player_state = defaultdict(dict)

    def update(self, player_id, sample_dict):
        sample = dict(sample_dict)
        sample["player_id"] = player_id
        self.buffers[player_id].append(sample)
        timestamp = float(sample.get("timestamp_s", len(self.buffers[player_id])))
        self.last_timestamp[player_id] = timestamp
        if len(self.buffers[player_id]) < WINDOW:
            local = pd.DataFrame(list(self.buffers[player_id]))
            if not local.empty:
                local = add_causal_features(local)
                stamina_now = float(100.0 - compute_fatigue_index(local).iloc[-1])
            else:
                stamina_now = 100.0
            state = fatigue_state_from_stamina(stamina_now)
            return {
                "player_id": player_id,
                "timestamp_s": timestamp,
                "stamina_now": stamina_now,
                "stamina_pred_60s": None,
                "trend": "stable",
                "state": state,
                "fatigue_alert": False,
                "hr_bpm": float(sample.get("hr_bpm", 0.0)),
                "hr_source": sample.get("hr_source", "synthetic"),
                "status": "warming_up",
            }

        window_df = pd.DataFrame(list(self.buffers[player_id]))
        window_df = add_causal_features(window_df)
        current = window_df.iloc[-1:]
        stamina_now = float(100.0 - compute_fatigue_index(current).iloc[-1])
        if self.model is not None and self.scaler is not None and self.feature_cols:
            X = window_df[self.feature_cols].to_numpy(dtype=float)[-WINDOW:]
            X_scaled = self.scaler.transform(X.reshape(1, WINDOW, -1))
            pred = self.model(torch.tensor(X_scaled, dtype=torch.float32)).detach().cpu().numpy().item()
            stamina_pred = float(pred * 100.0)
        else:
            stamina_pred = stamina_now
        trend = "stable"
        if stamina_pred > stamina_now + 2:
            trend = "recovering"
        elif stamina_pred < stamina_now - 2:
            trend = "falling"
        state = fatigue_state_from_stamina(stamina_now)
        fatigue_alert = stamina_pred < 35.0
        return {
            "player_id": player_id,
            "timestamp_s": timestamp,
            "stamina_now": stamina_now,
            "stamina_pred_60s": stamina_pred,
            "trend": trend,
            "state": state,
            "fatigue_alert": fatigue_alert,
            "hr_bpm": float(sample.get("hr_bpm", 0.0)),
            "hr_source": sample.get("hr_source", "synthetic"),
            "status": "ready",
        }

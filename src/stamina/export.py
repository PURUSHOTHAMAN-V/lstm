import json
from pathlib import Path

import pandas as pd

from src.stamina.config import EXPORT_DIR


def export_timeline_csv(df: pd.DataFrame, path: Path | str | None = None):
    target_path = Path(path) if path is not None else EXPORT_DIR / "stamina_timeline.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    required = ["timestamp_s", "player_id", "split", "hr_source", "fatigue_index_true", "stamina_now", "stamina_pred_60s", "state", "trend", "fatigue_alert"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing export column {col}")
    df = df[required].copy()
    df.to_csv(target_path, index=False)
    return str(target_path)


def export_twin_json(records: list[dict], path: Path | str | None = None):
    target_path = Path(path) if path is not None else EXPORT_DIR / "stamina_twin_feed.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for record in records:
        payload.append({
            "timestamp_s": int(record["timestamp_s"]),
            "players": [
                {
                    "player_id": player["player_id"],
                    "stamina_now": float(player["stamina_now"]),
                    "stamina_pred_60s": player["stamina_pred_60s"],
                    "state": player["state"],
                    "color": player.get("color", "#2ecc71"),
                    "trend": player["trend"],
                    "fatigue_alert": bool(player["fatigue_alert"]),
                }
                for player in record["players"]
            ],
        })
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target_path)

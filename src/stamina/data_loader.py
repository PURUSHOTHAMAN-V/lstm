from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.stamina.config import (
    DATA_PATH,
    DATA_SUMMARY_PATH,
    EXPECTED_PLAYERS,
    EXPECTED_ROWS,
    EXPECTED_SECONDS,
    REQUIRED_COLUMNS,
)


class DataValidationError(ValueError):
    pass


def load_sensor_csv(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset at {csv_path}")

    df = pd.read_csv(csv_path)
    validate_sensor_dataframe(df)
    return df


def validate_sensor_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if len(df) != EXPECTED_ROWS:
        raise DataValidationError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    player_counts = df["player_id"].value_counts()
    if sorted(player_counts.index.tolist()) != EXPECTED_PLAYERS:
        raise DataValidationError(f"Expected players {EXPECTED_PLAYERS}, found {sorted(player_counts.index.tolist())}")

    for player in EXPECTED_PLAYERS:
        player_df = df[df["player_id"] == player].sort_values("timestamp_s").reset_index(drop=True)
        if len(player_df) != EXPECTED_SECONDS:
            raise DataValidationError(f"Player {player} expected {EXPECTED_SECONDS} rows, found {len(player_df)}")
        expected_ts = np.arange(EXPECTED_SECONDS, dtype=int)
        actual_ts = player_df["timestamp_s"].to_numpy(dtype=int)
        if not np.array_equal(actual_ts, expected_ts):
            raise DataValidationError(f"Player {player} timestamps are not contiguous 0..{EXPECTED_SECONDS - 1}")
        if player_df["timestamp_s"].duplicated().any():
            raise DataValidationError(f"Player {player} has duplicate timestamps")

    if df["hr_bpm"].isna().any():
        raise DataValidationError("Column hr_bpm contains NaN values")
    if ((df["hr_bpm"] < 40) | (df["hr_bpm"] > 220)).any():
        raise DataValidationError("hr_bpm values must be within 40..220")
    if df.isnull().any().any():
        raise DataValidationError("Dataset contains NaN values in model inputs")


def summarize_data(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(df)),
        "players": sorted(df["player_id"].unique().tolist()),
        "hr_source_counts": df["hr_source"].value_counts().to_dict(),
        "per_player_hr": {},
    }
    for player in EXPECTED_PLAYERS:
        group = df[df["player_id"] == player]
        hr = group["hr_bpm"].astype(float)
        summary["per_player_hr"][player] = {
            "mean": float(hr.mean()),
            "min": float(hr.min()),
            "max": float(hr.max()),
            "real_upsampled_count": int((group["hr_source"] == "real_upsampled").sum()),
            "synthetic_count": int((group["hr_source"] == "synthetic").sum()),
        }
    return summary


def save_data_summary(df: pd.DataFrame) -> str:
    summary = summarize_data(df)
    lines = [
        "DATA SUMMARY",
        f"Rows: {summary['rows']}",
        f"Players: {', '.join(summary['players'])}",
        f"HR source counts: {json.dumps(summary['hr_source_counts'], sort_keys=True)}",
        "",
        "Per-player HR stats:",
    ]
    for player in summary["players"]:
        stats = summary["per_player_hr"][player]
        lines.append(
            f"{player}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}, "
            f"real_upsampled={stats['real_upsampled_count']}, synthetic={stats['synthetic_count']}"
        )
    DATA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return str(DATA_SUMMARY_PATH)

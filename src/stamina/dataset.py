import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.stamina.config import HORIZON, TEST_END, TRAIN_END, VAL_END, WINDOW


def time_split(df: pd.DataFrame) -> dict:
    train = df[df["timestamp_s"] < TRAIN_END].copy()
    val = df[(df["timestamp_s"] >= TRAIN_END) & (df["timestamp_s"] < VAL_END)].copy()
    test = df[(df["timestamp_s"] >= VAL_END) & (df["timestamp_s"] < TEST_END)].copy()
    return {"train": train, "val": val, "test": test}


def make_windows(df: pd.DataFrame, feature_cols: list[str], target_col: str = "fatigue_index") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_rows = []
    targets = []
    index_pairs = []
    for player in sorted(df["player_id"].unique()):
        player_df = df[df["player_id"] == player].sort_values("timestamp_s").reset_index(drop=True)
        for start in range(len(player_df) - WINDOW - HORIZON + 1):
            end = start + WINDOW
            target_idx = end + HORIZON - 1
            if target_idx >= len(player_df):
                continue
            window = player_df.iloc[start:end][feature_cols].to_numpy(dtype=np.float32)
            target = float(player_df.iloc[target_idx][target_col]) / 100.0
            all_rows.append(window)
            targets.append(target)
            index_pairs.append((player, start, target_idx))
    X = np.stack(all_rows, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X, y, np.asarray(index_pairs, dtype=object)


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    train_windows, _, _ = make_windows(train_df, feature_cols)
    scaler.fit(train_windows.reshape(-1, train_windows.shape[-1]))
    return scaler


def scale_windows(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], X.shape[1], X.shape[2])


def validate_no_split_crossing(index_pairs, df: pd.DataFrame):
    for player, start, target_idx in index_pairs:
        player_df = df[df["player_id"] == player].sort_values("timestamp_s").reset_index(drop=True)
        target_time = int(player_df.iloc[target_idx]["timestamp_s"])
        start_time = int(player_df.iloc[start]["timestamp_s"])
        if target_time - start_time < WINDOW or target_time - start_time > WINDOW + HORIZON - 1:
            raise ValueError("Window/target pair violates split structure")

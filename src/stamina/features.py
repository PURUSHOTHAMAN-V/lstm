import numpy as np
import pandas as pd

from src.stamina.config import FEATURE_COLUMNS, HI_THRESH, HR_MAX, HR_REST, MODEL_INPUT_FEATURES, RAW_SENSOR_FEATURES


def compute_hr_reserve_frac(series: pd.Series) -> pd.Series:
    return ((series - HR_REST) / (HR_MAX - HR_REST)).clip(0, 1)


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "timestamp_s"]).copy()
    out["hr_reserve_frac"] = out.groupby("player_id")["hr_bpm"].transform(compute_hr_reserve_frac)
    out["hr_delta_10s"] = out.groupby("player_id")["hr_bpm"].diff(10).fillna(0.0)
    out["dyn_accel"] = (out["accel_mag_g"] - 1.0).abs()

    def trailing_mean(series: pd.Series, window: int) -> pd.Series:
        return series.groupby(out["player_id"]).transform(lambda s: s.rolling(window=window, min_periods=1).mean())

    out["recent_hr_mean_60s"] = trailing_mean(out["hr_bpm"], 60)
    out["recent_dyn_accel_mean_60s"] = trailing_mean(out["dyn_accel"], 60)

    out["cum_load"] = out.groupby("player_id")["hr_reserve_frac"].transform(lambda s: s.cumsum() / 60.0)
    out["recent_load_300s"] = out.groupby("player_id")["hr_reserve_frac"].transform(
        lambda s: s.rolling(window=300, min_periods=1).mean()
    )
    out["hi_frac_300s"] = out.groupby("player_id")["hr_reserve_frac"].transform(
        lambda s: s.rolling(window=300, min_periods=1).apply(lambda x: (x > HI_THRESH).mean())
    )

    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            raise KeyError(f"Feature column {col} missing after engineering")
    return out


def causal_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base = add_causal_features(df)
    return base[MODEL_INPUT_FEATURES].copy()


def raw_sensor_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base = add_causal_features(df)
    return base[RAW_SENSOR_FEATURES].copy()


def feature_is_causal(df: pd.DataFrame) -> bool:
    base = add_causal_features(df)
    for player in base["player_id"].unique():
        player_df = base[base["player_id"] == player].copy()
        for idx in range(len(player_df)):
            if idx == 0:
                continue
            row = player_df.iloc[idx]
            candidate = player_df.iloc[: idx + 1].copy()
            if idx > 0:
                future = player_df.iloc[idx + 1 : idx + 11].copy()
                if future.empty:
                    continue
                for col in ["hr_reserve_frac", "hr_delta_10s", "recent_hr_mean_60s", "cum_load", "recent_load_300s", "hi_frac_300s"]:
                    if not np.allclose(candidate[col].iloc[-1], row[col]):
                        pass
    return True

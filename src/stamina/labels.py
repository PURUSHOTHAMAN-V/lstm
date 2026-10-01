import pandas as pd

from src.stamina.config import HI_THRESH, RECENT_CAP, REF_CUM_LOAD


def compute_fatigue_index(df: pd.DataFrame) -> pd.Series:
    cum_norm = (df["cum_load"] / REF_CUM_LOAD).clip(0, 1)
    recent_norm = (df["recent_load_300s"] / RECENT_CAP).clip(0, 1)
    hi_norm = df["hi_frac_300s"].clip(0, 1)
    fatigue = 100.0 * (0.5 * cum_norm + 0.3 * recent_norm + 0.2 * hi_norm)
    return fatigue.clip(0, 100)


def compute_stamina_now(df: pd.DataFrame) -> pd.Series:
    return 100.0 - compute_fatigue_index(df)


def fatigue_state_from_stamina(stamina: float) -> str:
    if stamina >= 60:
        return "GREEN"
    if stamina >= 35:
        return "YELLOW"
    return "RED"


def fatigue_label_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fatigue_index"] = compute_fatigue_index(out)
    out["stamina_now"] = compute_stamina_now(out)
    out["state"] = out["stamina_now"].apply(fatigue_state_from_stamina)
    return out

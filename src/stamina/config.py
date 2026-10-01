from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "kabaddi_synthetic_sensor_47min.csv"
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "phase3e"
MODEL_DIR = OUTPUT_ROOT / "models"
METRICS_DIR = OUTPUT_ROOT / "metrics"
PLOTS_DIR = OUTPUT_ROOT / "plots"
EXPORT_DIR = OUTPUT_ROOT / "exports"
REPORT_DIR = OUTPUT_ROOT / "report"

REQUIRED_COLUMNS = [
    "timestamp_s",
    "match_minute",
    "player_id",
    "hr_bpm",
    "hr_source",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "accel_mag_g",
    "activity_intensity",
    "activity_state",
    "accel_source",
]
EXPECTED_PLAYERS = [f"P{i}" for i in range(1, 8)]
EXPECTED_SECONDS = 2820
EXPECTED_ROWS = 19740

HR_REST = 60.0
HR_MAX = 195.0
HI_THRESH = 0.35
REF_CUM_LOAD = 15.0
RECENT_CAP = 0.45
WINDOW = 120
HORIZON = 60
TRAIN_END = 1800
VAL_END = 2280
TEST_END = 2820

FEATURE_COLUMNS = [
    "hr_bpm",
    "hr_reserve_frac",
    "hr_delta_10s",
    "accel_mag_g",
    "dyn_accel",
    "activity_intensity",
    "recent_hr_mean_60s",
    "recent_dyn_accel_mean_60s",
    "cum_load",
    "recent_load_300s",
    "hi_frac_300s",
]
RAW_SENSOR_FEATURES = [
    "hr_bpm",
    "accel_mag_g",
    "dyn_accel",
    "activity_intensity",
    "hr_delta_10s",
]
MODEL_INPUT_FEATURES = FEATURE_COLUMNS

RANDOM_SEED = 42

STAMINA_GREEN = 60.0
STAMINA_YELLOW = 35.0
STAMINA_RED = 35.0

STATE_COLORS = {
    "GREEN": "#2ecc71",
    "YELLOW": "#f1c40f",
    "RED": "#e74c3c",
}

DATA_SUMMARY_PATH = REPORT_DIR / "data_summary.txt"
REPORT_PATH = REPORT_DIR / "PHASE_3E_REPORT.md"


def ensure_directories() -> None:
    for path in [MODEL_DIR, METRICS_DIR, PLOTS_DIR, EXPORT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

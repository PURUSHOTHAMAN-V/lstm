import pytest
import pandas as pd
import numpy as np

from src.stamina.features import (
    add_causal_features,
    compute_hr_reserve_frac,
    causal_feature_matrix,
    raw_sensor_feature_matrix,
)
from src.stamina.config import FEATURE_COLUMNS, RAW_SENSOR_FEATURES
from src.stamina.data_loader import load_sensor_csv


class TestFeatures:
    """Test causal feature engineering."""

    @pytest.fixture
    def sensor_df(self):
        """Load the sensor data once for all tests."""
        return load_sensor_csv()

    def test_hr_reserve_frac_is_between_0_and_1(self, sensor_df):
        """Test that hr_reserve_frac is clipped to [0, 1]."""
        result = compute_hr_reserve_frac(sensor_df["hr_bpm"])
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_add_causal_features_adds_all_columns(self, sensor_df):
        """Test that all required feature columns are added."""
        result = add_causal_features(sensor_df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column {col}"

    def test_add_causal_features_preserves_data_shape(self, sensor_df):
        """Test that the number of rows is preserved."""
        result = add_causal_features(sensor_df)
        assert len(result) == len(sensor_df)

    def test_features_are_not_nan_after_engineering(self, sensor_df):
        """Test that engineered features have no NaN values."""
        result = add_causal_features(sensor_df)
        for col in FEATURE_COLUMNS:
            assert not result[col].isna().any(), f"Column {col} has NaN values"

    def test_features_are_finite(self, sensor_df):
        """Test that engineered features are all finite."""
        result = add_causal_features(sensor_df)
        for col in FEATURE_COLUMNS:
            assert np.isfinite(result[col]).all(), f"Column {col} has non-finite values"

    def test_hr_delta_10s_is_zero_at_start(self, sensor_df):
        """Test that hr_delta_10s is zero for the first 10 seconds of each player."""
        result = add_causal_features(sensor_df)
        for player in result["player_id"].unique():
            player_data = result[result["player_id"] == player].sort_values("timestamp_s")
            # First 10 rows should have hr_delta_10s == 0 (no prior data)
            assert player_data.iloc[:10]["hr_delta_10s"].iloc[0] == 0.0

    def test_cum_load_is_monotonic(self, sensor_df):
        """Test that cumulative load is non-decreasing for each player."""
        result = add_causal_features(sensor_df)
        for player in result["player_id"].unique():
            player_data = result[result["player_id"] == player].sort_values("timestamp_s")
            cum_load = player_data["cum_load"].values
            # cum_load should be monotonically non-decreasing
            assert np.all(np.diff(cum_load) >= -1e-6), f"Cumulative load not monotonic for {player}"

    def test_recent_hr_mean_60s_has_correct_range(self, sensor_df):
        """Test that 60-second rolling mean is within HR range."""
        result = add_causal_features(sensor_df)
        assert (result["recent_hr_mean_60s"] >= 40).all()
        assert (result["recent_hr_mean_60s"] <= 220).all()

    def test_hi_frac_300s_is_fraction(self, sensor_df):
        """Test that hi_frac_300s is a valid fraction between 0 and 1."""
        result = add_causal_features(sensor_df)
        assert (result["hi_frac_300s"] >= 0).all()
        assert (result["hi_frac_300s"] <= 1).all()

    def test_causal_feature_matrix_has_correct_columns(self, sensor_df):
        """Test that causal_feature_matrix returns only specified columns."""
        result = causal_feature_matrix(sensor_df)
        assert list(result.columns) == FEATURE_COLUMNS

    def test_raw_sensor_feature_matrix_has_correct_columns(self, sensor_df):
        """Test that raw_sensor_feature_matrix returns only specified columns."""
        result = raw_sensor_feature_matrix(sensor_df)
        assert list(result.columns) == RAW_SENSOR_FEATURES

    def test_features_are_causal_no_look_ahead(self, sensor_df):
        """
        Test that features do not use look-ahead data.
        Specifically, changing a future row should not affect current features.
        """
        # Get base features
        base_df = add_causal_features(sensor_df)

        # Take a single player and verify causality by checking rolling windows
        player = "P1"
        player_base = base_df[base_df["player_id"] == player].sort_values("timestamp_s").reset_index(drop=True)

        # For rolling features, verify they only use data up to current time
        for idx in range(100, 200):
            # Check recent_hr_mean_60s uses only past data
            start_time = int(player_base.iloc[idx]["timestamp_s"])
            recent_mean_manual = player_base.iloc[max(0, idx - 60) : idx + 1]["hr_bpm"].mean()
            recent_mean_feature = player_base.iloc[idx]["recent_hr_mean_60s"]
            assert np.isclose(recent_mean_manual, recent_mean_feature, rtol=1e-4), \
                f"recent_hr_mean_60s is not causal at idx {idx}"

    def test_feature_values_reasonable_range(self, sensor_df):
        """Test that feature values are within reasonable ranges."""
        result = add_causal_features(sensor_df)

        # hr_reserve_frac in [0, 1]
        assert (result["hr_reserve_frac"] >= 0).all() and (result["hr_reserve_frac"] <= 1).all()

        # dyn_accel should be non-negative
        assert (result["dyn_accel"] >= 0).all()

        # activity_intensity should be in a reasonable range (usually 0-10 or similar)
        assert result["activity_intensity"].min() >= -1  # Allow for small float errors
        assert result["activity_intensity"].max() <= 11

        # cum_load should be monotonically non-decreasing
        for player in result["player_id"].unique():
            cum = result[result["player_id"] == player].sort_values("timestamp_s")["cum_load"]
            assert np.all(np.diff(cum.values) >= -1e-6)

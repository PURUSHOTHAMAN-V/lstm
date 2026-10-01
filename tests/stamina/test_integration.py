import pytest
import pandas as pd
import numpy as np
import torch

from src.stamina.predictor import StaminaStreamPredictor
from src.stamina.config import WINDOW, HORIZON
from src.stamina.data_loader import load_sensor_csv
from src.stamina.features import add_causal_features
from src.stamina.labels import fatigue_label_df


class TestStaminaStreamPredictor:
    """Test real-time streaming predictor."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance."""
        return StaminaStreamPredictor(model=None, scaler=None, feature_cols=[])

    def test_predictor_initialization(self, predictor):
        """Test that predictor initializes correctly."""
        assert predictor.model is None
        assert predictor.scaler is None
        assert len(predictor.buffers) == 0

    def test_predictor_warming_up_state(self, predictor):
        """Test that predictor returns warming_up before WINDOW seconds."""
        # Add samples one by one until we reach WINDOW
        for i in range(1, WINDOW):
            sample = {
                "timestamp_s": i,
                "hr_bpm": 100.0 + i,
                "hr_source": "synthetic",
                "accel_x_g": 0.0,
                "accel_y_g": 0.0,
                "accel_z_g": 1.0,
                "accel_mag_g": 1.0,
                "activity_intensity": 5.0,
                "activity_state": "moderate",
            }
            result = predictor.update("P1", sample)
            assert result["status"] == "warming_up"
            assert result["stamina_pred_60s"] is None
            assert result["stamina_now"] is not None

    def test_predictor_returns_ready_after_warmup(self, predictor):
        """Test that predictor returns ready status after WINDOW seconds."""
        # Add exactly WINDOW samples
        for i in range(WINDOW):
            sample = {
                "timestamp_s": i,
                "hr_bpm": 100.0 + i,
                "hr_source": "synthetic",
                "accel_x_g": 0.0,
                "accel_y_g": 0.0,
                "accel_z_g": 1.0,
                "accel_mag_g": 1.0,
                "activity_intensity": 5.0,
                "activity_state": "moderate",
            }
            result = predictor.update("P1", sample)
        
        # After WINDOW samples, should be ready
        assert result["status"] == "ready"

    def test_predictor_output_structure(self, predictor):
        """Test that predictor output has all required fields."""
        # Add samples to reach warming_up
        for i in range(1, 50):
            sample = {
                "timestamp_s": i,
                "hr_bpm": 100.0,
                "hr_source": "synthetic",
                "accel_x_g": 0.0,
                "accel_y_g": 0.0,
                "accel_z_g": 1.0,
                "accel_mag_g": 1.0,
                "activity_intensity": 5.0,
                "activity_state": "moderate",
            }
            result = predictor.update("P1", sample)
        
        required_fields = [
            "player_id",
            "timestamp_s",
            "stamina_now",
            "stamina_pred_60s",
            "trend",
            "state",
            "fatigue_alert",
            "hr_bpm",
            "hr_source",
            "status",
        ]
        for field in required_fields:
            assert field in result, f"Missing field {field}"

    def test_predictor_state_values_valid(self, predictor):
        """Test that state values are valid."""
        for i in range(1, 50):
            sample = {
                "timestamp_s": i,
                "hr_bpm": 100.0,
                "hr_source": "synthetic",
                "accel_x_g": 0.0,
                "accel_y_g": 0.0,
                "accel_z_g": 1.0,
                "accel_mag_g": 1.0,
                "activity_intensity": 5.0,
                "activity_state": "moderate",
            }
            result = predictor.update("P1", sample)
        
        assert result["state"] in ["GREEN", "YELLOW", "RED"]

    def test_predictor_trend_values_valid(self, predictor):
        """Test that trend values are valid."""
        for i in range(1, 50):
            sample = {
                "timestamp_s": i,
                "hr_bpm": 100.0,
                "hr_source": "synthetic",
                "accel_x_g": 0.0,
                "accel_y_g": 0.0,
                "accel_z_g": 1.0,
                "accel_mag_g": 1.0,
                "activity_intensity": 5.0,
                "activity_state": "moderate",
            }
            result = predictor.update("P1", sample)
        
        assert result["trend"] in ["falling", "stable", "recovering"]

    def test_predictor_multiple_players(self, predictor):
        """Test that predictor handles multiple players independently."""
        # Add samples for two players
        for i in range(1, 50):
            sample1 = {
                "timestamp_s": i,
                "hr_bpm": 100.0,
                "hr_source": "synthetic",
                "accel_x_g": 0.0,
                "accel_y_g": 0.0,
                "accel_z_g": 1.0,
                "accel_mag_g": 1.0,
                "activity_intensity": 5.0,
                "activity_state": "moderate",
            }
            result1 = predictor.update("P1", sample1)
            
            sample2 = {
                "timestamp_s": i,
                "hr_bpm": 120.0,
                "hr_source": "synthetic",
                "accel_x_g": 0.0,
                "accel_y_g": 0.0,
                "accel_z_g": 1.0,
                "accel_mag_g": 1.0,
                "activity_intensity": 5.0,
                "activity_state": "moderate",
            }
            result2 = predictor.update("P2", sample2)
        
        # Both should have different stamina values due to different HRs
        assert result1["stamina_now"] != result2["stamina_now"]

    def test_predictor_stamina_in_valid_range(self, predictor):
        """Test that stamina values are in [0, 100] range."""
        for i in range(1, 50):
            sample = {
                "timestamp_s": i,
                "hr_bpm": 100.0,
                "hr_source": "synthetic",
                "accel_x_g": 0.0,
                "accel_y_g": 0.0,
                "accel_z_g": 1.0,
                "accel_mag_g": 1.0,
                "activity_intensity": 5.0,
                "activity_state": "moderate",
            }
            result = predictor.update("P1", sample)
            assert 0 <= result["stamina_now"] <= 100


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_data_pipeline(self):
        """Test loading, engineering, and labeling data."""
        df = load_sensor_csv()
        assert len(df) > 0
        
        feature_df = add_causal_features(df)
        assert len(feature_df) == len(df)
        
        labeled_df = fatigue_label_df(feature_df)
        assert len(labeled_df) == len(df)
        assert "fatigue_index" in labeled_df.columns
        assert "stamina_now" in labeled_df.columns
        assert "state" in labeled_df.columns

    def test_no_data_leakage_across_players(self):
        """Test that features for one player don't depend on another."""
        df = load_sensor_csv()
        feature_df = add_causal_features(df)
        
        # Get features for two players at the same timestamp
        p1_data = feature_df[(feature_df["player_id"] == "P1") & (feature_df["timestamp_s"] == 100)].iloc[0]
        p2_data = feature_df[(feature_df["player_id"] == "P2") & (feature_df["timestamp_s"] == 100)].iloc[0]
        
        # Similar HR should lead to similar reserve fraction
        if p1_data["hr_bpm"] == p2_data["hr_bpm"]:
            assert np.isclose(p1_data["hr_reserve_frac"], p2_data["hr_reserve_frac"])

    def test_feature_timestamp_consistency(self):
        """Test that features are computed for all timestamps properly."""
        df = load_sensor_csv()
        feature_df = add_causal_features(df)
        
        # Each player should have exactly 2820 rows
        for player in ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]:
            player_rows = len(feature_df[feature_df["player_id"] == player])
            assert player_rows == 2820

    def test_all_feature_columns_present(self):
        """Test that all required feature columns are present."""
        from src.stamina.config import FEATURE_COLUMNS
        
        df = load_sensor_csv()
        feature_df = add_causal_features(df)
        
        for col in FEATURE_COLUMNS:
            assert col in feature_df.columns

    def test_export_schema_validity(self):
        """Test that export data has valid schema."""
        df = load_sensor_csv()
        feature_df = add_causal_features(df)
        labeled_df = fatigue_label_df(feature_df)
        
        # Check that export-required columns exist
        required_cols = [
            "timestamp_s",
            "player_id",
            "hr_source",
            "fatigue_index",
            "stamina_now",
            "state",
        ]
        for col in required_cols:
            assert col in labeled_df.columns

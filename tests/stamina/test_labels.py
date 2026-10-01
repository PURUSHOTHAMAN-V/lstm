import pytest
import pandas as pd
import numpy as np

from src.stamina.labels import (
    compute_fatigue_index,
    compute_stamina_now,
    fatigue_state_from_stamina,
    fatigue_label_df,
)
from src.stamina.features import add_causal_features
from src.stamina.data_loader import load_sensor_csv


class TestLabels:
    """Test derived fatigue and stamina labels."""

    @pytest.fixture
    def feature_df(self):
        """Load and engineer features once for all tests."""
        df = load_sensor_csv()
        return add_causal_features(df)

    def test_compute_fatigue_index_in_valid_range(self, feature_df):
        """Test that fatigue_index is between 0 and 100."""
        fatigue = compute_fatigue_index(feature_df)
        assert (fatigue >= 0).all()
        assert (fatigue <= 100).all()

    def test_compute_stamina_now_in_valid_range(self, feature_df):
        """Test that stamina_now is between 0 and 100."""
        stamina = compute_stamina_now(feature_df)
        assert (stamina >= 0).all()
        assert (stamina <= 100).all()

    def test_fatigue_and_stamina_sum_to_100(self, feature_df):
        """Test that fatigue_index + stamina_now = 100."""
        fatigue = compute_fatigue_index(feature_df)
        stamina = compute_stamina_now(feature_df)
        assert np.allclose(fatigue + stamina, 100.0, atol=1e-6)

    def test_fatigue_state_from_stamina_green(self):
        """Test GREEN state is assigned correctly."""
        assert fatigue_state_from_stamina(60) == "GREEN"
        assert fatigue_state_from_stamina(100) == "GREEN"
        assert fatigue_state_from_stamina(70) == "GREEN"

    def test_fatigue_state_from_stamina_yellow(self):
        """Test YELLOW state is assigned correctly."""
        assert fatigue_state_from_stamina(50) == "YELLOW"
        assert fatigue_state_from_stamina(35) == "YELLOW"
        assert fatigue_state_from_stamina(45) == "YELLOW"

    def test_fatigue_state_from_stamina_red(self):
        """Test RED state is assigned correctly."""
        assert fatigue_state_from_stamina(34) == "RED"
        assert fatigue_state_from_stamina(0) == "RED"
        assert fatigue_state_from_stamina(20) == "RED"

    def test_fatigue_state_boundary_60(self):
        """Test boundary at 60 (GREEN threshold)."""
        assert fatigue_state_from_stamina(59.9) == "YELLOW"
        assert fatigue_state_from_stamina(60.0) == "GREEN"
        assert fatigue_state_from_stamina(60.1) == "GREEN"

    def test_fatigue_state_boundary_35(self):
        """Test boundary at 35 (RED threshold)."""
        assert fatigue_state_from_stamina(35.1) == "YELLOW"
        assert fatigue_state_from_stamina(35.0) == "YELLOW"
        assert fatigue_state_from_stamina(34.9) == "RED"

    def test_fatigue_label_df_adds_all_columns(self, feature_df):
        """Test that fatigue_label_df adds required columns."""
        result = fatigue_label_df(feature_df)
        assert "fatigue_index" in result.columns
        assert "stamina_now" in result.columns
        assert "state" in result.columns

    def test_fatigue_label_df_preserves_rows(self, feature_df):
        """Test that fatigue_label_df preserves the number of rows."""
        result = fatigue_label_df(feature_df)
        assert len(result) == len(feature_df)

    def test_fatigue_label_df_states_are_valid(self, feature_df):
        """Test that all assigned states are valid."""
        result = fatigue_label_df(feature_df)
        valid_states = {"GREEN", "YELLOW", "RED"}
        assert set(result["state"].unique()).issubset(valid_states)

    def test_fatigue_deterministic(self, feature_df):
        """Test that fatigue computation is deterministic."""
        fatigue1 = compute_fatigue_index(feature_df)
        fatigue2 = compute_fatigue_index(feature_df)
        assert np.allclose(fatigue1.values, fatigue2.values)

    def test_fatigue_formula_weights(self, feature_df):
        """Test that the fatigue formula uses the expected weighting."""
        # Verify the formula: 100 * (0.5*cum_norm + 0.3*recent_norm + 0.2*hi_norm)
        cum_norm = (feature_df["cum_load"] / 15.0).clip(0, 1)
        recent_norm = (feature_df["recent_load_300s"] / 0.45).clip(0, 1)
        hi_norm = feature_df["hi_frac_300s"].clip(0, 1)
        expected_fatigue = 100.0 * (0.5 * cum_norm + 0.3 * recent_norm + 0.2 * hi_norm)
        actual_fatigue = compute_fatigue_index(feature_df)
        assert np.allclose(expected_fatigue.values, actual_fatigue.values, atol=1e-6)

    def test_stamina_is_inverse_of_fatigue(self, feature_df):
        """Test that stamina_now = 100 - fatigue_index."""
        fatigue = compute_fatigue_index(feature_df)
        stamina = compute_stamina_now(feature_df)
        expected_stamina = 100.0 - fatigue
        assert np.allclose(stamina.values, expected_stamina.values, atol=1e-6)

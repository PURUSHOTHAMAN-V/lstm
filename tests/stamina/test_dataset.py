import pytest
import pandas as pd
import numpy as np

from src.stamina.dataset import (
    time_split,
    make_windows,
    fit_scaler,
    scale_windows,
)
from src.stamina.config import WINDOW, HORIZON, TRAIN_END, VAL_END, TEST_END, FEATURE_COLUMNS
from src.stamina.features import add_causal_features
from src.stamina.labels import fatigue_label_df
from src.stamina.data_loader import load_sensor_csv


class TestDataset:
    """Test dataset creation, splitting, and windowing."""

    @pytest.fixture
    def labeled_df(self):
        """Load, engineer, and label data once for all tests."""
        df = load_sensor_csv()
        feature_df = add_causal_features(df)
        return fatigue_label_df(feature_df)

    def test_time_split_creates_three_splits(self, labeled_df):
        """Test that time_split returns train, val, and test."""
        splits = time_split(labeled_df)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_time_split_no_overlap(self, labeled_df):
        """Test that splits do not overlap in time."""
        splits = time_split(labeled_df)
        train_times = set(splits["train"]["timestamp_s"].unique())
        val_times = set(splits["val"]["timestamp_s"].unique())
        test_times = set(splits["test"]["timestamp_s"].unique())
        assert len(train_times & val_times) == 0
        assert len(val_times & test_times) == 0
        assert len(train_times & test_times) == 0

    def test_time_split_covers_all_time(self, labeled_df):
        """Test that the splits cover all time points."""
        splits = time_split(labeled_df)
        all_times = set(splits["train"]["timestamp_s"].unique()) | \
                    set(splits["val"]["timestamp_s"].unique()) | \
                    set(splits["test"]["timestamp_s"].unique())
        expected_times = set(range(TRAIN_END)) | set(range(TRAIN_END, VAL_END)) | set(range(VAL_END, TEST_END))
        assert all_times == expected_times

    def test_time_split_correct_sizes(self, labeled_df):
        """Test that splits have correct approximate sizes."""
        splits = time_split(labeled_df)
        # Each split has 7 players, so multiply time range by 7
        train_expected = (TRAIN_END - 0) * 7
        val_expected = (VAL_END - TRAIN_END) * 7
        test_expected = (TEST_END - VAL_END) * 7
        assert len(splits["train"]) == train_expected
        assert len(splits["val"]) == val_expected
        assert len(splits["test"]) == test_expected

    def test_make_windows_returns_tuple(self, labeled_df):
        """Test that make_windows returns (X, y, index_pairs) tuple."""
        splits = time_split(labeled_df)
        X, y, pairs = make_windows(splits["train"], FEATURE_COLUMNS)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(pairs, np.ndarray)

    def test_make_windows_shapes(self, labeled_df):
        """Test that windows have correct shapes."""
        splits = time_split(labeled_df)
        X, y, pairs = make_windows(splits["train"], FEATURE_COLUMNS)
        assert X.shape[0] == y.shape[0] == len(pairs)
        assert X.shape[1] == WINDOW  # Sequence length
        assert X.shape[2] == len(FEATURE_COLUMNS)  # Feature dimension
        assert y.shape == (X.shape[0],)

    def test_make_windows_target_scaled_to_0_1(self, labeled_df):
        """Test that targets are scaled to [0, 1] range."""
        splits = time_split(labeled_df)
        X, y, pairs = make_windows(splits["train"], FEATURE_COLUMNS)
        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_make_windows_no_nan(self, labeled_df):
        """Test that windows contain no NaN values."""
        splits = time_split(labeled_df)
        X, y, pairs = make_windows(splits["train"], FEATURE_COLUMNS)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_make_windows_all_finite(self, labeled_df):
        """Test that all window values are finite."""
        splits = time_split(labeled_df)
        X, y, pairs = make_windows(splits["train"], FEATURE_COLUMNS)
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()

    def test_fit_scaler_on_train_only(self, labeled_df):
        """Test that the scaler is fitted on training data."""
        splits = time_split(labeled_df)
        train_X, train_y, _ = make_windows(splits["train"], FEATURE_COLUMNS)
        scaler = fit_scaler(splits["train"], FEATURE_COLUMNS)
        
        # The scaler mean and scale should match the training data statistics
        train_flat = train_X.reshape(-1, train_X.shape[-1])
        assert np.allclose(scaler.mean_, train_flat.mean(axis=0), rtol=1e-4)
        assert np.allclose(scaler.scale_, train_flat.std(axis=0), rtol=1e-4)

    def test_scale_windows_output_shape(self, labeled_df):
        """Test that scale_windows preserves shape."""
        splits = time_split(labeled_df)
        X, y, _ = make_windows(splits["train"], FEATURE_COLUMNS)
        scaler = fit_scaler(splits["train"], FEATURE_COLUMNS)
        X_scaled = scale_windows(X, scaler)
        assert X_scaled.shape == X.shape

    def test_scale_windows_output_standardized(self, labeled_df):
        """Test that scaled windows are approximately standardized."""
        splits = time_split(labeled_df)
        X, y, _ = make_windows(splits["train"], FEATURE_COLUMNS)
        scaler = fit_scaler(splits["train"], FEATURE_COLUMNS)
        X_scaled = scale_windows(X, scaler)
        
        # After scaling on training data, mean should be ~0 and std ~1
        X_scaled_flat = X_scaled.reshape(-1, X_scaled.shape[-1])
        assert np.allclose(X_scaled_flat.mean(axis=0), 0, atol=1e-1)
        assert np.allclose(X_scaled_flat.std(axis=0), 1, atol=1e-1)

    def test_windows_do_not_cross_splits(self, labeled_df):
        """Test that windows and targets do not cross split boundaries."""
        splits = time_split(labeled_df)
        for split_name, split_df in [("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]:
            X, y, pairs = make_windows(split_df, FEATURE_COLUMNS)
            for player_id, start_idx, target_idx in pairs:
                player_df = split_df[split_df["player_id"] == player_id].sort_values("timestamp_s").reset_index(drop=True)
                window_end_idx = start_idx + WINDOW
                # Both the window and target should be within the split
                assert start_idx >= 0
                assert target_idx < len(player_df)
                # The target time should be within the expected range (t + HORIZON)
                assert target_idx == start_idx + WINDOW + HORIZON - 1

    def test_make_windows_indices_are_valid(self, labeled_df):
        """Test that index pairs returned by make_windows are valid."""
        splits = time_split(labeled_df)
        X, y, pairs = make_windows(splits["train"], FEATURE_COLUMNS)
        for player_id, start_idx, target_idx in pairs:
            player_df = splits["train"][splits["train"]["player_id"] == player_id].sort_values("timestamp_s").reset_index(drop=True)
            # Indices should be within bounds
            assert 0 <= start_idx < len(player_df)
            assert 0 <= target_idx < len(player_df)

    def test_val_and_test_have_windows(self, labeled_df):
        """Test that validation and test splits produce windows."""
        splits = time_split(labeled_df)
        val_X, val_y, val_pairs = make_windows(splits["val"], FEATURE_COLUMNS)
        test_X, test_y, test_pairs = make_windows(splits["test"], FEATURE_COLUMNS)
        
        assert len(val_X) > 0, "Validation split produces no windows"
        assert len(test_X) > 0, "Test split produces no windows"

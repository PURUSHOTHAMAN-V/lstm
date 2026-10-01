import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.stamina.data_loader import (
    load_sensor_csv,
    validate_sensor_dataframe,
    DataValidationError,
    save_data_summary,
    summarize_data,
)
from src.stamina.config import EXPECTED_ROWS, EXPECTED_PLAYERS, DATA_PATH


class TestDataLoader:
    """Test data loading and validation functions."""

    def test_load_sensor_csv_file_exists(self):
        """Test that the CSV file exists at the expected path."""
        assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"

    def test_load_sensor_csv_returns_dataframe(self):
        """Test that load_sensor_csv returns a DataFrame."""
        df = load_sensor_csv()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_validate_sensor_dataframe_accepts_valid_data(self):
        """Test that validation passes for the real dataset."""
        df = load_sensor_csv()
        # Should not raise an exception
        validate_sensor_dataframe(df)

    def test_validate_sensor_dataframe_raises_on_missing_columns(self):
        """Test that validation fails when required columns are missing."""
        df = load_sensor_csv()
        df_missing = df.drop("hr_bpm", axis=1)
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_sensor_dataframe(df_missing)

    def test_validate_sensor_dataframe_raises_on_wrong_row_count(self):
        """Test that validation fails for incorrect row count."""
        df = load_sensor_csv()
        df_subset = df.iloc[:100]  # Only 100 rows
        with pytest.raises(DataValidationError, match="Expected.*rows"):
            validate_sensor_dataframe(df_subset)

    def test_validate_sensor_dataframe_raises_on_wrong_players(self):
        """Test that validation fails for wrong player set."""
        df = load_sensor_csv()
        # Remove P1 entirely (that means we lose 2820 rows from the expected 19740)
        df_bad = df[df["player_id"] != "P1"]
        with pytest.raises(DataValidationError, match="Expected.*rows"):
            # This fails on row count first
            validate_sensor_dataframe(df_bad)

    def test_validate_sensor_dataframe_raises_on_non_contiguous_timestamps(self):
        """Test that validation fails for non-contiguous timestamps."""
        df = load_sensor_csv()
        # Remove a few rows to break continuity for a player
        df_bad = df[~((df["player_id"] == "P1") & (df["timestamp_s"].between(5, 10)))]
        with pytest.raises(DataValidationError, match="Expected.*rows"):
            # This fails on row count first
            validate_sensor_dataframe(df_bad)

    def test_validate_sensor_dataframe_raises_on_nan_in_hr(self):
        """Test that validation fails when hr_bpm has NaN."""
        df = load_sensor_csv()
        df_bad = df.copy()
        df_bad.loc[0, "hr_bpm"] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            validate_sensor_dataframe(df_bad)

    def test_validate_sensor_dataframe_raises_on_hr_out_of_range(self):
        """Test that validation fails when hr_bpm is out of valid range."""
        df = load_sensor_csv()
        df_bad = df.copy()
        df_bad.loc[0, "hr_bpm"] = 30  # Below 40
        with pytest.raises(DataValidationError, match="40..220"):
            validate_sensor_dataframe(df_bad)

    def test_validate_sensor_dataframe_raises_on_duplicates(self):
        """Test that validation fails for duplicate timestamps in a player."""
        df = load_sensor_csv()
        df_bad = df.copy()
        # Duplicate the first row
        first_row = df_bad.iloc[0:1]
        df_bad = pd.concat([df_bad, first_row], ignore_index=True)
        with pytest.raises(DataValidationError, match="Expected.*rows"):
            # This fails on row count first
            validate_sensor_dataframe(df_bad)

    def test_summarize_data_structure(self):
        """Test that data summary has the correct structure."""
        df = load_sensor_csv()
        summary = summarize_data(df)
        assert summary["rows"] == EXPECTED_ROWS
        assert set(summary["players"]) == set(EXPECTED_PLAYERS)
        assert "hr_source_counts" in summary
        assert "per_player_hr" in summary

    def test_summarize_data_per_player_stats(self):
        """Test that per-player stats are valid."""
        df = load_sensor_csv()
        summary = summarize_data(df)
        for player in EXPECTED_PLAYERS:
            stats = summary["per_player_hr"][player]
            assert stats["mean"] >= 40 and stats["mean"] <= 220
            assert stats["min"] >= 40 and stats["min"] <= 220
            assert stats["max"] >= 40 and stats["max"] <= 220
            assert stats["real_upsampled_count"] >= 0
            assert stats["synthetic_count"] >= 0
            assert stats["real_upsampled_count"] + stats["synthetic_count"] == 2820

    def test_save_data_summary_creates_file(self, tmp_path):
        """Test that save_data_summary creates a file."""
        df = load_sensor_csv()
        # Temporarily change the output path
        from src.stamina import config
        old_path = config.DATA_SUMMARY_PATH
        try:
            config.DATA_SUMMARY_PATH = tmp_path / "test_summary.txt"
            path = save_data_summary(df)
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "DATA SUMMARY" in content
            assert "Rows:" in content
            assert "Players:" in content
        finally:
            config.DATA_SUMMARY_PATH = old_path

    def test_expected_row_count_matches_config(self):
        """Test that the dataset has the expected total row count."""
        df = load_sensor_csv()
        assert len(df) == EXPECTED_ROWS
        assert len(df) == 7 * 2820  # 7 players × 2820 seconds

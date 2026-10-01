from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stamina.config import PLOTS_DIR


def ensure_plot_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_loss(history):
    ensure_plot_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs validation loss")
    plt.tight_layout()
    path = PLOTS_DIR / "training_validation_loss.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_test_predictions(results_df):
    ensure_plot_dir()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    players = sorted(results_df["player_id"].unique())
    for ax, player in zip(axes.flatten(), players):
        subset = results_df[results_df["player_id"] == player]
        ax.scatter(subset["fatigue_index_true"], subset["fatigue_pred"], s=8)
        ax.set_title(player)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    for extra in range(len(players), len(axes.flatten())):
        axes.flatten()[extra].axis("off")
    plt.tight_layout()
    path = PLOTS_DIR / "test_predictions_by_player.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_stamina_over_time(df):
    ensure_plot_dir()
    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
    for ax, player in zip(axes, sorted(df["player_id"].unique())):
        subset = df[df["player_id"] == player].sort_values("timestamp_s")
        ax.plot(subset["timestamp_s"], subset["stamina_now"], label="stamina_now", alpha=0.8)
        ax.plot(subset["timestamp_s"], subset["stamina_pred_60s"], label="stamina_pred_60s", alpha=0.8)
        ax.set_title(player)
        ax.legend(loc="upper right")
    plt.tight_layout()
    path = PLOTS_DIR / "stamina_timeline_all_players.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_error_by_hr_source(results_df):
    ensure_plot_dir()
    plt.figure(figsize=(7, 5))
    grouped = results_df.groupby("hr_source")["abs_error"].mean()
    grouped.plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.ylabel("MAE fatigue index points")
    plt.title("Mean absolute error by HR source")
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = PLOTS_DIR / "error_by_hr_source.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_model_comparison(summary):
    ensure_plot_dir()
    labels = ["LSTM", "Raw-only", "Persistence", "Ridge"]
    vals = [summary["lstm"], summary["raw_only"], summary["persistence"], summary["ridge"]]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, vals)
    plt.ylabel("MAE")
    plt.title("Model comparison on test set")
    plt.tight_layout()
    path = PLOTS_DIR / "model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)

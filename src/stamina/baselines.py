import numpy as np
from sklearn.linear_model import Ridge


def persistence_baseline(y_true, y_prev):
    return np.asarray(y_true, dtype=float), np.asarray(y_prev, dtype=float)


def ridge_regression_baseline(X_windows, y_true):
    X_flat = np.asarray(X_windows, dtype=float)
    summary = np.concatenate([
        X_flat.mean(axis=1, keepdims=True),
        X_flat[:, -1, :],
        X_flat.min(axis=1),
        X_flat.max(axis=1),
    ], axis=1)
    model = Ridge(alpha=1.0)
    model.fit(summary, y_true)
    return model


def predict_ridge(model, X_windows):
    X_flat = np.asarray(X_windows, dtype=float)
    summary = np.concatenate([
        X_flat.mean(axis=1, keepdims=True),
        X_flat[:, -1, :],
        X_flat.min(axis=1),
        X_flat.max(axis=1),
    ], axis=1)
    return model.predict(summary)
